import json
import os
import argparse
from pathlib import Path
import requests
from tqdm.auto import tqdm
from datatrove.io import get_datafolder
from dotenv import load_dotenv

load_dotenv()

AUTH_BASE = "https://auth.enterprise.wikimedia.com/v1"
API_BASE = "https://api.enterprise.wikimedia.com/v2"

# Configuration constants
GCP_DESTINATION = "gs://wikipedia-graph/wikipedia/raw_html_dumps"

# Files are relative to this script directory
HERE = Path(__file__).resolve().parent
SNAPSHOTS_PATH = HERE / "available_snapshots.json"
TOKEN_PATH = HERE / "token.json"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Download Wikipedia snapshots to GCP')
parser.add_argument('--chunks', type=str, help='Comma-separated list of chunk indices to download (e.g., "125,176,47"). If not provided, all chunks will be downloaded.')
parser.add_argument('--wiki', type=str, default='enwiki_namespace_0', help='Wiki identifier (default: enwiki_namespace_0)')
parser.add_argument('--force', action='store_true', help='Force re-download even if file exists in GCP')
args = parser.parse_args()

# Parse chunk indices if provided
specific_chunks = None
if args.chunks:
    specific_chunks = [int(idx.strip()) for idx in args.chunks.split(',')]
    print(f"Will download specific chunks: {specific_chunks}")
else:
    print("Will download all chunks")

if args.force:
    print("Force mode: Will re-download existing files")

with SNAPSHOTS_PATH.open() as f:
    snapshots = json.load(f)


def load_access_token():
    """Load access token from file if present; otherwise login."""
    if TOKEN_PATH.exists():
        try:
            tok = json.loads(TOKEN_PATH.read_text())
            return tok["access_token"]
        except Exception:
            pass


def auth_headers():
    return {
        "Authorization": f"Bearer {load_access_token()}",
        "Accept": "application/json",
    }


out_df = get_datafolder(GCP_DESTINATION)

downloaded_count = 0
skipped_count = 0

for wiki in snapshots:
    # if any(wiki['is_part_of']['identifier'].endswith(y) for y in ['wikibooks', 'wiktionary', 'wikiquote', 'wikivoyage', 'wikiversity', 'wikisource', 'wikinews']):
    #   continue

    print(f"\nProcessing wiki: {wiki['identifier']}")
    print(f"Total chunks available: {len(wiki['chunks'])}")
    
    # Determine which chunks to process
    chunks_to_process = specific_chunks if specific_chunks else range(len(wiki["chunks"]))
    
    for chunk_idx in chunks_to_process:
        if chunk_idx >= len(wiki["chunks"]):
            print(f"Warning: Chunk index {chunk_idx} out of range (max: {len(wiki['chunks'])-1}), skipping")
            continue
            
        url = f"{API_BASE}/snapshots/{wiki['identifier']}/chunks/{wiki['chunks'][chunk_idx]}/download"
        filename = f"{wiki['identifier']}_{wiki['chunks'][chunk_idx]}.json.tar.gz"
        out_path = wiki["identifier"] + "/" + filename

        if out_df.exists(out_path) and not args.force:
            print(f"Skipping existing file: {out_path}")
            skipped_count += 1
            continue

        print(f"\n{'='*80}")
        print(f"Downloading chunk {chunk_idx}/{len(wiki['chunks'])-1}: {wiki['chunks'][chunk_idx]}")
        
        headers = auth_headers().copy()
        headers["Accept"] = "*/*"  # ensure binary ok

        try:
            with requests.get(url, headers=headers, stream=True, timeout=600) as r:
                r.raise_for_status()

                chunk_size = 20 * 1 << 20  # 20 MiB
                total_header = r.headers.get("Content-Length")
                total_bytes = (
                    int(total_header) if total_header and total_header.isdigit() else None
                )

                with (
                    out_df.open(out_path, "wb") as f,
                    tqdm(
                        total=total_bytes,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=filename,
                        dynamic_ncols=True,
                    ) as pbar,
                ):
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

                print(f"✓ Saved → {out_path} ({total_bytes / (1024*1024):.2f} MB)")
                downloaded_count += 1
        except Exception as e:
            print(f"✗ Error downloading chunk {chunk_idx}: {e}")
            continue

print(f"\n{'='*80}")
print(f"SUMMARY:")
print(f"  Downloaded: {downloaded_count}")
print(f"  Skipped: {skipped_count}")
print(f"{'='*80}")
