import json
from pathlib import Path
import requests
from tqdm.auto import tqdm
from datatrove.io import get_datafolder

AUTH_BASE = "https://auth.enterprise.wikimedia.com/v1"
API_BASE = "https://api.enterprise.wikimedia.com/v2"

# Configuration constants
S3_DESTINATION = "s3://wikipedia-bucket/wikipedia/raw_html_dumps/"

# Files are relative to this script directory
HERE = Path(__file__).resolve().parent
SNAPSHOTS_PATH = HERE / "available_snapshots.json"
TOKEN_PATH = HERE / "wm_enterprise_token.json"

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


out_df = get_datafolder(S3_DESTINATION)

for wiki in snapshots:
    # if any(wiki['is_part_of']['identifier'].endswith(y) for y in ['wikibooks', 'wiktionary', 'wikiquote', 'wikivoyage', 'wikiversity', 'wikisource', 'wikinews']):
    # continue

    for chunk_idx in range(len(wiki["chunks"])):
        url = f"{API_BASE}/snapshots/{wiki['identifier']}/chunks/{wiki['chunks'][chunk_idx]}/download"
        filename = f"{wiki['identifier']}_{wiki['chunks'][chunk_idx]}.json.tar.gz"
        out_path = wiki["identifier"] + "/" + filename

        if out_df.exists(out_path):
            continue

        headers = auth_headers().copy()
        headers["Accept"] = "*/*"  # ensure binary ok

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

        print(f"Saved → {out_path} ({total_bytes} bytes)")
