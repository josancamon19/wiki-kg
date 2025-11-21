"""
Upload filtered Wikipedia data to Hugging Face Hub.

Usage:
    python upload_hf.py --repo_id your-username/dataset-name --wiki enwiki_namespace_0

Environment variables:
    HF_TOKEN: Hugging Face API token (optional, will prompt if not set)
"""

import json
import os
from pathlib import Path

import gcsfs
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

load_dotenv()

# Configuration
GCP_FILTERED_PREFIX = "gs://wikipedia-graph/wikipedia/filtered_html"
CACHE_DIR = Path("./cache_parquet")


def authenticate() -> str:
    """
    Authenticate with Hugging Face Hub.
    
    Returns:
        str: The authenticated username
    """
    from huggingface_hub import whoami
    
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print("✓ Authenticated with HF_TOKEN from environment")
    else:
        login()  # Interactive login
        print("✓ Authenticated interactively")
    
    # Get username
    user_info = whoami()
    username = user_info["name"]
    print(f"✓ Logged in as: {username}")
    return username


def transform_document(data: dict) -> dict:
    """Transform a document to FineWiki format."""
    metadata = data.get("metadata", {})
    return {
        "url": metadata.get("url"),
        "title": metadata.get("title"),
        "text": data["text"],
        "id": data["id"],
        "language_code": metadata.get("in_language"),
        "wikidata_id": metadata.get("wikidata_id"),
        "bytes_html": metadata.get("bytes_html"),
        "wikitext": metadata.get("wikitext", ""),
        "version": metadata.get("version"),
        "infoboxes": json.dumps(metadata.get("infoboxes", [])),
        "has_math": metadata.get("has_math", False),
    }


def download_to_parquet(
    wiki: str,
    output_dir: Path,
    max_docs: int = None,
    max_files: int = None,
) -> list[Path]:
    """
    Download filtered documents from GCS and save as parquet files.
    Memory efficient - processes one file at a time.
    
    Args:
        wiki: Wiki identifier (e.g., 'enwiki_namespace_0')
        output_dir: Local directory to save parquet files
        max_docs: Maximum total documents to download (None for all)
        max_files: Maximum number of GCS files to process (None for all)
    
    Returns:
        List of created parquet file paths
    """
    fs = gcsfs.GCSFileSystem()
    input_path = f"{GCP_FILTERED_PREFIX}/{wiki}".replace("gs://", "")
    gcs_files = sorted(fs.ls(input_path, detail=False))
    
    if max_files:
        gcs_files = gcs_files[:max_files]
    
    print(f"Found {len(gcs_files)} files in gs://{input_path}")
    print(f"Downloading to: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = []
    total_docs = 0
    
    # Define schema for consistent parquet files
    schema = pa.schema([
        ("url", pa.string()),
        ("title", pa.string()),
        ("text", pa.string()),
        ("id", pa.string()),
        ("language_code", pa.string()),
        ("wikidata_id", pa.string()),
        ("bytes_html", pa.int64()),
        ("wikitext", pa.string()),
        ("version", pa.int64()),
        ("infoboxes", pa.string()),
        ("has_math", pa.bool_()),
    ])
    
    for file_idx, gcs_file in enumerate(tqdm(gcs_files, desc="Processing files")):
        if max_docs and total_docs >= max_docs:
            break
        
        # Read documents from this GCS file
        documents = []
        with fs.open(f"gs://{gcs_file}", "r") as f:
            for line in f:
                if max_docs and total_docs >= max_docs:
                    break
                
                data = json.loads(line)
                documents.append(transform_document(data))
                total_docs += 1
        
        if not documents:
            continue
        
        # Save as parquet
        output_file = output_dir / f"part_{file_idx:05d}.parquet"
        table = pa.Table.from_pylist(documents, schema=schema)
        pq.write_table(table, output_file)
        parquet_files.append(output_file)
        
        # Clear memory
        del documents
        del table
    
    print(f"\n✓ Downloaded {total_docs:,} documents to {len(parquet_files)} parquet files")
    return parquet_files


def upload_dataset(
    repo_id: str,
    wiki: str,
    max_docs: int = None,
    max_files: int = None,
    private: bool = False,
    cache_dir: Path = CACHE_DIR,
    skip_download: bool = False,
):
    """
    Upload a wiki dataset to Hugging Face Hub.
    Memory efficient - downloads to parquet first, then uploads.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        wiki: Wiki identifier (e.g., 'enwiki_namespace_0')
        max_docs: Maximum documents to upload (None for all)
        max_files: Maximum GCS files to process (None for all)
        private: Whether to create a private dataset
        cache_dir: Directory for caching parquet files
        skip_download: Skip download if parquet files already exist
    """
    print(f"\n{'='*80}")
    print(f"Uploading {wiki} to {repo_id}")
    print(f"{'='*80}\n")
    
    # Setup cache directory
    wiki_cache_dir = cache_dir / wiki
    
    # Step 1: Download to parquet (or use existing cache)
    if skip_download and wiki_cache_dir.exists():
        print(f"Using cached parquet files from {wiki_cache_dir}")
        parquet_files = sorted(wiki_cache_dir.glob("*.parquet"))
        if not parquet_files:
            print("No parquet files found, downloading...")
            skip_download = False
    
    if not skip_download:
        print("Step 1: Downloading from GCS to local parquet...")
        parquet_files = download_to_parquet(
            wiki=wiki,
            output_dir=wiki_cache_dir,
            max_docs=max_docs,
            max_files=max_files,
        )
    
    # Step 2: Load from parquet (memory efficient)
    print("\nStep 2: Loading dataset from parquet...")
    dataset = load_dataset(
        "parquet",
        data_files=str(wiki_cache_dir / "*.parquet"),
        split="train",
    )
    print(f"✓ Loaded dataset with {len(dataset):,} rows")
    
    # Extract language code for split name
    lang_code = wiki.removesuffix("wiki_namespace_0")
    
    # Step 3: Upload to Hub (streaming mode)
    print(f"\nStep 3: Uploading to {repo_id}...")
    dataset.push_to_hub(
        repo_id,
        split=lang_code,
        private=private,
        max_shard_size="500MB",
    )
    
    print(f"\n✓ Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
    print(f"   Cached parquet files at: {wiki_cache_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload filtered Wikipedia to HF Hub (memory efficient with parquet caching)"
    )
    parser.add_argument("--repo_id", help="HF repo (default: <username>/finewiki)")
    parser.add_argument("--wiki", default="enwiki_namespace_0", help="Wiki identifier (default: enwiki_namespace_0)")
    parser.add_argument("--max_docs", type=int, help="Max documents (for testing)")
    parser.add_argument("--max_files", type=int, help="Max GCS files to process (for testing)")
    parser.add_argument("--private", action="store_true", help="Create private dataset")
    parser.add_argument("--cache_dir", type=Path, default=CACHE_DIR, help="Local cache directory for parquet files")
    parser.add_argument("--skip_download", action="store_true", help="Skip download if parquet cache exists")
    parser.add_argument("--clean_cache", action="store_true", help="Delete cache after successful upload")
    
    args = parser.parse_args()
    
    # Authenticate and get username
    username = authenticate()
    
    # Set default repo_id if not provided
    repo_id = args.repo_id or f"{username}/finewiki"
    
    print(f"Target repository: {repo_id}")
    print(f"Source wiki: {args.wiki}")
    print(f"Cache directory: {args.cache_dir}")
    
    # Upload
    try:
        upload_dataset(
            repo_id=repo_id,
            wiki=args.wiki,
            max_docs=args.max_docs,
            max_files=args.max_files,
            private=args.private,
            cache_dir=args.cache_dir,
            skip_download=args.skip_download,
        )
        
        # Clean cache if requested
        if args.clean_cache:
            import shutil
            wiki_cache_dir = args.cache_dir / args.wiki
            if wiki_cache_dir.exists():
                print(f"\nCleaning cache: {wiki_cache_dir}")
                shutil.rmtree(wiki_cache_dir)
                print("✓ Cache cleaned")
    
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print(f"Parquet cache preserved at: {args.cache_dir / args.wiki}")
        raise

