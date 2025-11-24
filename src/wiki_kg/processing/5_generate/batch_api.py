import os
import json
import logging
import argparse

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
RESULTS_BASE_PATH = os.path.join(os.path.dirname(__file__), "results")


def get_batch_paths(batch_type: str) -> dict:
    """
    Get the standard file paths for a batch type.
    
    Args:
        batch_type: Type of batch (e.g., 'entities', 'relations')
        
    Returns:
        Dictionary with paths for batch files
    """
    batch_dir = os.path.join(RESULTS_BASE_PATH, batch_type)
    return {
        "dir": batch_dir,
        "batch_file": os.path.join(batch_dir, "batch.jsonl"),
        "info_file": os.path.join(batch_dir, "batch_info.json"),
        "results_file": os.path.join(batch_dir, "batch_results.jsonl"),
    }


def upload_batch(batch_type: str, force: bool = False) -> dict:
    """
    Upload a batch file to OpenAI and create a batch job.
    
    Args:
        batch_type: Type of batch (e.g., 'entities', 'relations')
        force: If True, upload even if batch info already exists
        
    Returns:
        Dictionary containing batch information
    """
    paths = get_batch_paths(batch_type)
    
    if not os.path.exists(paths["batch_file"]):
        raise FileNotFoundError(
            f"Batch file not found: {paths['batch_file']}\n"
            f"Run the generation script first to create the batch file."
        )
    
    # Check if batch info already exists
    if os.path.exists(paths["info_file"]) and not force:
        logger.info(f"Batch info already exists for '{batch_type}'")
        
        # Load and check existing batch status
        with open(paths["info_file"], "r") as f:
            existing_info = json.load(f)
        
        batch_id = existing_info.get("batch_id")
        logger.info(f"Existing batch ID: {batch_id}")
        logger.info("Checking current status...")
        
        try:
            batch = client.batches.retrieve(batch_id)
            logger.info(f"Current status: {batch.status}")
            
            if batch.status in ["validating", "in_progress", "finalizing", "completed"]:
                logger.info("Batch is active or completed. Skipping upload.")
                logger.info(f"Use: python _2_upload_batch.py upload {batch_type} --force")
                return existing_info
            else:
                logger.info(
                    f"Previous batch has status '{batch.status}'. Uploading new batch..."
                )
        except Exception as e:
            logger.warning(f"Could not retrieve existing batch: {e}")
            logger.info("Uploading new batch...")
    
    # Get file stats
    file_size = os.path.getsize(paths["batch_file"])
    with open(paths["batch_file"], "r") as f:
        num_requests = sum(1 for _ in f)
    
    logger.info(f"Uploading batch: {batch_type}")
    logger.info(f"File: {paths['batch_file']}")
    logger.info(f"File size: {file_size / 1024 / 1024:.2f} MB")
    logger.info(f"Number of requests: {num_requests}")
    
    # Step 1: Upload the file
    logger.info("Uploading file to OpenAI...")
    with open(paths["batch_file"], "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")
    
    logger.info(f"File uploaded successfully! File ID: {batch_input_file.id}")
    
    # Step 2: Create the batch job
    logger.info("Creating batch job...")
    description = f"{batch_type.capitalize()} extraction batch ({num_requests} requests)"
    
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "batch_type": batch_type,
            "description": description,
            "num_requests": str(num_requests),
        },
    )
    
    logger.info("=" * 60)
    logger.info("Batch created successfully!")
    logger.info(f"Batch Type: {batch_type}")
    logger.info(f"Batch ID: {batch.id}")
    logger.info(f"Status: {batch.status}")
    logger.info(f"Created at: {batch.created_at}")
    logger.info("=" * 60)
    
    # Save batch info
    batch_info = {
        "batch_type": batch_type,
        "batch_id": batch.id,
        "input_file_id": batch_input_file.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "num_requests": num_requests,
        "description": description,
    }
    
    with open(paths["info_file"], "w") as f:
        json.dump(batch_info, f, indent=2)
    logger.info(f"Batch info saved to: {paths['info_file']}")
    
    # Check initial status
    logger.info("\nChecking current status...")
    batch_status = client.batches.retrieve(batch.id)
    logger.info(f"Status: {batch_status.status}")
    logger.info(f"Request counts: {batch_status.request_counts}")
    
    return batch_info


def check_status(batch_type: str) -> dict:
    """
    Check the status of a batch job by batch type.
    
    Args:
        batch_type: Type of batch (e.g., 'entities', 'relations')
        
    Returns:
        Dictionary with batch status information
    """
    paths = get_batch_paths(batch_type)
    
    if not os.path.exists(paths["info_file"]):
        raise FileNotFoundError(
            f"Batch info not found for '{batch_type}'\n"
            f"Upload a batch first using: python _2_upload_batch.py upload {batch_type}"
        )
    
    # Load batch info
    with open(paths["info_file"], "r") as f:
        batch_info = json.load(f)
    
    batch_id = batch_info["batch_id"]
    
    logger.info(f"Checking status for batch type: {batch_type}")
    logger.info(f"Batch ID: {batch_id}")
    
    batch = client.batches.retrieve(batch_id)
    
    logger.info("=" * 60)
    logger.info(f"Batch Type: {batch_type}")
    logger.info(f"Batch ID: {batch.id}")
    logger.info(f"Status: {batch.status}")
    logger.info(f"Created at: {batch.created_at}")
    logger.info(f"Request counts: {batch.request_counts}")
    
    if batch.status == "completed":
        logger.info("✓ Batch completed!")
        logger.info(f"Output file ID: {batch.output_file_id}")
        logger.info(f"Download with: python _2_upload_batch.py download {batch_type}")
        if batch.error_file_id:
            logger.info(f"⚠ Error file ID: {batch.error_file_id}")
    elif batch.status in ["validating", "in_progress", "finalizing"]:
        logger.info(f"⏳ Batch is {batch.status}...")
    elif batch.status == "failed":
        logger.info("✗ Batch failed!")
        logger.info(f"Errors: {batch.errors}")
    
    logger.info("=" * 60)
    
    return {
        "batch_type": batch_type,
        "batch_id": batch.id,
        "status": batch.status,
        "request_counts": batch.request_counts,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }


def download_results(batch_type: str) -> None:
    """
    Download the results of a completed batch job.
    
    Args:
        batch_type: Type of batch (e.g., 'entities', 'relations')
    """
    paths = get_batch_paths(batch_type)
    
    if not os.path.exists(paths["info_file"]):
        raise FileNotFoundError(
            f"Batch info not found for '{batch_type}'\n"
            f"Upload a batch first using: python _2_upload_batch.py upload {batch_type}"
        )
    
    # Load batch info
    with open(paths["info_file"], "r") as f:
        batch_info = json.load(f)
    
    batch_id = batch_info["batch_id"]
    
    logger.info(f"Retrieving batch: {batch_type}")
    logger.info(f"Batch ID: {batch_id}")
    
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        logger.warning(f"Batch is not completed yet. Current status: {batch.status}")
        logger.info("Check status with: python _2_upload_batch.py status {batch_type}")
        return
    
    # Download output file
    if batch.output_file_id:
        logger.info(f"Downloading output file: {batch.output_file_id}")
        output_content = client.files.content(batch.output_file_id)
        
        with open(paths["results_file"], "wb") as f:
            f.write(output_content.content)
        logger.info(f"✓ Results saved to: {paths['results_file']}")
    
    # Download error file if it exists
    if batch.error_file_id:
        logger.info(f"Downloading error file: {batch.error_file_id}")
        error_content = client.files.content(batch.error_file_id)
        
        error_path = paths["results_file"].replace(".jsonl", "_errors.jsonl")
        with open(error_path, "wb") as f:
            f.write(error_content.content)
        logger.info(f"⚠ Errors saved to: {error_path}")
    
    logger.info("=" * 60)
    logger.info(f"Download complete for '{batch_type}'!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Manage OpenAI Batch API jobs by batch type (entities, relations, etc.)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Upload command
    upload_parser = subparsers.add_parser(
        "upload", 
        help="Upload a batch file (e.g., python _2_upload_batch.py upload entities)"
    )
    upload_parser.add_argument(
        "batch_type",
        type=str,
        help="Type of batch to upload (e.g., 'entities', 'relations')"
    )
    upload_parser.add_argument(
        "--force",
        action="store_true",
        help="Force upload even if batch already exists",
    )
    
    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Check batch status (e.g., python _2_upload_batch.py status entities)"
    )
    status_parser.add_argument(
        "batch_type",
        type=str,
        help="Type of batch to check (e.g., 'entities', 'relations')"
    )
    
    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download batch results (e.g., python _2_upload_batch.py download entities)"
    )
    download_parser.add_argument(
        "batch_type",
        type=str,
        help="Type of batch to download (e.g., 'entities', 'relations')"
    )
    
    args = parser.parse_args()
    
    if args.command == "upload":
        upload_batch(
            batch_type=args.batch_type,
            force=getattr(args, "force", False)
        )
    elif args.command == "status":
        check_status(args.batch_type)
    elif args.command == "download":
        download_results(args.batch_type)
        # TODO: add an option to identify the failed requests, and save them in a separate file of request and another of error messages, to either just re-run, or know what to change
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
