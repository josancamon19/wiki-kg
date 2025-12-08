import os
import json
import logging
from pathlib import Path
from typing import Optional, Annotated

import gcsfs
import typer
from openai import OpenAI
from dotenv import load_dotenv

try:
    from .utils import (
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
        get_batch_paths,
    )
except ImportError:
    from utils import (
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
        get_batch_paths,
    )

# Load environment variables
load_dotenv()

# Configure logging
HERE = Path(__file__).resolve().parent
LOG_FILE = HERE / "generation.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = typer.Typer()


def upload_batch(
    batch_type: str,
    wiki: str = "enwiki",
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    limit: Optional[int] = None,
    force: bool = False,
) -> dict:
    """
    Upload a batch file from GCS to OpenAI and create a batch job.

    Args:
        batch_type: Type of batch (e.g., 'entities', 'relations')
        wiki: Wiki identifier (default: 'enwiki')
        model: Model name used for generation
        reasoning_effort: Reasoning effort level used for generation
        limit: Optional limit used during generation
        force: If True, upload even if batch info already exists

    Returns:
        Dictionary containing batch information
    """
    fs = gcsfs.GCSFileSystem()
    paths = get_batch_paths(batch_type, wiki, model, reasoning_effort, limit)

    if not fs.exists(paths["batch_file"]):
        raise FileNotFoundError(
            f"Batch file not found: {paths['batch_file']}\n"
            f"Run the generation script first to create the batch file."
        )

    # Check if batch info already exists
    if fs.exists(paths["info_file"]) and not force:
        logger.info(f"Batch info already exists for '{batch_type}' in wiki '{wiki}'")

        # Load and check existing batch status
        with fs.open(paths["info_file"], "r") as f:
            existing_info = json.load(f)

        batch_id = existing_info.get("batch_id")
        logger.info(f"Existing batch ID: {batch_id}")
        logger.info("Checking current status...")

        try:
            batch = client.batches.retrieve(batch_id)
            logger.info(f"Current status: {batch.status}")

            if batch.status in ["validating", "in_progress", "finalizing", "completed"]:
                logger.info("Batch is active or completed. Skipping upload.")
                logger.info(
                    f"Use: python _2_upload_batch.py upload {batch_type} --force"
                )
                return existing_info
            else:
                logger.info(
                    f"Previous batch has status '{batch.status}'. Uploading new batch..."
                )
        except Exception as e:
            logger.warning(f"Could not retrieve existing batch: {e}")
            logger.info("Uploading new batch...")

    # Get file stats from GCS
    file_info = fs.info(paths["batch_file"])
    file_size = file_info.get("size", 0)

    with fs.open(paths["batch_file"], "r") as f:
        num_requests = sum(1 for _ in f)

    logger.info(f"Uploading batch: {batch_type} (wiki: {wiki})")
    logger.info(f"File: {paths['batch_file']}")
    logger.info(f"File size: {file_size / 1024 / 1024:.2f} MB")
    logger.info(f"Number of requests: {num_requests}")

    # Step 1: Download from GCS and upload to OpenAI
    logger.info("Downloading from GCS and uploading to OpenAI...")
    with fs.open(paths["batch_file"], "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    logger.info(f"File uploaded successfully! File ID: {batch_input_file.id}")

    # Step 2: Create the batch job
    logger.info("Creating batch job...")
    description = (
        f"{batch_type.capitalize()} extraction batch ({num_requests} requests)"
    )

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

    # Save batch info to GCS
    batch_info = {
        "batch_type": batch_type,
        "wiki": wiki,
        "batch_id": batch.id,
        "input_file_id": batch_input_file.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "num_requests": num_requests,
        "description": description,
    }

    with fs.open(paths["info_file"], "w") as f:
        f.write(json.dumps(batch_info, indent=2))
    logger.info(f"Batch info saved to: {paths['info_file']}")

    # Check initial status
    logger.info("\nChecking current status...")
    batch_status = client.batches.retrieve(batch.id)
    logger.info(f"Status: {batch_status.status}")
    logger.info(f"Request counts: {batch_status.request_counts}")

    return batch_info


def check_status(
    batch_type: str,
    wiki: str = "enwiki",
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    limit: Optional[int] = None,
) -> dict:
    """
    Check the status of a batch job by batch type.

    Args:
        batch_type: Type of batch (e.g., 'entities', 'relations')
        wiki: Wiki identifier (default: 'enwiki')
        model: Model name used for generation
        reasoning_effort: Reasoning effort level used for generation
        limit: Optional limit used during generation

    Returns:
        Dictionary with batch status information
    """
    fs = gcsfs.GCSFileSystem()
    paths = get_batch_paths(batch_type, wiki, model, reasoning_effort, limit)

    if not fs.exists(paths["info_file"]):
        raise FileNotFoundError(
            f"Batch info not found for '{batch_type}' in wiki '{wiki}'\n"
            f"Upload a batch first using: python batch_api.py upload {batch_type} --wiki {wiki}"
        )

    # Load batch info from GCS
    with fs.open(paths["info_file"], "r") as f:
        batch_info = json.load(f)

    batch_id = batch_info["batch_id"]

    logger.info(f"Checking status for batch type: {batch_type} (wiki: {wiki})")
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
        logger.info(
            f"Download with: python batch_api.py download {batch_type} --wiki {wiki}"
        )
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


def download_results(
    batch_type: str,
    wiki: str = "enwiki",
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    limit: Optional[int] = None,
) -> None:
    """
    Download the results of a completed batch job and save to GCS.

    Args:
        batch_type: Type of batch (e.g., 'entities', 'relations')
        wiki: Wiki identifier (default: 'enwiki')
        model: Model name used for generation
        reasoning_effort: Reasoning effort level used for generation
        limit: Optional limit used during generation
    """
    fs = gcsfs.GCSFileSystem()
    paths = get_batch_paths(batch_type, wiki, model, reasoning_effort, limit)

    if not fs.exists(paths["info_file"]):
        raise FileNotFoundError(
            f"Batch info not found for '{batch_type}' in wiki '{wiki}'\n"
            f"Upload a batch first using: python batch_api.py upload {batch_type} --wiki {wiki}"
        )

    # Load batch info from GCS
    with fs.open(paths["info_file"], "r") as f:
        batch_info = json.load(f)

    batch_id = batch_info["batch_id"]

    logger.info(f"Retrieving batch: {batch_type} (wiki: {wiki})")
    logger.info(f"Batch ID: {batch_id}")

    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        logger.warning(f"Batch is not completed yet. Current status: {batch.status}")
        logger.info(
            f"Check status with: python batch_api.py status {batch_type} --wiki {wiki}"
        )
        return

    # Download output file and save to GCS
    if batch.output_file_id:
        logger.info(f"Downloading output file: {batch.output_file_id}")
        output_content = client.files.content(batch.output_file_id)

        with fs.open(paths["results_file"], "wb") as f:
            f.write(output_content.content)
        logger.info(f"✓ Results saved to: {paths['results_file']}")

    # Download error file if it exists and save to GCS
    if batch.error_file_id:
        logger.info(f"Downloading error file: {batch.error_file_id}")
        error_content = client.files.content(batch.error_file_id)

        error_path = paths["results_file"].replace(".jsonl", "_errors.jsonl")
        with fs.open(error_path, "wb") as f:
            f.write(error_content.content)
        logger.info(f"⚠ Errors saved to: {error_path}")

    logger.info("=" * 60)
    logger.info(f"Download complete for '{batch_type}'!")
    logger.info("=" * 60)


@app.command()
def upload(
    batch_type: Annotated[
        str, typer.Argument(help="Type of batch (e.g., 'entities', 'relations')")
    ],
    wiki: Annotated[str, typer.Option(help="Wiki identifier")] = "enwiki",
    model: Annotated[str, typer.Option(help="Model name")] = DEFAULT_MODEL,
    reasoning_effort: Annotated[
        str, typer.Option(help="Reasoning effort level")
    ] = DEFAULT_REASONING_EFFORT,
    limit: Annotated[
        Optional[int], typer.Option(help="Limit used during generation")
    ] = None,
    force: Annotated[
        bool, typer.Option(help="Force upload even if batch exists")
    ] = False,
):
    """Upload a batch file from GCS to OpenAI."""
    logger.info(f"Script: {__file__}")
    logger.info("Command: upload")
    logger.info(
        f"Arguments: batch_type={batch_type}, wiki={wiki}, model={model}, "
        f"reasoning_effort={reasoning_effort}, limit={limit}, force={force}"
    )
    upload_batch(
        batch_type=batch_type,
        wiki=wiki,
        model=model,
        reasoning_effort=reasoning_effort,
        limit=limit,
        force=force,
    )


@app.command()
def status(
    batch_type: Annotated[
        str, typer.Argument(help="Type of batch (e.g., 'entities', 'relations')")
    ],
    wiki: Annotated[str, typer.Option(help="Wiki identifier")] = "enwiki",
    model: Annotated[str, typer.Option(help="Model name")] = DEFAULT_MODEL,
    reasoning_effort: Annotated[
        str, typer.Option(help="Reasoning effort level")
    ] = DEFAULT_REASONING_EFFORT,
    limit: Annotated[
        Optional[int], typer.Option(help="Limit used during generation")
    ] = None,
):
    """Check the status of a batch job."""
    logger.info(f"Script: {__file__}")
    logger.info("Command: status")
    logger.info(
        f"Arguments: batch_type={batch_type}, wiki={wiki}, model={model}, "
        f"reasoning_effort={reasoning_effort}, limit={limit}"
    )
    check_status(
        batch_type=batch_type,
        wiki=wiki,
        model=model,
        reasoning_effort=reasoning_effort,
        limit=limit,
    )


@app.command()
def download(
    batch_type: Annotated[
        str, typer.Argument(help="Type of batch (e.g., 'entities', 'relations')")
    ],
    wiki: Annotated[str, typer.Option(help="Wiki identifier")] = "enwiki",
    model: Annotated[str, typer.Option(help="Model name")] = DEFAULT_MODEL,
    reasoning_effort: Annotated[
        str, typer.Option(help="Reasoning effort level")
    ] = DEFAULT_REASONING_EFFORT,
    limit: Annotated[
        Optional[int], typer.Option(help="Limit used during generation")
    ] = None,
):
    """Download batch results from OpenAI to GCS."""
    logger.info(f"Script: {__file__}")
    logger.info("Command: download")
    logger.info(
        f"Arguments: batch_type={batch_type}, wiki={wiki}, model={model}, "
        f"reasoning_effort={reasoning_effort}, limit={limit}"
    )
    download_results(
        batch_type=batch_type,
        wiki=wiki,
        model=model,
        reasoning_effort=reasoning_effort,
        limit=limit,
    )
    # TODO: add an option to identify the failed requests, and save them in a separate file of request and another of error messages, to either just re-run, or know what to change


if __name__ == "__main__":
    app()
