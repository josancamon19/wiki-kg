"""
Parse relations from batch API results.

Reads the batch_results.jsonl file and extracts the relations list from successful responses.
Saves to a simplified JSONL format for downstream processing.
"""

import json
import logging
from typing import Optional, Annotated
from pathlib import Path

import gcsfs
import typer
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

try:
    from .utils import (
        GCP_KG_PREFIX,
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
        build_subdir,
    )
except ImportError:
    from utils import (
        GCP_KG_PREFIX,
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
        build_subdir,
    )


class RelationItem(BaseModel):
    """A single relation triple."""
    subject: str
    predicate: str
    object: str


class RelationsResponse(BaseModel):
    """Structured response for relation extraction."""
    relations: list[RelationItem]

load_dotenv()

# Logging
HERE = Path(__file__).resolve().parent
LOG_FILE = HERE / "generation.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = typer.Typer()


def parse_batch_results(input_file: str, output_file: str, fs: gcsfs.GCSFileSystem):
    """Parse batch results and extract relations."""
    output_dir = "/".join(output_file.replace("gs://", "").split("/")[:-1])
    fs.makedirs(output_dir, exist_ok=True)

    # Create failed IDs file path
    failed_file = output_file.replace("parsed.jsonl", "failed_ids.jsonl")

    successful = 0
    failed = 0
    parse_errors = 0

    with fs.open(input_file, "r") as f_in, fs.open(output_file, "w") as f_out, fs.open(failed_file, "w") as f_failed:
        for line_num, line in enumerate(f_in, 1):
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id")
                response = result.get("response", {})
                status_code = response.get("status_code")

                if status_code != 200:
                    failed += 1
                    logger.warning(f"Line {line_num}: Failed request for {custom_id}, status: {status_code}")
                    f_failed.write(json.dumps({"custom_id": custom_id, "reason": "failed_request", "status_code": status_code}) + "\n")
                    continue

                body = response.get("body", {})
                output = body.get("output", [])

                # Get the message content from the last output item (following kg_gen pattern)
                message_content = None
                if output:
                    last_output = output[-1]
                    content = last_output.get("content", [])
                    if content:
                        message_content = content[0].get("text")

                if not message_content:
                    parse_errors += 1
                    logger.warning(f"Line {line_num}: No message content found for {custom_id}")
                    f_failed.write(json.dumps({"custom_id": custom_id, "reason": "no_message_content"}) + "\n")
                    continue

                # Parse relations using Pydantic validation (following kg_gen pattern)
                try:
                    parsed = RelationsResponse.model_validate_json(message_content)
                    relations = [
                        {"subject": r.subject, "predicate": r.predicate, "object": r.object}
                        for r in parsed.relations
                    ]
                except ValidationError as e:
                    parse_errors += 1
                    logger.warning(f"Line {line_num}: Failed to validate relations for {custom_id}: {e}")
                    f_failed.write(json.dumps({"custom_id": custom_id, "reason": "validation_error", "error": str(e)}) + "\n")
                    continue

                output_data = {"custom_id": custom_id, "relations": relations}
                f_out.write(json.dumps(output_data) + "\n")
                successful += 1

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error: {e}")
                parse_errors += 1
                # Try to get custom_id if possible
                try:
                    partial_result = json.loads(line) if line else {}
                    custom_id = partial_result.get("custom_id", f"line_{line_num}")
                except Exception:
                    custom_id = f"line_{line_num}"
                f_failed.write(json.dumps({"custom_id": custom_id, "reason": "json_decode_error", "error": str(e)}) + "\n")
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error: {e}")
                parse_errors += 1
                # Try to get custom_id if possible
                try:
                    partial_result = json.loads(line) if line else {}
                    custom_id = partial_result.get("custom_id", f"line_{line_num}")
                except Exception:
                    custom_id = f"line_{line_num}"
                f_failed.write(json.dumps({"custom_id": custom_id, "reason": "unexpected_error", "error": str(e)}) + "\n")

    logger.info("=" * 80)
    logger.info("Parsing complete!")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed requests: {failed}")
    logger.info(f"  Parse errors: {parse_errors}")
    logger.info(f"  Total processed: {successful + failed + parse_errors}")
    logger.info(f"  Output saved to: {output_file}")
    logger.info(f"  Failed IDs saved to: {failed_file}")
    logger.info("=" * 80)


@app.command()
def main(
    wiki: Annotated[str, typer.Option(help="Wiki identifier")] = "enwiki",
    model: Annotated[str, typer.Option(help="Model name")] = DEFAULT_MODEL,
    reasoning_effort: Annotated[str, typer.Option(help="Reasoning effort")] = DEFAULT_REASONING_EFFORT,
    limit: Annotated[Optional[int], typer.Option(help="Limit used during generation")] = None,
    input_file: Annotated[Optional[str], typer.Option(help="GCS path to batch_results.jsonl")] = None,
    output_file: Annotated[Optional[str], typer.Option(help="GCS path to save parsed relations")] = None,
):
    """Parse relations from batch API results."""
    fs = gcsfs.GCSFileSystem()
    subdir = build_subdir(model, reasoning_effort, limit)

    resolved_input = input_file or f"{GCP_KG_PREFIX}/{wiki}/relations/{subdir}/batch_results.jsonl"
    resolved_output = output_file or f"{GCP_KG_PREFIX}/{wiki}/relations/{subdir}/parsed.jsonl"

    if not fs.exists(resolved_input):
        logger.error(f"Error: Input file not found: {resolved_input}")
        raise typer.Exit(1)

    logger.info(f"Script: {__file__}")
    logger.info(f"Args: wiki={wiki}, model={model}, reasoning_effort={reasoning_effort}, limit={limit}")
    logger.info("=" * 80)
    logger.info(f"Parsing relations from: {resolved_input}")
    logger.info(f"Output to: {resolved_output}")
    logger.info("=" * 80)

    parse_batch_results(resolved_input, resolved_output, fs)


if __name__ == "__main__":
    app()
