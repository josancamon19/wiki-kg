"""
Parse entities from batch API results.

Reads the batch_results.jsonl file and extracts the entities list from successful responses.
Saves to a simplified JSONL format for downstream processing.
Uses Google Cloud Storage for file operations.
"""

import json
import re
import logging
from typing import Optional, Annotated
from pathlib import Path

import gcsfs
import typer
from dotenv import load_dotenv

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

app = typer.Typer()


def extract_entities_from_text(text: str) -> list[str] | None:
    """
    Extract the entities list from the response text.

    The text format is:
    [[ ## entities ## ]]
    ["entity1", "entity2", ...]
    [[ ## completed ## ]]
    """
    # Find the content between markers
    pattern = r"\[\[ ## entities ## \]\]\s*\n?(.*?)\s*\n?\[\[ ## completed ## \]\]"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    entities_text = match.group(1).strip()

    # Remove any trailing comments (like "# note: the value you produce...")
    entities_text = re.sub(r"\s*#.*$", "", entities_text, flags=re.MULTILINE)
    entities_text = entities_text.strip()

    # Remove any trailing whitespace or newlines within the JSON
    entities_text = " ".join(entities_text.split())

    # Fix question marks outside quotes (e.g., "text"? -> "text")
    entities_text = re.sub(r'"(\?+)', r'"', entities_text)

    # Fix common malformations at the end
    # Remove trailing characters like ], ), }, etc. and rebuild proper ending
    entities_text = entities_text.rstrip("])}\t ")

    # Sometimes the model outputs wrong closing brackets
    # Make sure it ends with ]
    if not entities_text.endswith("]"):
        entities_text += "]"

    try:
        # Parse the JSON array
        entities = json.loads(entities_text)
        return entities if isinstance(entities, list) else None
    except json.JSONDecodeError as e:
        print(f"Failed to parse entities JSON: {e}")
        print(f"Text: {entities_text[:200]}")
        return None


def parse_batch_results(input_file: str, output_file: str, fs: gcsfs.GCSFileSystem):
    """
    Parse batch results and extract entities.

    Args:
        input_file: GCS path to batch_results.jsonl
        output_file: GCS path to save parsed entities
        fs: GCS filesystem instance
    """
    # Ensure output directory exists
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

                # Check if request was successful
                if status_code != 200:
                    failed += 1
                    logger.warning(
                        f"Line {line_num}: Failed request for {custom_id}, status: {status_code}"
                    )
                    f_failed.write(json.dumps({"custom_id": custom_id, "reason": "failed_request", "status_code": status_code}) + "\n")
                    continue

                # Extract the message content
                body = response.get("body", {})
                output = body.get("output", [])

                # Find the message in the output
                message_content = None
                for item in output:
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        if content and len(content) > 0:
                            message_content = content[0].get("text")
                            break

                if not message_content:
                    parse_errors += 1
                    logger.warning(
                        f"Line {line_num}: No message content found for {custom_id}"
                    )
                    f_failed.write(json.dumps({"custom_id": custom_id, "reason": "no_message_content"}) + "\n")
                    continue

                # Extract entities from the text
                entities = extract_entities_from_text(message_content)

                if entities is None:
                    parse_errors += 1
                    logger.warning(
                        f"Line {line_num}: Failed to extract entities for {custom_id}"
                    )
                    f_failed.write(json.dumps({"custom_id": custom_id, "reason": "failed_to_extract_entities"}) + "\n")
                    continue

                # Write the parsed result
                output_data = {"custom_id": custom_id, "entities": entities}
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
    reasoning_effort: Annotated[
        str, typer.Option(help="Reasoning effort level")
    ] = DEFAULT_REASONING_EFFORT,
    limit: Annotated[
        Optional[int], typer.Option(help="Limit used during generation")
    ] = None,
    input_file: Annotated[
        Optional[str], typer.Option(help="GCS path to batch_results.jsonl")
    ] = None,
    output_file: Annotated[
        Optional[str], typer.Option(help="GCS path to save parsed entities")
    ] = None,
):
    """Parse entities from batch API results."""
    fs = gcsfs.GCSFileSystem()
    subdir = build_subdir(model, reasoning_effort, limit)

    resolved_input = input_file or f"{GCP_KG_PREFIX}/{wiki}/entities/{subdir}/batch_results.jsonl"
    resolved_output = output_file or f"{GCP_KG_PREFIX}/{wiki}/entities/{subdir}/parsed.jsonl"

    if not fs.exists(resolved_input):
        logger.error(f"Error: Input file not found: {resolved_input}")
        raise typer.Exit(1)

    logger.info(f"Script: {__file__}")
    logger.info(
        f"Arguments: wiki={wiki}, model={model}, reasoning_effort={reasoning_effort}, limit={limit}"
    )
    logger.info("=" * 80)
    logger.info(f"Parsing entities from: {resolved_input}")
    logger.info(f"Output to: {resolved_output}")
    logger.info("=" * 80)

    parse_batch_results(resolved_input, resolved_output, fs)


if __name__ == "__main__":
    app()
