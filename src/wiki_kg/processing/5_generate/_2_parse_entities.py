"""
Parse entities from batch API results.

Reads the batch_results.jsonl file and extracts the entities list from successful responses.
Saves to a simplified JSONL format for downstream processing.
Uses Google Cloud Storage for file operations.
"""

import json
import re
import logging
import argparse
from pathlib import Path

import gcsfs
from dotenv import load_dotenv

try:
    from .utils import (
        GCP_KG_PREFIX,
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
        build_filename,
    )
except ImportError:
    from utils import (
        GCP_KG_PREFIX,
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
        build_filename,
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

    successful = 0
    failed = 0
    parse_errors = 0

    with fs.open(input_file, "r") as f_in, fs.open(output_file, "w") as f_out:
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
                    continue

                # Extract entities from the text
                entities = extract_entities_from_text(message_content)

                if entities is None:
                    parse_errors += 1
                    logger.warning(
                        f"Line {line_num}: Failed to extract entities for {custom_id}"
                    )
                    continue

                # Write the parsed result
                output_data = {"custom_id": custom_id, "entities": entities}
                f_out.write(json.dumps(output_data) + "\n")
                successful += 1

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error: {e}")
                parse_errors += 1
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error: {e}")
                parse_errors += 1

    logger.info("=" * 80)
    logger.info("Parsing complete!")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed requests: {failed}")
    logger.info(f"  Parse errors: {parse_errors}")
    logger.info(f"  Total processed: {successful + failed + parse_errors}")
    logger.info(f"  Output saved to: {output_file}")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parse entities from batch API results"
    )
    parser.add_argument(
        "--wiki",
        type=str,
        default="enwiki",
        help="Wiki identifier (default: enwiki)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name used for generation (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=DEFAULT_REASONING_EFFORT,
        choices=["minimal", "low", "medium", "high"],
        help=f"Reasoning effort level used during generation (default: {DEFAULT_REASONING_EFFORT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit used during generation (for filename matching)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="GCS path to batch_results.jsonl (overrides auto-generated path)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="GCS path to save parsed entities (overrides auto-generated path)",
    )
    args = parser.parse_args()

    # Initialize GCS filesystem
    fs = gcsfs.GCSFileSystem()

    # Generate default paths if not provided
    input_filename = build_filename(
        "batch_results", args.model, args.reasoning_effort, args.limit
    )
    output_filename = build_filename(
        "parsed_entities", args.model, args.reasoning_effort, args.limit
    )

    input_file = (
        args.input_file or f"{GCP_KG_PREFIX}/{args.wiki}/entities/{input_filename}"
    )
    output_file = (
        args.output_file or f"{GCP_KG_PREFIX}/{args.wiki}/entities/{output_filename}"
    )

    if not fs.exists(input_file):
        logger.error(f"Error: Input file not found: {input_file}")
        return

    logger.info(f"Script: {__file__}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info("=" * 80)
    logger.info(f"Parsing entities from: {input_file}")
    logger.info(f"Output to: {output_file}")
    logger.info("=" * 80)

    parse_batch_results(input_file, output_file, fs)


if __name__ == "__main__":
    main()
