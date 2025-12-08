"""
Parse relations from batch API results.

Reads the batch_results.jsonl file and extracts the relations list from successful responses.
Saves to a simplified JSONL format for downstream processing.
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


def extract_relations_from_text(text: str) -> list[dict] | None:
    """
    Extract the relations list from the response text.

    The text format is:
    [[ ## relations ## ]]
    [{"subject": "...", "predicate": "...", "object": "..."}, ...]
    [[ ## completed ## ]]
    """
    pattern = r"\[\[\s*##\s*relations\s*##\s*\]\]\s*\n?(.*?)\s*\n?\[\[\s*##\s*completed\s*##\s*\]\]"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    relations_text = match.group(1).strip()
    relations_text = re.sub(r"\s*#.*$", "", relations_text, flags=re.MULTILINE)
    relations_text = relations_text.strip()

    try:
        relations = json.loads(relations_text)
        return relations if isinstance(relations, list) else None
    except json.JSONDecodeError as e:
        print(f"Failed to parse relations JSON: {e}")
        print(f"Text preview: {relations_text[:300]}")
        return None


def parse_batch_results(input_file: str, output_file: str, fs: gcsfs.GCSFileSystem):
    """Parse batch results and extract relations."""
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

                if status_code != 200:
                    failed += 1
                    logger.warning(f"Line {line_num}: Failed request for {custom_id}, status: {status_code}")
                    continue

                body = response.get("body", {})
                output = body.get("output", [])

                message_content = None
                for item in output:
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        if content and len(content) > 0:
                            message_content = content[0].get("text")
                            break

                if not message_content:
                    parse_errors += 1
                    logger.warning(f"Line {line_num}: No message content found for {custom_id}")
                    continue

                relations = extract_relations_from_text(message_content)

                if relations is None:
                    parse_errors += 1
                    logger.warning(f"Line {line_num}: Failed to extract relations for {custom_id}")
                    continue

                output_data = {"custom_id": custom_id, "relations": relations}
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
