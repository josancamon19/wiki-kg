"""
Parse entities from batch API results.

Reads the batch_results.jsonl file and extracts the entities list from successful responses.
Saves to a simplified JSONL format for downstream processing.
"""

import json
import re
from pathlib import Path


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


def parse_batch_results(input_file: Path, output_file: Path):
    """
    Parse batch results and extract entities.

    Args:
        input_file: Path to batch_results.jsonl
        output_file: Path to save parsed entities
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0
    parse_errors = 0

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line_num, line in enumerate(f_in, 1):
            try:
                result = json.loads(line)

                custom_id = result.get("custom_id")
                response = result.get("response", {})
                status_code = response.get("status_code")

                # Check if request was successful
                if status_code != 200:
                    failed += 1
                    print(
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
                    print(f"Line {line_num}: No message content found for {custom_id}")
                    continue

                # Extract entities from the text
                entities = extract_entities_from_text(message_content)

                if entities is None:
                    parse_errors += 1
                    print(
                        f"Line {line_num}: Failed to extract entities for {custom_id}"
                    )
                    continue

                # Write the parsed result
                output_data = {"custom_id": custom_id, "entities": entities}
                f_out.write(json.dumps(output_data) + "\n")
                successful += 1

            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error: {e}")
                parse_errors += 1
            except Exception as e:
                print(f"Line {line_num}: Unexpected error: {e}")
                parse_errors += 1

    print("\nParsing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed requests: {failed}")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Total processed: {successful + failed + parse_errors}")
    print(f"\nOutput saved to: {output_file}")


def main():
    """Main entry point."""
    # Define paths
    base_dir = Path(__file__).parent
    input_file = base_dir / "results" / "entities" / "batch_results.jsonl"
    output_file = base_dir / "results" / "entities" / "parsed_entities.jsonl"

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Parsing entities from: {input_file}")
    parse_batch_results(input_file, output_file)


if __name__ == "__main__":
    main()
