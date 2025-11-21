from datatrove.data import Document
from typing import Tuple
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# ---- Configuration constants ----
HERE = Path(__file__).resolve().parent
GCP_RAW_PREFIX = "gs://wikipedia-graph/wikipedia/raw_html_dumps"
GCP_PARSED_PREFIX = "gs://wikipedia-graph/wikipedia/parsed_html"
GCP_FILTERED_PREFIX = "gs://wikipedia-graph/wikipedia/filtered_html"
LOGGING_DIR = HERE / "logs"
LANG_SCRIPTS_CSV = HERE / "language_scripts.csv"


class LangChecker:
    """Filter documents based on language and script detection."""

    def __init__(self, wiki: str, scripts: list[str]):
        from datatrove.utils.lid import GlotLID

        self.wiki = wiki
        self.scripts = scripts if isinstance(scripts, list) else [scripts]
        self.model = GlotLID()

    def should_keep(self, doc: Document) -> Tuple[bool, str]:
        """
        Check if document should be kept.
        Returns (keep: bool, reason: str)
        """
        # Filter out very short documents without infoboxes
        if (len(doc.text) < 20 or doc.text.count("\n") < 2) and len(
            doc.metadata.get("infoboxes", [])
        ) == 0:
            return False, "short"

        # Predict language and script
        (lang, lang_score), all_pairs = self.model.predict(doc)
        lang, script = lang.split("_")

        # Check if script matches expected scripts for this wiki
        if self.scripts and script not in self.scripts:
            doc.metadata["script"] = script
            return False, "script"

        # For non-English wikis, filter out English content
        if self.wiki != "en" and self.wiki != "simple":
            if lang == "eng" and lang_score > 0.7:
                doc.metadata["lang"] = lang
                doc.metadata["lang_score"] = lang_score
                return False, "eng"

        return True, "kept"


# Load wiki script mapping
df = pd.read_csv(LANG_SCRIPTS_CSV)
wiki_script_mapping = {
    row["subset"]: row["scripts"] for row in df.to_dict(orient="records")
}


def process_filtering_task(task_args):
    """Worker function to filter documents for a single file."""
    import json
    import gcsfs
    from loguru import logger
    from pathlib import Path

    wiki, task_id, total_tasks, logging_dir, scripts = task_args

    # Setup logging for this task
    log_file = Path(logging_dir) / "logs" / f"task_{task_id:05d}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), rotation="100 MB")

    try:
        logger.info(f"Starting filtering task {task_id}/{total_tasks} for {wiki}")

        # Initialize GCS filesystem
        fs = gcsfs.GCSFileSystem()

        # Create language checker
        basewikiname = wiki.removesuffix("wiki_namespace_0")
        checker = LangChecker(basewikiname, scripts)

        # Setup stats tracking
        stats = {
            "total": 0,
            "kept": 0,
            "short": 0,
            "script": 0,
            "eng": 0,
        }

        # Get file list for this task
        input_path = f"{GCP_PARSED_PREFIX}/{wiki}".replace("gs://", "")
        files = sorted(fs.ls(input_path, detail=False))

        if task_id >= len(files):
            logger.info(
                f"Task {task_id} has no file to process (only {len(files)} files)"
            )
            return stats

        # Process only the file assigned to this task
        file_path = files[task_id]
        logger.info(f"Processing file: gs://{file_path}")

        # Output file path
        output_path = f"{GCP_FILTERED_PREFIX}/{wiki}".replace("gs://", "")
        output_file = f"gs://{output_path}/{Path(file_path).name}"

        # Ensure output directory exists
        fs.makedirs(output_path, exist_ok=True)

        # Process documents from this file
        documents_written = 0
        with (
            fs.open(f"gs://{file_path}", "r") as in_f,
            fs.open(output_file, "w") as out_f,
        ):
            for line in in_f:
                stats["total"] += 1

                # Parse document
                data = json.loads(line)
                doc = Document(
                    text=data["text"], id=data["id"], metadata=data.get("metadata", {})
                )

                # Check if document should be kept
                keep, reason = checker.should_keep(doc)
                stats[reason] = stats.get(reason, 0) + 1

                if keep:
                    # Write filtered document
                    output_data = {
                        "text": doc.text,
                        "id": doc.id,
                        "metadata": doc.metadata,
                    }
                    out_f.write(json.dumps(output_data) + "\n")
                    documents_written += 1

        logger.info(
            f"Task {task_id} completed. Processed {stats['total']} documents, "
            f"kept {documents_written}"
        )
        return stats

    except Exception as e:
        logger.error(f"Task {task_id} failed with error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return {"error": str(e)}


if __name__ == "__main__":
    import multiprocessing as mp
    from datatrove.io import get_datafolder

    # Get list of wikis to process
    wikis = [
        wiki
        for wiki in get_datafolder(GCP_PARSED_PREFIX).ls("", detail=False)
        if wiki.removesuffix("_namespace_0").endswith("wiki")
    ]

    num_workers = os.cpu_count()
    print(f"Using {num_workers} workers")

    for wiki in wikis:
        basewikiname = wiki.removesuffix("wiki_namespace_0")
        scripts = wiki_script_mapping.get(basewikiname, None)

        if not scripts:
            print(f"Skipping {wiki} because it has no scripts mapping")
            continue

        # Parse scripts (normalize Hans/Hant to Hani)
        scripts = [
            x.replace("Hans", "Hani").replace("Hant", "Hani")
            for x in scripts.split("/")
            if x
        ]

        wiki_df = get_datafolder(GCP_PARSED_PREFIX + "/" + wiki)
        print(f"\n{'=' * 80}")
        print(f"Processing: {GCP_PARSED_PREFIX}/{wiki}")
        print(f"Output: {GCP_FILTERED_PREFIX}/{wiki}")
        print(f"Scripts: {scripts}")

        # Get number of files to process
        files = len(wiki_df.fs.ls(wiki_df.path, detail=False))
        print(f"Total files: {files}")

        # Create logging directory
        log_dir = LOGGING_DIR / wiki
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create task arguments for each file
        task_args = [
            (wiki, task_id, files, str(log_dir), scripts) for task_id in range(files)
        ]

        # Process files in parallel using multiprocessing
        print(f"Starting parallel processing with {num_workers} workers...")
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(process_filtering_task, task_args)

        # Aggregate statistics
        total_stats = {
            "total": 0,
            "kept": 0,
            "short": 0,
            "script": 0,
            "eng": 0,
            "errors": 0,
        }

        for result in results:
            if "error" in result:
                total_stats["errors"] += 1
            else:
                for key in ["total", "kept", "short", "script", "eng"]:
                    total_stats[key] += result.get(key, 0)

        print(f"\nCompleted filtering {wiki}")
        print(f"Statistics: {total_stats}")
        print(f"{'=' * 80}\n")
