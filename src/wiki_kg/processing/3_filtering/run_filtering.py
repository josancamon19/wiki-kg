from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.data import Document
from typing import Tuple
from datatrove.pipeline.readers import JsonlReader
from pathlib import Path
import pandas as pd

import os
import json

# ---- Configuration constants ----
HERE = Path(__file__).resolve().parent
S3_RAW_PREFIX = "s3://wikipedia-bucket/wikipedia/raw_html_dumps/"
S3_PARSED_PREFIX = "s3://wikipedia-bucket/wikipedia/parsed_html/"
S3_PARSED_FILTER_REMOVED_PREFIX = (
    "s3://wikipedia-bucket/wikipedia/parsed_html-lang_filter/removed/"
)
S3_PARSED_FILTER_OUTPUT_PREFIX = (
    "s3://wikipedia-bucket/wikipedia/parsed_html-lang_filter/output/"
)
EXTRACTION_LOGS_DIR = HERE.parent / "2_extraction" / "logs"
FILTERING_LOGS_DIR = HERE / "logs-filter"
LANG_SCRIPTS_CSV = HERE / "language_scripts.csv"

# Slurm defaults
SLURM_TIME = "10:00:00"
SLURM_PARTITION = "hopper-cpu"
SLURM_CPUS_PER_TASK = 5
SLURM_QOS = "normal"
SLURM_MEM_PER_CPU = "1950M"


class LangChecker(BaseFilter):
    def __init__(
        self, wiki: str, scripts: list[str], exclusion_writer=None, invert=False
    ):
        from datatrove.utils.lid import GlotLID

        super().__init__(exclusion_writer=exclusion_writer)
        self.wiki = wiki
        self.scripts = scripts if isinstance(scripts, list) else [scripts]
        self.model = GlotLID()
        self.invert = invert

    def filter(self, doc: Document) -> bool | Tuple[bool, str]:
        if (len(doc.text) < 20 or doc.text.count("\n") < 2) and len(
            doc.metadata.get("infoboxes", [])
        ) == 0:
            return False, "short"

        (lang, lang_score), all_pairs = self.model.predict(doc)
        lang, script = lang.split("_")

        if self.scripts and script not in self.scripts:
            doc.metadata["script"] = script
            return False, "script"
        if self.wiki != "en" and self.wiki != "simple":
            (lang, lang_score), all_pairs = self.model.predict(doc)
            lang, script = lang.split("_")
            if lang == "eng" and lang_score > 0.7:
                doc.metadata["lang"] = lang
                doc.metadata["lang_score"] = lang_score
                return False, "eng"
        return True


df = pd.read_csv(LANG_SCRIPTS_CSV)
wiki_script_mapping = {
    row["subset"]: row["scripts"] for row in df.to_dict(orient="records")
}


def is_job_complete(logging_dir):
    if not os.path.exists(os.path.join(logging_dir, "completions")):
        return False
    with open(os.path.join(logging_dir, "executor.json")) as f:
        executor_data = json.load(f)
    return (
        len(os.listdir(os.path.join(logging_dir, "completions")))
        == executor_data["world_size"]
    )


if __name__ == "__main__":
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.io import get_datafolder

    wikis = [
        wiki
        for wiki in get_datafolder(S3_RAW_PREFIX).ls("", detail=False)
        if wiki.removesuffix("_namespace_0").endswith("wiki")
    ]
    import os
    from datatrove.executor.slurm import SlurmPipelineExecutor

    for wiki in wikis:
        if not is_job_complete(os.path.join(EXTRACTION_LOGS_DIR, wiki)):
            print(f"Skipping {wiki} because it is not complete")
            continue
        files = len(get_datafolder(S3_RAW_PREFIX + wiki).ls("", detail=False))
        basewikiname = wiki.removesuffix("wiki_namespace_0")
        scripts = wiki_script_mapping.get(basewikiname, None)
        if not scripts:
            print(f"Skipping {wiki} because it has no scripts")
            continue
        scripts = [
            x.replace("Hans", "Hani").replace("Hant", "Hani")
            for x in scripts.split("/")
            if x
        ]

        SlurmPipelineExecutor(
            pipeline=[
                JsonlReader(f"{S3_PARSED_PREFIX}{wiki}"),
                LangChecker(
                    wiki=basewikiname,
                    scripts=scripts,
                    exclusion_writer=JsonlWriter(
                        f"{S3_PARSED_FILTER_REMOVED_PREFIX}{wiki}"
                    ),
                ),
                JsonlWriter(f"{S3_PARSED_FILTER_OUTPUT_PREFIX}{wiki}"),
            ],
            tasks=files,
            time=SLURM_TIME,
            partition=SLURM_PARTITION,
            cpus_per_task=SLURM_CPUS_PER_TASK,
            job_name=f"wkp_{wiki}",
            qos=SLURM_QOS,
            logging_dir=str(FILTERING_LOGS_DIR / wiki),
            sbatch_args={
                "mem-per-cpu": SLURM_MEM_PER_CPU,
            },
        ).run()
