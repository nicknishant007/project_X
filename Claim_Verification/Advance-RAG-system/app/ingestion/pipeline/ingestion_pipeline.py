
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from app.ingestion.scanner.local_scanner import LocalScanner
from app.ingestion.queue.worker import Worker


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "ingestion.yaml"


def load_config() -> dict:
    """Load and validate ingestion configuration."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Ingestion configuration not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    workers_config = config.get("workers", {})
    max_workers = workers_config.get("max_workers")

    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
    ):
        raise ValueError(
            "'workers.max_workers' in ingestion.yaml "
            "must be a positive integer."
        )

    return config


CONFIG = load_config()
MAX_WORKERS = CONFIG["workers"]["max_workers"]


# --------------------------------------------------------------------
# Ingestion Pipeline
# --------------------------------------------------------------------

class IngestionPipeline:

    @staticmethod
    def run() -> list:
        """Scan local files and process them concurrently."""

        all_chunks = []
        file_paths = list(LocalScanner.scan())

        if not file_paths:
            print("No files found for ingestion.")
            return all_chunks

        print(f"Found {len(file_paths)} files.")
        print(f"Using {MAX_WORKERS} ingestion workers.")

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            results = executor.map(
                Worker.process,
                file_paths,
            )

            for file_path, chunks in zip(file_paths, results):
                if chunks is None:
                    print(f"No chunks returned for: {file_path}")
                    continue

                all_chunks.extend(chunks)

                print(
                    f"Processed {file_path}: "
                    f"{len(chunks)} chunks"
                )

        print(
            f"Ingestion complete. "
            f"Total chunks collected: {len(all_chunks)}"
        )

        return all_chunks
