from concurrent.futures import (
    ThreadPoolExecutor
)

import yaml

from app.ingestion.scanner.local_scanner import (
    LocalScanner
)

from app.ingestion.queue.worker import (
    Worker
)


with open(
    "configs/ingestion.yaml",
    "r"
) as file:

    config = yaml.safe_load(file)


MAX_WORKERS = config["workers"]["max_workers"]


class IngestionPipeline:

    @staticmethod
    def run():

        all_chunks = []

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            results = executor.map(
                Worker.process,
                LocalScanner.scan()
            )

            for chunks in results:

                all_chunks.extend(chunks)

        return all_chunks