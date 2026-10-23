from pathlib import Path
import shutil
import uuid

ROOT = Path(__file__).resolve().parent.parent.parent

INCOMING_DIR = ROOT / "data" / "incoming"

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".csv",
    ".txt",
    ".jpg",
    ".jpeg",
    ".png",
}


class IngestService:

    # -------------------------------------------------------
    # Registry
    # -------------------------------------------------------

    @staticmethod
    def initialize_registry():

        try:

            from app.ingestion.registry.registry_manager import RegistryManager

            RegistryManager.initialize()

            print("✅ Registry initialized.")

        except Exception as e:

            print(f"Registry initialization warning: {e}")

    # -------------------------------------------------------
    # Save Uploaded File
    # -------------------------------------------------------

    @staticmethod
    def save_uploaded_file(filename: str, content: bytes) -> Path:

        INCOMING_DIR.mkdir(parents=True, exist_ok=True)

        filename = Path(filename).name

        destination = INCOMING_DIR / filename

        if destination.exists():

            stem = destination.stem
            suffix = destination.suffix

            destination = (
                INCOMING_DIR /
                f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
            )

        with open(destination, "wb") as f:
            f.write(content)

        return destination

    # -------------------------------------------------------
    # Process One File
    # -------------------------------------------------------

    @staticmethod
    def process_single_file(file_path: Path) -> dict:

        from app.ingestion.queue.worker import Worker

        chunks = Worker.process(str(file_path))

        skipped = chunks == []

        return {

            "file": file_path.name,

            "chunks_created": len(chunks),

            "skipped": skipped,

            "message": (
                f"{file_path.name} already exists."
                if skipped
                else f"{len(chunks)} chunks indexed."
            )

        }

    # -------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------

    @staticmethod
    def run_full_pipeline():

        from app.ingestion.pipeline.ingestion_pipeline import (
            IngestionPipeline
        )

        chunks = IngestionPipeline.run()

        return {

            "total_chunks_created": len(chunks),

            "message":
                f"Indexed {len(chunks)} chunks."

        }

    # -------------------------------------------------------
    # Validation
    # -------------------------------------------------------

    @staticmethod
    def validate_extension(filename: str):

        ext = Path(filename).suffix.lower()

        return ext in ALLOWED_EXTENSIONS

    @staticmethod
    def get_allowed_extensions():

        return sorted(ALLOWED_EXTENSIONS)