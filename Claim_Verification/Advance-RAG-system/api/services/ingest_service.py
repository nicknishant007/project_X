import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INCOMING_DIR = ROOT / "data" / "incoming"

# The processor (Docling clinical pipeline) is PDF-only.
ALLOWED_EXTENSIONS = {".pdf"}


class IngestService:

    @staticmethod
    def initialize_registry():
        try:
            from app.ingestion.registry.registry_manager import RegistryManager
            RegistryManager.initialize()
            print("Registry initialized.")
        except Exception as e:
            print(f"Registry initialization warning: {e}")

    @staticmethod
    def save_uploaded_file(filename: str, content: bytes) -> Path:
        INCOMING_DIR.mkdir(parents=True, exist_ok=True)
        destination = INCOMING_DIR / Path(filename).name

        if destination.exists():
            if destination.read_bytes() == content:
                return destination          # identical file: registry will skip it
            destination = INCOMING_DIR / f"{destination.stem}_{uuid.uuid4().hex[:8]}{destination.suffix}"

        destination.write_bytes(content)
        return destination

    @staticmethod
    def process_single_file(file_path: Path) -> dict:
        from app.ingestion.queue.worker import Worker

        chunks = Worker.process(str(file_path))
        skipped = not chunks
        return {
            "file": file_path.name,
            "chunks_created": len(chunks),
            "skipped": skipped,
            "message": (f"{file_path.name}: already indexed, or no extractable text."
                        if skipped else f"{len(chunks)} chunks indexed from {file_path.name}."),
        }

    @staticmethod
    def run_full_pipeline():
        from app.ingestion.pipeline.ingestion_pipeline import IngestionPipeline

        chunks = IngestionPipeline.run()
        return {"total_chunks_created": len(chunks),
                "message": f"Indexed {len(chunks)} chunks."}

    @staticmethod
    def validate_extension(filename: str):
        return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

    @staticmethod
    def get_allowed_extensions():
        return sorted(ALLOWED_EXTENSIONS)
