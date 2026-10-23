import os

from app.ingestion.loaders.csv_loader import load_csv
from app.ingestion.loaders.docx_loader import load_docx
from app.ingestion.loaders.image_loader import load_image
from app.ingestion.loaders.pdf_loader import load_pdf

EXTRACTORS = {
    '.csv': load_csv,
    '.docx': load_docx,
    '.pdf': load_pdf,
    '.jpg': load_image,
    '.jpeg': load_image,
    '.png': load_image,
}

class ExtractorRouter:
    @staticmethod
    def extract(file_path: str) -> str:
        extension = os.path.splitext(file_path)[1].lower()

        extractor = EXTRACTORS.get(extension)

        if not extractor:
            raise ValueError(f"Unsupported file type: {extension}")

        return extractor(file_path)