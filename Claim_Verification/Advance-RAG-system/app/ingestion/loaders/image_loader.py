from app.ingestion.orc.tesseract_orc import (
    extract_text_from_image
)

def load_image(file_path: str) -> str:
    return extract_text_from_image(file_path)