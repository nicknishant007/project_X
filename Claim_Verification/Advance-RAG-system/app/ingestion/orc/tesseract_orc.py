import pytesseract

from PIL import Image

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using Tesseract OCR.

    Args:
        image_path (str): The path to the image file."""
    # Open the image file
    image = Image.open(image_path)

    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(image)

    return text