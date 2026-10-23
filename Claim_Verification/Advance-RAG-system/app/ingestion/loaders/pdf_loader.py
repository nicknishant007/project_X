from langchain_community.document_loaders import (PyMuPDFLoader)


def load_pdf(file_path: str) -> str:

    loader = PyMuPDFLoader(file_path)

    documents = loader.load()

    extracted_text = []

    for doc in documents:
        extracted_text.append(doc.page_content)

    return "\n".join(extracted_text)