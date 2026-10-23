import yaml

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


with open(
    "configs/ingestion.yaml",
    "r"
) as file:

    config = yaml.safe_load(file)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["chunking"]["chunk_size"],
    chunk_overlap=config["chunking"]["chunk_overlap"]
)


class RecursiveChunker:

    @staticmethod
    def chunk(text, metadata):

        return splitter.create_documents(
            [text],
            metadatas=[metadata]
        )