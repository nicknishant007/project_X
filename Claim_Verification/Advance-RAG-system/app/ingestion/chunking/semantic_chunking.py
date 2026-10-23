import yaml

from langchain_experimental.text_splitter import SemanticChunker

from app.ingestion.embedding.embedder import Embedder



# Load Config

with open(
    "configs/ingestion.yaml",
    "r"
) as file:

    config = yaml.safe_load(file)

# Embedding Wrapper

class EmbeddingWrapper:

    def embed_documents(self, texts):

        return Embedder.embed(texts)

    def embed_query(self, text):

        return Embedder.embed([text])[0]


# Semantic Splitter

splitter = SemanticChunker(

    embeddings=EmbeddingWrapper(),

    breakpoint_threshold_type="percentile",

    breakpoint_threshold_amount=95,

)

# Chunker

class SemanticChunkerEngine:

    @staticmethod
    def chunk(text, metadata):

        return splitter.create_documents(

            [text],

            metadatas=[metadata]

        )