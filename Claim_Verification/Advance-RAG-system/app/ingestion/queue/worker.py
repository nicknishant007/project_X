import uuid

from app.ingestion.registry.hash import generate_file_hash
from app.ingestion.registry.registry_manager import RegistryManager
from app.ingestion.file_processing.clinical_document_processor import (
    ClinicalDocumentProcessor
)
from app.ingestion.embedding.embedder import Embedder
from app.ingestion.vectordb.qdrant_client import QdrantManager
from app.ingestion.retrieval.bm25_index import BM25Indexer


bm25_index = BM25Indexer()
qdrant = QdrantManager()


class Worker:

    @staticmethod
    def process(file_path: str):

        print(f"\nProcessing File: {file_path}")

        # 1. Generate file hash
        file_hash = generate_file_hash(file_path)

        # 2. Skip files already processed
        if RegistryManager.is_processed(file_hash):
            print(f"Skipping already processed file: {file_path}")
            return []

        # 3. Extract, clean, build evidence metadata,
        #    and semantically chunk the PDF.
        chunks = ClinicalDocumentProcessor.process(file_path)

        print(f"Evidence chunks created: {len(chunks)}")

        if not chunks:
            print(f"No evidence chunks extracted: {file_path}")
            return []

        # 4. Generate embeddings using the existing embedding model
        texts = [chunk.page_content for chunk in chunks]

        embeddings = Embedder.embed(texts)

        print(f"Embeddings created: {len(embeddings)}")

        # 5. Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in chunks]

        # 6. Store documents and evidence metadata in Qdrant
        qdrant.store_documents(
            ids,
            embeddings,
            chunks,
        )

        print(f"Stored in Qdrant: {len(ids)} vectors")

        # 7. Add the same evidence chunks to BM25
        bm25_index.add_documents(chunks)

        print(f"Indexed in BM25: {len(chunks)} documents")

        # 8. Mark the file as processed
        RegistryManager.mark_processed(
            file_path,
            file_hash,
        )

        print(f"Processed successfully: {file_path}")

        return chunks
