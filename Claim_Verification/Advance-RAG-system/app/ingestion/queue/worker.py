import uuid
from app.ingestion.registry.hash import (
    generate_file_hash
)

from app.ingestion.registry.registry_manager import (
    RegistryManager
)

from app.ingestion.extractor.extractor_router import (
    ExtractorRouter
)

from app.ingestion.extractor.text_cleaner import (
    TextCleaner
)

from app.ingestion.extractor.metadata_builder import (
    MetadataBuilder
)

from app.ingestion.chunking.recursive_chunker import (
    RecursiveChunker
)

from app.ingestion.embedding.embedder import (
    Embedder
)

from app.ingestion.vectordb.qdrant_client import (
    QdrantManager
)

from app.ingestion.retrieval.bm25_index import (
    BM25Indexer
)

bm25_index = BM25Indexer()
qdrant= QdrantManager()



class Worker:

    @staticmethod
    def process(file_path):

        print(f"\nProcessing File: {file_path}")

        file_hash = generate_file_hash(
            file_path
        )

        if RegistryManager.is_processed(
            file_hash
        ):

            print(f"Skipping: {file_path}")

            return []

        raw_text = ExtractorRouter.extract(
            file_path
        )

        print(f"Raw text length: {len(raw_text)}")

        cleaned_text = TextCleaner.clean(
            raw_text
        )

        metadata = MetadataBuilder.build(
            file_path
        )

        chunks = RecursiveChunker.chunk(
            cleaned_text,
            metadata
        )

        print(f"Chunks created: {len(chunks)}")

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = Embedder.embed(
            texts
        )

        print(f"Embeddings created: {len(embeddings)}")

        ids = [
            str(uuid.uuid4())
            for _ in chunks
        ]

        qdrant.store_documents(
            ids,
            embeddings,
            chunks
        )

        print(f"Stored in Qdrant: {len(ids)} vectors")

        bm25_index.add_documents(
            chunks
        )

        RegistryManager.mark_processed(
            file_path,
            file_hash
        )

        print(f"Processed Successfully: {file_path}")

        return chunks