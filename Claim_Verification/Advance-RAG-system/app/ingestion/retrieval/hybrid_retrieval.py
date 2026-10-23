from app.ingestion.embedding.embedder import (
    Embedder
)

from app.ingestion.vectordb.qdrant_client import (
    QdrantManager
)

from app.ingestion.retrieval.bm25_index import (
    BM25Indexer
)


class HybridRetriever:

    def __init__(self):

        self.qdrant = QdrantManager()

        self.bm25 = BM25Indexer()

    def retrieve(
        self,
        query,
        dense_top_k=15,
        sparse_top_k=7,
    ):

        # Query embedding for semantic search
        query_embedding = Embedder.embed(
            [query]
        )[0]

        # Dense retrieval from Qdrant
        dense_results = (
            self.qdrant.semantic_search(
                query_embedding,
                top_k=dense_top_k
            )
        )

        # Sparse retrieval from BM25
        sparse_results = (
            self.bm25.search(
                query,
                top_k=sparse_top_k
            )
        )

        return (
            query_embedding,
            dense_results,
            sparse_results
        )