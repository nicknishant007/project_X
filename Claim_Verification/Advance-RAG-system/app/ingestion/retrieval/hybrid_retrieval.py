from app.ingestion.embedding.embedder import Embedder
from app.ingestion.retrieval.bm25_index import BM25Indexer
from app.ingestion.vectordb.qdrant_client import QdrantManager


class HybridRetriever:

    def __init__(self):
        self.qdrant = QdrantManager()
        self.bm25 = BM25Indexer()

    def retrieve(self, query, dense_top_k=15, sparse_top_k=15, document_ids=None):
        query_embedding = Embedder.embed_query(query)

        dense_results = self.qdrant.semantic_search(
            query_embedding, top_k=dense_top_k, document_ids=document_ids)

        sparse_results = self.bm25.search(
            query, top_k=sparse_top_k, document_ids=document_ids)

        return query_embedding, dense_results, sparse_results
