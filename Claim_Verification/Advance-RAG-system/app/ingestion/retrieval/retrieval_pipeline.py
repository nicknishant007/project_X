from langchain_core.documents import Document

from app.ingestion.retrieval.hybrid_retrieval import HybridRetriever
from app.ingestion.retrieval.reranker import Reranker


class RetrievalPipeline:

    def __init__(self):
        self.retriever = HybridRetriever()

    @staticmethod
    def _deduplicate(chunks: list[Document]) -> list[Document]:
        """Remove duplicate chunks while preserving retrieval order."""
        unique_chunks = []
        seen = set()

        for chunk in chunks:
            # Include source to avoid merging identical text from
            # different clinical papers.
            key = (
                chunk.metadata.get("document_id")
                or chunk.metadata.get("source")
                or "",
                " ".join(chunk.page_content.split()).casefold(),
            )

            if key in seen:
                continue

            seen.add(key)
            unique_chunks.append(chunk)

        return unique_chunks

    def retrieve(self, query: str, top_k: int = 6) -> list[Document]:

        (
            query_embedding,
            dense_results,
            sparse_results,
        ) = self.retriever.retrieve(query)

        # 1. Convert Qdrant dense results to LangChain Documents.
        dense_chunks = []

        for result in dense_results:
            payload = result.payload or {}
            text = payload.get("text")

            if not text:
                continue

            metadata = payload.get("metadata") or {}

            dense_chunks.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        # 2. Merge dense and sparse results.
        # Dense results come first, preserving their priority on duplicates.
        candidates = self._deduplicate(
            dense_chunks + sparse_results
        )

        # 3. Rerank the unique candidates.
        reranked_chunks = Reranker.rerank(
            query=query,
            chunks=candidates,
            top_k=top_k,
        )

        return reranked_chunks
