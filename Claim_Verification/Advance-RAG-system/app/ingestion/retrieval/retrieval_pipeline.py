from langchain_core.documents import Document

from app.ingestion.retrieval.hybrid_retrieval import HybridRetriever
from app.ingestion.retrieval.reranker import Reranker


class RetrievalPipeline:

    def __init__(self):
        self.retriever = HybridRetriever()

    def list_documents(self) -> list[str]:
        """document_ids (PDF file stems) currently indexed."""
        return sorted(self.retriever.bm25.document_ids())

    @staticmethod
    def _deduplicate(chunks: list[Document]) -> list[Document]:
        unique, seen = [], set()
        for chunk in chunks:
            key = (
                chunk.metadata.get("document_id") or chunk.metadata.get("source") or "",
                " ".join(chunk.page_content.split()).casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(chunk)
        return unique

    def retrieve(self, query: str, top_k: int = 6, document_ids=None) -> list[Document]:
        ids = set(document_ids) if document_ids else None

        _, dense_results, sparse_results = self.retriever.retrieve(query, document_ids=ids)

        dense_chunks = []
        for result in dense_results:
            payload = result.payload or {}
            text = payload.get("text")
            if text:
                dense_chunks.append(Document(page_content=text, metadata=payload.get("metadata") or {}))

        candidates = self._deduplicate(dense_chunks + sparse_results)
        if not candidates:
            return []

        return Reranker.rerank(query=query, chunks=candidates, top_k=top_k)
