import threading
from typing import Generator, List

_rag_pipeline = None
_verify_lock = threading.Lock()   # local Qdrant + shared models: one verification at a time


def get_rag_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        from app.generation.rag_pipeline import RAGPipeline
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def get_retrieval_pipeline():
    # Share the SAME retrieval pipeline (one BM25 + one Qdrant handle).
    return get_rag_pipeline().retrieval_pipeline


class RAGService:

    @staticmethod
    def warmup():
        print("Loading embedding model...")
        try:
            from app.ingestion.embedding.embedder import Embedder
            Embedder.load_model()
        except Exception as e:
            print(f"Embedder warmup skipped: {e}")

        print("Loading reranker...")
        try:
            from app.ingestion.retrieval.reranker import Reranker
            Reranker.load_model()
        except Exception as e:
            print(f"Reranker warmup skipped: {e}")

        try:
            get_rag_pipeline()
        except Exception as e:
            print(f"Pipeline warmup skipped: {e}")
        print("RAG warmup complete.")

    @staticmethod
    def stream_answer(question: str, top_k: int = 5) -> Generator[str, None, None]:
        for token in get_rag_pipeline().stream_query(user_query=question, top_k=top_k):
            if token:
                yield token

    @staticmethod
    def verify_claims(data, top_k: int = 5):
        """Run the full claim-verification pipeline on parsed claims JSON."""
        from app.generation.extractor import parse_claims

        claims = parse_claims(data)            # ValueError on bad input
        with _verify_lock:
            predictions = get_rag_pipeline().verify_claims(claims, top_k=top_k)
        return claims, predictions

    @staticmethod
    def retrieve_sources(question: str, top_k: int = 5) -> List[dict]:
        docs = get_retrieval_pipeline().retrieve(query=question, top_k=top_k)

        seen, results = set(), []
        for doc in docs:
            text = getattr(doc, "page_content", "")
            if not text or text[:120] in seen:
                continue
            seen.add(text[:120])

            md = getattr(doc, "metadata", {}) or {}
            pages = md.get("page_numbers") or []
            name = md.get("file_name") or md.get("cited_paper") or "Unknown"
            label = f"{name} (p.{', '.join(map(str, pages))})" if pages else name

            results.append({
                "file_name": name,
                "label": label,
                "pages": pages,
                "source": md.get("source", ""),
                "file_size": md.get("file_size", 0),
                "preview": text[:350] + ("..." if len(text) > 350 else ""),
                "full_text": text,
            })
        return results
