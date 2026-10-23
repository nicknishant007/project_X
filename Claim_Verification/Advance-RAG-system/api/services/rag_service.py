from typing import Generator, List

# ----------------------------------------------------------------------
# Lazy-loaded singleton instances
# ----------------------------------------------------------------------

_rag_pipeline = None
_retrieval_pipeline = None


def get_rag_pipeline():
    """
    Returns a singleton RAGPipeline instance.
    The expensive initialization only happens once.
    """

    global _rag_pipeline

    if _rag_pipeline is None:
        from app.generation.rag_pipeline import RAGPipeline
        _rag_pipeline = RAGPipeline()

    return _rag_pipeline


def get_retrieval_pipeline():
    """
    Returns a singleton RetrievalPipeline.
    Used by /chat/retrieve.
    """

    global _retrieval_pipeline

    if _retrieval_pipeline is None:
        from app.ingestion.retrieval.retrieval_pipeline import RetrievalPipeline
        _retrieval_pipeline = RetrievalPipeline()

    return _retrieval_pipeline


class RAGService:

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

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

        print("RAG warmup complete.")

    # ------------------------------------------------------------------
    # Stream Answer
    # ------------------------------------------------------------------

    @staticmethod
    def stream_answer(
        question: str,
        top_k: int = 5
    ) -> Generator[str, None, None]:

        pipeline = get_rag_pipeline()

        for token in pipeline.stream_query(
            user_query=question,
            top_k=top_k
        ):

            if token:
                yield token

    # ------------------------------------------------------------------
    # Retrieve Sources
    # ------------------------------------------------------------------

    @staticmethod
    def retrieve_sources(
        question: str,
        top_k: int = 5
    ) -> List[dict]:

        pipeline = get_retrieval_pipeline()

        docs = pipeline.retrieve(
            query=question,
            top_k=top_k
        )

        seen = set()
        results = []

        for doc in docs:

            text = getattr(doc, "page_content", "")

            if not text:
                continue

            # Remove duplicates
            key = text[:120]

            if key in seen:
                continue

            seen.add(key)

            metadata = getattr(doc, "metadata", {})

            results.append({

                "file_name":
                    metadata.get(
                        "file_name",
                        "Unknown"
                    ),

                "source":
                    metadata.get(
                        "source",
                        ""
                    ),

                "file_size":
                    metadata.get(
                        "file_size",
                        0
                    ),

                "preview":
                    text[:350] + (
                        "..."
                        if len(text) > 350
                        else ""
                    ),

                "full_text":
                    text,

            })

        return results