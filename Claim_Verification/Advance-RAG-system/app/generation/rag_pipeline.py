from app.ingestion.retrieval.retrieval_pipeline import (
    RetrievalPipeline
)

from app.generation.response_generator import (
    ResponseGenerator)


class RAGPipeline:

    def __init__(self):

        self.retrieval_pipeline = (
            RetrievalPipeline()
        )

    def stream_query(
        self,
        user_query,
        top_k=5
    ):

        retrieved_chunks = (
            self.retrieval_pipeline.retrieve(
                query=user_query,
                top_k=top_k
            )
        )

        return ResponseGenerator.stream_response(
            query=user_query,
            retrieved_chunks=retrieved_chunks
        )