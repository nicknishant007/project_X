from langchain_core.documents import (
    Document
)

from app.ingestion.retrieval.hybrid_retrieval import (
    HybridRetriever
)

from app.ingestion.retrieval.mmr import (
    MMRRetriever
)

from app.ingestion.retrieval.reranker import (
    Reranker
)


class RetrievalPipeline:

    def __init__(self):

        self.retriever = HybridRetriever()

    def retrieve(
        self,
        query,
        top_k=5
    ):

        (
            query_embedding,
            dense_results,
            sparse_results
        ) = self.retriever.retrieve(query)

        # ------------------------
        # Dense Retrieval Branch
        # ------------------------

        dense_chunks = []

        dense_embeddings = []


        for result in dense_results:

            text = result.payload["text"]

            metadata = result.payload[
                "metadata"
            ]

            dense_chunks.append(

                Document(
                    page_content=text,
                    metadata=metadata
                )

            )

            dense_embeddings.append(
                result.vector
            )

        # MMR only on dense chunks

        mmr_chunks = MMRRetriever.rerank(
            query_embedding=query_embedding,
            candidate_embeddings=dense_embeddings,
            candidate_chunks=dense_chunks,
            top_k=8
        )

        # ------------------------
        # Merge Dense + Sparse
        # ------------------------

        candidates = []

        seen = set()

        for chunk in mmr_chunks:

            if chunk.page_content not in seen:

                seen.add(
                    chunk.page_content
                )

                candidates.append(
                    chunk
                )

        for chunk in sparse_results:

            if chunk.page_content not in seen:

                seen.add(
                    chunk.page_content
                )

                candidates.append(
                    chunk
                )

        # ------------------------
        # Final Reranking
        # ------------------------

        reranked_chunks = (

            Reranker.rerank(
                query=query,
                chunks=candidates,
                top_k=top_k
            )

        )

        return reranked_chunks