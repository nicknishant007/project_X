import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from difflib import (
    SequenceMatcher
)

from app.ingestion.retrieval.retrieval_pipeline import (
    RetrievalPipeline
)




class RetrievalEvaluator:

    def __init__(self):

        self.pipeline = (
            RetrievalPipeline()
        )
        self.model=SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate_question(
        self,
        question,
        expected_chunk,
        expected_metadata
    ):

        start_time = time.time()

        chunks = (
            self.pipeline.retrieve(
                question,
                top_k=5
            )
        )



        latency = (
            time.time()
            -
            start_time
        )

        # ----------------------------------
        # SOURCE MATCH
        # ----------------------------------

        expected_source = (
            expected_metadata[
                "source"
            ]
        )

        source_match = False

        for chunk in chunks:

            retrieved_source = (
                chunk.metadata.get(
                    "source",
                    ""
                )
            )

            if (
                retrieved_source
                ==
                expected_source
            ):

                source_match = True

                break

        # ----------------------------------
        # CHUNK SIMILARITY
        # ----------------------------------

        expected_embedding = self.model.encode(
        [expected_chunk]
        )

        retrieved_embeddings = self.model.encode(
        [chunk.page_content for chunk in chunks]
        )

        similarities = cosine_similarity(
        expected_embedding,
        retrieved_embeddings
        )[0]

        best_similarity = float(
        max(similarities)
        )

        # ----------------------------------
        # HIT
        # ----------------------------------

        hit = int(
            source_match
        )

        recall = hit

        # ----------------------------------
        # MRR
        # ----------------------------------

        mrr = 0

        for rank, chunk in enumerate(
            chunks,
            start=1
        ):

            if (
                chunk.metadata.get(
                    "source",
                    ""
                )
                ==
                expected_source
            ):

                mrr = (
                    1 / rank
                )

                break

        return {

            "question":
            question,

            "expected_source":
            expected_source,

            "source_match":
            source_match,

            "best_similarity":
            round(
                best_similarity,
                4
            ),

            "hit":hit,

            "recall":recall,

            "mrr":mrr,

            "latency":latency
        }