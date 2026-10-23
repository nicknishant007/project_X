
from sentence_transformers import (
    CrossEncoder
)


class Reranker:

    _model = None

    @classmethod
    def load_model(cls):

        if cls._model is None:

            cls._model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

    @classmethod
    def rerank(
        cls,
        query,
        chunks,
        top_k=6
    ):

        cls.load_model()

        pairs = [
            (query, chunk.page_content)
            for chunk in chunks
        ]

        scores = cls._model.predict(
            pairs
        )

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            chunk
            for chunk, _
            in ranked[:top_k]
        ]