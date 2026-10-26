import threading

from sentence_transformers import SentenceTransformer

# BGE retrieval models expect this prefix on QUERIES only (not on passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:

    _model = None
    _lock = threading.Lock()

    @classmethod
    def load_model(cls):
        with cls._lock:
            if cls._model is None:
                cls._model = SentenceTransformer(
                    "BAAI/bge-small-en-v1.5",
                    device="cpu",
                    cache_folder="storage/model_cache",
                )

    @classmethod
    def embed(cls, texts):
        cls.load_model()
        return cls._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )

    @classmethod
    def embed_query(cls, text):
        return cls.embed([QUERY_PREFIX + text])[0]
