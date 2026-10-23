from sentence_transformers import ( SentenceTransformer )


class Embedder:

    _model = None

    @classmethod
    def load_model(cls):

        if cls._model is None:

            cls._model = SentenceTransformer(
                "BAAI/bge-small-en-v1.5",
                device="cpu",
                cache_folder="storage/model_cache"
            )

    @classmethod
    def embed(cls, texts):

        cls.load_model()

        return cls._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True
        )