
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.ingestion.vectordb.schema import (
    COLLECTION_NAME,
    VECTOR_SIZE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QDRANT_PATH = PROJECT_ROOT / "storage" / "qdrant_data"


class QdrantManager:
    _client = None

    def __init__(self):
        if QdrantManager._client is None:
            QDRANT_PATH.mkdir(parents=True, exist_ok=True)

            QdrantManager._client = QdrantClient(
                path=str(QDRANT_PATH)
            )

        self.client = QdrantManager._client
        self.initialize_collection()

    def initialize_collection(self):
        collections = self.client.get_collections()

        existing = {
            collection.name
            for collection in collections.collections
        }

        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            return

        # Fail early if the existing collection has the wrong
        # vector dimension or distance configuration.
        info = self.client.get_collection(COLLECTION_NAME)
        vectors = info.config.params.vectors

        if isinstance(vectors, dict):
            raise ValueError(
                "Expected a single unnamed vector configuration, "
                "but the collection uses named vectors."
            )

        if vectors.size != VECTOR_SIZE:
            raise ValueError(
                f"Qdrant dimension mismatch: collection has "
                f"{vectors.size}, schema expects {VECTOR_SIZE}. "
                "Check schema.py and your embedding model."
            )

        if vectors.distance != Distance.COSINE:
            raise ValueError(
                f"Qdrant distance mismatch: collection uses "
                f"{vectors.distance}, expected COSINE."
            )

    def store_documents(self, ids, embeddings, chunks):
        ids = list(ids)
        embeddings = list(embeddings)
        chunks = list(chunks)

        if not (len(ids) == len(embeddings) == len(chunks)):
            raise ValueError(
                "ids, embeddings, and chunks must have equal lengths: "
                f"{len(ids)}, {len(embeddings)}, {len(chunks)}"
            )

        if not chunks:
            return

        points = []

        for point_id, embedding, chunk in zip(
            ids, embeddings, chunks
        ):
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            else:
                embedding = list(embedding)

            if len(embedding) != VECTOR_SIZE:
                raise ValueError(
                    f"Vector {point_id} has dimension {len(embedding)}; "
                    f"expected {VECTOR_SIZE}."
                )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk.page_content,
                        "metadata": dict(chunk.metadata or {}),
                    },
                )
            )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

    def semantic_search(self, query_vector, top_k=20):
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()
        else:
            query_vector = list(query_vector)

        if len(query_vector) != VECTOR_SIZE:
            raise ValueError(
                f"Query vector has dimension {len(query_vector)}; "
                f"expected {VECTOR_SIZE}."
            )

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return response.points
