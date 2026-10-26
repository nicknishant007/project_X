import threading
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from app.ingestion.vectordb.schema import COLLECTION_NAME, VECTOR_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QDRANT_PATH = PROJECT_ROOT / "storage" / "qdrant_data"


class QdrantManager:
    _client = None
    _lock = threading.RLock()

    def __init__(self):
        with QdrantManager._lock:
            if QdrantManager._client is None:
                QDRANT_PATH.mkdir(parents=True, exist_ok=True)
                # Local-mode Qdrant allows ONE process at a time: don't run
                # `python verify.py` while uvicorn is running.
                QdrantManager._client = QdrantClient(path=str(QDRANT_PATH))
        self.client = QdrantManager._client
        self.initialize_collection()

    def initialize_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}

        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            return

        vectors = self.client.get_collection(COLLECTION_NAME).config.params.vectors
        if isinstance(vectors, dict):
            raise ValueError("Expected a single unnamed vector configuration.")
        if vectors.size != VECTOR_SIZE:
            raise ValueError(
                f"Qdrant dimension mismatch: collection has {vectors.size}, "
                f"schema expects {VECTOR_SIZE}.")
        if vectors.distance != Distance.COSINE:
            raise ValueError(f"Qdrant distance mismatch: {vectors.distance}, expected COSINE.")

    def store_documents(self, ids, embeddings, chunks):
        ids, embeddings, chunks = list(ids), list(embeddings), list(chunks)
        if not (len(ids) == len(embeddings) == len(chunks)):
            raise ValueError(
                f"ids, embeddings, chunks must have equal lengths: "
                f"{len(ids)}, {len(embeddings)}, {len(chunks)}")
        if not chunks:
            return

        points = []
        for point_id, embedding, chunk in zip(ids, embeddings, chunks):
            vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            if len(vector) != VECTOR_SIZE:
                raise ValueError(f"Vector {point_id} has dimension {len(vector)}; expected {VECTOR_SIZE}.")
            points.append(PointStruct(
                id=point_id, vector=vector,
                payload={"text": chunk.page_content, "metadata": dict(chunk.metadata or {})},
            ))

        with QdrantManager._lock:
            self.client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

    def semantic_search(self, query_vector, top_k=20, document_ids=None):
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        vector = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        if len(vector) != VECTOR_SIZE:
            raise ValueError(f"Query vector has dimension {len(vector)}; expected {VECTOR_SIZE}.")

        query_filter = None
        if document_ids:
            # Native filter: top-k is taken INSIDE the cited paper, not after global search.
            query_filter = Filter(must=[FieldCondition(
                key="metadata.document_id", match=MatchAny(any=list(document_ids)))])

        with QdrantManager._lock:
            response = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        return response.points
