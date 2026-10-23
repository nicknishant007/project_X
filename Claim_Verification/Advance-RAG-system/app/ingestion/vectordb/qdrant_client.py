from qdrant_client import QdrantClient

from qdrant_client.models import (PointStruct)

from qdrant_client.models import (
    VectorParams,
    Distance
)

from app.ingestion.vectordb.schema import (
    COLLECTION_NAME,
    VECTOR_SIZE
)


class QdrantManager:

    _client = None




    def __init__(self):

        if QdrantManager._client is None:

            QdrantManager._client = QdrantClient(
                path="storage/qdrant_data"
            )
        
        self.client = QdrantManager._client



    def initialize_collection(self):

        collections = self.client.get_collections()

        existing = [
            collection.name
            for collection in collections.collections
        ]

        if COLLECTION_NAME not in existing:

            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )

    def store_documents(
        self,
        ids,
        embeddings,
        chunks
    ):

        payloads = []

        for chunk in chunks:

            payloads.append({
                "text": chunk.page_content,
                "metadata": chunk.metadata
            })
        points = [
            PointStruct(
                id=id,
                vector=embedding,
                payload=payload
            )
            for id, embedding, payload in zip(
                ids,
                embeddings,
                payloads
            )
        ]

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
            )


    def semantic_search(
        self,
        query_vector,
        top_k=20
    ):

        response=self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=True
        )
        return response.points
    