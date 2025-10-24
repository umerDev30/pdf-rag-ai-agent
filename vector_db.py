from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, url: str = "http://localhost:6333", collection: str = "docs", dims: int = 384):
        # Initialize Qdrant client
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        # Create collection if it doesn't already exist
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dims, distance=Distance.COSINE)
            )

    def upsert_vectors(self, ids: list[int], vectors: list[list[float]], payloads: list[dict]):
        """
        Insert or update vectors with payloads in the Qdrant collection.
        """
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    def search_vectors(self, query_vector: list[float], top_k: int = 5):
        """
        Search for the most similar vectors and return their text contexts and sources.
        """
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k
        )

        contexts = []
        sources = set()
        for res in results:
            payload = getattr(res, 'payload', {}) or {}
            text = payload.get('text', '')
            source = payload.get('source', '')
            if text:
                contexts.append(text)
                if source:
                    sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
