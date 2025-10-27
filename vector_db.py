from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

class QdrantStorage:
    def __init__(self, url: str = "http://localhost:6333", collection: str = "docs", dims: int = 384):
        """
        Initialize and manage connection to a Qdrant vector database.
        Creates a new collection if one does not exist.
        """
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        # Create collection if missing
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dims,
                    distance=Distance.COSINE
                )
            )

    def upsert_vectors(self, ids: list[int], vectors: list[list[float]], payloads: list[dict], source_id: str = None):
        """
        Insert or update vectors into the collection, tagging them with a source_id (e.g., PDF name).
        """
        # Attach source_id to payloads if provided
        if source_id:
            for p in payloads:
                p["source_id"] = source_id

        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    def search_vectors(self, query_vector: list[float], top_k: int = 5, source_id: str = None):
        """
        Search for most similar vectors to query_vector.
        Optionally filter results by source_id to get answers from a specific PDF only.
        """
        search_filter = None
        if source_id:
            search_filter = Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
            )

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=search_filter,
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
