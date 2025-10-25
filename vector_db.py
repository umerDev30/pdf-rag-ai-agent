from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, url: str = "http://localhost:6333", collection: str = "docs", dims: int = 384):
        """
        Initialize and manage connection to a Qdrant vector database.
        Creates a new collection if one does not exist.
        """
        # Connect to Qdrant instance (default localhost)
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        # Check if the collection exists — if not, create one
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dims,              # Vector size (dimension of embeddings)
                    distance=Distance.COSINE  # Use cosine similarity for comparisons
                )
            )

    def upsert_vectors(self, ids: list[int], vectors: list[list[float]], payloads: list[dict]):
        """
        Insert or update vectors with associated metadata (payloads) into the Qdrant collection.
        Each vector gets an ID, embedding vector, and related payload.
        """
        # Combine each vector, id, and payload into a Qdrant PointStruct
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]

        # Send the data to Qdrant to insert or update existing points
        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    def search_vectors(self, query_vector: list[float], top_k: int = 5):
        """
        Search for the most similar vectors to a given query vector.
        Returns both the matching text chunks (contexts) and their sources.
        """
        # Perform similarity search in Qdrant
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,  # Include stored metadata (text, source)
            limit=top_k          # Number of top matches to return
        )

        # Initialize result containers
        contexts = []
        sources = set()

        # Extract text and source info from search results
        for res in results:
            payload = getattr(res, 'payload', {}) or {}
            text = payload.get('text', '')
            source = payload.get('source', '')
            if text:
                contexts.append(text)
                if source:
                    sources.add(source)

        # Return matched chunks and unique source list
        return {"contexts": contexts, "sources": list(sources)}
