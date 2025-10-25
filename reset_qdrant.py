from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")

collection_name = "docs"

# 1️⃣ Check if old collection exists
if client.collection_exists(collection_name):
    info = client.get_collection(collection_name)
    print("Old collection dimension:", info.config.params.vectors.size)
    print("Deleting old collection...")
    client.delete_collection(collection_name)

# 2️⃣ Recreate collection with correct dimension (384)
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

print(f"✅ Recreated '{collection_name}' collection with 384 dimensions (COSINE distance).")
