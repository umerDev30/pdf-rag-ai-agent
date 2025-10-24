import os
from sentence_transformers import SentenceTransformer
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# Load SentenceTransformer globally
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    # Load PDF and split into chunks
    reader = PDFReader()
    documents = reader.load_data(file=path)
    text = [docs.text for docs in documents if getattr(docs, 'text', None)]
    all_chunks = []
    for t in text:
        chunks = splitter.split_text(t)
        all_chunks.extend(chunks)
    return all_chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    # Embed texts using SentenceTransformer embeddings
    if isinstance(texts, str):
        texts = [texts]
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
