import os
from sentence_transformers import SentenceTransformer
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the SentenceTransformer model once (used globally)
# "all-MiniLM-L6-v2" is a lightweight model that outputs 384-dimensional embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize the SentenceSplitter to divide text into manageable chunks
# Each chunk is up to 1000 characters long, with 200 characters overlapping between chunks
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    """
    Load a PDF file from the given path and split its text into smaller chunks.
    Returns a list of text chunks.
    """
    reader = PDFReader()  # Initialize the PDF reader
    documents = reader.load_data(file=path)  # Extract text from the PDF file

    # Extract text content from each page/document (skip if 'text' attribute is missing)
    text = [docs.text for docs in documents if getattr(docs, 'text', None)]

    all_chunks = []  # Store all text chunks here
    for t in text:
        # Split each text into overlapping chunks
        chunks = splitter.split_text(t)
        all_chunks.extend(chunks)  # Add all chunks to the list
    return all_chunks  # Return list of text chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into their vector embeddings using SentenceTransformer.
    Returns a list of embedding vectors (as lists of floats).
    """
    # If a single string is passed instead of a list, convert it into a list
    if isinstance(texts, str):
        texts = [texts]

    # Generate embeddings for the input texts
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Convert embeddings from NumPy arrays to regular Python lists for compatibility
    return embeddings.tolist()