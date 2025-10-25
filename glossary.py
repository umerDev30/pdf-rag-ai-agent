"""
📘 AI & RAG Project Glossary
Simple explanations of key terms used in the RAG AI PDF Agent.
"""

# ==============================
# 🧠 CORE CONCEPTS
# ==============================

# 1️⃣ RAG (Retrieval-Augmented Generation)
# Combines search + AI generation. The system first retrieves relevant text from a database
# (like Qdrant), then sends that context to an LLM (like Gemini) to generate an answer.
# Example: You upload a PDF, ask a question — RAG retrieves the relevant page
# and Gemini answers based on that page.

# 2️⃣ Embedding
# A numerical representation (vector) of text, created by a model like SentenceTransformer.
# Words/sentences with similar meaning have vectors close to each other.
# Example: "dog" and "puppy" → similar embeddings.

# 3️⃣ Vector
# A list of numbers that represents data (like a text chunk) in high-dimensional space.
# Example: [0.12, -0.45, 0.88, ...] — 384 numbers for "all-MiniLM-L6-v2".

# 4️⃣ Vector Database (Qdrant)
# A special database that stores embeddings (vectors) and allows similarity searches.
# Instead of matching text exactly, it finds the most *semantically similar* data.
# Example: Searching "weather forecast" might also match "climate prediction".

# 5️⃣ Chunking
# Breaking long text into smaller, overlapping pieces for better search and embedding.
# Example: Splitting a 10-page PDF into 200-word chunks.

# 6️⃣ Overlap
# Repeating a small part of the previous chunk in the next one to keep context connected.
# Example: Last 2 sentences of chunk A appear again at start of chunk B.

# 7️⃣ Similarity Search
# Finds vectors (text chunks) most similar to a given query vector.
# Example: "What is AI?" → finds chunks talking about artificial intelligence.

# ==============================
# ⚙️ TOOLS & FRAMEWORKS
# ==============================

# 8️⃣ FastAPI
# A Python web framework for building APIs quickly.
# Example: You define endpoints like /ingest or /query to trigger backend functions.

# 9️⃣ Inngest
# Event-driven workflow engine that automates steps (like load → embed → upsert).
# Example: When a PDF upload event happens, Inngest automatically runs the RAG ingestion pipeline.

# 🔟 Qdrant
# Open-source vector database used for semantic search.
# Stores (id, vector, payload) triples.
# Example: Stores chunks of PDF text + their embeddings.

# 11️⃣ SentenceTransformer
# Model that converts text into embeddings.
# Example: model.encode(["Hello world"]) → vector of 384 numbers.

# 12️⃣ Payload
# Extra info stored with a vector (metadata like source, text, or ID).
# Example: payload = {"source": "file1.pdf", "text": "AI is intelligence in machines."}

# 13️⃣ UUID
# Unique identifier for each chunk, often generated from source name and index.
# Example: "c839c9b6-efaa-4f8e-bec9-9326bb7c25d3"

# 14️⃣ Environment Variables (.env)
# Secure way to store API keys or configuration values.
# Example: GEMINI_API_KEY="abcdef123"

# 15️⃣ Gemini (via OpenAI API)
# Google’s AI model, used for generating answers using the context retrieved from Qdrant.
# Example: Model reads relevant chunks and answers your question.

# 16️⃣ COSINE Distance
# Measures similarity between two vectors — smaller angle = more similar meaning.
# Example: cosine(“dog”, “puppy”) ≈ high similarity.

# 17️⃣ Context
# The retrieved text given to the LLM so it can generate an accurate answer.
# Example: “According to page 4, AI was first coined in 1956...”

# ==============================
# 💻 PIPELINE OVERVIEW
# ==============================

# Step 1: Upload PDF
# Step 2: Split into chunks (chunking)
# Step 3: Create embeddings (SentenceTransformer)
# Step 4: Store vectors in Qdrant
# Step 5: Ask a question → embed query → search similar chunks
# Step 6: Send found chunks + question → Gemini → get answer
