🧠 RAG AI Agent

A Retrieval-Augmented Generation (RAG) system that lets you upload and query PDFs — combining semantic search with generative AI answers.
This project uses SentenceTransformer for embeddings, Qdrant (via Docker) as a vector database, and Gemini API (through OpenAI-compatible endpoints) for answer generation.
All dependencies are managed using UV, the next-generation Python package manager.

🚀 Features

📄 PDF Ingestion — Chunk and embed document content using sentence-transformers.

🔍 Semantic Search — Store and retrieve document vectors from Qdrant.

🤖 AI-Powered Answers — Use Gemini (via OpenAI endpoints) to generate contextual answers from retrieved text.

⚙️ Event-Driven Workflow — Powered by Inngest for scalable function execution.

🌐 FastAPI backend and optional Streamlit frontend for interactive querying.

🧩 Containerized Vector DB — Qdrant runs fully in Docker.

🧰 UV-based project — Modern, reproducible dependency management.

🏗️ Tech Stack
Component	Tool / Library
Language	Python 3.12+
Package Manager	UV

Web Framework	FastAPI

Vector Database	Qdrant
 (Docker)
Embeddings	SentenceTransformers

LLM	Google Gemini API
 (via OpenAI-compatible endpoints)
Event Handling	Inngest

Frontend	Streamlit
🧩 Project Structure
rag-ai-agent/
│
├── main.py                 # FastAPI app entrypoint
├── data_loader.py          # Loads and chunks PDF content
├── vector_db.py            # Handles Qdrant operations
├── custom_types.py         # Pydantic models for data
├── .env                    # Environment variables
├── Dockerfile / docker-run # Qdrant setup
├── uv.lock / pyproject.toml# Dependency management via UV
└── README.md               # Project documentation

⚙️ Setup Instructions
1. Clone the repo
git clone https://github.com/yourusername/rag-ai-agent.git
cd rag-ai-agent

2. Start Qdrant in Docker
docker run -d --name qdrantRagDb \
  -p 6333:6333 \
  -v "${PWD}/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant

3. Set up environment variables

Create a .env file in the project root:

GEMINI_API_KEY=your_gemini_key
QDRANT_URL=http://localhost:6333

4. Install dependencies using UV
uv sync

5. Run the FastAPI server
uvicorn main:app --reload

6. (Optional) Run Streamlit UI
streamlit run app.py

🧠 How It Works

PDF Upload → Chunking:
data_loader.py splits documents into overlapping text chunks.

Embedding → Storage:
Chunks are embedded via SentenceTransformer and stored in Qdrant.

User Query → Retrieval:
The query is embedded and matched against stored vectors to retrieve context.

Context → Generation:
Retrieved text is passed to Gemini (via OpenAI endpoints) for generating human-like answers.

Response → User:
The system returns a concise, context-aware answer.

🧩 Example Usage

Query Example:

"What does the document say about revenue growth?"

Response:

"The report indicates that revenue grew by 15% year-over-year due to strong Q2 performance."

🧱 Future Enhancements

🧬 Support for multiple document formats (DOCX, TXT)

🕸️ Multi-source retrieval (web + PDFs)

☁️ Cloud deployment (AWS / GCP)

🧭 Improved ranking using cross-encoders