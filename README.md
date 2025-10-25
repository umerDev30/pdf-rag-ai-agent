🧠 RAG AI Agent

A Retrieval-Augmented Generation (RAG) system that lets you upload and query PDFs — combining semantic search with generative AI answers.
This project uses SentenceTransformer for embeddings, Qdrant (via Docker) as a vector database, and Gemini API (through OpenAI-compatible endpoints) for answer generation.
All dependencies are managed using UV, the next-generation Python package manager.

🚀 Features

📄 PDF Ingestion — Chunk and embed document content using SentenceTransformers

🔍 Semantic Search — Store and retrieve document vectors from Qdrant

🤖 AI-Powered Answers — Use Gemini (via OpenAI endpoints) to generate contextual answers

⚙️ Event-Driven Workflow — Powered by Inngest for scalable function execution

🌐 FastAPI Backend and optional Streamlit Frontend for interactive querying

🧩 Containerized Vector DB — Qdrant runs fully in Docker

🧰 UV-Based Project — Modern, reproducible dependency management

🏗️ Tech Stack
Component	Tool / Library
Language	Python 3.12+
Package Manager	UV
Web Framework	FastAPI
Vector Database	Qdrant (Docker)
Embeddings	SentenceTransformers
LLM	Google Gemini API (via OpenAI-compatible endpoints)
Event Handling	Inngest
Frontend	Streamlit
🧩 Project Structure
rag-ai-agent/
│
├── __pycache__/            # Python cache files
├── .venv/                  # Virtual environment created by UV
│
├── qdrant_storage/         # Local folder mounted to Docker for persistent Qdrant data
├── uploads/                # Folder for uploaded PDFs
│
├── .env                    # Environment variables
├── .gitignore              # Git ignore rules
├── .python-version         # Python version specification (for UV/pyenv)
│
├── custom_types.py         # Pydantic models for data schemas
├── data_loader.py          # Loads and chunks PDF content
├── glossary.py             # Contains simplified definitions for technical terms
├── main.py                 # FastAPI app entrypoint
├── reset_qdrant.py         # Script to reset or clear Qdrant collections
├── streamlit_app.py        # Streamlit frontend interface
├── vector_db.py            # Handles Qdrant vector operations
│
├── pyproject.toml          # Project dependencies (UV)
├── requirements.txt        # Optional standard dependency list
├── uv.lock                 # Dependency lockfile for reproducibility
└── README.md               # Project documentation


🗂️ Note:

qdrant_storage/ — created by Docker for persistent Qdrant data.

uploads/ — stores uploaded PDFs for processing.

reset_qdrant.py — helper to clear or reset your Qdrant database.

⚙️ Setup & Run Commands (Sequential)
1. Clone the Repository
git clone https://github.com/umerDev30/rag-ai-agent.git
cd rag-ai-agent

2. Start Qdrant via Docker
docker run -d --name qdrantRagDb \
  -p 6333:6333 \
  -v "${PWD}/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant


The qdrant_storage folder is automatically created for persistent Qdrant data.

3. Create the .env File
GEMINI_API_KEY=your_gemini_key
QDRANT_URL=http://localhost:6333

4. Initialize the Project
uv init

5. Install Dependencies
uv sync

6. Run the FastAPI Server
uv run uvicorn main:app


Server will start at: http://127.0.0.1:8000

7. Start the Inngest Dev Server (for ingestion events)
inngest dev -u http://127.0.0.1:8000/api/inngest --no-discovery

8. (Optional) Launch Streamlit Frontend
uv run streamlit run .\streamlit_app.py


Streamlit will open at: http://localhost:8501

🧠 How It Works

PDF Upload → Chunking — data_loader.py splits documents into overlapping text chunks.

Embedding → Storage — Chunks are embedded via SentenceTransformer and stored in Qdrant.

User Query → Retrieval — The query is embedded and matched against stored vectors.

Context → Generation — Retrieved text is passed to Gemini for generating answers.

Response → User — Returns a concise, context-aware answer.

🧩 Example Usage

Query Example:

“What does the document say about revenue growth?”

Response:

“The report indicates that revenue grew by 15% year-over-year due to strong Q2 performance.”

🧱 Future Enhancements

🧬 Support for multiple document formats (DOCX, TXT)

🕸️ Multi-source retrieval (web + PDFs)

☁️ Cloud deployment (AWS / GCP)

🧭 Improved ranking using cross-encoders
