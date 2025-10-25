import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import os
import datetime
import uuid
from openai import OpenAI  
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult

# Load environment variables from .env file (like API keys)
load_dotenv()

# Fetch Google Gemini API key from environment
google_api_key = os.getenv("GEMINI_API_KEY")

# Initialize the Inngest client — used for background event-driven workflows
inngest_client = inngest.Inngest(
    app_id="rag_ai_app",                     # App identifier
    logger=logging.getLogger("uvicorn"),     # Use uvicorn logger for console logging
    is_production=False,                     # Development mode (no real throttling)
    serializer=inngest.PydanticSerializer()  # Serializer for Pydantic-based data
)

# -----------------------------------------------------------
# INGEST FUNCTION: Triggered when a new PDF needs to be added
# -----------------------------------------------------------
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",                               # Unique function ID for Inngest
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),  # Event name to trigger this function
)
# Optional advanced controls (commented for now):
# throttle=inngest.Throttle(count=2, period=datetime.timedelta(minutes=1))
# rate_limit=inngest.RateLimit(limit=1, period=datetime.timedelta(hours=2))
# key="event.data.source_id"
# → Throttle = total requests limit per time frame (global)
# → Rate limit = per-source limit (per key)
# → key defines which field (source_id) is used for per-key limits

async def rag_ingest_pdf(ctx: inngest.Context):
    """
    Step 1 of RAG pipeline:
    Load, chunk, embed, and upsert PDF data into Qdrant vector database.
    """
    logger = logging.getLogger("uvicorn")

    # -------------------------------
    # Step 1A: Load and chunk the PDF
    # -------------------------------
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]  # Get file path from incoming event
        logger.info(f"🔹 Loading PDF from path: {pdf_path}")

        # Ensure file exists before proceeding
        if not os.path.exists(pdf_path):
            logger.error(f"❌ PDF not found at: {pdf_path}")
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        # Use provided source_id if exists, else use file path as fallback
        source_id = ctx.event.data.get("source_id", pdf_path)

        # Load PDF content and split into smaller chunks
        chunks = load_and_chunk_pdf(pdf_path)
        logger.info(f"✅ Loaded and chunked PDF into {len(chunks)} pieces")

        # Return structured result object
        return RAGChunkAndSrc(chunks=chunks, sources_id=source_id)

    # -----------------------------------------
    # Step 1B: Embed and upsert into Qdrant DB
    # -----------------------------------------
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.sources_id

        # Create embeddings using SentenceTransformer
        logger.info(f"🔹 Creating embeddings for {len(chunks)} chunks from source {source_id}")
        vecs = embed_texts(chunks)
        logger.info(f"✅ Generated embeddings")

        # Generate deterministic UUIDs for each chunk (based on source_id + index)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]

        # Each vector’s metadata (for tracing and referencing later)
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]

        # Initialize connection to Qdrant (vector database)
        storage = QdrantStorage(dims=384)
        logger.info("🧠 Upserting into Qdrant...")

        # Upload embeddings with metadata to Qdrant collection
        storage.upsert_vectors(ids, vecs, payloads)
        logger.info("✅ Upsert completed")

        # Return summary of ingestion
        return RAGUpsertResult(ingested=len(chunks))

    # Log start of the ingestion workflow
    logger.info("🚀 Starting RAG PDF ingestion...")

    # Execute step 1: load and chunk PDF
    chunks_and_src = await ctx.step.run(
        "Load and Chunk PDF",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc
    )

    # Execute step 2: embed and upsert into vector DB
    ingested = await ctx.step.run(
        "Embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult
    )

    logger.info(f"🎉 Ingestion completed. Total chunks: {ingested.ingested}")
    return ingested.model_dump()


# ------------------------------------------------------------
# QUERY FUNCTION: Triggered to search and answer using context
# ------------------------------------------------------------
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    """
    Step 2 of RAG pipeline:
    Given a user question, retrieve similar chunks and use Gemini to answer.
    """

    # ------------------------
    # Inner function: Searching
    # ------------------------
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        # Convert the input question into an embedding vector
        query_vec = embed_texts([question])[0]

        # Connect to Qdrant and perform similarity search
        store = QdrantStorage(dims=384)
        found = store.search_vectors(query_vec, top_k)

        # Return matched contexts and their sources
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    # Extract question and top_k parameter from event data
    questions = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    # Execute embedding and vector search as a step
    found = await ctx.step.run(
        "Embed-and-search",
        lambda: _search(questions, top_k),
        output_type=RAGSearchResult
    )

    # Build a context block string for the AI model
    content_block = "\n\n".join(f"- {c}" for c in found.contexts)

    # Combine context and question for the model prompt
    user_context = (
        "Use the following context to answer the question.\n"
        f"Context:\n{content_block}\n\n"
        f"Question: {questions}\n"
        "Answer concisely based on the context provided."
    )

    # -------------------------------------
    # Step 2B: Send request to Gemini model
    # -------------------------------------
    # OpenAI-compatible Gemini client (uses OpenAI SDK but points to Gemini API)
    gemini = OpenAI(
        api_key=google_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # Select Gemini model
    model_name = "gemini-2.0-flash"

    # Compose chat messages
    messages = [
        {"role": "system", "content": "Answer the question using only the provided context."},
        {"role": "user", "content": user_context}
    ]

    # Send prompt to Gemini and get response
    response = gemini.chat.completions.create(model=model_name, messages=messages)

    # Extract the AI's answer text
    answer = response.choices[0].message.content

    # Return structured result including sources and answer
    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }


# --------------------------------
# FASTAPI APP + INNGEST ENDPOINTS
# --------------------------------
# Create a FastAPI app and register Inngest functions as HTTP routes
app = FastAPI()

inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf, rag_query_pdf_ai]
)
