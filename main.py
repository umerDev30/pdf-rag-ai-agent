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

load_dotenv()

google_api_key = os.getenv("GEMINI_API_KEY")

inngest_client = inngest.Inngest(
    app_id="rag_ai_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(count=2, period= datetime.timedelta(minutes=1)),
    rate_limit=inngest.RateLimit(limit=1, period=datetime.timedelta(hours=2)),
    key="event.data.source_id"
)
# Throttle: Global limit — max 2 PDFs ingested per minute, no matter the source.
# Rate limit: Per-key limit — each source_id can be ingested only once every 2 hours.
# Key: Determines which field (source_id) is used for the per-key rate limit.

async def rag_ingest_pdf(ctx: inngest.Context):
    logger = logging.getLogger("uvicorn")

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        logger.info(f"🔹 Loading PDF from path: {pdf_path}")
        if not os.path.exists(pdf_path):
            logger.error(f"❌ PDF not found at: {pdf_path}")
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        logger.info(f"✅ Loaded and chunked PDF into {len(chunks)} pieces")
        return RAGChunkAndSrc(chunks=chunks, sources_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.sources_id
        logger.info(f"🔹 Creating embeddings for {len(chunks)} chunks from source {source_id}")
        vecs = embed_texts(chunks)  # SentenceTransformer embeddings
        logger.info(f"✅ Generated embeddings")

        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]

        storage = QdrantStorage(dims=384)
        logger.info("🧠 Upserting into Qdrant...")
        storage.upsert_vectors(ids, vecs, payloads)
        logger.info("✅ Upsert completed")
        return RAGUpsertResult(ingested=len(chunks))

    logger.info("🚀 Starting RAG PDF ingestion...")
    chunks_and_src = await ctx.step.run(
        "Load and Chunk PDF",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc
    )
    ingested = await ctx.step.run(
        "Embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult
    )
    logger.info(f"🎉 Ingestion completed. Total chunks: {ingested.ingested}")
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage(dims=384)
        found = store.search_vectors(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    questions = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run(
        "Embed-and-search",
        lambda: _search(questions, top_k),
        output_type=RAGSearchResult
    )

    content_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_context = (
        "Use the following context to answer the question.\n"
        f"Context:\n{content_block}\n\n"
        f"Question: {questions}\n"
        "Answer concisely based on the context provided."
    )

    # OpenAI-compatible Gemini client
    gemini = OpenAI(api_key=google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    model_name = "gemini-2.0-flash"

    # Generate answer
    messages = [
        {"role": "system", "content": "Answer the question using only the provided context."},
        {"role": "user", "content": user_context}
    ]

    response = gemini.chat.completions.create(model=model_name, messages=messages)
    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }


app = FastAPI()
inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf, rag_query_pdf_ai]
)
