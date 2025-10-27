# frontend_rag_dashboard.py
"""
RAG Dashboard — a streamlined Streamlit interface for managing
document ingestion and AI-assisted querying via Inngest.
"""

import asyncio
import os
import time
from pathlib import Path

import requests
import streamlit as st
import inngest
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

load_dotenv()
st.set_page_config(page_title="RAG Dashboard", page_icon="📚", layout="wide")

# ------------------------------------------------------------------
# Inngest Helpers
# ------------------------------------------------------------------

@st.cache_resource
def inngest_client() -> inngest.Inngest:
    """Create (and cache) a local Inngest client."""
    return inngest.Inngest(app_id="rag_dashboard", is_production=False)


async def trigger_ingest_event(filepath: Path) -> None:
    """Send a document ingestion event to Inngest."""
    client = inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(filepath.resolve()),
                "source_id": filepath.name,
            },
        )
    )


async def trigger_query_event(prompt: str, top_n: int) -> str:
    """Send a question to Inngest and return the event ID."""
    client = inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": prompt, "top_k": top_n},
        )
    )
    return result[0]


# ------------------------------------------------------------------
# Local utility functions
# ------------------------------------------------------------------

def save_temp_file(uploaded_file) -> Path:
    """Save an uploaded PDF file locally."""
    save_dir = Path("user_uploads")
    save_dir.mkdir(exist_ok=True)
    path = save_dir / uploaded_file.name
    path.write_bytes(uploaded_file.getbuffer())
    return path


def inngest_api_url() -> str:
    """Return base Inngest API endpoint (uses local by default)."""
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_run_data(event_id: str) -> list[dict]:
    """Fetch all runs for a given event."""
    resp = requests.get(f"{inngest_api_url()}/events/{event_id}/runs")
    resp.raise_for_status()
    return resp.json().get("data", [])


def poll_for_output(event_id: str, timeout: int = 120, delay: float = 0.6) -> dict:
    """Poll until Inngest run completes or fails."""
    start_time = time.time()
    last_state = None
    while True:
        runs = fetch_run_data(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_state = status
            if status in ("Completed", "Succeeded", "Finished", "Success"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Run ended with status: {status}")
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timed out while waiting (last={last_state})")
        time.sleep(delay)


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------

st.title("📚 RAG Dashboard")
st.markdown("Interact with your document-based retrieval system powered by **Inngest**.")

tab_ingest, tab_query = st.tabs(["📄 Upload & Ingest", "💬 Ask Questions"])

# =========================
# Tab 1 — Upload & Ingest
# =========================
with tab_ingest:
    st.subheader("Upload a new document")

    uploaded_pdf = st.file_uploader("Select a PDF to upload", type=["pdf"])

    if uploaded_pdf:
        st.info(f"Selected file: `{uploaded_pdf.name}`")
        if st.button("Ingest Document 🚀"):
            with st.spinner("Uploading and triggering backend ingestion..."):
                pdf_path = save_temp_file(uploaded_pdf)
                asyncio.run(trigger_ingest_event(pdf_path))
                time.sleep(0.5)
            st.success(f"Ingestion started for **{uploaded_pdf.name}** ✅")
            st.caption("You can now switch to the *Ask Questions* tab once processing is done.")

# =========================
# Tab 2 — Query PDFs
# =========================
with tab_query:
    st.subheader("Ask a question about your indexed documents")

    question = st.text_area("Type your question below:")
    top_k = st.slider("Number of chunks to retrieve", 1, 15, 5)

    if st.button("Submit Question 🤖"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Sending question to backend..."):
                event_id = asyncio.run(trigger_query_event(question.strip(), top_k))
                result = poll_for_output(event_id)

            st.success("Received response from backend.")
            st.markdown("### Answer")
            st.write(result.get("answer", "_No answer returned._"))

            srcs = result.get("sources", [])
            if srcs:
                st.markdown("### References")
                st.write(f"- {srcs[-1]}")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.divider()
st.caption(
    "Built with Streamlit & Inngest — this interface coordinates ingestion and retrieval tasks asynchronously."
)
