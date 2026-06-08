"""
api/main.py
===========
FastAPI wrapper around the retrieval pipeline.

Endpoints:
  POST /query   — run RAG or Graph RAG on a question
  GET  /health  — liveness check for Cloud Run

Usage (local):
  uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# GCS download — pull latest manifest files before loading the retriever.
# When GCS_BUCKET is set (Cloud Run), downloads from GCS so data can be
# updated without rebuilding the container.
# When GCS_BUCKET is not set (local dev), skips and uses files on disk.
# ---------------------------------------------------------------------------

def _download_manifest_from_gcs():
    bucket_name = os.getenv("GCS_BUCKET")
    if not bucket_name:
        return

    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    manifest_dir = ROOT / "manifest"
    manifest_dir.mkdir(exist_ok=True)

    for filename in ["manifest.json", "embeddings.json", "clusters.json"]:
        blob = bucket.blob(f"manifest/{filename}")
        blob.download_to_filename(manifest_dir / filename)


# ---------------------------------------------------------------------------
# Startup — load retriever once when the server starts, not per request.
# FAISS index, Neo4j driver, cross-encoder, and Gemini client are all
# expensive to initialise — this keeps them alive across all requests.
# ---------------------------------------------------------------------------

_retrieve = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retrieve
    _download_manifest_from_gcs()
    from retrieval.retrieve import retrieve
    _retrieve = retrieve
    yield


app = FastAPI(
    title="Insurance Graph RAG",
    description="RAG and Graph RAG retrieval over insurance policies and claims.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# Pydantic validates incoming JSON and generates the /docs OpenAPI UI.
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer")
    mode: str = Field("graph_rag", pattern="^(rag|graph_rag)$", description="rag or graph_rag")
    top_k: int = Field(7, ge=1, le=20, description="Paths (graph_rag) or chunks (rag) passed to Gemini")


class QueryResponse(BaseModel):
    answer: str
    mode: str
    latency_ms: int
    clusters_used: list[int]
    supporting_paths: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check — Cloud Run calls this to confirm the container is ready."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Run the retrieval pipeline on a question.

    - mode=rag       — cosine top_k chunks → Gemini (no graph)
    - mode=graph_rag — cluster routing → graph expansion → cross-encoder → Gemini
    """
    if _retrieve is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")

    result = _retrieve(req.question, mode=req.mode, top_k=req.top_k)

    return QueryResponse(
        answer=result.answer,
        mode=result.mode,
        latency_ms=result.latency_breakdown.get("total_ms", 0),
        clusters_used=result.clusters_used,
        supporting_paths=result.supporting_paths,
    )
