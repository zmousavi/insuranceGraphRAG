"""
retrieval/retrieve.py
=====================
Graph RAG retrieval pipeline for insurance policies and claims.

RUNTIME WORKFLOW
----------------
# Step 1 — Embed query
#   Vertex AI text-embedding-004 → q_vec (768d)

# Step 2 — Cluster routing  [route_clusters()]
#   cosine_search() → top 50 seeds + scores
#   shadow_id → doc_id (shadow_to_doc dict) → cluster_id (clusters dict)
#   Group seeds by cluster → compute mean cosine score per cluster
#   Select top 2 clusters

# Step 3 — Anchor selection  [get_anchor_shadows()]
#   Filter top 50 seeds to those whose doc_id is in top 2 clusters → anchor shadows

# Step 4 — Graph expansion  [expand_anchor()]
#   For each anchor shadow, one Neo4j session:
#     a) Resolve: Shadow <-[HAS_SHADOW]- Section <-[HAS_SECTION]- doc
#        Collect section title + summary (summary = router; always relevant for anchor's own section)
#        Fetch ALL sibling shadows in same section (anchor's section is always relevant)
#        Fetch NEXT_CHUNK neighbor for text continuity across section boundary
#     b) If doc is ClaimDocument:
#        Follow -[REFERENCES_POLICY]-> Policy
#        Read ALL policy section summaries → _score_relevance(query, summary) → pick top 1
#        Fetch that section's shadows (up to 2 chunks) as policy_terminal
#        Read endorsement summaries → if relevant → fetch endorsement section shadows
#     c) If doc is Policy:
#        Read other section summaries → _score_relevance() → pick top 1
#        Fetch that section's shadows (up to 2 chunks) as policy_section_terminal
#        Read endorsement summaries → if relevant → fetch endorsement section shadows
#     Returns path: {node_ids, edge_types, tiered_texts, anchor_score, terminal_node_id}
#     terminal_node_id = deepest node reached (policy_id or doc_id) — used for dedup in Step 6

# Step 5 — Condense paths  [condense_path()]
#   anchor text (≤200 words) + "Path: A → B → C" edge types + terminal text (≤100 words)
#   Keeps each condensed path under ~512 tokens so cross-encoder can read query+path together

# Step 6 — Cross-encoder reranking  [rerank_paths()]
#   cross_encoder.predict([(query, condensed_path), ...]) → scores in one batch call
#   Sort by score descending → deduplicate on terminal_node_id (keep best path per destination)
#   Keep top 5 paths

# Step 7 — Synthesize  [synthesize()]
#   Flatten tiered_texts from top 5 paths → source-tagged passages
#   Load system prompt from prompts dict (rag_v2 or graph_rag_v2)
#   gemini.models.generate_content(model, contents=user_msg, config={system_instruction})

# Step 8 — Return RetrievalResult  [retrieve()]
#   answer, mode, supporting_paths, clusters_used, latency_breakdown
#   latency_breakdown keys: embed_ms, search_ms, cluster_ms, expand_ms, rerank_ms, synth_ms, total_ms

# RAG BASELINE MODE  [retrieve(mode="rag")]
#   Steps 1 → cosine_search top 5 → shadow_texts lookup → synthesize directly
#   Skips: cluster routing, graph expansion, path condensing, cross-encoder reranking

# TODO: add Redis cache (Step 0: cache check before embed, Step 9: cache write after synthesis)
# TODO: upgrade to ReAct agent (replace fixed traversal with Gemini function calling loop)
"""

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
import numpy as np
import faiss
import yaml
from neo4j import GraphDatabase
from google import genai
from sentence_transformers import CrossEncoder

load_dotenv()

PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = "global"
MODEL    = "gemini-2.5-flash"
EMBED_MODEL = "text-embedding-004"

ROOT            = Path(__file__).parent.parent
EMBEDDINGS_FILE = ROOT / "manifest" / "embeddings.json"
CLUSTERS_FILE   = ROOT / "manifest" / "clusters.json"
PROMPTS_FILE    = ROOT / "prompts.yaml"

VECTOR_TOP_K       = 50   # seeds for cluster routing
CLUSTER_TOP_N      = 2    # clusters selected per query
RERANK_TOP_K       = 5    # paths kept after cross-encoder reranking
ANCHOR_MAX_WORDS   = 200  # words from anchor shadow in condensed path
TERMINAL_MAX_WORDS = 100  # words from terminal node in condensed path
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RetrievalResult:
    answer:            str
    mode:              str
    supporting_paths:  list[dict] = field(default_factory=list)
    clusters_used:     list[int]  = field(default_factory=list)
    latency_breakdown: dict       = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Startup — assets loaded at import time
# TODO: switch to lazy loading when embeddings grow large or cross-encoder
#       load time becomes a bottleneck (e.g. multiple workers, serverless cold starts)
# ---------------------------------------------------------------------------

embeddings    = json.loads(EMBEDDINGS_FILE.read_text())   # shadow_id → 768d vector
clusters      = json.loads(CLUSTERS_FILE.read_text())     # node_id   → cluster_id
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

# Build FAISS index — IndexFlatIP (exact inner product search).
# We normalize all vectors so inner product = cosine similarity.
# Rebuilt from embeddings.json at startup — fast for 436 vectors, no file needed.
# TODO: for millions of vectors switch to IndexIVFFlat (approximate, needs training)
shadow_ids = list(embeddings.keys())
_vectors   = np.array(list(embeddings.values()), dtype=np.float32)
faiss.normalize_L2(_vectors)                              # normalize in-place → unit vectors
faiss_index = faiss.IndexFlatIP(768)                      # 768 = embedding dimension
faiss_index.add(_vectors)                                 # add all shadow vectors

_manifest = json.loads((ROOT / "manifest" / "manifest.json").read_text())

# RAG baseline needs shadow text without touching Neo4j:
#   cosine_search returns shadow_ids → look up text here → synthesize directly.
shadow_texts: dict[str, str] = {
    shadow["shadow_id"]: shadow["text"]
    for doc in _manifest
    for shadow in doc.get("shadows", [])
}

# Cluster routing: cosine_search returns shadow_ids → look up their doc_id here
#   → look up cluster_id in clusters dict → group seeds by cluster → rank clusters.
# Built from manifest so we don't need Neo4j for this hop.
shadow_to_doc: dict[str, str] = {
    shadow["shadow_id"]: doc["doc_id"]
    for doc in _manifest
    for shadow in doc.get("shadows", [])
}

# prompts.yaml has rag_v1/v2, graph_rag_v1/v2 system prompts.
# Loaded once at startup — no disk I/O per query.
prompts: dict = yaml.safe_load(PROMPTS_FILE.read_text())


def get_driver():
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    driver.verify_connectivity()
    return driver


def init_gemini() -> genai.Client:
    return genai.Client(vertexai=True, project=PROJECT, location=LOCATION)


driver = get_driver()
gemini = init_gemini()


# ---------------------------------------------------------------------------
# Step 1 — Embed query
# ---------------------------------------------------------------------------

def embed_query(question: str) -> np.ndarray:
    response = gemini.models.embed_content(model=EMBED_MODEL, contents=question)
    return np.array(response.embeddings[0].values)


# ---------------------------------------------------------------------------
# Step 2 — Cosine search via FAISS
# ---------------------------------------------------------------------------

def cosine_search(q_vec: np.ndarray, top_k: int = VECTOR_TOP_K) -> list[tuple[str, float]]:
    """
    Search the FAISS index for the top_k most similar shadows.
    q_vec must be normalized (unit vector) so inner product = cosine similarity.
    Returns list of (shadow_id, score) sorted descending.

    FAISS search returns:
      scores  shape: (1, top_k) — similarity score per result
      indices shape: (1, top_k) — position in faiss_index (maps to shadow_ids list)
    """
    q = q_vec.astype(np.float32).reshape(1, -1)   # FAISS expects shape (1, dim)
    faiss.normalize_L2(q)                          # normalize query → unit vector
    scores, indices = faiss_index.search(q, top_k)
    return [
        (shadow_ids[idx], float(scores[0][i]))
        for i, idx in enumerate(indices[0])
        if idx != -1   # FAISS returns -1 for empty slots when top_k > index size
    ]


# ---------------------------------------------------------------------------
# Step 7 — Synthesize with Gemini
# ---------------------------------------------------------------------------

def synthesize(question: str, texts: list[str], mode: str, prompt_version: str = "v2") -> str:
    """
    Call Gemini with a system prompt from prompts.yaml and the retrieved passages.

    mode + prompt_version → key in prompts dict, e.g. "rag_v2" or "graph_rag_v2".
    texts are already source-tagged by the caller ("[section_title]\nchunk text").
    Falls back to a minimal instruction if the key is not found.
    """
    prompt_key    = f"{mode}_{prompt_version}"
    system_prompt = prompts.get(prompt_key, {}).get("system", "Answer using only the provided passages.")

    passages = "\n\n---\n\n".join(texts)
    user_msg = f"Passages:\n{passages}\n\nQuestion: {question}"

    response = gemini.models.generate_content(
        model=MODEL,
        contents=user_msg,
        config={"system_instruction": system_prompt},
    )
    return response.text


# ---------------------------------------------------------------------------
# Main retrieval entry point
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    mode: str = "rag",
    prompt_version: str = "v2",
    top_k: int = RERANK_TOP_K,
) -> RetrievalResult:
    """
    Run the retrieval pipeline and return a RetrievalResult.

    mode="rag"       — cosine top_k → shadow texts → synthesize (no graph, no reranking)
    mode="graph_rag" — full 8-step pipeline (cluster routing, graph expansion, reranking)
                       Graph RAG steps are added in the next phase.
    top_k            — number of shadows passed to Gemini (RAG mode only; Graph RAG uses RERANK_TOP_K)
    """
    t0 = time.time()

    q_vec = embed_query(question)
    t1 = time.time()

    if mode == "rag":
        seeds = cosine_search(q_vec, top_k=top_k)
        texts = [
            f"[{sid}]\n{shadow_texts[sid]}"
            for sid, _ in seeds
            if sid in shadow_texts
        ]
        answer = synthesize(question, texts, mode="rag", prompt_version=prompt_version)
        return RetrievalResult(
            answer=answer,
            mode="rag",
            latency_breakdown={
                "embed_ms": round((t1 - t0) * 1000),
                "total_ms": round((time.time() - t0) * 1000),
            },
        )

    raise NotImplementedError("graph_rag mode not yet implemented — coming next")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run retrieval pipeline on a question.")
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--mode", choices=["rag", "graph_rag"], default="rag")
    parser.add_argument("--prompt-version", default="v2", choices=["v1", "v2"])
    args = parser.parse_args()

    result = retrieve(args.question, mode=args.mode, prompt_version=args.prompt_version)

    print(f"\nMode   : {result.mode}")
    print(f"Latency: {result.latency_breakdown}")
    print(f"\nAnswer :\n{result.answer}")
