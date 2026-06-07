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
os.environ.setdefault("OMP_NUM_THREADS", "1")       # prevent FAISS/PyTorch OpenMP conflict on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
ANCHOR_MAX_WORDS   = 150  # words from anchor section in condensed path (cross-encoder 512-token limit)
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

_all_embeddings = json.loads(EMBEDDINGS_FILE.read_text())   # shadow_id + sec_<section_id> → 768d vector
clusters        = json.loads(CLUSTERS_FILE.read_text())     # node_id → cluster_id
cross_encoder   = CrossEncoder(CROSS_ENCODER_MODEL)

# Split embeddings into shadows and sections by key prefix.
# Shadows go into the FAISS index for cosine search.
# Sections are kept as a dict for _best_section cosine scoring.
embeddings         = {k: v for k, v in _all_embeddings.items() if not k.startswith("sec_")}
section_embeddings = {k[4:]: np.array(v, dtype=np.float32)      # strip sec_ prefix → section_id
                      for k, v in _all_embeddings.items() if k.startswith("sec_")}

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
# Steps 2-3 — Cluster routing + anchor selection
# ---------------------------------------------------------------------------

def route_clusters(seeds: list[tuple[str, float]]) -> list[int]:
    """
    Group top-50 cosine seeds by cluster_id, rank clusters by mean cosine score,
    return top CLUSTER_TOP_N cluster ids.
    shadow_id → doc_id (shadow_to_doc) → cluster_id (clusters)
    """
    cluster_scores: dict[int, list[float]] = {}
    for sid, score in seeds:
        doc_id = shadow_to_doc.get(sid)
        if doc_id is None:
            continue
        cid = clusters.get(doc_id)
        if cid is None:
            continue
        cluster_scores.setdefault(cid, []).append(score)
    ranked = sorted(
        cluster_scores.items(),
        key=lambda x: sum(x[1]) / len(x[1]),
        reverse=True,
    )
    return [cid for cid, _ in ranked[:CLUSTER_TOP_N]]


def get_anchor_shadows(seeds: list[tuple[str, float]], top_clusters: list[int]) -> list[tuple[str, float]]:
    """Filter seeds to those whose doc_id belongs to one of the top clusters."""
    top_set = set(top_clusters)
    return [
        (sid, score) for sid, score in seeds
        if clusters.get(shadow_to_doc.get(sid)) in top_set
    ]


# ---------------------------------------------------------------------------
# Step 4 helpers
# ---------------------------------------------------------------------------

def _score_relevance(query: str, text: str) -> float:
    """Word-overlap score — used to rank section summaries without an extra API call.
    TODO: replace with cosine similarity. Section summary embeddings should be computed once
    in pipeline/embed.py and stored in embeddings.json alongside shadow embeddings. At query
    time, dot product against the query vector instead of word overlap — no extra API call,
    handles synonyms that word overlap misses (e.g. 'pipe threshold' vs 'plumbing age exclusion').
    """
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    if not q_words or not t_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)


def _fetch_section_shadows(session, section_id: str, label: str, limit: int = 2) -> list[str]:
    """Return up to `limit` shadow texts for a given section id."""
    rows = session.run(
        "MATCH (sec:Section {id: $sid})-[:HAS_SHADOW]->(sw:Shadow) "
        "RETURN sw.text AS text LIMIT $lim",
        sid=section_id, lim=limit,
    ).data()
    return [f"[{label}]\n{r['text']}" for r in rows if r["text"]]


def _first(result) -> dict | None:
    """Take the first record from a Neo4j result without raising a warning for multi-row results."""
    rows = result.data()
    return rows[0] if rows else None


def _best_section(query: str, sections: list[dict]) -> dict | None:
    """Pick the section whose title+summary has the highest word-overlap with query.
    TODO: upgrade to cosine similarity using section_embeddings. Attempted but caused
    regressions — cosine scores are too similar across sections (0.737 vs 0.743), causing
    wrong sections to win. Needs better section summary quality or a hybrid approach.
    """
    if not sections:
        return None
    return max(
        sections,
        key=lambda s: _score_relevance(query, (s.get("summary") or "") + " " + (s.get("title") or "")),
    )


# ---------------------------------------------------------------------------
# Step 4 — Graph expansion
# ---------------------------------------------------------------------------

def expand_anchor(anchor_id: str, anchor_score: float, query: str) -> dict | None:
    """
    One Neo4j session per anchor shadow. Traversal:
      Shadow ←HAS_SHADOW← Section ←HAS_SECTION← doc
        (a) fetch all sibling shadows + NEXT_CHUNK neighbor
        (b) ClaimDocument: follow REFERENCES_POLICY → Policy (as-filed) → best section + endorsements
        (c) Policy: score other sections → best section + endorsements
    Returns path dict or None if anchor not found in graph.
    """
    with driver.session() as session:
        row = _first(session.run("""
            MATCH (sw:Shadow {id: $sid})
            MATCH (sec:Section)-[:HAS_SHADOW]->(sw)
            MATCH (doc)-[:HAS_SECTION]->(sec)
            OPTIONAL MATCH (sec)-[:HAS_SHADOW]->(sib:Shadow)
            OPTIONAL MATCH (sw)-[:NEXT_CHUNK]->(nxt:Shadow)
            WITH sec, doc, sw, sib, nxt
            ORDER BY sib.chunk_index
            RETURN
                sec.id          AS sec_id,
                sec.title       AS sec_title,
                sec.summary     AS sec_summary,
                labels(doc)     AS doc_labels,
                doc.id          AS doc_id,
                doc.superseded  AS superseded,
                collect(DISTINCT {chunk_index: sib.chunk_index, text: sib.text}) AS siblings,
                nxt.text        AS next_text
        """, sid=anchor_id))

        if row is None:
            return None

        sec_id      = row["sec_id"]
        sec_title   = row["sec_title"] or ""
        doc_labels  = row["doc_labels"]
        doc_id      = row["doc_id"] or anchor_id
        siblings    = [s["text"] for s in sorted(row["siblings"], key=lambda x: x.get("chunk_index") or 0) if s["text"]]
        next_text   = row["next_text"]

        # anchor shadow text comes first in anchor_section — cosine selected it as the
        # most relevant chunk, so it should lead the condensed path fed to the cross-encoder
        anchor_text = shadow_texts.get(anchor_id, "")
        ordered = [anchor_text] + [t for t in siblings if t != anchor_text]

        tiered_texts: dict[str, list[str]] = {
            "anchor_section": [f"[{sec_title}]\n{t}" for t in ordered],
        }
        if next_text:
            tiered_texts["next_chunk"] = [next_text]

        node_ids   = [anchor_id, sec_id, doc_id]
        edge_types = ["HAS_SHADOW", "HAS_SECTION"]
        terminal   = doc_id

        # (b) ClaimDocument → REFERENCES_POLICY → Policy
        if "ClaimDocument" in doc_labels:
            pol_row = _first(session.run("""
                MATCH (cd:ClaimDocument {id: $cid})-[:REFERENCES_POLICY]->(pol:Policy)
                MATCH (pol)-[:HAS_SECTION]->(psec:Section)
                RETURN pol.id AS pol_id,
                       collect({id: psec.id, title: psec.title, summary: psec.summary}) AS sections
            """, cid=doc_id))

            if pol_row:
                pol_id    = pol_row["pol_id"]
                best_sec = _best_section(query, pol_row["sections"])
                if best_sec:
                    tiered_texts["policy_terminal"] = _fetch_section_shadows(
                        session, best_sec["id"], best_sec["title"] or pol_id
                    )
                node_ids.extend([pol_id])
                edge_types.append("REFERENCES_POLICY")
                terminal = pol_id

                # endorsements
                end_rows = _first(session.run("""
                    MATCH (pol:Policy {id: $pid})-[:HAS_ENDORSEMENT]->(end:Endorsement)
                    MATCH (end)-[:HAS_SECTION]->(esec:Section)
                    RETURN collect({id: esec.id, title: esec.title, summary: esec.summary}) AS sections
                """, pid=pol_id))
                if end_rows:
                    best_end = _best_section(query, end_rows["sections"])
                    if best_end and _score_relevance(query, (best_end.get("summary") or "") + " " + (best_end.get("title") or "")) > 0:
                        tiered_texts["endorsement"] = _fetch_section_shadows(
                            session, best_end["id"], best_end["title"] or "endorsement"
                        )

        # TODO: entity traversal not implemented (Q16, Q18, Q21 fail because of this).
        # Entity nodes (Contractor, Person, Organization) have no shadows so FAISS never
        # lands on them. Fix: at pipeline time, generate a text description for each entity
        # from its graph edges (e.g. "FastFix Restoration repaired claims X, Y, Z") and embed
        # it. Then FAISS can land on entity nodes, and expand_anchor can traverse REPAIRED_BY,
        # OWNS, MANAGED_BY edges here as a new branch. Keeps it pure RAG — no text-to-Cypher needed.

        # (c) Policy → other sections + endorsements
        elif "Policy" in doc_labels:
            sec_rows = _first(session.run("""
                MATCH (pol:Policy {id: $pid})-[:HAS_SECTION]->(psec:Section)
                WHERE psec.id <> $skip_sec
                RETURN collect({id: psec.id, title: psec.title, summary: psec.summary}) AS sections
            """, pid=doc_id, skip_sec=sec_id))

            if sec_rows:
                best_sec = _best_section(query, sec_rows["sections"])
                if best_sec:
                    tiered_texts["policy_section_terminal"] = _fetch_section_shadows(
                        session, best_sec["id"], best_sec["title"] or doc_id
                    )

            end_rows = _first(session.run("""
                MATCH (pol:Policy {id: $pid})-[:HAS_ENDORSEMENT]->(end:Endorsement)
                MATCH (end)-[:HAS_SECTION]->(esec:Section)
                RETURN collect({id: esec.id, title: esec.title, summary: esec.summary}) AS sections
            """, pid=doc_id))
            if end_rows:
                best_end = _best_section(query, end_rows["sections"])
                if best_end and _score_relevance(query, (best_end.get("summary") or "") + " " + (best_end.get("title") or "")) > 0:
                    tiered_texts["endorsement"] = _fetch_section_shadows(
                        session, best_end["id"], best_end["title"] or "endorsement"
                    )

    return {
        "node_ids":         node_ids,
        "edge_types":       edge_types,
        "tiered_texts":     tiered_texts,
        "anchor_score":     anchor_score,
        "terminal_node_id": terminal,
        "sec_summary":      row.get("sec_summary") or "",
    }


# ---------------------------------------------------------------------------
# Steps 5-6 — Condense paths + cross-encoder reranking
# ---------------------------------------------------------------------------

def condense_path(path: dict) -> str:
    """
    Flatten a path into a short string for cross-encoder scoring.
    section summary (compact, key facts first) + raw anchor text up to ANCHOR_MAX_WORDS total
    + edge chain + terminal text (≤TERMINAL_MAX_WORDS)
    """
    sec_summary = path.get("sec_summary", "")
    raw_anchor  = " ".join(path["tiered_texts"].get("anchor_section", []))
    # Prepend section summary so key facts (e.g. "25 years") reach the cross-encoder
    # even when raw shadow text has a long preamble before the relevant sentence.
    combined    = (sec_summary + "\n" + raw_anchor) if sec_summary else raw_anchor
    anchor_str  = " ".join(combined.split()[:ANCHOR_MAX_WORDS])

    terminal_texts = (
        path["tiered_texts"].get("policy_terminal") or
        path["tiered_texts"].get("policy_section_terminal") or
        path["tiered_texts"].get("endorsement") or
        []
    )
    terminal_str = " ".join(" ".join(terminal_texts).split()[:TERMINAL_MAX_WORDS])

    edge_str = " → ".join(path["edge_types"])
    parts = [p for p in [anchor_str, f"Path: {edge_str}", terminal_str] if p]
    return "\n".join(parts)


def rerank_paths(query: str, paths: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    Score (query, condensed_path) pairs with cross-encoder in one batch call.
    Deduplicate on terminal_node_id (keep best-scoring path per destination).
    Return top_k paths.
    """
    if not paths:
        return []
    condensed = [condense_path(p) for p in paths]
    # TODO: NaN root cause not fully diagnosed — likely empty condense_path output (no anchor text
    # or no terminal section shadows). Proper fix: validate condense_path output and skip empty
    # paths before predict() rather than masking NaN after the fact.
    scores    = np.nan_to_num(cross_encoder.predict([(query, c) for c in condensed]), nan=-999.0)
    ranked    = sorted(zip(scores, paths), key=lambda x: x[0], reverse=True)

    seen, result = set(), []
    for _, path in ranked:
        tid = path["terminal_node_id"]
        if tid not in seen:
            seen.add(tid)
            result.append(path)
        if len(result) >= top_k:
            break
    return result


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
    top_k            — chunks passed to Gemini (RAG) or paths kept after reranking (Graph RAG)
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
            supporting_paths=[{"shadow_id": sid, "score": round(score, 3)} for sid, score in seeds],
            latency_breakdown={
                "embed_ms": round((t1 - t0) * 1000),
                "total_ms": round((time.time() - t0) * 1000),
            },
        )

    # graph_rag — full 8-step pipeline
    seeds = cosine_search(q_vec, top_k=VECTOR_TOP_K)
    t_search = time.time()

    top_clusters = route_clusters(seeds)
    anchors      = get_anchor_shadows(seeds, top_clusters)
    t_cluster    = time.time()

    paths = [p for sid, score in anchors if (p := expand_anchor(sid, score, question))]
    t_expand = time.time()

    top_paths = rerank_paths(question, paths, top_k=top_k)
    t_rerank  = time.time()

    texts = [t for path in top_paths for tier in path["tiered_texts"].values() for t in tier]
    answer = synthesize(question, texts, mode="graph_rag", prompt_version=prompt_version)
    t_synth = time.time()

    return RetrievalResult(
        answer=answer,
        mode="graph_rag",
        supporting_paths=top_paths,
        clusters_used=top_clusters,
        latency_breakdown={
            "embed_ms":   round((t1 - t0) * 1000),
            "search_ms":  round((t_search - t1) * 1000),
            "cluster_ms": round((t_cluster - t_search) * 1000),
            "expand_ms":  round((t_expand - t_cluster) * 1000),
            "rerank_ms":  round((t_rerank - t_expand) * 1000),
            "synth_ms":   round((t_synth - t_rerank) * 1000),
            "total_ms":   round((t_synth - t0) * 1000),
        },
    )


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
