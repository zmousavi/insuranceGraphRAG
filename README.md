# Insurance Graph RAG

A production-quality Graph RAG system for the insurance domain. Answers questions that require crossing claim documents and policy documents — a task where graph traversal demonstrably outperforms flat vector search.

**Live demo:** `https://insurance-graph-rag-97967676278.us-central1.run.app/docs`

---

## The Problem

An adjuster asks: *"Was the denial on claim HO-002 consistent with the policy exclusion?"*

Flat vector search returns chunks semantically similar to the question — usually claim text. The policy exclusion section, written in legal language far from the question's vocabulary, scores too low to surface. The answer is incomplete.

Graph RAG follows the `REFERENCES_POLICY` edge from the claim document to the policy, then traverses to the Exclusions section regardless of vocabulary distance. Both sides of the question reach the LLM.

---

## Benchmark Results

| Mode | Score (25 questions) |
|------|---------------------|
| RAG (cosine k=15) | 8 / 25 |
| Graph RAG (k=7) | 11 / 25 |

Graph RAG wins on questions that require crossing from a claim to its referenced policy — endorsements, exclusion thresholds, coverage limits. Both modes fail on entity traversal questions (fraud ring detection, cross-entity ownership) which require a ReAct agent — a planned next step.

### Best demo questions

| Question | Why it's a good demo |
|----------|---------------------|
| Does Sarah Mitchell have water backup coverage? | Requires finding endorsement E01 attached to her policy |
| Why was claim HO-002 denied? | Requires crossing from adjuster notes to the Exclusions section |
| What is the medical payment limit on David Chen's policy? | Clean single-hop retrieval — fast and correct |
| Is Maple Street Holdings listed as an additional insured? | Endorsement traversal |

---

## Architecture

```
Query
  │
  ▼
embed_query()          Vertex AI text-embedding-004 → 768d vector
  │
  ▼
cosine_search()        FAISS IndexFlatIP — top-50 candidate shadows
  │
  ▼
route_clusters()       Louvain cluster IDs → rank clusters by mean cosine → top-2
  │
  ▼
expand_anchor()        Per anchor: Neo4j traversal
  │                      ClaimDocument → REFERENCES_POLICY → Policy → Sections
  │                      Policy → HAS_ENDORSEMENT → Endorsement
  │                      Shadow → NEXT_CHUNK → Shadow
  ▼
condense_path()        Section summary + anchor text + terminal text → cross-encoder input
  │
  ▼
rerank_paths()         ms-marco-MiniLM cross-encoder → dedup per doc → top-7
  │
  ▼
synthesize()           Gemini 2.5 Flash → answer with provenance
```

### Graph schema (key edges)

```
ClaimDocument ─[:REFERENCES_POLICY]────► Policy          ← primary cross-link
Policy        ─[:HAS_ENDORSEMENT]──────► Endorsement
Policy        ─[:HAS_SECTION]──────────► Section
Section       ─[:HAS_SHADOW]───────────► Shadow
Shadow        ─[:NEXT_CHUNK]───────────► Shadow
Policy        ─[:SUPERSEDES]───────────► Policy          ← document versioning
Claim         ─[:REPAIRED_BY]─────────► Contractor       ← fraud signal
Person        ─[:OWNS]────────────────► Organization     ← hidden ownership
```

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Graph DB | Neo4j AuraDB Free |
| Embeddings | Vertex AI `text-embedding-004` (768d) |
| LLM | Vertex AI Gemini 2.5 Flash |
| Vector search | FAISS IndexFlatIP |
| Clustering | python-louvain + networkx (no GDS required) |
| Cross-encoder | `ms-marco-MiniLM-L-6-v2` |
| API | FastAPI on Cloud Run |
| UI | Streamlit |

---

## Demo Data

Synthetic corpus generated with Gemini — no licensing concerns, full control over fraud patterns:

- **7 policies**: HO-3 homeowners (×2), personal auto, CGL (×2), commercial property, plus one amended HO-3 for versioning tests
- **~30 claims** across all policies: FNOL reports, adjuster notes, outcome letters
- **Fraud scenarios baked in**:
  - James Carter owns 3 businesses under unrelated names — all share one address and one agent
  - FastFix Restoration appears as contractor in claims from 3 unrelated policyholders within 45 days
  - Marcus Webb appears as injured party in claims against 2 separate policyholders

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (copy .env.example → .env, fill in credentials)
cp .env.example .env

# Run Streamlit UI
streamlit run app/streamlit_app.py

# Or run the API
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Required credentials (in `.env`)
- `GOOGLE_CLOUD_PROJECT` — GCP project ID
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` — AuraDB connection
- `GCS_BUCKET` — GCS bucket name (local dev: leave unset, files read from disk)

---

## API

```bash
# Health check
curl https://insurance-graph-rag-97967676278.us-central1.run.app/health

# Ask a question
curl -X POST .../query \
  -H "Content-Type: application/json" \
  -d '{"question": "Does Sarah Mitchell have water backup coverage?", "mode": "graph_rag"}'
```

Interactive docs: `/docs`

---

## Pipeline (offline, one-time setup)

```bash
python data_gen/generate_data.py     # generate synthetic docs
python pipeline/ingest.py            # chunk into shadows
python pipeline/summarize.py         # Gemini section summaries
python pipeline/extract_entities.py  # extract persons, orgs, agents, contractors
python pipeline/load_neo4j.py        # load graph
python pipeline/embed.py             # Vertex AI embeddings
python pipeline/cluster.py           # Louvain community detection
python evaluation/evaluate.py        # run 25-question benchmark
```

---

## Known Limitations / Next Steps

- **Entity traversal not implemented** — questions about cross-entity ownership (who owns all 3 Carter businesses?) and fraud ring detection (FastFix appearing across unrelated policyholders) require a ReAct agent that can call `traverse_entity`. The fixed pipeline handles document traversal only.
- **ReAct agent upgrade** — replace the fixed 8-step pipeline with a Gemini function-calling agent that selects traversal tools dynamically based on the question. Simple questions get fast single-tool answers; complex cross-document questions chain multiple tools.
- **Redis path cache** — designed but not implemented. Repeat queries would skip the full pipeline and return cached paths in ~50ms instead of ~5s.
- **Q14 regression** — amended policy path (v2) loses to claim paths in cross-encoder after the dedup fix. Known tradeoff: fixing it improves Q10/Q12 at the cost of Q14.
