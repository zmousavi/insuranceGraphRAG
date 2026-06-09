import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


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

_download_manifest_from_gcs()


@st.cache_resource
def load_retriever():
    from retrieval.retrieve import retrieve
    return retrieve


def run_query(question: str, rag_top_k: int = 15, graph_top_k: int = 5):
    retrieve = load_retriever()
    rag_result       = retrieve(question, mode="rag",       top_k=rag_top_k)
    graph_rag_result = retrieve(question, mode="graph_rag", top_k=graph_top_k)
    return rag_result, graph_rag_result


def render_answer(result):
    st.markdown(result.answer)
    total_ms = result.latency_breakdown.get("total_ms", 0)
    st.caption(f"Latency: {total_ms:,} ms")

    if result.mode == "rag" and result.supporting_paths:
        with st.expander("Chunks retrieved"):
            st.caption("RAG retrieves flat chunks with no document or section context.")
            for p in result.supporting_paths:
                st.text(f"{p['shadow_id']}  (score: {p['score']})")


def render_paths(result):
    if not result.supporting_paths:
        st.caption("No paths available.")
        return

    st.subheader("How Graph RAG found this answer")
    for i, path in enumerate(result.supporting_paths, 1):
        edge_chain = " → ".join(path["edge_types"])
        nodes      = " → ".join(path["node_ids"])
        with st.expander(f"Path {i}: {nodes}"):
            st.markdown(f"**Traversal:** `{edge_chain}`")
            for tier, texts in path["tiered_texts"].items():
                st.markdown(f"**{tier}**")
                for t in texts:
                    st.text(t[:300])


def main():
    st.set_page_config(page_title="Insurance Graph RAG", layout="wide")
    st.title("Insurance Graph RAG")
    st.caption("RAG vs Graph RAG on insurance policies and claims.")

    question = st.text_input("Ask a question about a policy or claim:")
    run = st.button("Run")

    if run and question:
        with st.spinner("Running RAG and Graph RAG..."):
            rag_result, graph_result = run_query(question)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("RAG (15 chunks)")
            render_answer(rag_result)

        with col2:
            st.subheader("Graph RAG (5 paths)")
            render_answer(graph_result)

        render_paths(graph_result)


if __name__ == "__main__":
    main()
