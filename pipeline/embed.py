"""
pipeline/embed.py
=================
Reads shadow text from manifest.json, generates 768-dimensional embeddings
using Vertex AI text-embedding-004, and saves them to manifest/embeddings.json.

Embeddings are stored locally so Neo4j can be wiped and reloaded without
re-calling Vertex AI. load_neo4j.py reads embeddings.json and writes the
vectors onto Shadow nodes.

Incremental: shadow_ids already in embeddings.json are skipped, so re-runs
only process new or updated chunks.

Usage:
  python pipeline/embed.py
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = "global"
MODEL    = "text-embedding-004"

BATCH_SIZE      = 25  # ~500 words/shadow * 25 = ~16k tokens, under 20k limit
ROOT            = Path(__file__).parent.parent
MANIFEST_FILE   = ROOT / "manifest" / "manifest.json"
EMBEDDINGS_FILE = ROOT / "manifest" / "embeddings.json"


def init_gemini() -> genai.Client:
    return genai.Client(vertexai=True, project=PROJECT, location=LOCATION)


def load_existing_embeddings() -> dict[str, list[float]]:
    """Load already-computed embeddings from disk. Returns empty dict if none."""
    if not EMBEDDINGS_FILE.exists():
        return {}
    return json.loads(EMBEDDINGS_FILE.read_text())


def fetch_shadows_from_manifest() -> list[dict]:
    """Read all shadow id + text pairs from manifest.json."""
    manifest = json.loads(MANIFEST_FILE.read_text())
    shadows = []
    for doc in manifest:
        for shadow in doc.get("shadows", []):
            shadows.append({"id": shadow["shadow_id"], "text": shadow["text"]})
    return shadows


def fetch_sections_from_manifest() -> list[dict]:
    """Read all section id + summary pairs from manifest.json."""
    manifest = json.loads(MANIFEST_FILE.read_text())
    sections = []
    for doc in manifest:
        for section in doc.get("sections", []):
            summary = section.get("summary") or ""
            if summary:
                sections.append({"id": section["section_id"], "text": summary})
    return sections


def embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """
    Call Vertex AI text-embedding-004 for a batch of texts.

    Why batch and not one call per shadow?
        A single round trip takes ~200ms. Batching 100 at a time
        reduces 183 shadows from ~37s to 2 round trips.
    """
    response = client.models.embed_content(model=MODEL, contents=texts)
    return [e.values for e in response.embeddings]


def embed_items(client: genai.Client, items: list[dict], existing: dict, label: str):
    """Embed a list of {id, text} dicts, skipping already-embedded ids. Saves after every batch."""
    to_embed = [s for s in items if s["id"] not in existing]
    print(f"  {len(items)} total {label}, {len(existing)} already embedded, {len(to_embed)} to embed.")

    if not to_embed:
        print(f"  Nothing to do for {label}.")
        return

    for i in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[i:i + BATCH_SIZE]
        texts = [s["text"] for s in batch]

        for attempt in range(3):
            try:
                vectors = embed_batch(client, texts)
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"  retry in {wait}s ({e})")
                    time.sleep(wait)
                else:
                    raise

        for s, v in zip(batch, vectors):
            existing[s["id"]] = v

        EMBEDDINGS_FILE.write_text(json.dumps(existing))
        print(f"  {min(i + BATCH_SIZE, len(to_embed))}/{len(to_embed)} {label} embedded")


def main():
    print("Initializing Vertex AI...")
    client = init_gemini()

    existing = load_existing_embeddings()

    print("\nEmbedding shadows...")
    shadows = fetch_shadows_from_manifest()
    embed_items(client, shadows, existing, "shadows")

    print("\nEmbedding section summaries...")
    sections = fetch_sections_from_manifest()
    # prefix sec_ so retrieve.py can distinguish section embeddings from shadow embeddings
    sec_items = [{"id": f"sec_{s['id']}", "text": s["text"]} for s in sections]
    embed_items(client, sec_items, existing, "sections")

    print(f"\nDone. embeddings.json has {len(existing)} vectors.")
    print("Next: python pipeline/load_neo4j.py  (to write embeddings into Neo4j)")


if __name__ == "__main__":
    main()
