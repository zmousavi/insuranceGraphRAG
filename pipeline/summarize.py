"""
pipeline/summarize.py
=====================
Reads the manifest produced by ingest.py and generates LLM summaries for
every Section and Document node that does not already have one.

Why summaries?
  1. Storage efficiency: full policy text is too large to store on every Neo4j
     node. Summaries (~250 words) go into Neo4j; full text stays in GCS.
  2. Retrieval context: when the agent traverses up from a Shadow chunk to its
     parent Section or Document, it needs something readable to pass to the LLM
     for synthesis. A summary gives the LLM the gist without blowing up the
     context window.
  3. Embedding quality: embedding a 250-word summary gives a better semantic
     signal than embedding 3000 words. Long text dilutes the embedding.

Why not summarize Shadow chunks?
  Shadow chunks already contain raw text which is exactly what we want for
  vector search. Summarizing them would lose the specific wording that makes
  cosine similarity work.

Idempotent: sections and documents that already have a summary are skipped.
Progress is saved after every document so a crash does not lose prior work.

Usage:
  python pipeline/summarize.py
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT         = os.getenv("GOOGLE_CLOUD_PROJECT")
GEMINI_LOCATION = "global"
MODEL           = "gemini-2.5-flash"

MANIFEST_DIR = Path(__file__).parent.parent / "manifest"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """
You are an insurance document analyst. Write a concise summary of the following
insurance document section in 250 words or less.

Focus on:
- Key coverage amounts, limits, or thresholds (include exact dollar amounts)
- Specific exclusions or conditions that affect coverage
- Any named parties, policy numbers, or dates

Do not include generic filler. Every sentence must carry specific information
from the text.

Document section:
{text}

Summary:
"""


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def init_gemini() -> genai.Client:
    """
    Initialize the Google Gen AI client with Vertex AI backend.

    Why location='global'?
        Gemini 2.x models on Vertex AI are served from the global endpoint,
        not a regional one. Using 'us-central1' returns a 404.
    """
    return genai.Client(vertexai=True, project=PROJECT, location=GEMINI_LOCATION)


def call_gemini(client: genai.Client, prompt: str, retries: int = 3) -> str:
    """
    Call Gemini with exponential backoff retry.

    Why retry?
        Gemini API calls can fail transiently due to rate limits, network
        timeouts, or server errors. A single failure should not crash a
        100-call summarization job.

    Why re-raise on the third failure?
        Three consecutive failures on the same call means something is
        seriously wrong (quota exceeded, auth expired). Stopping and surfacing
        the error is better than silently skipping sections and producing an
        incomplete manifest.

    Args:
        client:  Initialized genai.Client.
        prompt:  Full prompt string.
        retries: Maximum number of attempts before raising.

    Returns:
        Stripped response text from Gemini.
    """
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                # thinking_budget=0 disables reasoning mode — summarization
                # doesn't need it and thinking adds latency + cost.
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            return response.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def summarize_text(client: genai.Client, text: str) -> str:
    """
    Generate a concise summary of a document section or full document.

    Why 250 words?
        Sections can span multiple 1000-word Shadow chunks. 250 words gives
        enough room to capture key coverage amounts, exclusions, thresholds,
        and named parties without being so long it dilutes the embedding signal.

    Why send the full text without truncation?
        Gemini 2.5 Flash has a 1M token context window. Our largest section is
        a few thousand words — well within limit. Truncating would risk missing
        critical exclusions or coverage terms that appear later in the section.

    Args:
        client: Initialized genai.Client.
        text:   Full section or document text to summarize.

    Returns:
        Summary string (≤250 words, returned by Gemini).
    """
    prompt = SUMMARY_PROMPT.format(text=text)
    return call_gemini(client, prompt)


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest(manifest_file: Path) -> list[dict]:
    """
    Load the manifest JSON produced by ingest.py into a Python list.

    Args:
        manifest_file: Path to manifest.json.

    Returns:
        List of document dicts, each containing sections and shadows.
    """
    return json.loads(manifest_file.read_text())


def save_manifest(manifest: list[dict], manifest_file: Path) -> None:
    """
    Write the updated manifest back to disk.

    Why save after every document instead of at the end?
        If the script crashes mid-run (network error, quota exceeded), all
        summaries generated so far are preserved. On re-run, already-summarized
        sections are skipped, so no API calls are wasted.

    Args:
        manifest:      Updated list of document dicts.
        manifest_file: Path to write to.
    """
    manifest_file.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.json",
                        help="Manifest filename inside manifest/ dir. Default: manifest.json")
    args = parser.parse_args()
    manifest_file = MANIFEST_DIR / args.manifest

    client = init_gemini()
    manifest = load_manifest(manifest_file)

    print(f"Loaded {len(manifest)} documents from manifest.")

    for doc in manifest:
        doc_id = doc["doc_id"]
        # changed tracks whether any summary was added for this document.
        # We only write to disk when something actually changed — avoids
        # unnecessary writes for documents already fully summarized.
        changed = False

        # Summarize each section that does not already have a summary.
        for section in doc.get("sections", []):
            if "summary" in section:
                print(f"  skip (exists): {section['section_id']}")
                continue
            print(f"  summarizing section: {section['section_id']}")
            section["summary"] = summarize_text(client, section["text"])
            changed = True
            time.sleep(0.5)  # avoid hitting Gemini rate limits

        # Summarize the document itself if not already done.
        # The document summary is built from all its section texts concatenated
        # so Gemini sees the full document context, not just one section.
        if "summary" not in doc:
            print(f"  summarizing doc: {doc_id}")
            full_text = " ".join(s["text"] for s in doc.get("sections", []))
            doc["summary"] = summarize_text(client, full_text)
            changed = True
            time.sleep(0.5)

        if changed:
            save_manifest(manifest, manifest_file)
            print(f"  saved: {doc_id}")

    print("\nDone.")
    print("Next: python pipeline/load_neo4j.py")


if __name__ == "__main__":
    main()
