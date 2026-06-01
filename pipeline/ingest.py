"""
pipeline/ingest.py
==================
Reads all documents from demo_docs/ and chunks them into a manifest JSON.

Two strategies are supported:
  --strategy structure_aware  (default) splits on insurance section headers.
                              Each section (DECLARATIONS, COVERAGE, EXCLUSIONS,
                              etc.) becomes its own Section node. Oversized
                              sections are sub-divided into Shadow chunks with
                              20% overlap.
  --strategy fixed            Sliding-window 1000-word chunks, no structural
                              awareness. Used only as an evaluation baseline to
                              show that structure-aware chunking outperforms
                              naive splitting on the benchmark questions.

Why not use LangChain text splitters?
  Insurance policy documents have explicit section headers that carry domain
  meaning. A generic splitter discards that structure. By splitting on headers
  we guarantee that EXCLUSIONS text never mixes with COVERAGE text in the same
  chunk, which is critical for the cross-section benchmark questions.

Why word count instead of tiktoken?
  Chunking precision matters when you are right up against a model context
  window. Here chunks go into Neo4j for retrieval, not directly into an LLM
  prompt. Word count (avg ~1.3 tokens/word) is close enough and avoids an
  extra dependency.

Outputs:
  manifest/manifest.json        (structure_aware)
  manifest/manifest_fixed.json  (fixed)

Each manifest entry contains:
  doc_id, source_file, strategy, hash, sections[], shadows[]

The hash field enables incremental re-runs: unchanged files are skipped,
updated files are fully re-chunked. Section-level change detection is deferred
to embed.py where the API cost of re-embedding makes granularity worthwhile.

Usage:
  python pipeline/ingest.py --strategy structure_aware
  python pipeline/ingest.py --strategy fixed
"""

import re
import json
import hashlib
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Insurance-specific section headers used by the structure-aware splitter.
# A set is used because order does not matter here — we detect all headers
# and then sort the results by character offset to recover document order.
SECTION_HEADERS = {
    "DECLARATIONS",
    "INSURING AGREEMENT",
    "COVERAGE",
    "EXCLUSIONS",
    "CONDITIONS",
    "DEFINITIONS",
    "ENDORSEMENT",
}

# Shadow chunk size in words. 1000 words ≈ 1300 tokens — well within the
# Neo4j 4KB storage target and the embedding model's input limit.
SHADOW_MAX_TOKENS = 500

# 20% overlap means 200 words repeat between consecutive chunks so that
# sentences near a boundary are not severed and lost to retrieval.
SHADOW_OVERLAP = 0.2

MANIFEST_DIR = Path(__file__).parent.parent / "manifest"
DEMO_DOCS_DIR = Path(__file__).parent.parent / "demo_docs"


# ---------------------------------------------------------------------------
# Structure-aware chunking
# ---------------------------------------------------------------------------

def detect_section_boundaries(text: str) -> list[tuple[int, str]]:
    """
    Scan the document text for known insurance section headers and return
    their positions.

    Args:
        text: Full document text.

    Returns:
        List of (char_offset, header_name) tuples sorted by position.
        Empty list if no headers are found (e.g. claim FNOL documents).

    Why a list and not a set?
        We need the boundaries in document order so we can slice text between
        consecutive positions. A set does not preserve order.

    Why re.MULTILINE?
        The ^ anchor matches the start of any line, not just the start of the
        string. This ensures we only match headers that begin a line, not
        the word "COVERAGE" appearing mid-sentence.
    """
    boundaries = []
    for header in SECTION_HEADERS:
        for match in re.finditer(rf"^{header}\b", text, re.MULTILINE | re.IGNORECASE):
            boundaries.append((match.start(), header.upper()))
    boundaries.sort(key=lambda x: x[0])
    return boundaries


def split_into_sections(text: str, doc_id: str) -> list[dict]:
    """
    Split a document into sections using detected header boundaries.

    Each section spans from its header to the start of the next header (or
    end of document). Both the header line and the body text are included in
    the section text so that Shadow chunks inherit the section title as
    context.

    Args:
        text:   Full document text.
        doc_id: Stable document identifier (e.g. "policy_HO3_001").

    Returns:
        List of section dicts with keys: section_id, title, text.
        Falls back to a single FULL_DOCUMENT section if no headers are found.
        This handles claim documents (FNOL, adjuster notes, outcome letters)
        that do not use the standard policy header format.

    Why deterministic section_id?
        IDs like "policy_HO3_001_exclusions" are referenced directly in
        evaluation/ground_truth.json as required_sources. They must be stable
        across re-runs so the benchmark hit-rate metric stays valid.
    """
    boundaries = detect_section_boundaries(text)

    if not boundaries:
        return [{
            "section_id": f"{doc_id}_full",
            "title": "FULL_DOCUMENT",
            "text": text.strip(),
        }]

    sections = []
    for i, (start, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append({
                "section_id": f"{doc_id}_{title.lower().replace(' ', '_')}",
                "title": title,
                "text": section_text,
            })
    return sections


def create_shadows(section: dict) -> list[dict]:
    """
    Sub-divide a section into Shadow chunks if it exceeds SHADOW_MAX_TOKENS.

    Shadow nodes are the vector search entry point in Neo4j. They store raw
    chunk text (~4KB each) and carry embeddings. Keeping them under
    SHADOW_MAX_TOKENS ensures they fit within embedding model input limits
    and stay manageable in size.

    Args:
        section: A section dict produced by split_into_sections().

    Returns:
        List of shadow dicts with keys: shadow_id, section_id, text,
        chunk_index, token_count.

    Why 20% overlap (step = 800 words)?
        Without overlap, a sentence split at a chunk boundary loses half its
        context. With 200-word overlap, both neighboring chunks contain the
        full sentence, so retrieval can find it regardless of which chunk
        the query embedding scores highest.

    Why word count instead of a tokenizer?
        Chunking precision only matters when filling a model context window.
        These chunks go into Neo4j for retrieval, not directly into an LLM.
        Word count (avg ~1.3 tokens/word) is accurate enough and avoids
        importing tiktoken.
    """
    words = section["text"].split()
    # Step size is 80% of chunk size — the remaining 20% is the overlap.
    step = int(SHADOW_MAX_TOKENS * (1 - SHADOW_OVERLAP))  # 800 words

    # If the section fits in a single chunk, no splitting needed.
    if len(words) <= SHADOW_MAX_TOKENS:
        return [{
            "shadow_id": f"{section['section_id']}_s0",
            "section_id": section["section_id"],
            "text": section["text"],
            "chunk_index": 0,
            "token_count": len(words),
        }]

    overlap = SHADOW_MAX_TOKENS - step  # words already in previous chunk's tail
    shadows = []
    for i, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start:start + SHADOW_MAX_TOKENS]
        if not chunk_words:
            break
        # Skip tail chunks that are fully covered by the previous chunk's overlap.
        if i > 0 and len(chunk_words) <= overlap:
            break
        shadows.append({
            "shadow_id": f"{section['section_id']}_s{i}",
            "section_id": section["section_id"],
            "text": " ".join(chunk_words),
            "chunk_index": i,
            "token_count": len(chunk_words),
        })
    return shadows


# ---------------------------------------------------------------------------
# Fixed chunking (evaluation baseline)
# ---------------------------------------------------------------------------

def create_fixed_chunks(text: str, doc_id: str) -> list[dict]:
    """
    Split a full document into fixed-size overlapping chunks with no
    structural awareness.

    This is the naive baseline used only for evaluation comparison. It
    deliberately ignores section headers so that the benchmark can quantify
    how much structure-aware chunking improves retrieval hit rate.

    Args:
        text:   Full document text.
        doc_id: Stable document identifier.

    Returns:
        List of shadow dicts using the same schema as create_shadows() so
        both strategies can be processed identically downstream.

    Why no sections list?
        Fixed chunking has no concept of sections. The entire document is
        treated as a single flat blob. The section_id field uses a synthetic
        "_fixed" suffix to distinguish these chunks in Neo4j.
    """
    words = text.split()
    step = int(SHADOW_MAX_TOKENS * (1 - SHADOW_OVERLAP))

    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start:start + SHADOW_MAX_TOKENS]
        if not chunk_words:
            break
        chunks.append({
            "shadow_id": f"{doc_id}_fixed_s{i}",
            "section_id": f"{doc_id}_fixed",
            "text": " ".join(chunk_words),
            "chunk_index": i,
            "token_count": len(chunk_words),
        })
    return chunks


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------

def file_hash(file_path: Path) -> str:
    """
    Compute the MD5 hash of a file's contents.

    Used for incremental ingestion: if the hash stored in the manifest
    matches the current file hash, the file has not changed and can be
    skipped. MD5 is used here for speed — collision resistance is not
    required since this is a change-detection mechanism, not a security check.

    Args:
        file_path: Path to the file.

    Returns:
        Hex-encoded MD5 digest string.
    """
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def load_existing_manifest(manifest_file: Path) -> dict:
    """
    Load an existing manifest and index it by doc_id for fast lookup.

    Args:
        manifest_file: Path to the manifest JSON file.

    Returns:
        Dict mapping doc_id → manifest entry. Empty dict if no manifest exists.

    Why a dict keyed by doc_id?
        O(1) lookup when checking whether a file has already been processed.
        The manifest on disk is a list (ordered), but we need random access
        here to check individual documents by ID.
    """
    if not manifest_file.exists():
        return {}
    existing = json.loads(manifest_file.read_text())
    return {entry["doc_id"]: entry for entry in existing}


def process_document(file_path: Path, strategy: str) -> dict:
    """
    Read a document file and produce its full manifest entry.

    For structure_aware strategy: detects section boundaries, splits into
    Section nodes, then sub-divides oversized sections into Shadow chunks.

    For fixed strategy: skips section splitting and chunks the whole document
    as a flat blob. sections list is empty in this case.

    Args:
        file_path: Path to the source .txt file.
        strategy:  "structure_aware" or "fixed".

    Returns:
        Manifest entry dict with keys: doc_id, source_file, strategy, hash,
        sections, shadows.

    Why store both sections and shadows?
        Sections → Neo4j Section nodes (store LLM summary, link to parent doc)
        Shadows  → Neo4j Shadow nodes (store raw chunk text, carry embedding)
        They are linked by section_id. Keeping both in the manifest means
        load_neo4j.py can create the full graph in one pass without re-reading
        source files.
    """
    text = file_path.read_text()
    # For files in claim subfolders (e.g. claims/claim_CGL_001/adjuster.txt),
    # include the parent folder name so doc_id is unique: "claim_CGL_001_adjuster".
    # For policies and endorsements the filename stem is already unique.
    parent = file_path.parent.name
    if parent not in ("policies", "endorsements", "claims"):
        doc_id = f"{parent}_{file_path.stem}"
    else:
        doc_id = file_path.stem

    if strategy == "structure_aware":
        sections = split_into_sections(text, doc_id)
        shadows = []
        for section in sections:
            # extend (not append) so shadows stays a flat list, not nested.
            shadows.extend(create_shadows(section))
    else:
        sections = []
        shadows = create_fixed_chunks(text, doc_id)

    # Determine document type from doc_id prefix.
    if doc_id.startswith("policy_"):
        doc_type = "policy"
    elif doc_id.startswith("endorsement_"):
        doc_type = "endorsement"
    else:
        doc_type = "claim_document"

    entry = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "source_file": str(file_path),
        "strategy": strategy,
        "hash": file_hash(file_path),
        "sections": sections,
        "shadows": shadows,
    }

    if doc_type == "claim_document":
        # Derive claim_id: "claim_CGL_001_fnol" → "claim_CGL_001"
        parts = doc_id.split("_")
        entry["claim_id"] = f"{parts[0]}_{parts[1]}_{parts[2]}"

        # Extract policy_id — every claim doc header has "Policy Number: policy_XYZ".
        # Adjuster notes don't have this field so we also check the sibling fnol/outcome
        # via the claim folder. For now extract where available; load_neo4j propagates
        # policy_id across all docs in the same claim using claim_id.
        policy_id_match = re.search(r"Policy Number:\s*(policy_\S+)", text)
        if policy_id_match:
            entry["policy_id"] = policy_id_match.group(1)

        # Extract date_of_loss for NEXT_PERIOD edge creation in load_temporal.py.
        date_match = re.search(r"Date of Loss:\s*(.+)", text)
        if date_match:
            entry["date_of_loss"] = date_match.group(1).strip()

    return entry


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chunk insurance documents into manifest.")
    parser.add_argument(
        "--strategy",
        choices=["structure_aware", "fixed"],
        default="structure_aware",
        help="Chunking strategy. structure_aware splits on section headers; fixed uses a sliding window.",
    )
    args = parser.parse_args()

    manifest_file = MANIFEST_DIR / (
        "manifest.json" if args.strategy == "structure_aware"
        else "manifest_fixed.json"
    )
    MANIFEST_DIR.mkdir(exist_ok=True)

    # Load existing manifest so unchanged files can be skipped.
    existing = load_existing_manifest(manifest_file)

    doc_files = sorted(DEMO_DOCS_DIR.rglob("*.txt"))
    print(f"Strategy : {args.strategy}")
    print(f"Documents: {len(doc_files)} found")

    manifest = []
    skipped = 0
    processed = 0
    seen_doc_ids: dict[str, Path] = {}

    for file_path in doc_files:
        parent = file_path.parent.name
        if parent not in ("policies", "endorsements", "claims"):
            doc_id = f"{parent}_{file_path.stem}"
        else:
            doc_id = file_path.stem

        if doc_id in seen_doc_ids:
            raise ValueError(
                f"doc_id collision: '{doc_id}' is produced by both "
                f"'{seen_doc_ids[doc_id]}' and '{file_path}'. "
                f"Rename one of the source files."
            )
        seen_doc_ids[doc_id] = file_path

        current_hash = file_hash(file_path)

        # Skip if file is unchanged since last run.
        # Re-chunk the whole file if its hash changed — chunking is fast
        # (milliseconds). Section-level change detection is handled in
        # embed.py where re-embedding has real API cost.
        if doc_id in existing and existing[doc_id].get("hash") == current_hash:
            print(f"  skip (unchanged): {file_path.name}")
            manifest.append(existing[doc_id])
            skipped += 1
            continue

        print(f"  processing: {file_path.name}")
        manifest.append(process_document(file_path, args.strategy))
        processed += 1

    manifest_file.write_text(json.dumps(manifest, indent=2))

    total_shadows = sum(len(d["shadows"]) for d in manifest)
    total_sections = sum(len(d["sections"]) for d in manifest)
    print(f"\nWritten : {manifest_file}")
    print(f"Processed: {processed} | Skipped: {skipped}")
    print(f"Sections : {total_sections}")
    print(f"Shadows  : {total_shadows}")
    print(f"\nNext: python pipeline/summarize.py")


if __name__ == "__main__":
    main()
