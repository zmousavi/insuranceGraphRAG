"""
pipeline/load_neo4j.py
======================
Reads manifest.json and loads all nodes and edges into Neo4j AuraDB.

Load order matters — parent nodes must exist before child nodes and edges:
  1. Schema (constraints + vector index)
  2. Policy + Policyholder nodes
  3. Section + Shadow nodes (reference parent Policy/ClaimDocument)
  4. Claim + ClaimDocument nodes
  5. Endorsement nodes
  6. NEXT_CHUNK edges (between consecutive Shadows in same section)
  7. SUPERSEDES edge (amended HO-3 v2 → original HO-3)

Edges that require data not yet available are built in later steps:
  - SIMILAR_TO    → cluster.py     (needs embeddings)
  - HAS_KEYWORD   → load_keywords.py (needs keyword extraction)
  - NEXT_PERIOD   → load_temporal.py (needs date parsing across claim docs)

All Cypher queries use MERGE (not CREATE) so re-runs are safe — existing
nodes and edges are matched, not duplicated.

Usage:
  python pipeline/load_neo4j.py
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

MANIFEST_FILE  = Path(__file__).parent.parent / "manifest" / "manifest.json"
ENTITIES_FILE  = Path(__file__).parent.parent / "manifest" / "entities.json"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_driver():
    """
    Create and return a Neo4j driver using credentials from .env.

    Why verify_connectivity()?
        Fails fast if credentials are wrong or the instance isn't ready,
        rather than discovering the problem mid-load after partial data
        has already been written.
    """
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    driver.verify_connectivity()
    return driver


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def create_schema(driver):
    """
    Create uniqueness constraints and the Shadow vector index.
    Safe to re-run — IF NOT EXISTS prevents errors on repeat runs.

    Why constraints first?
        Uniqueness constraints create an implicit B-tree index on the id
        field, making MERGE operations fast. Without them, every MERGE
        scans all nodes of that label looking for a match.

    Why 768 dimensions?
        Vertex AI text-embedding-004 produces 768-dimensional vectors.
        The index dimension must exactly match — wrong dimension means
        the index is unusable for search.

    Why cosine similarity?
        Cosine measures the angle between vectors, not their magnitude.
        Two chunks about the same topic score high even if one is longer.
        Dot product would bias toward longer chunks with larger magnitudes.
    """
    with driver.session() as session:
        for label in ["Policyholder", "Policy", "Endorsement", "Claim",
                      "ClaimDocument", "Section", "Shadow", "Keyword",
                      "Person", "Organization", "Agent", "Location",
                      "Contractor", "ThirdParty"]:
            session.run(f"""
                CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label})
                REQUIRE n.id IS UNIQUE
            """)

    print("  Schema created.")


# ---------------------------------------------------------------------------
# Node loaders
# ---------------------------------------------------------------------------

def load_policies(driver, manifest):
    """
    Create Policyholder and Policy nodes and link them with HAS_POLICY edges.

    Why UNWIND + MERGE instead of one MERGE per node?
        UNWIND sends all nodes in a single Cypher query instead of one
        query per node. For 100 documents that's 100 round trips vs 1.
        Much faster over a network connection to AuraDB.

    Why SET after MERGE?
        Updates properties if the node already exists — handles re-runs
        after a summary or source_file has been updated.
    """
    policies = [d for d in manifest if d.get("doc_type") == "policy"]

    with driver.session() as session:
        session.run("""
            UNWIND $policies AS p
            MERGE (ph:Policyholder {id: p.doc_id + '_holder'})
            MERGE (pol:Policy {id: p.doc_id})
            SET pol.summary = p.summary,
                pol.source_file = p.source_file,
                pol.superseded = false
            MERGE (ph)-[:HAS_POLICY]->(pol)
        """, policies=policies)
    print(f"  Loaded {len(policies)} policies.")


def load_sections_and_shadows(driver, manifest):
    """
    Create Section and Shadow nodes and link them to their parent documents.

    Why sections before shadows?
        Shadow nodes reference their parent Section via HAS_SHADOW. The
        Section must exist before the edge can be created.

    Why HAS_SHADOW on Section and not directly on Policy/ClaimDocument?
        The traversal path is Shadow → Section → Document. This preserves
        the structural hierarchy so the agent can retrieve section-level
        context (title, summary) before document-level context.
    """
    sections_batch = []
    shadows_batch = []

    for doc in manifest:
        for section in doc.get("sections", []):
            sections_batch.append({
                "id": section["section_id"],
                "parent_id": doc["doc_id"],
                "title": section["title"],
                "summary": section.get("summary", ""),
                "text": section["text"],
            })
            for shadow in [s for s in doc["shadows"]
                           if s["section_id"] == section["section_id"]]:
                shadows_batch.append({
                    "id": shadow["shadow_id"],
                    "section_id": shadow["section_id"],
                    "text": shadow["text"],
                    "chunk_index": shadow["chunk_index"],
                    "token_count": shadow["token_count"],
                })

    with driver.session() as session:
        session.run("""
            UNWIND $sections AS s
            MERGE (sec:Section {id: s.id})
            SET sec.title = s.title,
                sec.summary = s.summary,
                sec.text = s.text
            WITH sec, s
            MATCH (parent {id: s.parent_id})
            MERGE (parent)-[:HAS_SECTION]->(sec)
        """, sections=sections_batch)

        session.run("""
            UNWIND $shadows AS sh
            MERGE (s:Shadow {id: sh.id})
            SET s.text = sh.text,
                s.chunk_index = sh.chunk_index,
                s.token_count = sh.token_count
            WITH s, sh
            MATCH (sec:Section {id: sh.section_id})
            MERGE (sec)-[:HAS_SHADOW]->(s)
        """, shadows=shadows_batch)

    print(f"  Loaded {len(sections_batch)} sections, {len(shadows_batch)} shadows.")


def load_claims(driver, manifest):
    """
    Create Claim and ClaimDocument nodes and link them to their parent Policy
    via REFERENCES_POLICY — the key cross-link for Graph RAG.

    Why REFERENCES_POLICY on ClaimDocument and not Claim?
        ClaimDocument is the unit that contains retrievable text. The policy
        reference lives at this level so the agent can traverse directly from
        a retrieved chunk back to the governing policy.

    policy_id and claim_id come from the manifest (set by ingest.py).
    We propagate policy_id within a claim group so adjuster notes — which
    don't have a Policy Number header — still get REFERENCES_POLICY edges.
    """
    claim_docs = [d for d in manifest if d.get("doc_type") == "claim_document"]

    # Build claim_id → policy_id from docs that have it (fnol + outcome).
    claim_policy_map: dict[str, str] = {}
    for doc in claim_docs:
        if doc.get("policy_id") and doc.get("claim_id"):
            claim_policy_map[doc["claim_id"]] = doc["policy_id"]

    batch = []
    for doc in claim_docs:
        claim_id = doc.get("claim_id", "")
        policy_id = doc.get("policy_id") or claim_policy_map.get(claim_id, "")

        batch.append({
            "id": doc["doc_id"],
            "claim_id": claim_id,
            "policy_id": policy_id,
            "summary": doc.get("summary", ""),
            "source_file": doc["source_file"],
        })

    with driver.session() as session:
        session.run("""
            UNWIND $claims AS c
            MERGE (claim:Claim {id: c.claim_id})
            MERGE (cd:ClaimDocument {id: c.id})
            SET cd.summary = c.summary,
                cd.source_file = c.source_file
            MERGE (claim)-[:HAS_DOCUMENT]->(cd)
            WITH cd, c
            MATCH (pol:Policy {id: c.policy_id})
            MERGE (cd)-[:REFERENCES_POLICY]->(pol)
        """, claims=batch)

    print(f"  Loaded {len(batch)} claim documents.")


def load_endorsements(driver, manifest):
    """
    Create Endorsement nodes and link them to their parent Policy
    via HAS_ENDORSEMENT edges.

    Why endorsements as separate nodes and not policy sections?
        Endorsements are the answer to several hard benchmark questions
        (Q07, Q08, Q09, Q10). Separate nodes let the agent retrieve them
        directly via get_policy_endorsements() tool without parsing the
        full policy text.

    Policy id is derived from endorsement id naming convention:
        endorsement_HO3_001_E01 → policy_HO3_001
        endorsement_PAP_001_E01 → policy_PAP_001
        endorsement_CGL_001_E01 → policy_CGL_001
    """
    endorsements = [d for d in manifest if d.get("doc_type") == "endorsement"]

    batch = []
    for doc in endorsements:
        parts = doc["doc_id"].split("_")
        policy_id = f"policy_{parts[1]}_{parts[2]}"
        batch.append({
            "id": doc["doc_id"],
            "policy_id": policy_id,
            "summary": doc.get("summary", ""),
            "source_file": doc["source_file"],
        })

    with driver.session() as session:
        session.run("""
            UNWIND $endorsements AS e
            MERGE (end:Endorsement {id: e.id})
            SET end.summary = e.summary,
                end.source_file = e.source_file
            WITH end, e
            MATCH (pol:Policy {id: e.policy_id})
            MERGE (pol)-[:HAS_ENDORSEMENT]->(end)
        """, endorsements=batch)

    print(f"  Loaded {len(batch)} endorsements.")


# ---------------------------------------------------------------------------
# Entity loaders  (reads manifest/entities.json produced by extract_entities.py)
# ---------------------------------------------------------------------------

def load_entities(driver, entities: dict):
    """
    Create Person, Organization, Agent, Location, Contractor, and ThirdParty
    nodes from the deduplicated entities extracted by extract_entities.py.

    Why separate from policy/claim loading?
        Entity extraction requires a Gemini call per document and produces a
        separate JSON file. Loading them in their own step keeps this file
        idempotent and easy to re-run after entity extraction is updated.

    Why MERGE and not CREATE?
        The same person or agent can appear across multiple policies
        (e.g. James Carter owns three businesses). MERGE ensures we get one
        node per entity regardless of how many policies reference them.
    """
    with driver.session() as session:
        session.run("""
            UNWIND $persons AS p
            MERGE (n:Person {id: p.id})
            SET n.name = p.name
        """, persons=entities.get("persons", []))

        session.run("""
            UNWIND $orgs AS o
            MERGE (n:Organization {id: o.id})
            SET n.name = o.name
        """, orgs=entities.get("organizations", []))

        session.run("""
            UNWIND $agents AS a
            MERGE (n:Agent {id: a.id})
            SET n.name = a.name, n.agency = a.agency
        """, agents=entities.get("agents", []))

        session.run("""
            UNWIND $locs AS l
            MERGE (n:Location {id: l.id})
            SET n.address = l.address
        """, locs=entities.get("locations", []))

        session.run("""
            UNWIND $contractors AS c
            MERGE (n:Contractor {id: c.id})
            SET n.name = c.name
        """, contractors=entities.get("contractors", []))

        session.run("""
            UNWIND $tps AS t
            MERGE (n:ThirdParty {id: t.id})
            SET n.name = t.name
        """, tps=entities.get("third_parties", []))

    counts = {k: len(v) for k, v in entities.items() if isinstance(v, list)}
    print(f"  Persons: {counts.get('persons',0)}  Orgs: {counts.get('organizations',0)}  "
          f"Agents: {counts.get('agents',0)}  Locations: {counts.get('locations',0)}  "
          f"Contractors: {counts.get('contractors',0)}  ThirdParties: {counts.get('third_parties',0)}")


def load_entity_edges(driver, entities: dict):
    """
    Create edges connecting entity nodes to Policy and Claim nodes.

    Edge types created here:
      Person/Org  -[:INSURED_UNDER]->  Policy      (named insured)
      Person      -[:OWNS]->           Organization (business owner link)
      Policy      -[:MANAGED_BY]->     Agent
      Policy      -[:LOCATED_AT]->     Location
      Claim       -[:REPAIRED_BY]->    Contractor
      Claim       -[:INVOLVES_THIRD_PARTY]->  ThirdParty  (role=claimant)
      Claim       -[:HAS_SUBROGATION_RIGHT]-> ThirdParty  (role=subrogation_target)

    Why INSURED_UNDER on the entity → Policy rather than Policy → entity?
        The traversal direction matters for Cypher: "find all policies for
        James Carter" reads naturally as MATCH (p:Person)-[:INSURED_UNDER]->(pol)
        rather than a reverse MATCH. Both directions work in Neo4j but
        entity-to-policy feels more natural for portfolio queries.

    Why owner via OWNS edge instead of adding owner_id to the policy link?
        An OWNS edge lets us traverse: Person → Organization → Policy with
        a simple 2-hop path. Storing owner_id as a property would require
        property filtering and can't be followed by a relationship traversal.
    """
    policy_links = entities.get("policy_links", [])
    claim_links  = entities.get("claim_links", [])

    with driver.session() as session:
        # Named insured → Policy
        person_links = [lk for lk in policy_links if lk.get("insured_type") != "organization" and lk.get("insured_id")]
        session.run("""
            UNWIND $links AS lk
            MATCH (person:Person {id: lk.insured_id})
            MATCH (pol:Policy    {id: lk.policy_id})
            MERGE (person)-[:INSURED_UNDER]->(pol)
        """, links=person_links)

        org_links = [lk for lk in policy_links if lk.get("insured_type") == "organization" and lk.get("insured_id")]
        session.run("""
            UNWIND $links AS lk
            MATCH (org:Organization {id: lk.insured_id})
            MATCH (pol:Policy       {id: lk.policy_id})
            MERGE (org)-[:INSURED_UNDER]->(pol)
        """, links=org_links)

        # Business owner → Organization
        owner_links = [lk for lk in policy_links if lk.get("owner_id") and lk.get("insured_id") and lk.get("insured_type") == "organization"]
        session.run("""
            UNWIND $links AS lk
            MATCH (owner:Person       {id: lk.owner_id})
            MATCH (org:Organization   {id: lk.insured_id})
            MERGE (owner)-[:OWNS]->(org)
        """, links=owner_links)

        # Policy → Agent
        agent_links = [lk for lk in policy_links if lk.get("agent_id")]
        session.run("""
            UNWIND $links AS lk
            MATCH (pol:Policy {id: lk.policy_id})
            MATCH (agent:Agent {id: lk.agent_id})
            MERGE (pol)-[:MANAGED_BY]->(agent)
        """, links=agent_links)

        # Policy → Location
        loc_links = [lk for lk in policy_links if lk.get("location_id")]
        session.run("""
            UNWIND $links AS lk
            MATCH (pol:Policy    {id: lk.policy_id})
            MATCH (loc:Location  {id: lk.location_id})
            MERGE (pol)-[:LOCATED_AT]->(loc)
        """, links=loc_links)

        # Claim → Contractor (REPAIRED_BY)
        repair_links = [lk for lk in claim_links if lk.get("link_type") == "repaired_by"]
        session.run("""
            UNWIND $links AS lk
            MATCH (claim:Claim       {id: lk.claim_id})
            MATCH (con:Contractor    {id: lk.entity_id})
            MERGE (claim)-[:REPAIRED_BY]->(con)
        """, links=repair_links)

        # Claim → ThirdParty (claimant)
        claimant_links = [lk for lk in claim_links if lk.get("link_type") == "claimant"]
        session.run("""
            UNWIND $links AS lk
            MATCH (claim:Claim      {id: lk.claim_id})
            MATCH (tp:ThirdParty    {id: lk.entity_id})
            MERGE (claim)-[:INVOLVES_THIRD_PARTY]->(tp)
        """, links=claimant_links)

        # Claim → ThirdParty (subrogation target)
        subrogation_links = [lk for lk in claim_links if lk.get("link_type") == "subrogation_target"]
        session.run("""
            UNWIND $links AS lk
            MATCH (claim:Claim   {id: lk.claim_id})
            MATCH (tp:ThirdParty {id: lk.entity_id})
            MERGE (claim)-[:HAS_SUBROGATION_RIGHT]->(tp)
        """, links=subrogation_links)

    print(f"  Policy links: {len(policy_links)}  Claim links: {len(claim_links)}")


# ---------------------------------------------------------------------------
# Edge loaders
# ---------------------------------------------------------------------------

def load_next_chunk_edges(driver, manifest):
    """
    Create NEXT_CHUNK edges between consecutive Shadow nodes within
    the same section.

    Why NEXT_CHUNK?
        When a relevant chunk is found via vector search, the surrounding
        chunks often contain continuation of the same thought. NEXT_CHUNK
        lets the agent fetch neighboring chunks for additional context
        without a second vector search.

    Why sort by chunk_index?
        Shadows are stored in manifest order but chunk_index is the
        authoritative order assigned during ingest. Sorting guarantees
        correct sequential linking regardless of manifest ordering.

    Note: this is NOT a temporal edge. NEXT_CHUNK is text continuity
    within a document. NEXT_PERIOD (built in load_next_period_edges) is
    chronological order across separate claim filings over time.
    """
    edges = []
    for doc in manifest:
        sections = {}
        for shadow in doc["shadows"]:
            sid = shadow["section_id"]
            if sid not in sections:
                sections[sid] = []
            sections[sid].append(shadow)

        for sid, shadows in sections.items():
            sorted_shadows = sorted(shadows, key=lambda s: s["chunk_index"])
            for i in range(len(sorted_shadows) - 1):
                edges.append({
                    "from_id": sorted_shadows[i]["shadow_id"],
                    "to_id": sorted_shadows[i + 1]["shadow_id"],
                })

    with driver.session() as session:
        session.run("""
            UNWIND $edges AS e
            MATCH (a:Shadow {id: e.from_id})
            MATCH (b:Shadow {id: e.to_id})
            MERGE (a)-[:NEXT_CHUNK]->(b)
        """, edges=edges)

    print(f"  Created {len(edges)} NEXT_CHUNK edges.")


def load_next_period_edges(driver, manifest):
    """
    Create NEXT_PERIOD edges between consecutive FNOL ClaimDocuments for the
    same policy, ordered by date_of_loss.

    Why FNOL docs and not all claim docs?
        Only FNOL docs carry date_of_loss. Using one representative doc per
        claim (the FNOL) keeps the chain simple: one node per claim in the
        temporal sequence.

    Why group by policy_id and not claim_id?
        NEXT_PERIOD is a cross-claim edge — it chains separate claims filed
        against the same policy over time. Grouping by claim_id would only
        look within a single claim (one FNOL = no pairs).
    """
    from collections import defaultdict
    from dateutil import parser as dateparser

    # Use only FNOL docs — they carry date_of_loss and represent each claim.
    policy_groups: dict[str, list[dict]] = defaultdict(list)
    for doc in manifest:
        if doc.get("doc_type") != "claim_document":
            continue
        if not doc["doc_id"].endswith("_fnol"):
            continue
        policy_id = doc.get("policy_id")
        date_str = doc.get("date_of_loss")
        if not policy_id or not date_str:
            continue
        try:
            doc["_parsed_date"] = dateparser.parse(date_str)
            policy_groups[policy_id].append(doc)
        except Exception:
            continue

    edges = []
    for policy_id, docs in policy_groups.items():
        sorted_docs = sorted(docs, key=lambda d: d["_parsed_date"])
        for i in range(len(sorted_docs) - 1):
            edges.append({
                "from_id": sorted_docs[i]["doc_id"],
                "to_id": sorted_docs[i + 1]["doc_id"],
            })

    if not edges:
        print("  No NEXT_PERIOD edges to create (no docs with date_of_loss).")
        return

    with driver.session() as session:
        session.run("""
            UNWIND $edges AS e
            MATCH (a:ClaimDocument {id: e.from_id})
            MATCH (b:ClaimDocument {id: e.to_id})
            MERGE (a)-[:NEXT_PERIOD]->(b)
        """, edges=edges)

    print(f"  Created {len(edges)} NEXT_PERIOD edges.")


def load_supersedes(driver):
    """
    Create the SUPERSEDES edge from the amended HO-3 policy to the original,
    and mark the original as superseded.

    Why mark superseded=true on the old policy?
        retrieve.py filters out superseded nodes:
            WHERE NOT doc.superseded = true
        This ensures retrieval always returns the current version unless
        the user explicitly asks about the old one. The old node stays in
        the graph as an audit trail — required for benchmark Q14 and Q15
        which test version comparison reasoning.
    """
    with driver.session() as session:
        session.run("""
            MATCH (new:Policy {id: 'policy_HO3_001_v2'})
            MATCH (old:Policy {id: 'policy_HO3_001'})
            MERGE (new)-[:SUPERSEDES]->(old)
            SET old.superseded = true
        """)
    print("  SUPERSEDES edge created.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Connecting to Neo4j...")
    driver = get_driver()
    print("  Connected.")

    print("Creating schema...")
    create_schema(driver)

    manifest = json.loads(MANIFEST_FILE.read_text())
    print(f"Manifest loaded: {len(manifest)} documents")

    print("Loading policies...")
    load_policies(driver, manifest)

    print("Loading claims...")
    load_claims(driver, manifest)

    # Claims must load before sections — load_sections_and_shadows matches parent
    # nodes by id, so ClaimDocument nodes must exist before HAS_SECTION edges
    # can be created for claim documents.
    print("Loading sections and shadows...")
    load_sections_and_shadows(driver, manifest)

    print("Loading endorsements...")
    load_endorsements(driver, manifest)

    if ENTITIES_FILE.exists():
        entities = json.loads(ENTITIES_FILE.read_text())
        print("Loading entity nodes...")
        load_entities(driver, entities)
        print("Loading entity edges...")
        load_entity_edges(driver, entities)
    else:
        print("  entities.json not found — skipping entity load (run extract_entities.py first)")

    print("Loading NEXT_CHUNK edges...")
    load_next_chunk_edges(driver, manifest)

    print("Loading NEXT_PERIOD edges...")
    load_next_period_edges(driver, manifest)

    print("Creating SUPERSEDES edge...")
    load_supersedes(driver)

    driver.close()
    print("\nDone. Next: python pipeline/extract_keywords.py")


if __name__ == "__main__":
    main()
