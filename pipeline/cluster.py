"""
pipeline/cluster.py
===================
Builds a networkx graph from manifest.json + entities.json, runs Louvain
community detection, and saves cluster assignments to manifest/clusters.json.

load_neo4j.py reads clusters.json and writes cluster_id onto every node
so retrieve.py can expand context by pulling nodes in the same community.

Why build from manifest and not Neo4j?
    Neo4j is derived state — it can be wiped and reloaded at any time.
    Clustering runs offline on the source files so it never depends on
    the database being healthy.

Usage:
  python pipeline/cluster.py
"""

import json
from pathlib import Path
from collections import defaultdict
import networkx as nx
import community as community_louvain

ROOT           = Path(__file__).parent.parent
MANIFEST_FILE  = ROOT / "manifest" / "manifest.json"
ENTITIES_FILE  = ROOT / "manifest" / "entities.json"
CLUSTERS_FILE  = ROOT / "manifest" / "clusters.json"


def build_graph(manifest: list[dict], entities: dict) -> nx.Graph:
    """
    Build an undirected networkx graph from manifest + entities.

    Nodes: every doc_id (policy, endorsement, claim doc) and every
           entity id (person, org, agent, location, contractor, third party).

    Edges: all relationships from manifest and entities.json.
           Shadow/Section nodes are excluded — too granular for community
           detection; we want clusters at the document and entity level.
    """
    G = nx.Graph()

    # --- Nodes from manifest ---
    for doc in manifest:
        G.add_node(doc["doc_id"], label=doc.get("doc_type", "document"))

    # --- Nodes from entities ---
    for person in entities.get("persons", []):
        G.add_node(person["id"], label="person")
    for org in entities.get("organizations", []):
        G.add_node(org["id"], label="organization")
    for agent in entities.get("agents", []):
        G.add_node(agent["id"], label="agent")
    for loc in entities.get("locations", []):
        G.add_node(loc["id"], label="location")
    for contractor in entities.get("contractors", []):
        G.add_node(contractor["id"], label="contractor")
    for tp in entities.get("third_parties", []):
        G.add_node(tp["id"], label="third_party")

    # --- Claim nodes (claim_id level, bridges doc_ids to entity links) ---
    for doc in manifest:
        if doc.get("doc_type") == "claim_document":
            claim_id = doc.get("claim_id")
            if claim_id:
                G.add_node(claim_id, label="claim")
                G.add_edge(claim_id, doc["doc_id"], rel="has_document")

    # --- Edges from manifest ---

    # Policy → Endorsement
    endorsement_docs = [d for d in manifest if d.get("doc_type") == "endorsement"]
    for doc in endorsement_docs:
        parts = doc["doc_id"].split("_")
        policy_id = f"policy_{parts[1]}_{parts[2]}"
        if G.has_node(policy_id):
            G.add_edge(doc["doc_id"], policy_id, rel="has_endorsement")

    # ClaimDocument → Policy (REFERENCES_POLICY)
    claim_policy_map: dict[str, str] = {}
    for doc in manifest:
        if doc.get("doc_type") == "claim_document" and doc.get("policy_id") and doc.get("claim_id"):
            claim_policy_map[doc["claim_id"]] = doc["policy_id"]

    for doc in manifest:
        if doc.get("doc_type") != "claim_document":
            continue
        policy_id = doc.get("policy_id") or claim_policy_map.get(doc.get("claim_id", ""), "")
        if policy_id and G.has_node(policy_id):
            G.add_edge(doc["doc_id"], policy_id, rel="references_policy")

    # NEXT_PERIOD — consecutive FNOL docs per policy
    from dateutil import parser as dateparser
    policy_fnols: dict[str, list[dict]] = defaultdict(list)
    for doc in manifest:
        if doc.get("doc_type") == "claim_document" and doc["doc_id"].endswith("_fnol"):
            policy_id = doc.get("policy_id") or claim_policy_map.get(doc.get("claim_id", ""), "")
            date_str = doc.get("date_of_loss")
            if policy_id and date_str:
                try:
                    doc["_parsed_date"] = dateparser.parse(date_str)
                    policy_fnols[policy_id].append(doc)
                except Exception:
                    pass

    for policy_id, fnols in policy_fnols.items():
        sorted_fnols = sorted(fnols, key=lambda d: d["_parsed_date"])
        for i in range(len(sorted_fnols) - 1):
            G.add_edge(sorted_fnols[i]["doc_id"], sorted_fnols[i + 1]["doc_id"], rel="next_period")

    # --- Edges from entities ---
    for link in entities.get("policy_links", []):
        policy_id   = link.get("policy_id")
        insured_id  = link.get("insured_id")
        owner_id    = link.get("owner_id")
        agent_id    = link.get("agent_id")
        location_id = link.get("location_id")

        if insured_id and G.has_node(insured_id) and G.has_node(policy_id):
            G.add_edge(insured_id, policy_id, rel="insured_under")
        if owner_id and insured_id and G.has_node(owner_id) and G.has_node(insured_id):
            G.add_edge(owner_id, insured_id, rel="owns")
        if agent_id and G.has_node(agent_id) and G.has_node(policy_id):
            G.add_edge(agent_id, policy_id, rel="managed_by")
        if location_id and G.has_node(location_id) and G.has_node(policy_id):
            G.add_edge(location_id, policy_id, rel="located_at")

    for link in entities.get("claim_links", []):
        claim_id  = link.get("claim_id")
        entity_id = link.get("entity_id")
        if claim_id and entity_id and G.has_node(claim_id) and G.has_node(entity_id):
            G.add_edge(claim_id, entity_id, rel=link.get("link_type", "linked"))

    return G


def main():
    manifest = json.loads(MANIFEST_FILE.read_text())
    entities = json.loads(ENTITIES_FILE.read_text()) if ENTITIES_FILE.exists() else {}

    print("Building graph...")
    G = build_graph(manifest, entities)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Running Louvain community detection...")
    partition = community_louvain.best_partition(G)

    num_communities = len(set(partition.values()))
    print(f"  {num_communities} communities found")

    from collections import Counter
    sizes = Counter(partition.values())
    size_dist = Counter(sizes.values())
    for size in sorted(size_dist):
        print(f"    {size_dist[size]} communities of size {size}")
    print(f"  Largest: {max(sizes.values())} nodes  |  Smallest: {min(sizes.values())} nodes")

    singletons = [node for node, cid in partition.items() if sizes[cid] == 1]
    if singletons:
        print(f"\n  Singleton nodes ({len(singletons)} isolated — no edges):")
        for node in sorted(singletons):
            print(f"    {node}")

    CLUSTERS_FILE.write_text(json.dumps(partition, indent=2))
    print(f"\nWritten: {CLUSTERS_FILE}")
    print("Next: python pipeline/load_neo4j.py  (to write cluster_ids into Neo4j)")


if __name__ == "__main__":
    main()
