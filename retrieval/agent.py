"""
retrieval/agent.py
==================
ReAct retrieval agent — Gemini function-calling loop.

Gemini decides which tools to call based on the question:

  Simple question  → search_vectors only (like RAG)
  Claim question   → search_vectors → get_policy_for_claim (REFERENCES_POLICY)
  Endorsement Q    → search_vectors → get_endorsements
  Fraud/entity Q   → search_vectors → traverse_entity (REPAIRED_BY, FILED_AGAINST, OWNS)

Tools:
  search_vectors(query, top_k)       — FAISS cosine search → shadow texts
  get_policy_for_claim(shadow_id)    — REFERENCES_POLICY → policy section texts
  get_endorsements(policy_id)        — policy → endorsement texts
  traverse_entity(entity_name)       — entity graph edges → linked claims/policies
"""

import time
from google.genai import types

# Lazy imports to avoid circular dependency — agent.py imports from retrieve.py,
# retrieve.py imports from agent.py only inside retrieve() when mode="agent".
from retrieval.retrieve import (
    embed_query,
    cosine_search,
    shadow_texts,
    shadow_to_doc,
    driver,
    gemini,
    MODEL,
    RetrievalResult,
    _first,
    _fetch_section_shadows,
)

SYSTEM_PROMPT = """You are an insurance expert with access to a knowledge graph of insurance policies and claims.

Answer questions by calling the available tools. Always traverse the graph to get actual text — never stop at IDs.

Tool selection:
- Simple factual question: call search_vectors
- Cross-document question needing graph traversal: call graph_rag_search
- Entity/fraud/ownership question: call traverse_entity to get policy and claim IDs, then call get_policy_sections or get_claim_sections to read the actual content
- Question about a specific policy's coverage or exclusions: call get_policy_sections(policy_id)
- Question about what happened in a specific claim: call get_claim_sections(claim_id)
- Endorsement question (water backup, rideshare, scheduled property): call get_endorsements
- Claim denial question needing the referenced policy: call get_policy_for_claim(shadow_id)

Chaining pattern for entity questions:
1. traverse_entity → returns direct_policies (personal policies) AND org_policies (business policies owned through organizations)
2. Call get_policy_sections for EVERY org_policy returned — these are the business policies, do not skip them
3. Call get_policy_sections for direct policies only if the question is about personal coverage
4. Call get_claim_sections(claim_id) for each relevant claim
5. Answer from the retrieved text — never conclude "no coverage" without first reading the org_policies

Stop calling tools once you have enough to answer. Answer using only retrieved evidence — never guess.

Policy IDs look like: policy_HO3_001, policy_PAP_001, policy_CGL_001
Claim IDs look like: claim_HO_001, claim_HC_003
Entity names are real names: "FastFix Restoration", "Marcus Webb", "James Carter", "Carlos Mendez\""""

_TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_vectors",
        description="Search for relevant insurance document chunks using semantic similarity. Use for simple factual questions.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(type=types.Type.STRING, description="Search query"),
                "top_k": types.Schema(type=types.Type.INTEGER, description="Number of results (default 10)"),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_policy_for_claim",
        description="Given a shadow ID from a claim document, retrieve all sections of the policy the claim references. Use when you need policy coverage terms, exclusions, or conditions to answer a question about a claim.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "shadow_id": types.Schema(type=types.Type.STRING, description="Shadow ID from a claim document found via search_vectors"),
            },
            required=["shadow_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_endorsements",
        description="Get all endorsement texts for a policy. Endorsements add or modify coverage on top of the base policy. Use when asking about additional coverage like water backup, rideshare, or scheduled property.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "policy_id": types.Schema(type=types.Type.STRING, description="Policy ID, e.g. policy_HO3_001"),
            },
            required=["policy_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="traverse_entity",
        description="Find all claims and policies connected to a named entity (contractor, person, agent, or organization). Use for fraud detection, repeat claimant analysis, ownership structure, or shared agent questions. This traverses graph edges that vector search cannot find.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "entity_name": types.Schema(type=types.Type.STRING, description="Full name of the entity, e.g. 'FastFix Restoration', 'Marcus Webb', 'James Carter', 'Carlos Mendez'"),
            },
            required=["entity_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="graph_rag_search",
        description="Run the full Graph RAG pipeline: routes the query through document clusters, traverses Neo4j graph edges, and re-ranks results with a cross-encoder. Use for cross-document questions that require connecting related policies and claims through the knowledge graph.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(type=types.Type.STRING, description="The question to answer using graph-based retrieval"),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_policy_sections",
        description="Get the full text of all sections and endorsements for a specific policy by following graph edges Policy→Section→Shadow. Use after traverse_entity returns a policy ID, or whenever you need the complete content of a known policy.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "policy_id": types.Schema(type=types.Type.STRING, description="Policy ID, e.g. policy_CGL_001, policy_HO3_002"),
            },
            required=["policy_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_claim_sections",
        description="Get the full text of all sections in a claim document by following graph edges Claim→ClaimDocument→Section→Shadow. Use after traverse_entity returns a claim ID, or whenever you need the complete content of a known claim.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "claim_id": types.Schema(type=types.Type.STRING, description="Claim ID, e.g. claim_HO_001, claim_HC_003"),
            },
            required=["claim_id"],
        ),
    ),
])]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_search_vectors(query: str, top_k: int = 10) -> str:
    q_vec = embed_query(query)
    results = cosine_search(q_vec, top_k=min(top_k, 20))
    lines = []
    for shadow_id, score in results:
        text = shadow_texts.get(shadow_id, "")[:600]
        doc_id = shadow_to_doc.get(shadow_id, "unknown")
        lines.append(f"[shadow_id={shadow_id} | doc={doc_id} | score={score:.3f}]\n{text}")
    return "\n\n---\n\n".join(lines) if lines else "No results found."


def _tool_get_policy_for_claim(shadow_id: str) -> str:
    doc_id = shadow_to_doc.get(shadow_id)
    if not doc_id:
        return f"Shadow ID {shadow_id!r} not found in manifest."

    with driver.session() as session:
        pol_row = _first(session.run("""
            MATCH (cd:ClaimDocument {id: $cid})-[:REFERENCES_POLICY]->(pol:Policy)
            MATCH (pol)-[:HAS_SECTION]->(sec:Section)
            RETURN pol.id AS pol_id,
                   collect({id: sec.id, title: sec.title, summary: sec.summary}) AS sections
        """, cid=doc_id))

        if not pol_row:
            return f"No REFERENCES_POLICY link found for document {doc_id!r}. This shadow may not be from a claim document."

        pol_id = pol_row["pol_id"]
        lines = [f"Policy: {pol_id}\n"]
        for sec in pol_row["sections"]:
            shadow_list = _fetch_section_shadows(session, sec["id"], sec["title"] or sec["id"], limit=2)
            if shadow_list:
                lines.append("\n".join(shadow_list))

        return "\n\n".join(lines)


def _tool_get_endorsements(policy_id: str) -> str:
    with driver.session() as session:
        rows = session.run("""
            MATCH (pol:Policy {id: $pid})-[:HAS_ENDORSEMENT]->(end:Endorsement)
            MATCH (end)-[:HAS_SECTION]->(sec:Section)-[:HAS_SHADOW]->(sw:Shadow)
            RETURN end.id AS end_id, sec.title AS sec_title, sw.text AS text
            ORDER BY end.id, sec.title, sw.chunk_index
        """, pid=policy_id).data()

        if not rows:
            return f"No endorsements found for policy {policy_id!r}."

        lines = []
        for row in rows:
            lines.append(f"[{row['end_id']} — {row['sec_title']}]\n{(row['text'] or '')[:500]}")
        return "\n\n---\n\n".join(lines)


def _tool_traverse_entity(entity_name: str) -> str:
    with driver.session() as session:
        lines = []

        # Contractor → claims it repaired (REPAIRED_BY edges, reversed)
        rows = session.run("""
            MATCH (c:Contractor)
            WHERE toLower(c.name) CONTAINS toLower($name)
            MATCH (claim:Claim)-[:REPAIRED_BY]->(c)
            OPTIONAL MATCH (claim)-[:HAS_DOCUMENT]->(cd:ClaimDocument)
            RETURN c.name AS entity, 'Contractor' AS type,
                   collect(DISTINCT {claim_id: claim.id, doc_id: cd.id, outcome: claim.outcome}) AS links
        """, name=entity_name).data()
        for row in rows:
            lines.append(f"Contractor: {row['entity']}")
            for link in row["links"]:
                lines.append(f"  → Claim {link['claim_id']} (outcome: {link['outcome']}, doc: {link['doc_id']})")

        # ThirdParty → claims filed against (FILED_AGAINST edges)
        rows = session.run("""
            MATCH (tp:ThirdParty)
            WHERE toLower(tp.name) CONTAINS toLower($name)
            MATCH (tp)-[:FILED_AGAINST]->(claim:Claim)
            OPTIONAL MATCH (claim)-[:HAS_DOCUMENT]->(cd:ClaimDocument)
            RETURN tp.name AS entity,
                   collect(DISTINCT {claim_id: claim.id, doc_id: cd.id}) AS links
        """, name=entity_name).data()
        for row in rows:
            lines.append(f"Third Party: {row['entity']}")
            for link in row["links"]:
                lines.append(f"  → Filed against Claim {link['claim_id']} (doc: {link['doc_id']})")

        # Person → policies directly + organizations they own → policies
        rows = session.run("""
            MATCH (p:Person)
            WHERE toLower(p.name) CONTAINS toLower($name)
            OPTIONAL MATCH (p)-[:HAS_POLICY]->(pol:Policy)
            OPTIONAL MATCH (p)-[:OWNS]->(org:Organization)-[:HAS_POLICY]->(org_pol:Policy)
            RETURN p.name AS entity,
                   collect(DISTINCT pol.id) AS direct_policies,
                   collect(DISTINCT {org: org.name, policy: org_pol.id}) AS org_policies
        """, name=entity_name).data()
        for row in rows:
            lines.append(f"Person: {row['entity']}")
            for pid in row["direct_policies"]:
                if pid:
                    lines.append(f"  → Direct policy: {pid}")
            for link in row["org_policies"]:
                if not (link["org"] and link["policy"]):
                    continue
                lines.append(f"  → Owns {link['org']} → policy {link['policy']}")
                lines.append(_tool_get_policy_sections(link["policy"]))

        # Agent → all policies they manage
        rows = session.run("""
            MATCH (a:Agent)
            WHERE toLower(a.name) CONTAINS toLower($name)
            MATCH (pol:Policy)-[:MANAGED_BY]->(a)
            RETURN a.name AS entity,
                   collect(DISTINCT pol.id) AS policies
        """, name=entity_name).data()
        for row in rows:
            lines.append(f"Agent: {row['entity']}")
            for pid in row["policies"]:
                lines.append(f"  → Manages policy: {pid}")

        if not lines:
            return f"No entity found matching '{entity_name}'. Try a partial name."
        return "\n".join(lines)


def _tool_get_policy_sections(policy_id: str) -> str:
    with driver.session() as session:
        rows = session.run("""
            MATCH (pol:Policy {id: $pid})-[:HAS_SECTION]->(sec:Section)-[:HAS_SHADOW]->(sw:Shadow)
            RETURN 'base' AS source, sec.title AS title, sw.text AS text, sw.chunk_index AS idx
            ORDER BY sec.title, sw.chunk_index
        """, pid=policy_id).data()

        end_rows = session.run("""
            MATCH (pol:Policy {id: $pid})-[:HAS_ENDORSEMENT]->(end:Endorsement)
            MATCH (end)-[:HAS_SECTION]->(sec:Section)-[:HAS_SHADOW]->(sw:Shadow)
            RETURN end.id AS source, sec.title AS title, sw.text AS text, sw.chunk_index AS idx
            ORDER BY end.id, sec.title, sw.chunk_index
        """, pid=policy_id).data()

        all_rows = rows + end_rows
        if not all_rows:
            return f"No sections found for policy {policy_id!r}."

        lines = [f"Policy: {policy_id}\n"]
        for row in all_rows:
            label = f"[{row['source']}] {row['title'] or ''}"
            lines.append(f"{label}\n{(row['text'] or '')[:600]}")
        return "\n\n---\n\n".join(lines)


def _tool_get_claim_sections(claim_id: str) -> str:
    with driver.session() as session:
        rows = session.run("""
            MATCH (claim:Claim {id: $cid})-[:HAS_DOCUMENT]->(cd:ClaimDocument)
            MATCH (cd)-[:HAS_SECTION]->(sec:Section)-[:HAS_SHADOW]->(sw:Shadow)
            RETURN cd.id AS doc_id, sec.title AS title, sw.text AS text, sw.chunk_index AS idx
            ORDER BY sec.title, sw.chunk_index
        """, cid=claim_id).data()

        if not rows:
            return f"No sections found for claim {claim_id!r}."

        lines = [f"Claim: {claim_id}\n"]
        for row in rows:
            label = f"[{row['doc_id']}] {row['title'] or ''}"
            lines.append(f"{label}\n{(row['text'] or '')[:600]}")
        return "\n\n---\n\n".join(lines)


def _tool_graph_rag_search(query: str) -> str:
    from retrieval.retrieve import retrieve
    result = retrieve(query, mode="graph_rag", top_k=7)
    lines = [f"Graph RAG answer: {result.answer}", ""]
    for i, path in enumerate(result.supporting_paths, 1):
        nodes = " → ".join(path.get("node_ids", []))
        edges = " → ".join(path.get("edge_types", []))
        lines.append(f"Path {i}: {nodes} ({edges})")
        for tier, texts in path.get("tiered_texts", {}).items():
            for t in texts:
                lines.append(f"  [{tier}] {t[:400]}")
    return "\n".join(lines)


def _execute_tool(name: str, args: dict) -> str:
    if name == "search_vectors":
        return _tool_search_vectors(args.get("query", ""), int(args.get("top_k", 10)))
    if name == "get_policy_for_claim":
        return _tool_get_policy_for_claim(args.get("shadow_id", ""))
    if name == "get_endorsements":
        return _tool_get_endorsements(args.get("policy_id", ""))
    if name == "traverse_entity":
        return _tool_traverse_entity(args.get("entity_name", ""))
    if name == "graph_rag_search":
        return _tool_graph_rag_search(args.get("query", ""))
    if name == "get_policy_sections":
        return _tool_get_policy_sections(args.get("policy_id", ""))
    if name == "get_claim_sections":
        return _tool_get_claim_sections(args.get("claim_id", ""))
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def retrieve_agent(question: str, max_turns: int = 6) -> RetrievalResult:
    """
    Run the ReAct agent loop.
    Each turn: Gemini sees the conversation so far, calls tools or writes final answer.
    Loop ends when Gemini produces a response with no function calls.
    """
    t0 = time.time()
    supporting_paths: list[dict] = []

    contents: list = [
        types.Content(role="user", parts=[types.Part(text=question)])
    ]

    answer = "Agent did not produce an answer."

    for _ in range(max_turns):
        response = gemini.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=_TOOLS,
                system_instruction=SYSTEM_PROMPT,
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.AUTO
                    )
                ),
            ),
        )

        # Add model turn to conversation history
        contents.append(types.Content(
            role="model",
            parts=response.candidates[0].content.parts,
        ))

        # Collect any function calls in this response
        function_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call
        ]

        if not function_calls:
            # No tool calls → Gemini is done, take the text as the answer
            answer = response.text or answer
            break

        # Execute each tool call and collect results
        tool_result_parts = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            result_text = _execute_tool(fc.name, args)
            supporting_paths.append({"tool": fc.name, "args": args})
            tool_result_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result_text},
                )
            )

        contents.append(types.Content(role="user", parts=tool_result_parts))

    return RetrievalResult(
        answer=answer,
        mode="agent",
        supporting_paths=supporting_paths,
        clusters_used=[],
        latency_breakdown={"total_ms": round((time.time() - t0) * 1000)},
    )
