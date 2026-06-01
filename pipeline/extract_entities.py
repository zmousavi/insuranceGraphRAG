"""
pipeline/extract_entities.py
=============================
Reads manifest.json, extracts structured entities from policy DECLARATIONS
sections and claim adjuster notes using Gemini.

Outputs:
  manifest/entities.json

Entities extracted:
  From policy DECLARATIONS:
    Person       — named insured of personal policies (HO-3, PAP)
    Organization — named insured of business policies (CGL, Commercial Property)
    Agent        — agent name + agency
    Location     — property or business address

  From claim adjuster notes:
    Contractor   — repair/restoration company named in site inspection
    ThirdParty   — injured claimant or liable subrogation target

Usage:
  python pipeline/extract_entities.py
"""

import os
import re
import json
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from google import genai

load_dotenv()

PROJECT         = os.getenv("GOOGLE_CLOUD_PROJECT")
GEMINI_LOCATION = "global"
MODEL           = "gemini-2.5-flash"

ROOT          = Path(__file__).parent.parent
MANIFEST_FILE = ROOT / "manifest" / "manifest.json"
ENTITIES_FILE = ROOT / "manifest" / "entities.json"


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def init_gemini() -> genai.Client:
    return genai.Client(vertexai=True, project=PROJECT, location=GEMINI_LOCATION)


def call_gemini(client: genai.Client, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={"thinking_config": {"thinking_budget": 0}},
            )
            return response.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e


def parse_json(text: str) -> dict | None:
    """Strip markdown fences and parse JSON from Gemini response."""
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def slugify(name: str) -> str:
    """Convert a name to a stable lowercase ID slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# Noise patterns that should never become ThirdParty nodes.
_TP_NOISE = re.compile(
    r"^(unknown|perpetrator|john doe|jane doe|mr\.?\s*john doe|"
    r"ms\.?\s*jane doe|unidentified|n/a)$",
    re.IGNORECASE,
)

def normalize_address(addr: str) -> str:
    """Normalize common address abbreviation variants before slugifying."""
    a = addr.lower()
    a = re.sub(r"\bstreet\b",   "st",  a)
    a = re.sub(r"\bavenue\b",   "ave", a)
    a = re.sub(r"\bboulevard\b","blvd",a)
    a = re.sub(r"\broad\b",     "rd",  a)
    a = re.sub(r"\bdrive\b",    "dr",  a)
    a = re.sub(r"\billinois\b", "il",  a)
    a = re.sub(r"[^a-z0-9]+",  "_",   a)
    return a.strip("_")


# ---------------------------------------------------------------------------
# Extraction prompts
# ---------------------------------------------------------------------------

POLICY_PROMPT = """
Extract entities from this insurance policy DECLARATIONS section.
Return ONLY a valid JSON object — no markdown, no explanation.

Text:
{text}

Return exactly this structure:
{{
  "named_insured": {{
    "name": "full name exactly as written",
    "type": "person or organization"
  }},
  "owner": "full name of business owner or principal if explicitly mentioned, else null",
  "agent": {{
    "name": "agent full name",
    "agency": "agency name"
  }},
  "address": "full property or business address as written"
}}
"""

CLAIM_PROMPT = """
Extract entities from this insurance claim adjuster notes.
Return ONLY a valid JSON object — no markdown, no explanation.

Text:
{text}

Return exactly this structure:
{{
  "contractor": "repair or restoration company name if mentioned in site inspection, else null",
  "third_parties": [
    {{
      "name": "party name",
      "role": "claimant or subrogation_target"
    }}
  ]
}}

Strict rules:
- contractor: a repair/restoration company that quoted or performed repairs (e.g. FastFix Restoration).
  Must be a named company — null if only vaguely referenced or unnamed.
- claimant: a third party who was injured or filed the claim against the insured.
  Must be a real named individual (first + last name) or a named company.
  Do NOT include: unnamed roles ("perpetrator", "unknown driver"), placeholder names ("John Doe"),
  government bodies (police department, municipality), suppliers/vendors unrelated to the claim,
  or the policyholder themselves.
- subrogation_target: a named company or individual the insurer explicitly has recovery rights against.
- Return empty list if no qualifying third parties.
"""


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_from_policies(client: genai.Client, manifest: list[dict]) -> dict[str, dict]:
    """Extract Person/Org/Agent/Location from each policy DECLARATIONS section."""
    results: dict[str, dict] = {}

    policy_docs = [d for d in manifest if d.get("doc_type") == "policy"]
    print(f"  {len(policy_docs)} policy docs to process...")

    for doc in policy_docs:
        policy_id = doc["doc_id"]
        sections = doc.get("sections", [])
        decl = next((s for s in sections if s["title"] == "DECLARATIONS"), None)
        if not decl and sections:
            decl = sections[0]
        if not decl:
            print(f"  {policy_id}: no DECLARATIONS section — skipping")
            continue

        text = decl.get("text", "")[:5000]
        response = call_gemini(client, POLICY_PROMPT.format(text=text))
        parsed = parse_json(response)

        if parsed:
            results[policy_id] = parsed
            name = parsed.get("named_insured", {}).get("name", "?")
            print(f"  {policy_id}: {name}")
        else:
            print(f"  {policy_id}: parse failed — {response[:80]}")

        time.sleep(1)

    return results


def extract_from_claims(client: genai.Client, manifest: list[dict]) -> dict[str, dict]:
    """Extract Contractor/ThirdParty from each claim adjuster doc."""
    results: dict[str, dict] = {}

    adjuster_docs = [
        d for d in manifest
        if d.get("doc_type") == "claim_document" and d["doc_id"].endswith("_adjuster")
    ]
    print(f"  {len(adjuster_docs)} adjuster docs to process...")

    for doc in adjuster_docs:
        claim_id = doc.get("claim_id", "")
        sections = doc.get("sections", [])
        text = " ".join(s.get("text", "") for s in sections)[:5000]
        if not text:
            continue

        response = call_gemini(client, CLAIM_PROMPT.format(text=text))
        parsed = parse_json(response)

        if parsed:
            contractor = parsed.get("contractor")
            tps = parsed.get("third_parties", [])
            if contractor or tps:
                results[claim_id] = parsed
                print(f"  {claim_id}: contractor={contractor} third_parties={[t['name'] for t in tps]}")
        else:
            print(f"  {doc['doc_id']}: parse failed")

        time.sleep(0.5)

    return results


# ---------------------------------------------------------------------------
# Entity deduplication + link building
# ---------------------------------------------------------------------------

def build_entities(policy_data: dict, claim_data: dict) -> dict:
    """
    Deduplicate extracted entities by name and build the final entities.json
    structure consumed by load_neo4j.py.
    """
    persons:       dict[str, dict] = {}
    organizations: dict[str, dict] = {}
    agents:        dict[str, dict] = {}
    locations:     dict[str, dict] = {}
    contractors:   dict[str, dict] = {}
    third_parties: dict[str, dict] = {}
    policy_links:  list[dict] = []
    claim_links:   list[dict] = []

    for policy_id, ext in policy_data.items():
        named     = ext.get("named_insured", {})
        owner_str = ext.get("owner")
        agent_ext = ext.get("agent", {})
        address   = ext.get("address")

        # Named insured — Person or Organization
        insured_name = named.get("name", "").strip()
        insured_type = named.get("type", "person").lower()
        insured_id   = None

        if insured_name:
            if insured_type == "organization":
                insured_id = f"org_{slugify(insured_name)}"
                if insured_id not in organizations:
                    organizations[insured_id] = {"id": insured_id, "name": insured_name}
            else:
                insured_id = f"person_{slugify(insured_name)}"
                if insured_id not in persons:
                    persons[insured_id] = {"id": insured_id, "name": insured_name}

        # Business owner — always a Person
        owner_id = None
        if owner_str and owner_str.strip():
            owner_id = f"person_{slugify(owner_str.strip())}"
            if owner_id not in persons:
                persons[owner_id] = {"id": owner_id, "name": owner_str.strip()}

        # Agent
        agent_id   = None
        agent_name = (agent_ext.get("name") or "").strip()
        if agent_name:
            agent_id = f"agent_{slugify(agent_name)}"
            if agent_id not in agents:
                agents[agent_id] = {
                    "id":     agent_id,
                    "name":   agent_name,
                    "agency": (agent_ext.get("agency") or "").strip(),
                }

        # Location — deduplicate by normalized address slug
        location_id = None
        if address and address.strip():
            location_id = f"loc_{normalize_address(address.strip())[:50]}"
            if location_id not in locations:
                locations[location_id] = {"id": location_id, "address": address.strip()}

        policy_links.append({
            "policy_id":    policy_id,
            "insured_id":   insured_id,
            "insured_type": insured_type,
            "owner_id":     owner_id,
            "agent_id":     agent_id,
            "location_id":  location_id,
        })

    for claim_id, ext in claim_data.items():
        contractor_name = ext.get("contractor")
        if contractor_name and contractor_name.strip():
            cid = f"contractor_{slugify(contractor_name.strip())}"
            if cid not in contractors:
                contractors[cid] = {"id": cid, "name": contractor_name.strip()}
            claim_links.append({
                "claim_id":  claim_id,
                "link_type": "repaired_by",
                "entity_id": cid,
            })

        # TEMP FIX: basic noise filter for third parties.
        # Problem: the LLM cannot reliably distinguish contractors (repair companies
        # working for the insured) from third parties (adverse claimants/subrogation
        # targets) because both appear in unstructured adjuster note text.
        # This causes contractors to be double-listed as third parties, and vendors/
        # bystanders mentioned in passing to be incorrectly extracted as third parties.
        # Proper fix: split adjuster notes by section header before extraction —
        # look for contractors only in SITE INSPECTION FINDINGS, and third parties
        # only in COVERAGE DETERMINATION / POLICY REVIEW sections. This gives the
        # LLM the right context window for each entity type.
        for tp in ext.get("third_parties", []):
            tp_name = (tp.get("name") or "").strip()
            tp_role = tp.get("role", "claimant")
            if not tp_name:
                continue
            if _TP_NOISE.match(tp_name):
                continue
            if len(tp_name.split()) < 2:
                continue
            tp_id = f"third_party_{slugify(tp_name)}"
            if f"contractor_{slugify(tp_name)}" in contractors:
                continue  # already captured as contractor — skip duplicate
            if tp_id not in third_parties:
                third_parties[tp_id] = {"id": tp_id, "name": tp_name}
            claim_links.append({
                "claim_id":  claim_id,
                "link_type": tp_role,
                "entity_id": tp_id,
            })

    return {
        "persons":       list(persons.values()),
        "organizations": list(organizations.values()),
        "agents":        list(agents.values()),
        "locations":     list(locations.values()),
        "contractors":   list(contractors.values()),
        "third_parties": list(third_parties.values()),
        "policy_links":  policy_links,
        "claim_links":   claim_links,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Initializing Gemini...")
    client = init_gemini()

    manifest = json.loads(MANIFEST_FILE.read_text())
    print(f"Loaded {len(manifest)} documents from manifest.\n")

    print("[1] Extracting entities from policy DECLARATIONS...")
    policy_data = extract_from_policies(client, manifest)

    print("\n[2] Extracting contractor + third-party from adjuster notes...")
    claim_data = extract_from_claims(client, manifest)

    print("\n[3] Building entities.json...")
    entities = build_entities(policy_data, claim_data)

    ENTITIES_FILE.write_text(json.dumps(entities, indent=2))

    print(f"\nWritten: {ENTITIES_FILE}")
    print(f"  Persons:       {len(entities['persons'])}")
    print(f"  Organizations: {len(entities['organizations'])}")
    print(f"  Agents:        {len(entities['agents'])}")
    print(f"  Locations:     {len(entities['locations'])}")
    print(f"  Contractors:   {len(entities['contractors'])}")
    print(f"  ThirdParties:  {len(entities['third_parties'])}")
    print(f"  Policy links:  {len(entities['policy_links'])}")
    print(f"  Claim links:   {len(entities['claim_links'])}")
    print(f"\nNext: python pipeline/load_neo4j.py")


if __name__ == "__main__":
    main()
