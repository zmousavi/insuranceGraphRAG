"""
generate_data.py
================
Generates synthetic insurance demo data using Gemini.

Outputs:
  demo_docs/policies/     — 7 policy documents
  demo_docs/endorsements/ — 1-2 endorsements per policy
  demo_docs/claims/       — ~32 claims across all policies
                            each claim: fnol.txt, adjuster.txt, outcome.txt
  evaluation/ground_truth.json — 25 benchmark questions with expected answers

Policies generated:
  policy_HO3_001  — Sarah Mitchell, personal homeowners HO-3
  policy_HO3_001_v2 — amended HO-3 (for SUPERSEDES test)
  policy_PAP_001  — David Chen, personal auto PAP
  policy_HO3_002  — James Carter, personal homeowners HO-3
  policy_CGL_001  — Golden Slice LLC (owned by Carter), commercial general liability
  policy_CGL_002  — Premier Occasions LLC (owned by Carter), commercial general liability
  policy_CGL_003  — Maple Street Holdings LLC (owned by Carter), commercial property

Entity model baked into DECLARATIONS:
  Persons:      Sarah Mitchell, David Chen, James Carter
  Orgs:         Golden Slice LLC, Premier Occasions LLC, Maple Street Holdings LLC
  Location:     800 Industrial Ave (shared by all 3 Carter entities — co-location signal)
  Agent:        Carlos Mendez manages ALL 4 Carter policies (fraud signal)
  Contractor:   FastFix Restoration appears in 3 claims within 33 days (fraud ring)
  ThirdParty:   Marcus Webb files against 2 Carter businesses (repeat claimant)
  Subrogation:  Rivera Delivery Co is liable party in a delivery damage claim

Design goals for synthetic data:
  1. Policies have clear section headers so structure-aware chunking splits correctly.
  2. Claims reference their policy by ID in the text.
  3. Cross-document questions require traversing claim → policy → endorsement.
  4. Temporal reasoning questions require adjacent claims in a sequence.
  5. Entity/fraud questions require graph traversal impossible for vector search.
  6. FastFix Restoration, Marcus Webb, and owner James Carter are never named
     together in any single document — connections only exist in the graph.

Usage:
  python data_gen/generate_data.py
  python data_gen/generate_data.py --force   # regenerate existing files
"""

import os
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

from google import genai

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT         = os.getenv("GOOGLE_CLOUD_PROJECT")
GEMINI_LOCATION = "global"
MODEL           = "gemini-2.5-flash"

ROOT             = Path(__file__).parent.parent
DEMO_DOCS        = ROOT / "demo_docs"
EVAL_DIR         = ROOT / "evaluation"

POLICIES_DIR     = DEMO_DOCS / "policies"
ENDORSEMENTS_DIR = DEMO_DOCS / "endorsements"
CLAIMS_DIR       = DEMO_DOCS / "claims"


# ---------------------------------------------------------------------------
# Policy definitions — drives all generation
# ---------------------------------------------------------------------------

POLICIES = [
    # ── Personal policies ─────────────────────────────────────────────────
    {
        "id": "policy_HO3_001",
        "insurer": "ACME_HOME",
        "policy_type": "HO-3",
        "policyholder": "Sarah Mitchell",
        "agent": "Jennifer Walsh, Midwest Home Insurance Agency",
        "effective_date": "2023-01-01",
        "expiry_date": "2024-01-01",
        "property": "42 Maple Street, Springfield, IL 62701",
        "dwelling_limit": 350000,
        "personal_property_limit": 140000,
        "liability_limit": 300000,
        "deductible": 2500,
        "key_exclusions": [
            "flood damage",
            "earthquake damage",
            "gradual deterioration or wear and tear",
            "pipes more than 20 years old",
            "intentional acts",
        ],
        "key_coverages": [
            "sudden and accidental discharge from plumbing",
            "windstorm and hail",
            "fire and smoke",
            "theft",
            "personal liability",
        ],
        "endorsements": [
            {
                "id": "endorsement_HO3_001_E01",
                "type": "Water Backup and Sump Pump Overflow",
                "limit": 10000,
                "description": "Adds coverage for water backup through sewers or drains and sump pump overflow.",
            },
            {
                "id": "endorsement_HO3_001_E02",
                "type": "Scheduled Personal Property",
                "items": "Diamond engagement ring valued at $8,500",
                "description": "Extends coverage to specifically listed high-value personal property beyond standard limits.",
            },
        ],
    },
    {
        "id": "policy_PAP_001",
        "insurer": "ACME_AUTO",
        "policy_type": "PAP",
        "policyholder": "David Chen",
        "agent": "Jennifer Walsh, Midwest Home Insurance Agency",
        "effective_date": "2023-03-15",
        "expiry_date": "2024-03-15",
        "vehicle": "2021 Toyota Camry, VIN 4T1BF1FK5MU123456",
        "bodily_injury_limit": "100/300",
        "property_damage_limit": 100000,
        "collision_deductible": 1000,
        "comprehensive_deductible": 500,
        "uninsured_motorist": "100/300",
        "key_exclusions": [
            "use of vehicle for commercial delivery or ride-share",
            "racing or speed contests",
            "intentional damage",
            "wear and tear",
            "mechanical breakdown",
        ],
        "key_coverages": [
            "collision with another vehicle",
            "collision with a fixed object",
            "comprehensive (theft, weather, vandalism)",
            "bodily injury liability",
            "property damage liability",
            "uninsured motorist",
            "medical payments up to $5,000 per person",
        ],
        "endorsements": [
            {
                "id": "endorsement_PAP_001_E01",
                "type": "Ride-Share Coverage Extension",
                "description": "Extends liability during Period 1 of ride-share activity (app on, no passenger). Does NOT extend to Periods 2 and 3.",
            },
            {
                "id": "endorsement_PAP_001_E02",
                "type": "New Car Replacement",
                "description": "If total loss occurs within 24 months of original purchase, pays replacement cost of a new vehicle of same make and model rather than actual cash value.",
            },
        ],
    },
    {
        "id": "policy_HO3_002",
        "insurer": "ACME_HOME",
        "policy_type": "HO-3",
        "policyholder": "James Carter",
        "agent": "Carlos Mendez, Pacific Coast Insurance Agency",
        "effective_date": "2023-04-01",
        "expiry_date": "2024-04-01",
        "property": "12 Birch Road, Springfield, IL 62704",
        "dwelling_limit": 425000,
        "personal_property_limit": 170000,
        "liability_limit": 300000,
        "deductible": 2500,
        "key_exclusions": [
            "flood damage",
            "earthquake damage",
            "business pursuits conducted from premises (commercial activities not covered)",
            "gradual deterioration or wear and tear",
            "intentional acts",
        ],
        "key_coverages": [
            "sudden and accidental discharge from plumbing",
            "windstorm and hail",
            "fire and smoke",
            "theft",
            "personal liability",
        ],
        "endorsements": [
            {
                "id": "endorsement_HO3_002_E01",
                "type": "Home Business Equipment Coverage",
                "limit": 15000,
                "description": "Extends coverage for business equipment stored at the insured residence. Does NOT extend liability coverage to business activities conducted at the premises.",
            },
        ],
    },
    # ── Commercial policies — all owned by James Carter ───────────────────
    {
        "id": "policy_CGL_001",
        "insurer": "ACME_COMM",
        "policy_type": "CGL",
        "policyholder": "Golden Slice LLC",
        "owner": "James Carter",
        "agent": "Carlos Mendez, Pacific Coast Insurance Agency",
        "effective_date": "2023-06-01",
        "expiry_date": "2024-06-01",
        "business": "Pizza restaurant with delivery operations",
        "business_address": "800 Industrial Avenue, Springfield, IL 62703",
        "occurrence_limit": 1000000,
        "aggregate_limit": 2000000,
        "personal_injury_limit": 1000000,
        "deductible": 5000,
        "key_exclusions": [
            "employee bodily injury (covered by workers compensation)",
            "damage to insured's own property",
            "professional liability (errors and omissions)",
            "pollution",
            "liquor liability",
            "expected or intended injury",
        ],
        "key_coverages": [
            "bodily injury to third parties on premises",
            "property damage to third party property",
            "personal and advertising injury",
            "products and completed operations",
            "medical payments up to $10,000 per person",
        ],
        "endorsements": [
            {
                "id": "endorsement_CGL_001_E01",
                "type": "Additional Insured — Landlord",
                "additional_insured": "Maple Street Holdings LLC",
                "description": "Adds the building owner (Maple Street Holdings LLC) as additional insured for claims arising from the named insured's operations at the leased premises at 800 Industrial Avenue.",
            },
            {
                "id": "endorsement_CGL_001_E02",
                "type": "Food Contamination Extension",
                "limit": 50000,
                "description": "Covers costs of product recall, disposal, and third-party illness claims arising from food contamination events at the business premises.",
            },
        ],
    },
    {
        "id": "policy_CGL_002",
        "insurer": "ACME_COMM",
        "policy_type": "CGL",
        "policyholder": "Premier Occasions LLC",
        "owner": "James Carter",
        "agent": "Carlos Mendez, Pacific Coast Insurance Agency",
        "effective_date": "2023-09-01",
        "expiry_date": "2024-09-01",
        "business": "Event catering and venue services",
        "business_address": "800 Industrial Avenue, Springfield, IL 62703",
        "occurrence_limit": 1000000,
        "aggregate_limit": 2000000,
        "personal_injury_limit": 1000000,
        "deductible": 5000,
        "key_exclusions": [
            "employee bodily injury (covered by workers compensation)",
            "damage to insured's own property",
            "professional liability (errors and omissions)",
            "pollution",
            "liquor liability (unless liquor liability endorsement purchased)",
            "expected or intended injury",
        ],
        "key_coverages": [
            "bodily injury to third parties at events and premises",
            "property damage to third party property",
            "personal and advertising injury",
            "products and completed operations (food served at events)",
            "medical payments up to $10,000 per person",
        ],
        "endorsements": [
            {
                "id": "endorsement_CGL_002_E01",
                "type": "Liquor Liability Extension",
                "limit": 500000,
                "description": "Extends liability to claims arising from the service of alcohol at catered events. Applies only to events where Premier Occasions LLC is the licensed server.",
            },
            {
                "id": "endorsement_CGL_002_E02",
                "type": "Additional Insured — Venue Owners",
                "description": "Adds event venue owners as additional insureds for claims arising from Premier Occasions LLC operations at their premises.",
            },
        ],
    },
    {
        "id": "policy_CGL_003",
        "insurer": "ACME_PROP",
        "policy_type": "Commercial Property",
        "policyholder": "Maple Street Holdings LLC",
        "owner": "James Carter",
        "agent": "Carlos Mendez, Pacific Coast Insurance Agency",
        "effective_date": "2023-06-01",
        "expiry_date": "2024-06-01",
        "business": "Commercial property holding company — owner of 800 Industrial Avenue, Springfield, IL 62703",
        "business_address": "800 Industrial Avenue, Springfield, IL 62703",
        "building_limit": 2500000,
        "business_income_limit": 500000,
        "deductible": 10000,
        "key_exclusions": [
            "flood and surface water",
            "earthquake",
            "gradual deterioration or wear and tear",
            "intentional acts by insured",
            "vacancy (building unoccupied more than 60 consecutive days)",
        ],
        "key_coverages": [
            "building structure and permanent fixtures",
            "business income loss during restoration period",
            "debris removal following a covered loss",
            "fire and smoke damage",
            "windstorm and hail",
            "vandalism and malicious mischief",
        ],
        "endorsements": [
            {
                "id": "endorsement_CGL_003_E01",
                "type": "Tenant Improvement Coverage",
                "limit": 200000,
                "description": "Covers tenant-installed improvements and betterments that become part of the building structure, up to $200,000.",
            },
        ],
    },
]

# Amended version of HO-3 policy (for SUPERSEDES testing)
AMENDED_POLICY = {
    "id": "policy_HO3_001_v2",
    "supersedes": "policy_HO3_001",
    "policyholder": "Sarah Mitchell",
    "agent": "Jennifer Walsh, Midwest Home Insurance Agency",
    "effective_date": "2024-01-01",
    "change_summary": "Pipe age exclusion raised from 20 years to 25 years. Deductible reduced from $2500 to $2000. Water backup endorsement limit increased from $10,000 to $15,000.",
    "key_change": "pipes more than 25 years old (was: 20 years)",
}


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def init_gemini() -> genai.Client:
    return genai.Client(vertexai=True, project=PROJECT, location=GEMINI_LOCATION)


def call_gemini(client: genai.Client, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e


# ---------------------------------------------------------------------------
# Policy generation
# ---------------------------------------------------------------------------

POLICY_PROMPT = """
Generate a realistic insurance policy document for the following parameters.
The document must be plain text (no markdown, no asterisks, no bullet symbols).
Use ALL CAPS for section headers exactly as listed.

Policy parameters:
{params}

Required section structure (use these exact headers in ALL CAPS):
DECLARATIONS
INSURING AGREEMENT
COVERAGE
EXCLUSIONS
CONDITIONS
DEFINITIONS

Requirements:
- DECLARATIONS: named insured, owner/principal (if business policy — include the owner's full
  name explicitly), agent name and agency, property or business address, policy number,
  effective and expiry dates, coverage limits, and deductible amounts.
- COVERAGE: describe each covered peril with specific conditions and dollar amounts.
- EXCLUSIONS: describe each exclusion with specific language. For HO-3 policies, include the
  pipe age threshold as: "pipes or plumbing systems more than XX years of age".
- CONDITIONS: claims reporting requirements (must report within 60 days), cooperation clause,
  subrogation rights (insurer may pursue recovery against liable third parties), cancellation terms.
- DEFINITIONS: define key terms relevant to this policy type.

Length: 3000-4000 words. Write in formal insurance policy language.
Do NOT use any markdown formatting. Use plain text only.
"""


def generate_policy(client: genai.Client, policy: dict,
                    output_path: Path, force: bool = False) -> None:
    if output_path.exists() and not force:
        print(f"  skip (exists): {output_path.name}")
        return
    prompt = POLICY_PROMPT.format(params=json.dumps(policy, indent=2))
    text = call_gemini(client, prompt)
    output_path.write_text(text)
    print(f"  written: {output_path.name}")


ENDORSEMENT_PROMPT = """
Generate a realistic insurance endorsement document for the following parameters.
Plain text only, no markdown.

Base policy ID: {policy_id}
Policy type: {policy_type}
Policyholder: {policyholder}
Endorsement parameters:
{endorsement}

Format:
ENDORSEMENT — [ENDORSEMENT TYPE]
Policy Number: [policy_id]
Effective Date: [same as base policy]

AMENDMENT TO POLICY:
[Describe exactly what coverage is added, modified, or excluded. Include specific
dollar limits, conditions, and any restrictions. 2-3 paragraphs.]

ALL OTHER TERMS AND CONDITIONS REMAIN UNCHANGED.

Length: 600-800 words.
"""


def generate_endorsement(client: genai.Client, policy: dict,
                          endorsement: dict, output_path: Path,
                          force: bool = False) -> None:
    if output_path.exists() and not force:
        print(f"  skip (exists): {output_path.name}")
        return
    prompt = ENDORSEMENT_PROMPT.format(
        policy_id=policy["id"],
        policy_type=policy["policy_type"],
        policyholder=policy["policyholder"],
        endorsement=json.dumps(endorsement, indent=2),
    )
    text = call_gemini(client, prompt)
    output_path.write_text(text)
    print(f"  written: {output_path.name}")


AMENDED_POLICY_PROMPT = """
Generate an amended insurance policy document. This is a replacement for the original
policy with specific changes noted below. Plain text only, no markdown.

Original policy: {original_id}
Amendment details: {amendment}
Use the same full structure as the original (DECLARATIONS, INSURING AGREEMENT, COVERAGE,
EXCLUSIONS, CONDITIONS, DEFINITIONS) but apply the stated changes. In the DECLARATIONS
section, note "AMENDED POLICY — replaces {original_id} effective {effective_date}".
Include agent name: {agent}.
In the EXCLUSIONS section, use the updated pipe age threshold: "pipes or plumbing systems
more than 25 years of age" (changed from 20 years in original policy).
Length: 3000-4000 words.
"""


def generate_amended_policy(client: genai.Client, output_path: Path,
                              force: bool = False) -> None:
    if output_path.exists() and not force:
        print(f"  skip (exists): {output_path.name}")
        return
    prompt = AMENDED_POLICY_PROMPT.format(
        original_id=AMENDED_POLICY["supersedes"],
        amendment=json.dumps(AMENDED_POLICY, indent=2),
        effective_date=AMENDED_POLICY["effective_date"],
        agent=AMENDED_POLICY["agent"],
    )
    text = call_gemini(client, prompt)
    output_path.write_text(text)
    print(f"  written: {output_path.name}")


# ---------------------------------------------------------------------------
# Claim scenarios — keyed by policy_id
# Each scenario is a dict; contractor/third_party/subrogation_target are
# written into the generated documents so retrieval can surface them.
# ---------------------------------------------------------------------------

CLAIM_SCENARIOS = {
    "policy_HO3_001": [
        # claim_HO_001: FastFix Restoration — fraud signal (same contractor as CGL_007 and PO_001)
        {
            "id": "claim_HO_001",
            "scenario": "burst pipe causing water damage to kitchen floor and cabinets; pipe installed 18 years ago",
            "outcome": "approved",
            "note": "pipe age under 20-year threshold — covered",
            "contractor": "FastFix Restoration",
            "date_of_loss": "February 15, 2024",
        },
        {
            "id": "claim_HO_002",
            "scenario": "water damage from burst pipe; pipe was installed 23 years ago per plumber's report",
            "outcome": "denied",
            "note": "pipe age over 20-year threshold in original policy",
            "date_of_loss": "March 10, 2023",
        },
        {
            "id": "claim_HO_003",
            "scenario": "roof damage from windstorm; shingles blown off, interior water damage",
            "outcome": "approved",
            "note": "windstorm covered",
        },
        {
            "id": "claim_HO_004",
            "scenario": "theft of jewelry from home; diamond ring taken",
            "outcome": "partially_approved",
            "note": "ring covered under scheduled property endorsement E02; other jewelry over standard limit",
        },
        {
            "id": "claim_HO_005",
            "scenario": "sewer backup flooded basement after heavy rain",
            "outcome": "approved_with_endorsement",
            "note": "requires water backup endorsement E01",
        },
        {
            "id": "claim_HO_006",
            "scenario": "gradual water stain on ceiling from slow roof leak over several months",
            "outcome": "denied",
            "note": "gradual deterioration exclusion",
        },
        {
            "id": "claim_HO_007",
            "scenario": "fire damage to kitchen following stove malfunction",
            "outcome": "approved",
            "note": "fire covered",
        },
        {
            "id": "claim_HO_008",
            "scenario": "slip and fall injury to visitor on icy front steps",
            "outcome": "approved",
            "note": "personal liability covered",
        },
        {
            "id": "claim_HO_009",
            "scenario": "mold damage discovered during renovation",
            "outcome": "denied",
            "note": "mold from gradual moisture not covered",
        },
        {
            "id": "claim_HO_010",
            "scenario": "water damage from burst pipe; pipe installed 22 years ago; filed after policy amendment v2",
            "outcome": "approved",
            "note": "pipe age now under 25-year threshold in amended policy v2",
            "date_of_loss": "February 5, 2024",
        },
    ],

    "policy_PAP_001": [
        {
            "id": "claim_PAP_001",
            "scenario": "rear-end collision at stop light; vehicle struck from behind by uninsured driver",
            "outcome": "approved",
            "note": "uninsured motorist coverage",
        },
        {
            "id": "claim_PAP_002",
            "scenario": "collision while delivering food for a ride-share app with passenger in vehicle",
            "outcome": "denied",
            "note": "ride-share Period 2/3 not covered by base policy or endorsement",
        },
        {
            "id": "claim_PAP_003",
            "scenario": "windshield cracked by road debris on highway",
            "outcome": "approved",
            "note": "comprehensive coverage",
        },
        {
            "id": "claim_PAP_004",
            "scenario": "vehicle stolen from apartment parking lot overnight",
            "outcome": "approved",
            "note": "comprehensive theft coverage",
        },
        {
            "id": "claim_PAP_005",
            "scenario": "collision with deer on rural road causing front-end damage",
            "outcome": "approved",
            "note": "comprehensive animal collision",
        },
        {
            "id": "claim_PAP_006",
            "scenario": "vehicle total loss in intersection collision; vehicle purchased 18 months ago",
            "outcome": "approved_replacement_cost",
            "note": "new car replacement endorsement E02 applies within 24-month window",
        },
        {
            "id": "claim_PAP_007",
            "scenario": "engine failure due to lack of oil maintenance",
            "outcome": "denied",
            "note": "mechanical breakdown exclusion",
        },
        {
            "id": "claim_PAP_008",
            "scenario": "minor collision while using app as delivery driver with no passenger",
            "outcome": "partially_approved",
            "note": "ride-share Period 1 covered by endorsement E01",
        },
        {
            "id": "claim_PAP_009",
            "scenario": "hail damage to hood and roof of vehicle",
            "outcome": "approved",
            "note": "comprehensive weather coverage",
        },
        {
            "id": "claim_PAP_010",
            "scenario": "bodily injury claim from third party after policyholder at-fault collision",
            "outcome": "approved",
            "note": "bodily injury liability",
        },
    ],

    "policy_HO3_002": [
        {
            "id": "claim_HC_001",
            "scenario": "theft of home office laptop and equipment from insured residence",
            "outcome": "partially_approved",
            "note": "personal property limit applies; home business endorsement E01 covers up to $15,000 for business equipment",
        },
        {
            "id": "claim_HC_002",
            "scenario": "windstorm damage to roof; several sections of shingles blown off",
            "outcome": "approved",
            "note": "windstorm covered under standard HO-3",
        },
        {
            "id": "claim_HC_003",
            "scenario": "business client visiting home office slipped on wet entryway floor and fractured wrist",
            "outcome": "denied",
            "note": "business pursuits exclusion — liability arising from commercial activities at residence is excluded",
        },
    ],

    "policy_CGL_001": [
        # claim_CGL_001: Marcus Webb — repeat claimant (also appears in claim_PO_002)
        {
            "id": "claim_CGL_001",
            "scenario": "customer slipped on wet floor near display case; fractured wrist",
            "outcome": "approved",
            "note": "bodily injury on premises",
            "third_party": "Marcus Webb",
        },
        # claim_CGL_002: Rivera Delivery Co — subrogation target
        {
            "id": "claim_CGL_002",
            "scenario": "delivery driver from an independent delivery company damaged a customer's fence post while making a delivery on behalf of the business",
            "outcome": "approved",
            "note": "property damage liability; subrogation right against delivery company",
            "subrogation_target": "Rivera Delivery Co",
        },
        {
            "id": "claim_CGL_003",
            "scenario": "employee injured back lifting heavy supply boxes in the storage room",
            "outcome": "denied",
            "note": "employee bodily injury excluded; covered by workers compensation",
        },
        {
            "id": "claim_CGL_004",
            "scenario": "three customers reported food poisoning after eating a pizza order",
            "outcome": "approved_with_endorsement",
            "note": "food contamination extension E02 covers recall costs and illness claims",
        },
        {
            "id": "claim_CGL_005",
            "scenario": "customer alleged defamatory statement in social media post by the business account",
            "outcome": "approved",
            "note": "personal and advertising injury coverage",
        },
        # claim_CGL_006: additional insured endorsement — Maple Street Holdings LLC
        {
            "id": "claim_CGL_006",
            "scenario": "building owner (Maple Street Holdings LLC) sued after customer tripped on uneven flooring at the restaurant entrance",
            "outcome": "approved",
            "note": "additional insured endorsement E01 covers building owner Maple Street Holdings LLC",
        },
        # claim_CGL_007: FastFix Restoration — fraud signal (same contractor as claim_HO_001 and claim_PO_001)
        {
            "id": "claim_CGL_007",
            "scenario": "oven fire caused smoke and soot damage to the kitchen and neighboring tenant's storage space",
            "outcome": "approved",
            "note": "property damage to third party; smoke restoration required",
            "contractor": "FastFix Restoration",
            "date_of_loss": "March 1, 2024",
        },
        {
            "id": "claim_CGL_008",
            "scenario": "customer claimed allergic reaction to undisclosed ingredient in a pizza",
            "outcome": "approved",
            "note": "products and completed operations",
        },
        {
            "id": "claim_CGL_009",
            "scenario": "business owner's personal vehicle damaged in the restaurant parking lot",
            "outcome": "denied",
            "note": "damage to insured's own property excluded",
        },
        {
            "id": "claim_CGL_010",
            "scenario": "customer claimed financial loss after relying on business owner's advice about a catering contract",
            "outcome": "denied",
            "note": "professional liability (errors and omissions) excluded from CGL",
        },
    ],

    "policy_CGL_002": [
        # claim_PO_001: FastFix Restoration — fraud signal (same contractor as claim_HO_001 and claim_CGL_007)
        {
            "id": "claim_PO_001",
            "scenario": "kitchen fire during event preparation caused smoke and water damage to the catering kitchen and adjacent storage area",
            "outcome": "approved",
            "note": "property damage covered; smoke restoration required",
            "contractor": "FastFix Restoration",
            "date_of_loss": "March 20, 2024",
        },
        # claim_PO_002: Marcus Webb — repeat claimant (also appears in claim_CGL_001)
        {
            "id": "claim_PO_002",
            "scenario": "event guest alleged he tripped over an unsecured cable at a catered corporate event and injured his ankle",
            "outcome": "approved",
            "note": "bodily injury at event covered",
            "third_party": "Marcus Webb",
            "date_of_loss": "April 8, 2024",
        },
        {
            "id": "claim_PO_003",
            "scenario": "guest at a wedding reception alleged intoxication and subsequent injury; Premier Occasions was the licensed alcohol server at the event",
            "outcome": "approved_with_endorsement",
            "note": "liquor liability extension E01 covers alcohol-related injury claims",
        },
        {
            "id": "claim_PO_004",
            "scenario": "client demanded reimbursement for event deposits after Premier Occasions cancelled due to staff illness",
            "outcome": "denied",
            "note": "event cancellation and contractual liability not covered under CGL",
        },
        {
            "id": "claim_PO_005",
            "scenario": "event venue owner's antique mirror damaged during event setup by catering staff",
            "outcome": "approved",
            "note": "property damage to third party covered; venue owner qualifies as additional insured under endorsement E02",
        },
        {
            "id": "claim_PO_006",
            "scenario": "multiple attendees at a corporate luncheon reported food poisoning symptoms attributed to catered food",
            "outcome": "approved",
            "note": "products and completed operations coverage applies",
        },
    ],

    "policy_CGL_003": [
        {
            "id": "claim_MS_001",
            "scenario": "severe windstorm caused structural damage to the roof of the commercial building at 800 Industrial Avenue; two tenant spaces affected",
            "outcome": "approved",
            "note": "windstorm damage to building structure covered",
        },
        {
            "id": "claim_MS_002",
            "scenario": "burst water main in the building's utility room caused flooding to lower-floor tenant spaces",
            "outcome": "approved",
            "note": "sudden and accidental water damage to building structure covered; business income loss during repair period also covered",
        },
        {
            "id": "claim_MS_003",
            "scenario": "fire originating in one tenant's space spread and caused smoke damage to the common corridor and adjacent unit",
            "outcome": "approved",
            "note": "fire and smoke damage to building covered; tenant improvement endorsement E01 applies to affected improvements",
        },
    ],
}


# ---------------------------------------------------------------------------
# Claim generation prompts
# ---------------------------------------------------------------------------

FNOL_PROMPT = """
Write a First Notice of Loss (FNOL) claim report for an insurance claim.
Plain text only, no markdown. Use formal insurance reporting language.

Policy ID: {policy_id}
Policy Type: {policy_type}
Policyholder: {policyholder}
Claim ID: {claim_id}
Incident scenario: {scenario}
{contractor_line}
{third_party_line}

Format:
FIRST NOTICE OF LOSS

Claim Number: {claim_id}
Policy Number: {policy_id}
Date of Loss: {date_of_loss}
Date Reported: [1-3 days after date of loss]
Reported By: {policyholder}

DESCRIPTION OF LOSS:
[3-4 sentences describing what happened in the policyholder's own words.
Do NOT use technical insurance terms.
Do NOT mention the policy exclusion or coverage outcome.
If a third party is involved, name them naturally in the description.
If a contractor was contacted for repairs, name them in the description.]

ESTIMATED DAMAGES:
[Plausible dollar estimate]

SUPPORTING DOCUMENTS ATTACHED:
[2-3 plausible document types]

Length: 600-800 words.
"""

ADJUSTER_PROMPT = """
Write an insurance adjuster's investigation notes for a claim.
Plain text only, no markdown. Use formal adjuster language.

Policy ID: {policy_id}
Policy Type: {policy_type}
Claim ID: {claim_id}
Incident scenario: {scenario}
Expected outcome: {outcome}
{contractor_line}
{third_party_line}
{subrogation_line}

Format:
ADJUSTER INVESTIGATION NOTES

Claim Number: {claim_id}
Adjuster: [plausible adjuster name and ID]
Inspection Date: [plausible date, 5-10 days after FNOL]

SITE INSPECTION FINDINGS:
[2-3 sentences with specific findings. If a contractor performed or quoted repairs,
name them explicitly (e.g. "Repairs quoted by FastFix Restoration, $12,400").
If a third party is involved, name them and describe their role.]

POLICY REVIEW:
[Reference the specific policy section and language that applies.
If an exclusion applies, quote the relevant exclusion language precisely.
If an endorsement applies, name it and describe how it changes coverage.]

COVERAGE DETERMINATION RECOMMENDATION:
[State the coverage decision and the exact policy basis.
If a subrogation right exists against a third party, identify the liable party
and note the insurer's right of recovery under the CONDITIONS section.]

Length: 600-800 words.
"""

OUTCOME_PROMPT = """
Write a formal insurance claim outcome letter to the policyholder.
Plain text only, no markdown.

Policy ID: {policy_id}
Claim ID: {claim_id}
Policyholder: {policyholder}
Outcome: {outcome}
Incident scenario: {scenario}
{subrogation_line}

Format:
CLAIM DETERMINATION LETTER

Date: [plausible date, 15-20 days after FNOL]
Claim Number: {claim_id}
Policy Number: {policy_id}
Insured: {policyholder}

Dear {policyholder},

[Opening: state coverage decision clearly.]

[Second paragraph: cite the specific policy section, exclusion language, or endorsement
that supports the decision. Include exact dollar amounts if applicable.]

[Third paragraph: if denied, explain using policy language. If approved, state payment
amount and next steps. If subrogation rights apply, note that the insurer may pursue
recovery against the liable third party.]

[Closing: appeal rights and contact information.]

Sincerely,
[Plausible claims adjuster name]
Claims Department
[Insurer name]

Length: 600-800 words.
"""


def generate_claim(client: genai.Client, policy: dict, scenario: dict,
                   claims_base: Path, force: bool = False) -> None:
    claim_id    = scenario["id"]
    sc_text     = scenario["scenario"]
    outcome     = scenario["outcome"]
    contractor  = scenario.get("contractor")
    third_party = scenario.get("third_party")
    sub_target  = scenario.get("subrogation_target")
    date_loss   = scenario.get("date_of_loss", "[a plausible date in 2023 or 2024]")

    claim_dir = claims_base / claim_id
    claim_dir.mkdir(exist_ok=True)

    policyholder   = policy.get("policyholder", "")
    contractor_line   = f"Contractor performing or quoting repairs: {contractor}" if contractor else ""
    third_party_line  = f"Third party involved: {third_party}" if third_party else ""
    subrogation_line  = f"Potential subrogation target (liable third party): {sub_target}" if sub_target else ""

    for doc_type, prompt_template in [
        ("fnol",     FNOL_PROMPT),
        ("adjuster", ADJUSTER_PROMPT),
        ("outcome",  OUTCOME_PROMPT),
    ]:
        out_path = claim_dir / f"{doc_type}.txt"
        if out_path.exists() and not force:
            print(f"    skip (exists): {claim_id}/{doc_type}.txt")
            continue
        prompt = prompt_template.format(
            policy_id        = policy["id"],
            policy_type      = policy["policy_type"],
            policyholder     = policyholder,
            claim_id         = claim_id,
            scenario         = sc_text,
            outcome          = outcome,
            contractor_line  = contractor_line,
            third_party_line = third_party_line,
            subrogation_line = subrogation_line,
            date_of_loss     = date_loss,
        )
        text = call_gemini(client, prompt)
        out_path.write_text(text)
        print(f"    written: {claim_id}/{doc_type}.txt")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Benchmark questions
# ---------------------------------------------------------------------------

GROUND_TRUTH_QUESTIONS = [
    # ── Policy-only (5) ──────────────────────────────────────────────────────
    {
        "id": "Q01",
        "question": "Under a homeowners policy, what is the maximum dollar amount covered for personal property stolen or damaged while located away from the insured premises?",
        "expected_answer": "10% of the dwelling coverage limit ($35,000 on a $350,000 dwelling limit policy)",
        "required_sources": ["policy_HO3_001_coverage"],
        "traversal_type": "traverse_up",
        "difficulty": "medium",
        "note": "Answer is a percentage of dwelling limit; question asks for dollar amount — vocabulary gap",
    },
    {
        "id": "Q02",
        "question": "What specific age threshold for plumbing systems determines whether a water damage claim from a burst pipe is excluded under the homeowners policy?",
        "expected_answer": "20 years (pipes more than 20 years of age are excluded under the original policy)",
        "required_sources": ["policy_HO3_001_exclusions"],
        "traversal_type": "traverse_up",
        "difficulty": "easy",
    },
    {
        "id": "Q03",
        "question": "Under the personal auto policy, what is the maximum medical payment benefit available per person injured in a covered accident, regardless of fault?",
        "expected_answer": "$5,000 per person",
        "required_sources": ["policy_PAP_001_coverage"],
        "traversal_type": "traverse_up",
        "difficulty": "easy",
    },
    {
        "id": "Q04",
        "question": "Under the Golden Slice LLC commercial general liability policy, what is the per-occurrence limit and what is the general aggregate limit?",
        "expected_answer": "$1,000,000 per occurrence; $2,000,000 aggregate",
        "required_sources": ["policy_CGL_001_declarations"],
        "traversal_type": "traverse_up",
        "difficulty": "easy",
    },
    {
        "id": "Q05",
        "question": "Under the homeowners policy, which specific type of water event is explicitly excluded from coverage unless a separate endorsement is purchased?",
        "expected_answer": "Water backup through sewers or drains and sump pump overflow",
        "required_sources": ["policy_HO3_001_exclusions", "endorsement_HO3_001_E01"],
        "traversal_type": "traverse_similar",
        "difficulty": "medium",
        "note": "Requires finding both exclusion and endorsement",
    },
    # ── Cross-policy-claim via REFERENCES_POLICY (5) ─────────────────────────
    {
        "id": "Q06",
        "question": "A homeowner filed a water damage claim after a pipe burst. The adjuster denied the claim citing the age of the plumbing. What was the specific age of the pipe and which policy exclusion was cited?",
        "expected_answer": "The pipe was 23 years old. The policy excludes pipes more than 20 years of age.",
        "required_sources": ["claim_HO_002_adjuster", "policy_HO3_001_exclusions"],
        "traversal_type": "traverse_policy",
        "difficulty": "hard",
        "note": "Requires crossing from claim adjuster notes to policy exclusions section",
    },
    {
        "id": "Q07",
        "question": "A restaurant received a claim from multiple customers who became ill after eating its food. Was this covered, and if so under what specific provision?",
        "expected_answer": "Yes, covered under the Food Contamination Extension endorsement (endorsement_CGL_001_E02) up to $50,000.",
        "required_sources": ["claim_CGL_004_adjuster", "endorsement_CGL_001_E02"],
        "traversal_type": "traverse_policy",
        "difficulty": "hard",
    },
    {
        "id": "Q08",
        "question": "A ride-share driver filed a claim for a collision that occurred while they had a passenger in the vehicle. Was this loss covered?",
        "expected_answer": "No. The ride-share endorsement (E01) only covers Period 1 (app on, no passenger). Coverage during Period 2 and 3 is explicitly excluded.",
        "required_sources": ["claim_PAP_002_adjuster", "endorsement_PAP_001_E01"],
        "traversal_type": "traverse_policy",
        "difficulty": "hard",
    },
    {
        "id": "Q09",
        "question": "An injured visitor filed a claim against a commercial restaurant after slipping on the premises. Was the building's owner also protected under the restaurant's insurance policy?",
        "expected_answer": "Yes. The Additional Insured endorsement (E01) covers the building owner (Maple Street Holdings LLC) for claims arising from the named insured's operations at the leased premises.",
        "required_sources": ["claim_CGL_006_adjuster", "endorsement_CGL_001_E01"],
        "traversal_type": "traverse_policy",
        "difficulty": "hard",
    },
    {
        "id": "Q10",
        "question": "A vehicle owner filed a total loss claim eighteen months after purchasing a new vehicle. What was the basis for the settlement and why was it different from the standard calculation?",
        "expected_answer": "Settled at new vehicle replacement cost rather than actual cash value because the New Car Replacement endorsement applies within 24 months of original purchase.",
        "required_sources": ["claim_PAP_006_adjuster", "endorsement_PAP_001_E02"],
        "traversal_type": "traverse_policy",
        "difficulty": "hard",
    },
    # ── Temporal chain via NEXT_PERIOD (3) ───────────────────────────────────
    {
        "id": "Q11",
        "question": "A homeowner had a water damage claim denied in 2023. They filed a second claim for a similar pipe-related loss in 2024. What changed between the two claims that affected the coverage outcome?",
        "expected_answer": "The policy was amended in 2024, raising the pipe age exclusion from 20 to 25 years. The second claim involved a 22-year-old pipe, excluded under the original but covered under the amended policy.",
        "required_sources": ["claim_HO_002_outcome", "claim_HO_010_adjuster", "policy_HO3_001_v2_exclusions"],
        "traversal_type": "traverse_temporal",
        "difficulty": "hard",
    },
    {
        "id": "Q12",
        "question": "A policyholder had their auto theft claim approved. Shortly after, they filed a hail damage claim. What deductibles applied to each claim?",
        "expected_answer": "Theft: $500 comprehensive deductible. Hail: $500 comprehensive deductible. Both fall under comprehensive coverage.",
        "required_sources": ["claim_PAP_004_outcome", "claim_PAP_009_adjuster"],
        "traversal_type": "traverse_temporal",
        "difficulty": "medium",
    },
    {
        "id": "Q13",
        "question": "A commercial restaurant filed a bodily injury claim followed by a property damage claim for smoke damage to a neighboring tenant. What was the combined exposure against the policy aggregate?",
        "expected_answer": "Bodily injury (claim_CGL_001): within $1M per-occurrence limit. Property damage (claim_CGL_007): within $1M per-occurrence limit. Combined $2M equals the aggregate limit.",
        "required_sources": ["claim_CGL_001_outcome", "claim_CGL_007_outcome"],
        "traversal_type": "traverse_temporal",
        "difficulty": "hard",
    },
    # ── Document versioning via SUPERSEDES (2) ───────────────────────────────
    {
        "id": "Q14",
        "question": "Under the current version of the homeowners policy effective January 2024, what is the pipe age threshold that triggers the water damage exclusion?",
        "expected_answer": "25 years. The amended policy raised the threshold from 20 to 25 years.",
        "required_sources": ["policy_HO3_001_v2_exclusions"],
        "traversal_type": "supersedes",
        "difficulty": "medium",
    },
    {
        "id": "Q15",
        "question": "A claim was denied in early 2023 because the damaged pipe was 22 years old. If the same loss had occurred under the policy terms in effect from January 2024, would the outcome have been different?",
        "expected_answer": "Yes. The original policy excluded pipes over 20 years so the 22-year-old pipe was denied. The amended policy raised the threshold to 25 years so the same pipe would be covered.",
        "required_sources": ["policy_HO3_001_exclusions", "policy_HO3_001_v2_exclusions"],
        "traversal_type": "supersedes",
        "difficulty": "hard",
    },
    # ── Entity / fraud detection (5) ─────────────────────────────────────────
    {
        "id": "Q16",
        "question": "What is the total commercial general liability aggregate coverage limit across all businesses owned by the principal of Golden Slice LLC?",
        "expected_answer": "$4,000,000 total: $2,000,000 aggregate for Golden Slice LLC (policy_CGL_001) plus $2,000,000 aggregate for Premier Occasions LLC (policy_CGL_002), both owned by James Carter.",
        "required_sources": ["policy_CGL_001_declarations", "policy_CGL_002_declarations"],
        "traversal_type": "traverse_entity",
        "difficulty": "hard",
        "note": "Requires discovering hidden shared ownership then summing across policies. No document states this.",
    },
    {
        "id": "Q17",
        "question": "Which other insured businesses operate at the same address as Golden Slice LLC, and what policies cover them?",
        "expected_answer": "Premier Occasions LLC and Maple Street Holdings LLC also operate at 800 Industrial Avenue. Premier Occasions is covered under policy_CGL_002. Maple Street Holdings is covered under policy_CGL_003.",
        "required_sources": ["policy_CGL_001_declarations", "policy_CGL_002_declarations", "policy_CGL_003_declarations"],
        "traversal_type": "traverse_entity",
        "difficulty": "hard",
        "note": "Requires Location node shared by 3 organizations. No single document names all three.",
    },
    {
        "id": "Q18",
        "question": "Has FastFix Restoration appeared as the repair contractor in claims involving more than one policyholder in our portfolio?",
        "expected_answer": "Yes. FastFix Restoration appears in claims from three separate policyholders: Sarah Mitchell (claim_HO_001, February 2024), Golden Slice LLC (claim_CGL_007, March 2024), and Premier Occasions LLC (claim_PO_001, March 2024) — all within 33 days.",
        "required_sources": ["claim_HO_001_adjuster", "claim_CGL_007_adjuster", "claim_PO_001_adjuster"],
        "traversal_type": "traverse_entity",
        "difficulty": "hard",
        "note": "No single document connects these three claims. Connection exists only through Contractor node.",
    },
    {
        "id": "Q19",
        "question": "Has the same individual filed bodily injury claims against more than one business in our insured portfolio?",
        "expected_answer": "Yes. Marcus Webb filed a slip-and-fall claim against Golden Slice LLC (claim_CGL_001, April 2024) and a separate injury claim against Premier Occasions LLC (claim_PO_002, April 2024) within the same month.",
        "required_sources": ["claim_CGL_001_adjuster", "claim_PO_002_adjuster"],
        "traversal_type": "traverse_entity",
        "difficulty": "hard",
        "note": "Requires ThirdParty node traversal. No document states Marcus Webb filed against two policyholders.",
    },
    {
        "id": "Q20",
        "question": "Does the same insurance agent manage both personal and commercial policies for any individual in our system?",
        "expected_answer": "Yes. Carlos Mendez of Pacific Coast Insurance manages James Carter's personal homeowners policy (policy_HO3_002) and all three of his commercial policies: Golden Slice LLC, Premier Occasions LLC, and Maple Street Holdings LLC.",
        "required_sources": ["policy_HO3_002_declarations", "policy_CGL_001_declarations", "policy_CGL_002_declarations", "policy_CGL_003_declarations"],
        "traversal_type": "traverse_entity",
        "difficulty": "hard",
        "note": "Requires Agent → Policies → Person traversal. James Carter not named as the linking factor in any single document.",
    },
    # ── Subrogation + third-party (3) ─────────────────────────────────────────
    {
        "id": "Q21",
        "question": "In the Golden Slice LLC delivery damage claim, does the insurer have a right of subrogation, and against whom?",
        "expected_answer": "Yes. The loss was caused by a driver employed by Rivera Delivery Co. The insurer paid the property damage claim and has subrogation rights to recover costs from Rivera Delivery Co under the policy's conditions section.",
        "required_sources": ["claim_CGL_002_adjuster", "policy_CGL_001_conditions"],
        "traversal_type": "traverse_entity",
        "difficulty": "hard",
        "note": "Requires crossing from claim adjuster notes to policy conditions subrogation clause.",
    },
    {
        "id": "Q22",
        "question": "Could both James Carter's personal homeowners policy and his commercial general liability policy potentially respond to the same incident at his home office?",
        "expected_answer": "Potentially. The HO-3 covers personal liability at the insured premises but excludes business pursuits. The CGL covers business operations but only at scheduled locations. A home office incident may fall in a gap — the HO-3 excludes it as a business activity and the CGL may not cover a non-scheduled location. Both policies must be reviewed for coordination.",
        "required_sources": ["policy_HO3_002_exclusions", "policy_CGL_001_exclusions"],
        "traversal_type": "traverse_entity",
        "difficulty": "hard",
        "note": "Requires reasoning across personal and business policies for the same individual.",
    },
    {
        "id": "Q23",
        "question": "What is the full insurance coverage in place for all entities operating at 800 Industrial Avenue, Springfield?",
        "expected_answer": "Three entities operate at 800 Industrial Avenue: Golden Slice LLC (CGL, $1M/$2M), Premier Occasions LLC (CGL, $1M/$2M), and Maple Street Holdings LLC (commercial property, $2.5M building). All three are owned by James Carter.",
        "required_sources": ["policy_CGL_001_declarations", "policy_CGL_002_declarations", "policy_CGL_003_declarations"],
        "traversal_type": "traverse_entity",
        "difficulty": "hard",
        "note": "Requires Location → Organizations → Policies traversal. No single document lists all three.",
    },
    # ── Temporal — new entities (2) ───────────────────────────────────────────
    {
        "id": "Q24",
        "question": "Premier Occasions LLC filed a kitchen fire claim in March 2024 and a guest injury claim in April 2024. What is the total amount paid out and how much aggregate limit remains?",
        "expected_answer": "The fire claim (claim_PO_001) and injury claim (claim_PO_002) are both approved within the $1M per-occurrence limit. Combined exposure reduces the $2M aggregate. Remaining aggregate depends on exact payout amounts stated in the outcome letters.",
        "required_sources": ["claim_PO_001_outcome", "claim_PO_002_outcome", "policy_CGL_002_declarations"],
        "traversal_type": "traverse_temporal",
        "difficulty": "hard",
        "note": "Requires NEXT_PERIOD traversal across PO claims plus REFERENCES_POLICY to get aggregate limit.",
    },
    {
        "id": "Q25",
        "question": "Three claims were all repaired by the same contractor within a 33-day window. What are the dates of these claims, which policyholders are involved, and why is this pattern significant?",
        "expected_answer": "FastFix Restoration appears in: Sarah Mitchell claim_HO_001 (February 15, 2024), Golden Slice LLC claim_CGL_007 (March 1, 2024), and Premier Occasions LLC claim_PO_001 (March 20, 2024). Three separate policyholders using the same contractor within 33 days is a potential fraud ring indicator.",
        "required_sources": ["claim_HO_001_adjuster", "claim_CGL_007_adjuster", "claim_PO_001_adjuster"],
        "traversal_type": "traverse_entity",
        "difficulty": "hard",
        "note": "Requires Contractor node + date filtering across three unrelated policyholders. Impossible for naive RAG.",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic insurance demo data.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate files that already exist. Without this flag, existing files are skipped.",
    )
    args = parser.parse_args()

    print("Initializing Gemini...")
    client = init_gemini()

    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    ENDORSEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Policies ──────────────────────────────────────────────────────────────
    print("\n[1] Generating policies...")
    for policy in POLICIES:
        out = POLICIES_DIR / f"{policy['id']}.txt"
        print(f"  Policy: {policy['id']}")
        generate_policy(client, policy, out, force=args.force)
        time.sleep(1)

    print("\n[2] Generating amended policy (v2 for SUPERSEDES test)...")
    generate_amended_policy(
        client,
        POLICIES_DIR / "policy_HO3_001_v2.txt",
        force=args.force,
    )

    # ── Endorsements ──────────────────────────────────────────────────────────
    print("\n[3] Generating endorsements...")
    for policy in POLICIES:
        for end in policy.get("endorsements", []):
            out = ENDORSEMENTS_DIR / f"{end['id']}.txt"
            print(f"  Endorsement: {end['id']}")
            generate_endorsement(client, policy, end, out, force=args.force)
            time.sleep(1)

    # ── Claims ────────────────────────────────────────────────────────────────
    print("\n[4] Generating claims...")
    policy_map = {p["id"]: p for p in POLICIES}
    for policy_id, scenarios in CLAIM_SCENARIOS.items():
        policy = policy_map[policy_id]
        print(f"\n  {policy_id} ({policy['policyholder']}):")
        for scenario in scenarios:
            print(f"  Claim: {scenario['id']}")
            generate_claim(client, policy, scenario, CLAIMS_DIR, force=args.force)
            time.sleep(1)

    # ── Ground truth ──────────────────────────────────────────────────────────
    print("\n[5] Writing ground_truth.json...")
    gt_path = EVAL_DIR / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(GROUND_TRUTH_QUESTIONS, f, indent=2)
    print(f"  Written: {gt_path} ({len(GROUND_TRUTH_QUESTIONS)} questions)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Data generation complete.")
    policy_files     = list(POLICIES_DIR.glob("*.txt"))
    endorsement_files = list(ENDORSEMENTS_DIR.glob("*.txt"))
    claim_dirs       = [d for d in CLAIMS_DIR.iterdir() if d.is_dir()]
    total_claims     = sum(len(s) for s in CLAIM_SCENARIOS.values())
    print(f"  Policies:     {len(policy_files)} files in demo_docs/policies/")
    print(f"  Endorsements: {len(endorsement_files)} files in demo_docs/endorsements/")
    print(f"  Claims:       {len(claim_dirs)} claim folders ({total_claims} total scenarios)")
    print(f"  Benchmark:    {len(GROUND_TRUTH_QUESTIONS)} questions in evaluation/ground_truth.json")
    print("\nFiles that must be regenerated if previously generated with old schema:")
    print("  demo_docs/policies/policy_CGL_001.txt  (renamed: Riverside Bakery → Golden Slice LLC)")
    print("  demo_docs/claims/claim_HO_001/          (added contractor: FastFix Restoration)")
    print("  demo_docs/claims/claim_CGL_001/         (added third_party: Marcus Webb)")
    print("  demo_docs/claims/claim_CGL_002/         (added subrogation: Rivera Delivery Co)")
    print("  demo_docs/claims/claim_CGL_006/         (updated landlord: Maple Street Holdings LLC)")
    print("  demo_docs/claims/claim_CGL_007/         (added contractor: FastFix Restoration)")
    print("Use --force to regenerate these files automatically.")
    print("\nNext: python pipeline/ingest.py --strategy structure_aware")


if __name__ == "__main__":
    main()
