"""
generate_data.py
================
Generates synthetic insurance demo data using Gemini.

Outputs:
  demo_docs/policies/     — 3 policy documents (HO-3, PAP, CGL)
  demo_docs/endorsements/ — 2 endorsements per policy
  demo_docs/claims/       — 30 claims (10 per policy type)
                            each claim: fnol.txt, adjuster.txt, outcome.txt
  evaluation/ground_truth.json — 15 benchmark questions with expected answers

Design goals for synthetic data:
  1. Policies have clear section headers (DECLARATIONS, COVERAGE, EXCLUSIONS,
     CONDITIONS, DEFINITIONS) so structure-aware chunking splits correctly.
  2. Claims reference their policy by ID in the text.
  3. Some benchmark questions require crossing claim ↔ policy documents
     (vocabulary gap: question uses claim language, answer lives in policy).
  4. Some questions require temporal reasoning (adjacent claims in a sequence).
  5. Two policies are versioned: original + amended, with SUPERSEDES relationship.

Usage:
  python data_gen/generate_data.py

Requirements:
  pip install google-cloud-aiplatform vertexai python-dotenv
  gcloud auth application-default login
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

from google import genai

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GEMINI_LOCATION = "global"
MODEL    = "gemini-2.5-flash"

ROOT       = Path(__file__).parent.parent
DEMO_DOCS  = ROOT / "demo_docs"
EVAL_DIR   = ROOT / "evaluation"

POLICIES_DIR     = DEMO_DOCS / "policies"
ENDORSEMENTS_DIR = DEMO_DOCS / "endorsements"
CLAIMS_DIR       = DEMO_DOCS / "claims"


# ---------------------------------------------------------------------------
# Policy definitions — drives all generation
# ---------------------------------------------------------------------------

POLICIES = [
    {
        "id": "policy_HO3_001",
        "insurer": "ACME_HOME",
        "policy_type": "HO-3",
        "policyholder": "Sarah Mitchell",
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
                "description": "Adds coverage for water backup through sewers or drains and sump pump overflow. Not included in base policy.",
            },
            {
                "id": "endorsement_HO3_001_E02",
                "type": "Scheduled Personal Property",
                "items": "Diamond engagement ring valued at $8,500",
                "description": "Extends coverage to specifically listed high-value personal property items beyond standard limits.",
            },
        ],
    },
    {
        "id": "policy_PAP_001",
        "insurer": "ACME_AUTO",
        "policy_type": "PAP",
        "policyholder": "David Chen",
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
                "description": "Extends liability coverage during Period 1 of ride-share activity (app on, no passenger). Does NOT extend to Periods 2 and 3.",
            },
            {
                "id": "endorsement_PAP_001_E02",
                "type": "New Car Replacement",
                "description": "If total loss occurs within 24 months of original purchase, pays replacement cost of a new vehicle of same make and model rather than actual cash value.",
            },
        ],
    },
    {
        "id": "policy_CGL_001",
        "insurer": "ACME_COMM",
        "policy_type": "CGL",
        "policyholder": "Riverside Bakery LLC",
        "effective_date": "2023-06-01",
        "expiry_date": "2024-06-01",
        "business": "Retail bakery with delivery operations",
        "occurrence_limit": 1000000,
        "aggregate_limit": 2000000,
        "personal_injury_limit": 1000000,
        "deductible": 5000,
        "key_exclusions": [
            "employee bodily injury (covered by workers comp)",
            "damage to your own property",
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
                "additional_insured": "Springfield Properties LLC",
                "description": "Adds the building landlord as an additional insured for claims arising from the named insured's operations at the leased premises.",
            },
            {
                "id": "endorsement_CGL_001_E02",
                "type": "Food Contamination Extension",
                "limit": 50000,
                "description": "Covers costs of product recall, disposal, and third-party illness claims arising from food contamination events at the business premises.",
            },
        ],
    },
]

# Amended version of HO-3 policy (for SUPERSEDES testing)
AMENDED_POLICY = {
    "id": "policy_HO3_001_v2",
    "supersedes": "policy_HO3_001",
    "effective_date": "2024-01-01",
    "change_summary": "Pipe age exclusion raised from 20 years to 25 years. Deductible reduced from $2500 to $2000. Water backup endorsement limit increased from $10,000 to $15,000.",
    "key_change": "pipes more than 25 years old (was: 20 years)",
}


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def init_vertex() -> genai.Client:
    return genai.Client(vertexai=True, project=PROJECT, location=GEMINI_LOCATION)


def call_gemini(model: genai.Client, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = model.models.generate_content(model=MODEL, contents=prompt)
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
- DECLARATIONS: policyholder name, address/vehicle/business, policy number, effective/expiry dates,
  coverage limits, and deductible amounts.
- COVERAGE: describe each covered peril with specific conditions. Include dollar amounts and
  percentages where relevant (e.g. personal property away from premises: 10% of dwelling limit).
- EXCLUSIONS: describe each exclusion with specific language. Include the pipe age threshold
  (if applicable) as: "pipes or plumbing systems more than XX years of age".
- CONDITIONS: claims reporting requirements (must report within 60 days), cooperation clause,
  subrogation rights, cancellation terms.
- DEFINITIONS: define "occurrence", "property damage", "bodily injury", "insured", and
  any policy-specific terms.

Length: 800-1000 words. Write in formal insurance policy language.
Do NOT use any markdown formatting. Use plain text only.
"""

def generate_policy(model: genai.Client, policy: dict, output_path: Path) -> None:
    if output_path.exists():
        print(f"  skip (exists): {output_path.name}")
        return
    prompt = POLICY_PROMPT.format(params=json.dumps(policy, indent=2))
    text = call_gemini(model, prompt)
    output_path.write_text(text)
    print(f"  written: {output_path.name}")


ENDORSEMENT_PROMPT = """
Generate a realistic insurance endorsement document for the following parameters.
Plain text only, no markdown.

Base policy ID: {policy_id}
Policy type: {policy_type}
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

Length: 200-300 words.
"""

def generate_endorsement(model: genai.Client, policy: dict,
                          endorsement: dict, output_path: Path) -> None:
    if output_path.exists():
        print(f"  skip (exists): {output_path.name}")
        return
    prompt = ENDORSEMENT_PROMPT.format(
        policy_id=policy["id"],
        policy_type=policy["policy_type"],
        endorsement=json.dumps(endorsement, indent=2),
    )
    text = call_gemini(model, prompt)
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
In the EXCLUSIONS section, use the updated pipe age threshold: "pipes or plumbing systems
more than 25 years of age" (changed from 20 years in original policy).
Length: 800-1000 words.
"""

def generate_amended_policy(model: genai.Client, output_path: Path) -> None:
    if output_path.exists():
        print(f"  skip (exists): {output_path.name}")
        return
    prompt = AMENDED_POLICY_PROMPT.format(
        original_id=AMENDED_POLICY["supersedes"],
        amendment=json.dumps(AMENDED_POLICY, indent=2),
        effective_date=AMENDED_POLICY["effective_date"],
    )
    text = call_gemini(model, prompt)
    output_path.write_text(text)
    print(f"  written: {output_path.name}")


# ---------------------------------------------------------------------------
# Claim generation
# ---------------------------------------------------------------------------

CLAIM_SCENARIOS = {
    "HO-3": [
        # (claim_id, scenario, outcome, vocabulary_gap_note)
        ("claim_HO_001", "burst pipe causing water damage to kitchen floor and cabinets; pipe installed 18 years ago", "approved", "pipe age under threshold → covered"),
        ("claim_HO_002", "water damage from burst pipe; pipe was installed 23 years ago per plumber's report", "denied", "pipe age over 20-year threshold in original policy"),
        ("claim_HO_003", "roof damage from windstorm; shingles blown off, interior water damage", "approved", "windstorm covered"),
        ("claim_HO_004", "theft of jewelry from home; diamond ring taken", "partially_approved", "ring covered under scheduled property endorsement; other jewelry over limit"),
        ("claim_HO_005", "sewer backup flooded basement after heavy rain", "approved_with_endorsement", "requires water backup endorsement E01"),
        ("claim_HO_006", "gradual water stain on ceiling from slow roof leak over several months", "denied", "gradual deterioration exclusion"),
        ("claim_HO_007", "fire damage to kitchen following stove malfunction", "approved", "fire covered"),
        ("claim_HO_008", "slip and fall injury to visitor on icy front steps", "approved", "personal liability covered"),
        ("claim_HO_009", "mold damage discovered during renovation", "denied", "mold resulting from gradual moisture not covered"),
        ("claim_HO_010", "water damage from burst pipe; pipe installed 22 years ago; filed after policy amendment", "approved", "pipe age now under 25-year threshold in amended policy v2"),
    ],
    "PAP": [
        ("claim_PAP_001", "rear-end collision at stop light; vehicle struck from behind by uninsured driver", "approved", "uninsured motorist coverage"),
        ("claim_PAP_002", "collision while delivering food for a ride-share app with passenger in vehicle", "denied", "ride-share Period 2/3 not covered by base policy or endorsement"),
        ("claim_PAP_003", "windshield cracked by road debris on highway", "approved", "comprehensive coverage"),
        ("claim_PAP_004", "vehicle stolen from apartment parking lot overnight", "approved", "comprehensive theft coverage"),
        ("claim_PAP_005", "collision with deer on rural road causing front-end damage", "approved", "comprehensive animal collision"),
        ("claim_PAP_006", "vehicle total loss in intersection collision; vehicle purchased 18 months ago", "approved_replacement_cost", "new car replacement endorsement applies within 24-month window"),
        ("claim_PAP_007", "engine failure due to lack of oil maintenance", "denied", "mechanical breakdown exclusion"),
        ("claim_PAP_008", "minor collision while using app as delivery driver (no passenger)", "partially_approved", "ride-share Period 1 covered by endorsement E01"),
        ("claim_PAP_009", "hail damage to hood and roof of vehicle", "approved", "comprehensive weather coverage"),
        ("claim_PAP_010", "bodily injury claim from third party after policyholder at-fault collision", "approved", "bodily injury liability"),
    ],
    "CGL": [
        ("claim_CGL_001", "customer slipped on wet floor near display case; fractured wrist", "approved", "bodily injury on premises"),
        ("claim_CGL_002", "delivery driver damaged customer's gate post while making a delivery", "approved", "property damage liability"),
        ("claim_CGL_003", "employee injured back lifting flour bags in warehouse", "denied", "employee injury excluded; covered by workers compensation"),
        ("claim_CGL_004", "three customers reported food poisoning after eating birthday cake", "approved_with_endorsement", "food contamination extension E02 covers recall costs and illness claims"),
        ("claim_CGL_005", "customer alleged defamatory statement in social media post by business", "approved", "personal and advertising injury coverage"),
        ("claim_CGL_006", "landlord sued after customer tripped on uneven flooring at bakery entrance", "approved", "additional insured endorsement E01 covers landlord"),
        ("claim_CGL_007", "oven fire caused smoke damage to neighboring tenant's space", "approved", "property damage to third party"),
        ("claim_CGL_008", "customer claimed allergic reaction to undisclosed ingredient", "approved", "products and completed operations"),
        ("claim_CGL_009", "business owner's personal vehicle damaged in parking lot", "denied", "damage to insured's own property excluded"),
        ("claim_CGL_010", "professional advice claim: customer claimed financial loss from catering recommendation", "denied", "professional liability (E&O) excluded from CGL"),
    ],
}

FNOL_PROMPT = """
Write a First Notice of Loss (FNOL) claim report for an insurance claim.
Plain text only, no markdown. Use formal insurance reporting language.

Policy ID: {policy_id}
Policy Type: {policy_type}
Policyholder: {policyholder}
Claim ID: {claim_id}
Incident scenario: {scenario}

Format:
FIRST NOTICE OF LOSS

Claim Number: {claim_id}
Policy Number: {policy_id}
Date of Loss: [a plausible date in 2023 or 2024]
Date Reported: [1-3 days after date of loss]
Reported By: {policyholder}

DESCRIPTION OF LOSS:
[3-4 sentences describing what happened. Use the policyholder's own words.
Do NOT use technical insurance terms — this is the policyholder's account.
Do NOT mention the policy exclusion or coverage outcome.]

ESTIMATED DAMAGES:
[Plausible dollar estimate]

SUPPORTING DOCUMENTS ATTACHED:
[2-3 plausible document types, e.g. photos, repair estimates, police report]

Length: 200-250 words.
"""

ADJUSTER_PROMPT = """
Write an insurance adjuster's investigation notes for a claim.
Plain text only, no markdown. Use formal adjuster language.

Policy ID: {policy_id}
Policy Type: {policy_type}
Claim ID: {claim_id}
Incident scenario: {scenario}
Expected outcome: {outcome}

Format:
ADJUSTER INVESTIGATION NOTES

Claim Number: {claim_id}
Adjuster: [plausible adjuster name and ID]
Inspection Date: [plausible date, 5-10 days after FNOL]

SITE INSPECTION FINDINGS:
[2-3 sentences describing physical findings. Be specific — include measurements,
age estimates, photos taken. If pipe age is relevant, state the specific age found.]

POLICY REVIEW:
[Reference the specific policy section and language that applies to this claim.
If an exclusion applies, quote the relevant exclusion language precisely.
If an endorsement applies, name it and describe how it changes coverage.]

COVERAGE DETERMINATION RECOMMENDATION:
[State the coverage decision and the exact policy basis for it. Be specific.]

Length: 250-300 words.
"""

OUTCOME_PROMPT = """
Write a formal insurance claim outcome letter to the policyholder.
Plain text only, no markdown.

Policy ID: {policy_id}
Claim ID: {claim_id}
Policyholder: {policyholder}
Outcome: {outcome}
Incident scenario: {scenario}

Format:
CLAIM DETERMINATION LETTER

Date: [plausible date, 15-20 days after FNOL]
Claim Number: {claim_id}
Policy Number: {policy_id}
Insured: {policyholder}

Dear {policyholder},

[Opening paragraph: state coverage decision clearly — approved, denied, or partially approved.]

[Second paragraph: cite the specific policy section, exclusion language, or endorsement
that supports the decision. Include exact dollar amounts if applicable.]

[Third paragraph: if denied, explain the specific reason using policy language.
If approved, state the payment amount and next steps.]

[Closing: appeal rights and contact information.]

Sincerely,
[Plausible claims adjuster name]
Claims Department
[Insurer name]

Length: 250-300 words.
"""

def generate_claim(model: genai.Client, policy: dict, claim_id: str,
                   scenario: str, outcome: str, claims_base: Path) -> None:
    claim_dir = claims_base / claim_id
    claim_dir.mkdir(exist_ok=True)

    policyholder = policy.get("policyholder", "")

    for doc_type, prompt_template in [
        ("fnol", FNOL_PROMPT),
        ("adjuster", ADJUSTER_PROMPT),
        ("outcome", OUTCOME_PROMPT),
    ]:
        out_path = claim_dir / f"{doc_type}.txt"
        if out_path.exists():
            print(f"    skip (exists): {claim_id}/{doc_type}.txt")
            continue
        prompt = prompt_template.format(
            policy_id=policy["id"],
            policy_type=policy["policy_type"],
            policyholder=policyholder,
            claim_id=claim_id,
            scenario=scenario,
            outcome=outcome,
        )
        text = call_gemini(model, prompt)
        out_path.write_text(text)
        print(f"    written: {claim_id}/{doc_type}.txt")
        time.sleep(0.5)  # respect rate limits


# ---------------------------------------------------------------------------
# Benchmark question generation
# ---------------------------------------------------------------------------

GROUND_TRUTH_QUESTIONS = [
    # ── Policy-only (5) ──────────────────────────────────────────────────────
    # Test structure-aware chunking finds the right policy section.
    {
        "id": "Q01",
        "question": "Under a homeowners policy, what is the maximum dollar amount covered for personal property that is stolen or damaged while located away from the insured premises?",
        "expected_answer": "10% of the dwelling coverage limit (i.e. $35,000 on a $350,000 dwelling limit policy)",
        "required_sources": ["policy_HO3_001_coverage"],
        "traversal_type": "traverse_up",
        "difficulty": "medium",
        "note": "Answer is a percentage of dwelling limit; question asks for dollar amount — vocabulary gap",
    },
    {
        "id": "Q02",
        "question": "What specific age threshold for plumbing systems is used to determine whether a water damage claim from a burst pipe is excluded under the homeowners policy?",
        "expected_answer": "20 years (pipes more than 20 years of age are excluded under the original policy)",
        "required_sources": ["policy_HO3_001_exclusions"],
        "traversal_type": "traverse_up",
        "difficulty": "easy",
        "note": "Direct factual retrieval from Exclusions section",
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
        "question": "Under the commercial general liability policy, what is the per-occurrence limit and what is the general aggregate limit?",
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
        "question": "A homeowner filed a water damage claim after a pipe burst. The adjuster denied the claim citing the age of the plumbing. What was the specific age of the pipe found during inspection, and which policy exclusion was cited as the basis for denial?",
        "expected_answer": "The pipe was 23 years old. The policy excludes pipes more than 20 years of age.",
        "required_sources": ["claim_HO_002_adjuster", "policy_HO3_001_exclusions"],
        "traversal_type": "traverse_policy",
        "difficulty": "hard",
        "note": "Requires crossing from claim adjuster notes to policy exclusions section",
    },
    {
        "id": "Q07",
        "question": "A bakery received a claim from multiple customers who became ill after eating its products. Was this type of claim covered, and if so, under what specific provision?",
        "expected_answer": "Yes, covered under the Food Contamination Extension endorsement (endorsement_CGL_001_E02) up to $50,000.",
        "required_sources": ["claim_CGL_004_adjuster", "endorsement_CGL_001_E02"],
        "traversal_type": "traverse_policy",
        "difficulty": "hard",
        "note": "Answer requires finding the endorsement, not the base policy",
    },
    {
        "id": "Q08",
        "question": "A ride-share driver filed a claim for a collision that occurred while they had a passenger in the vehicle. Was this loss covered under their auto policy?",
        "expected_answer": "No. The ride-share endorsement (E01) only covers Period 1 (app on, no passenger). Coverage during Period 2 and 3 (passenger in vehicle) is explicitly excluded.",
        "required_sources": ["claim_PAP_002_adjuster", "endorsement_PAP_001_E01"],
        "traversal_type": "traverse_policy",
        "difficulty": "hard",
        "note": "Requires endorsement text to understand Period 1 vs Period 2/3 distinction",
    },
    {
        "id": "Q09",
        "question": "An injured visitor filed a claim against a commercial bakery after slipping on the premises. Was the building's landlord also protected under the bakery's insurance policy?",
        "expected_answer": "Yes. The Additional Insured endorsement (E01) covers the landlord (Springfield Properties LLC) for claims arising from the named insured's operations at the leased premises.",
        "required_sources": ["claim_CGL_006_adjuster", "endorsement_CGL_001_E01"],
        "traversal_type": "traverse_policy",
        "difficulty": "hard",
    },
    {
        "id": "Q10",
        "question": "A vehicle owner filed a total loss claim eighteen months after purchasing a new vehicle. What was the basis for the settlement amount, and why was it different from the standard calculation?",
        "expected_answer": "Settled at new vehicle replacement cost (same make and model) rather than actual cash value, because the New Car Replacement endorsement applies within 24 months of original purchase.",
        "required_sources": ["claim_PAP_006_adjuster", "endorsement_PAP_001_E02"],
        "traversal_type": "traverse_policy",
        "difficulty": "hard",
    },
    # ── Temporal chain via NEXT_PERIOD (3) ───────────────────────────────────
    {
        "id": "Q11",
        "question": "A homeowner had a water damage claim denied in 2023. They filed a second claim for a similar pipe-related loss in 2024. What changed between the two claims that affected the coverage outcome?",
        "expected_answer": "The policy was amended in 2024, raising the pipe age exclusion from 20 years to 25 years. The second claim was for a 22-year-old pipe, which was excluded under the original policy but covered under the amended policy.",
        "required_sources": ["claim_HO_002_outcome", "claim_HO_010_adjuster", "policy_HO3_001_v2_exclusions"],
        "traversal_type": "traverse_temporal",
        "difficulty": "hard",
        "note": "Requires temporal traversal across two claims AND policy version comparison",
    },
    {
        "id": "Q12",
        "question": "A policyholder had their auto claim for a theft approved. Shortly after, they filed a separate claim for hail damage. What were the deductibles that applied to each claim?",
        "expected_answer": "Theft claim: $500 comprehensive deductible. Hail damage claim: $500 comprehensive deductible. Both fall under comprehensive coverage.",
        "required_sources": ["claim_PAP_004_outcome", "claim_PAP_009_adjuster"],
        "traversal_type": "traverse_temporal",
        "difficulty": "medium",
    },
    {
        "id": "Q13",
        "question": "A commercial bakery filed a bodily injury claim for a customer who slipped on premises, followed by a property damage claim for smoke damage to a neighboring tenant. What was the combined exposure against the policy aggregate?",
        "expected_answer": "Bodily injury claim (claim_CGL_001): within $1M per occurrence limit. Property damage claim (claim_CGL_007): within $1M per occurrence limit. Combined exposure $2M, which equals the aggregate limit.",
        "required_sources": ["claim_CGL_001_outcome", "claim_CGL_007_outcome"],
        "traversal_type": "traverse_temporal",
        "difficulty": "hard",
    },
    # ── Document versioning via SUPERSEDES (2) ───────────────────────────────
    {
        "id": "Q14",
        "question": "Under the current version of the homeowners policy (effective January 2024), what is the pipe age threshold that triggers the water damage exclusion?",
        "expected_answer": "25 years. The amended policy (effective 2024-01-01) raised the threshold from 20 years to 25 years.",
        "required_sources": ["policy_HO3_001_v2_exclusions"],
        "traversal_type": "supersedes",
        "difficulty": "medium",
        "note": "Must retrieve amended policy, not original — tests SUPERSEDES + superseded=true filter",
    },
    {
        "id": "Q15",
        "question": "A claim was denied in early 2023 because the damaged pipe was 22 years old. If the same loss had occurred under the policy terms in effect from January 2024 onward, would the outcome have been different?",
        "expected_answer": "Yes. The original policy excluded pipes over 20 years, so the 22-year-old pipe was denied. The amended policy (effective 2024-01-01) raised the threshold to 25 years, so the same 22-year-old pipe would be covered.",
        "required_sources": ["policy_HO3_001_exclusions", "policy_HO3_001_v2_exclusions"],
        "traversal_type": "supersedes",
        "difficulty": "hard",
        "note": "Requires comparing original and amended policy — most complex question",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Initializing Vertex AI...")
    model = init_vertex()

    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    ENDORSEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Policies ──────────────────────────────────────────────────────────────
    print("\n[1] Generating policies...")
    for policy in POLICIES:
        out = POLICIES_DIR / f"{policy['id']}.txt"
        print(f"  Policy: {policy['id']}")
        generate_policy(model, policy, out)
        time.sleep(1)

    print("\n[2] Generating amended policy (v2 for SUPERSEDES test)...")
    generate_amended_policy(model, POLICIES_DIR / "policy_HO3_001_v2.txt")

    # ── Endorsements ──────────────────────────────────────────────────────────
    print("\n[3] Generating endorsements...")
    for policy in POLICIES:
        for end in policy["endorsements"]:
            out = ENDORSEMENTS_DIR / f"{end['id']}.txt"
            print(f"  Endorsement: {end['id']}")
            generate_endorsement(model, policy, end, out)
            time.sleep(1)

    # ── Claims ────────────────────────────────────────────────────────────────
    print("\n[4] Generating claims...")
    policy_map = {p["policy_type"]: p for p in POLICIES}
    for policy_type, scenarios in CLAIM_SCENARIOS.items():
        policy = policy_map[policy_type]
        print(f"\n  {policy_type} claims:")
        for claim_id, scenario, outcome, _ in scenarios:
            print(f"  Claim: {claim_id}")
            generate_claim(model, policy, claim_id, scenario, outcome, CLAIMS_DIR)
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
    policy_files = list(POLICIES_DIR.glob("*.txt"))
    endorsement_files = list(ENDORSEMENTS_DIR.glob("*.txt"))
    claim_dirs = [d for d in CLAIMS_DIR.iterdir() if d.is_dir()]
    print(f"  Policies:     {len(policy_files)} files in demo_docs/policies/")
    print(f"  Endorsements: {len(endorsement_files)} files in demo_docs/endorsements/")
    print(f"  Claims:       {len(claim_dirs)} claim folders in demo_docs/claims/")
    print(f"  Benchmark:    {len(GROUND_TRUTH_QUESTIONS)} questions in evaluation/ground_truth.json")
    print("\nNext: python pipeline/ingest.py --strategy structure_aware")


if __name__ == "__main__":
    main()
