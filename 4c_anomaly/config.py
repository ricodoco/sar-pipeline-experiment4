"""
Experiment 4c: Anomaly Detection and Explanation from Accumulated SARs
Tests Claim 3 of the MIEM/LLM paper (Section 2.4): given an accumulated set
of SAR frames (a knowledge-graph excerpt), the system must (a) correctly
detect whether a genuine cross-frame threat/anomaly pattern is present,
(b) NOT manufacture a false threat from benign accumulated frames, and
(c) when a threat IS present, cite the SPECIFIC frames that support it
("the LLM can cite the specific SAR chains that support each hypothesis").

Scored with the same signal detection framework as 4b: a missed threat and
a false alarm are both "wrong picture" errors and are weighted symmetrically
via d-prime. Haystack density = number of irrelevant/benign distractor SAR
frames mixed in with the (2-4) frames that jointly constitute the genuine
pattern (or, in NO_THREAT cases, mixed in with frames that look superficially
concerning but do not jointly constitute a real pattern).

Condition B: varied phrasing/structure of the SAR frames presented.
Condition C: templated/fixed-order SAR frame presentation.
Condition A: untrained control (no worked hypothesis-ranking examples).
"""

SEED_TRAIN = 344
SEED_TEST  = 345
TRANCHE_TRAIN = 20
TRANCHE_TEST  = 20
MAX_TRANCHES  = 5
MODEL         = "llama3.2"

HAYSTACK_DENSITIES = [0, 5, 15, 30]

# ── Threat pattern scenarios (ground truth THREAT or NO_THREAT) ─────────────
# Each pattern is a SET of 2-4 SAR frames that, taken jointly, support (or
# in the NO_THREAT case, superficially resemble but do not support) a
# threat hypothesis. Modeled directly on the paper's two worked examples.

CONCEPTS = {
    "preattack_indicator_cluster": {
        "name": "preattack_indicator_cluster",
        "desc": ("partial flight training + extremist association + fuel "
                "quantity query + visa anomaly, jointly indicating aircraft "
                "weaponization planning"),
        "ground_truth": "THREAT",
    },
    "cargo_ied_cluster": {
        "name": "cargo_ied_cluster",
        "desc": ("tampered cargo seal + missing crewman matched to a flagged "
                "alias + precursor-material retail purchase, jointly "
                "indicating IED assembly"),
        "ground_truth": "THREAT",
    },
    "benign_training_pattern": {
        "name": "benign_training_pattern",
        "desc": ("ordinary flight training enrollment frames that superficially "
                "resemble a threat cluster (foreign student, normal curriculum "
                "questions) but lack the joint co-occurrence of disqualifying "
                "features"),
        "ground_truth": "NO_THREAT",
    },
    "benign_cargo_pattern": {
        "name": "benign_cargo_pattern",
        "desc": ("a cargo discrepancy (seal status anomaly, inventory "
                "mismatch) with an innocuous explanation (paperwork error, "
                "routine inspection reseal) and no flagged personnel or "
                "precursor materials involved"),
        "ground_truth": "NO_THREAT",
    },
}
FRAME_KEYS = list(CONCEPTS.keys())

DOMAINS = {
    "aviation_security": "aviation security and flight school oversight",
    "maritime_security": "maritime cargo and port security",
    "border_intel":      "border and customs intelligence",
    "law_enforcement":    "domestic law enforcement threat assessment",
}
DOMAIN_KEYS = list(DOMAINS.keys())

# ── Distractor SAR frame pool (benign, unrelated to any threat pattern) ─────

DISTRACTOR_FRAME_TYPES = ["CargoLoading", "Movement", "RetailPurchase",
                          "FlaggedEntity", "RiskAssessment"]
DISTRACTOR_FIRST_NAMES = ["Marcus", "Elena", "Yusuf", "Priya", "Daniel",
                          "Aisha", "Viktor", "Maria", "Kenji", "Fatima"]
DISTRACTOR_LAST_NAMES  = ["Reyes", "Novak", "Larsen", "Okafor", "Petrov",
                          "Tan", "Schmidt", "Haddad", "Costa", "Lindgren"]
DISTRACTOR_VESSELS     = ["MV Horizon", "SS Calypso Star", "MV Northern Dawn",
                          "MV Baltic Pearl", "SS Coral Voyager"]
DISTRACTOR_PORTS       = ["Rotterdam", "Singapore", "Valencia", "Dubai",
                          "Hamburg", "Busan", "Santos"]
DISTRACTOR_ITEMS       = ["machine parts", "textiles", "frozen seafood",
                          "lumber", "consumer electronics", "produce"]

DISTRACTOR_TEMPLATES = [
    "{{ FrameName: CargoLoading, Cargo: Crate_{n}, Vessel: {vessel}, Port: {port}, Time: routine, SealStatus: Intact }}",
    "{{ FrameName: Movement, Entity: {name}, From: {port}, To: home, Mode: air, Source: routine_travel_log }}",
    "{{ FrameName: RetailPurchase, Buyer: {name}, Store: GroceryMart_{port}, Item1: groceries }}",
    "{{ FrameName: FlaggedEntity, Entity: {name}, FlaggedBy: routine_audit, Reason: expired_document, Confidence: 0.1 }}",
    "{{ FrameName: RiskAssessment, Subject: {vessel}, RiskLevel: low, Basis: routine_inspection, Analyst: duty_officer }}",
]

# ── Modelfile templates ──────────────────────────────────────────────────────

TASK_SPEC = (
    "You will be given a set of Semantic Assertion Records (SAR frames), "
    "presented in random order, accumulated from multiple intelligence "
    "sources. Most frames are benign distractors unrelated to any threat. "
    "Among them may be a small cluster of frames (marked with "
    ">>CLUSTER<< tags) that, taken JOINTLY, either DO or DO NOT "
    "constitute a genuine threat pattern.\n\n"
    "Your task: examine ALL frames, focus on the ones tagged "
    ">>CLUSTER<<, and decide whether they jointly support a THREAT "
    "hypothesis or whether the pattern is explainable as NO_THREAT (benign, "
    "even if individual frames look superficially concerning in isolation). "
    "Ignore distractor frames -- they are noise, not signal.\n\n"
    "Respond in EXACTLY this format:\n"
    "DECISION: THREAT or NO_THREAT\n"
    "HYPOTHESIS: <one-sentence threat hypothesis if THREAT, else 'none'>\n"
    "SUPPORTING_FRAMES: <list the specific FrameName values from the "
    ">>CLUSTER<< that jointly support your decision>\n"
    "EVIDENCE: <one sentence explaining why these frames jointly indicate "
    "THREAT or why they do NOT, despite superficial concern>\n\n"
    "Do not flag THREAT based on a single frame in isolation, or based on "
    "superficial concern alone (foreign nationality, unusual purchase) -- "
    "require the JOINT co-occurrence pattern across the relevant cluster."
)

MODELFILE_B = """FROM {base_model}
SYSTEM \"\"\"
You are an intelligence fusion assistant specializing in cross-source
anomaly detection and threat hypothesis ranking.

""" + TASK_SPEC + """

EXAMPLE (THREAT):
>>CLUSTER<< {{ FrameName: TrainingRecord, Subject: Atta_M, Sought: takeoff_cruise_only, Avoided: landing_procedures }}
>>CLUSTER<< {{ FrameName: FlaggedEntity, Entity: Atta_M, FlaggedBy: CIA, Reason: AlQaeda_Hamburg_Cell_Association, Confidence: 0.8 }}
>>CLUSTER<< {{ FrameName: VisaStatus, Subject: Atta_M, Type: Tourist, Inconsistency: Extended_Flight_Training }}
>>CLUSTER<< {{ FrameName: Inquiry, Subject: Atta_M, Topic: Fuel_Quantity_At_Cruise_Large_Aircraft }}
Response:
DECISION: THREAT
HYPOTHESIS: Subject is planning to use a commercial aircraft as a weapon.
SUPPORTING_FRAMES: TrainingRecord, FlaggedEntity, VisaStatus, Inquiry
EVIDENCE: Partial training avoiding landing only makes sense for a one-way
mission; combined with extremist association, visa anomaly, and fuel
quantity interest, the joint pattern is highly atypical of legitimate
flight training.

EXAMPLE (NO_THREAT):
>>CLUSTER<< {{ FrameName: TrainingRecord, Subject: Chen_L, Sought: full_curriculum_including_landing }}
>>CLUSTER<< {{ FrameName: VisaStatus, Subject: Chen_L, Type: Student_F1, Inconsistency: none }}
Response:
DECISION: NO_THREAT
HYPOTHESIS: none
SUPPORTING_FRAMES: TrainingRecord, VisaStatus
EVIDENCE: Full curriculum including landing and a consistent, valid visa
status indicate ordinary legitimate flight training with no joint
indicator of weaponization intent.
\"\"\"
"""

MODELFILE_C = """FROM {base_model}
SYSTEM \"\"\"
You are an intelligence fusion assistant. Follow this fixed procedure:

""" + TASK_SPEC + """

Always check the >>CLUSTER<< frames for: (1) any FlaggedEntity frame
with Confidence above 0.5, (2) any frame indicating an omitted or avoided
required step, (3) any frame indicating a document/status inconsistency.
If at least two of these three are present jointly, decide THREAT.
Otherwise decide NO_THREAT.

EXAMPLE: DECISION: THREAT / HYPOTHESIS: X / SUPPORTING_FRAMES: A, B / EVIDENCE: two of three checks matched.
\"\"\"
"""

MODELFILE_A = """FROM {base_model}
SYSTEM \"\"\"
You are a helpful assistant. You will be given a set of structured records.
Answer any question asked about them as best you can.
\"\"\"
"""
