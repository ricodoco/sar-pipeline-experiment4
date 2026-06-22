"""
Experiment 4b: Multi-Report Identity Integration
Tests Claim 2 of the MIEM/LLM paper (Section 2.3): given multiple SAR-bearing
reports, the system must decide which entity-references refer to the SAME
real-world entity (merge) and which refer to DIFFERENT entities (do not
merge) -- including under realistic "haystack" conditions where many
irrelevant distractor reports surround the relevant pair.

Scored with signal detection theory (d-prime), not simple pass/fail, because
false merges and missed merges are the SAME error class (wrong identity
picture) and must be weighted symmetrically. See evaluate.py.

Condition B: varied phrasing identity cues (name spelling variants, partial
             descriptions, role-based references) -- the realistic case.
Condition C: templated/canonical phrasing only (same surface form every time).
Condition A: untrained control (no merge-judgment examples shown).
"""

SEED_TRAIN = 244
SEED_TEST  = 245
TRANCHE_TRAIN = 20   # cases are PAIRS of reports + haystack, not single reports
TRANCHE_TEST  = 20
MAX_TRANCHES  = 5
MODEL         = "llama3.2"

# Haystack densities to sweep across test cases (number of distractor SAR
# frames/reports injected alongside the 2 relevant reports in each case).
# 0 = clean pair only; higher = more realistic operational clutter.
HAYSTACK_DENSITIES = [0, 5, 15, 30]

# ── Entity pair scenarios: ground truth SAME or DIFFERENT person/vessel ─────
# "CONCEPTS" naming kept for structural parity with the reused scaffold.

CONCEPTS = {
    "same_person_alias": {
        "name": "same_person_alias",
        "desc": "two reports describing the same person under different "
                "names/aliases or partial descriptions",
        "ground_truth": "MERGE",
    },
    "same_vessel_variant_spelling": {
        "name": "same_vessel_variant_spelling",
        "desc": "two reports describing the same vessel with variant name "
                "spelling, transliteration, or partial hull/registry ID",
        "ground_truth": "MERGE",
    },
    "different_person_similar_features": {
        "name": "different_person_similar_features",
        "desc": "two reports describing genuinely different people who "
                "happen to share nationality, rough age, or a common name",
        "ground_truth": "NO_MERGE",
    },
    "different_vessel_similar_class": {
        "name": "different_vessel_similar_class",
        "desc": "two reports describing genuinely different vessels of the "
                "same class, flag, or operating similar routes",
        "ground_truth": "NO_MERGE",
    },
}
FRAME_KEYS = list(CONCEPTS.keys())

# ── Domains ──────────────────────────────────────────────────────────────────

DOMAINS = {
    "customs":      "customs and border inspection reporting",
    "coastguard":   "coast guard vessel and crew tracking",
    "intel":        "intelligence agency source reporting",
    "interpol":     "international law enforcement watch-list reporting",
    "port_auth":    "port authority logistics reporting",
}
DOMAIN_KEYS = list(DOMAINS.keys())

# ── Distractor entity pool (used to build haystack reports) ─────────────────
# Generic short report templates about unrelated entities, varied across
# frame types, used to pad context without being relevant to the merge
# decision under test. Filled with random distractor names/ports/times.

DISTRACTOR_FIRST_NAMES = ["Marcus", "Elena", "Yusuf", "Priya", "Daniel",
                           "Aisha", "Viktor", "Maria", "Kenji", "Fatima"]
DISTRACTOR_LAST_NAMES  = ["Reyes", "Novak", "Larsen", "Okafor", "Petrov",
                           "Tan", "Schmidt", "Haddad", "Costa", "Lindgren"]
DISTRACTOR_VESSELS     = ["MV Horizon", "SS Calypso Star", "MV Northern Dawn",
                           "MV Baltic Pearl", "SS Coral Voyager", "MV Atlas Crown"]
DISTRACTOR_PORTS       = ["Rotterdam", "Singapore", "Valencia", "Dubai",
                           "Hamburg", "Busan", "Santos", "Piraeus"]

DISTRACTOR_TEMPLATES = [
    "Customs at {port} cleared routine cargo of {item} from {vessel}, "
    "no irregularities noted.",
    "{name} was logged departing {port} on a scheduled commercial flight, "
    "routine travel, no flags.",
    "{vessel} reported standard crew muster at {port}, all {n} crew present "
    "and accounted for.",
    "A retail purchase by {name} at a hardware store in {port} included "
    "ordinary home repair supplies, no items of concern.",
    "Port authority at {port} logged {vessel} arrival on schedule, manifest "
    "matched declared cargo of {item}.",
]
DISTRACTOR_ITEMS = ["machine parts", "textiles", "frozen seafood",
                    "construction materials", "electronics", "produce"]

# ── Modelfile templates ──────────────────────────────────────────────────────
# Task: given a SET of reports (2 relevant + N distractors, shuffled), decide
# which (if any) pairs of entity-references should be merged as the same
# real-world entity, and output the merged SAR identity record, or state
# that no merge applies among the candidate pair under test.

TASK_SPEC = (
    "You will be given a set of intelligence/maritime reports, presented in "
    "random order. Most reports describe unrelated entities (distractors). "
    "Exactly one PAIR of reports may or may not describe the SAME real-world "
    "entity (a person or vessel) referred to differently across the two "
    "reports (e.g., name variant, alias, partial description, role-based "
    "reference).\n\n"
    "Your task: examine ALL reports, identify the one candidate pair "
    "explicitly marked with >>CANDIDATE<< tags, and decide whether they "
    "describe the SAME entity or DIFFERENT entities. Ignore distractor "
    "reports -- they are not part of the decision.\n\n"
    "Respond in EXACTLY this format:\n"
    "DECISION: MERGE or NO_MERGE\n"
    "ENTITY: <unified identity if MERGE, or 'N/A' if NO_MERGE>\n"
    "EVIDENCE: <one sentence citing the specific overlapping detail(s) that "
    "justify your decision>\n\n"
    "Do not merge based on superficial similarity alone (same nationality, "
    "similar name, same vessel class). Require a specific, distinguishing "
    "overlap (e.g., consistent unusual detail, explicit alias statement, "
    "matching registry/passport fragment) before deciding MERGE."
)

MODELFILE_B = """FROM {base_model}
SYSTEM \"\"\"
You are an intelligence fusion assistant specializing in entity identity
resolution across multiple reports.

""" + TASK_SPEC + """

EXAMPLE (MERGE):
Reports include: "Coast Guard muster shows passport name Tariq Al-Rashid
missing from MV Meridian crew at Charleston anchorage." and >>CANDIDATE<<
"A credit card in the name Omar Farouk, one of three known aliases for the
missing Meridian crewman per Interpol, was declined at a Jacksonville
hardware store."
Response:
DECISION: MERGE
ENTITY: Al_Rashid_T (aliases: Omar_Farouk, Hassan_Khalil)
EVIDENCE: The second report explicitly states Omar Farouk is a known alias
of the missing Meridian crewman, directly linking the two references.

EXAMPLE (NO_MERGE):
Reports include: "A Syrian national was flagged at port customs for
document irregularities." and >>CANDIDATE<< "An unrelated Syrian
national was interviewed routinely as a crew replacement candidate at the
same port, no flags."
Response:
DECISION: NO_MERGE
ENTITY: N/A
EVIDENCE: Shared nationality alone does not establish identity; no name,
alias, document, or distinguishing detail links the two individuals.
\"\"\"
"""

MODELFILE_C = """FROM {base_model}
SYSTEM \"\"\"
You are an intelligence fusion assistant. Follow this fixed procedure:

""" + TASK_SPEC + """

Always check for: identical name, then identical passport/registry number,
then explicit alias statement. If any one of these three exact matches is
present, decide MERGE. Otherwise decide NO_MERGE.

EXAMPLE: DECISION: MERGE / ENTITY: X / EVIDENCE: explicit alias match.
\"\"\"
"""

MODELFILE_A = """FROM {base_model}
SYSTEM \"\"\"
You are a helpful assistant. You will be given several reports. Answer any
question asked about them as best you can.
\"\"\"
"""
