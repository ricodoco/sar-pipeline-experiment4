"""
Experiment 4a: Natural Language -> SAR Translation Efficiency
Tests Claim 1 of the MIEM/LLM paper: an LLM can translate natural language
observations into well-formed Semantic Assertion Records (SARs) -- named
n-ary case frames -- from a small training set.

Condition B: trained on varied, naturally-phrased NL reports -> correct SAR
Condition C: trained on repetitive, templated NL reports -> correct SAR
Condition A: untrained control (no SAR examples, asked to "structure" reports)

IMPORTANT: MODELFILE_* strings are passed through .format(base_model=...) in
build_models.py. Every literal "{" and "}" below (in SAR examples) is therefore
escaped as "{{" and "}}" so .format() leaves it as a single brace in output.
"""

SEED_TRAIN = 144
SEED_TEST  = 145
TRANCHE_TRAIN = 20   # 15 positive + 5 negative per tranche
TRANCHE_TEST  = 20
MAX_TRANCHES  = 5
MODEL         = "llama3.2"

# ── SAR frame types (from the MIEM/LLM paper, Section 3) ────────────────────
# Reused as "CONCEPTS" so generate.py / evaluate.py need minimal changes.

CONCEPTS = {
    "CargoLoading": {
        "name": "CargoLoading",
        "desc": "a cargo loading or transfer event at a port",
        "required": ["Cargo", "Vessel", "Port", "Time"],
        "optional": ["Agent", "SealStatus", "Manifest"],
    },
    "Movement": {
        "name": "Movement",
        "desc": "movement of a person, vessel, or vehicle between locations",
        "required": ["Entity"],
        "optional": ["From", "To", "DepartTime", "ArriveTime", "Mode", "Source"],
    },
    "FlaggedEntity": {
        "name": "FlaggedEntity",
        "desc": "an entity flagged by a watch list or intelligence source",
        "required": ["Entity", "FlaggedBy", "Reason", "Confidence"],
        "optional": ["Date"],
    },
    "RiskAssessment": {
        "name": "RiskAssessment",
        "desc": "an analyst's risk rating of a subject",
        "required": ["Subject", "RiskLevel", "Basis", "Analyst"],
        "optional": ["Confidence", "Date"],
    },
    "RetailPurchase": {
        "name": "RetailPurchase",
        "desc": "a retail purchase transaction",
        "required": ["Buyer", "Store", "Time"],
        "optional": ["Item1", "Item2", "Item3", "PaymentCard"],
    },
}
FRAMES = CONCEPTS
FRAME_KEYS = list(FRAMES.keys())

# ── Negative case taxonomy ───────────────────────────────────────────────────

_NEG_TYPES = ["wrong_frame", "missing_required", "ambiguous_entity",
              "non_event", "contradictory"]
NEGATIVES = {k: list(_NEG_TYPES) for k in FRAME_KEYS}

# ── Domains ──────────────────────────────────────────────────────────────────

DOMAINS = {
    "customs":      "customs and border inspection reporting",
    "coastguard":   "coast guard vessel and crew tracking",
    "intel":        "intelligence agency source reporting",
    "retail_sec":   "retail security and fraud alert reporting",
    "port_auth":    "port authority logistics reporting",
    "interpol":     "international law enforcement watch-list reporting",
}
DOMAIN_KEYS = list(DOMAINS.keys())

# ── Modelfile templates (all literal braces doubled for .format() safety) ────

MODELFILE_B = """FROM {base_model}
SYSTEM \"\"\"
You are a Semantic Assertion Record (SAR) translator for intelligence and
maritime tracking reports. You convert natural language observations into
well-formed SAR case frames.

Available SAR frame types:

CargoLoading: {{ FrameName: CargoLoading, Cargo(R), Vessel(R), Port(R), Time(R), Agent(O), SealStatus(O), Manifest(O) }}
Movement: {{ FrameName: Movement, Entity(R), From(O), To(O), DepartTime(O), ArriveTime(O), Mode(O), Source(O) }}
FlaggedEntity: {{ FrameName: FlaggedEntity, Entity(R), FlaggedBy(R), Reason(R), Confidence(R), Date(O) }}
RiskAssessment: {{ FrameName: RiskAssessment, Subject(R), RiskLevel(R), Basis(R), Analyst(R), Confidence(O), Date(O) }}
RetailPurchase: {{ FrameName: RetailPurchase, Buyer(R), Store(R), Time(R), Item1(O), Item2(O), Item3(O), PaymentCard(O) }}

(R) = required slot, must be filled. (O) = optional slot, omit if not in the report.
Slots are named and unordered: list only the slots you can fill, in any order.
If the report does not clearly support any single frame with all required
slots filled, respond: "INCOMPLETE: <which required slot is missing or unclear>"
instead of a frame.

When given a report, identify the single best-fitting frame type, then output
ONLY the frame in this exact format:
{{ FrameName: <type>, Slot1: value1, Slot2: value2, ... }}

Do not invent information not present or clearly implied in the report.
Use slot names exactly as given. Slots may appear in any order -- order does
not matter, only that required slots are present and correctly named.

EXAMPLE 1:
Report: "Customs inspectors at Barcelona watched dockworkers load crate
XYZ-447 onto the MV Meridian this morning around 8:30. The seal looked
intact at the time."
Response: {{ FrameName: CargoLoading, Cargo: XYZ-447, Vessel: MV_Meridian, Port: Barcelona, Time: 08:30_today, SealStatus: Intact }}

EXAMPLE 2:
Report: "Coast Guard boarding party did a crew muster on the Meridian and
came up one man short -- a Syrian national, last seen aboard while at sea,
not present once they reached the Charleston anchorage."
Response: {{ FrameName: Movement, Entity: Syrian_crewman, From: at_sea, To: missing_at_anchorage, Mode: vessel, Source: CoastGuard_muster }}

EXAMPLE 3:
Report: "Someone mentioned a guy might be trouble, not sure who flagged him
or why."
Response: INCOMPLETE: Entity, FlaggedBy, and Reason are not clearly identified
\"\"\"
"""

MODELFILE_C = """FROM {base_model}
SYSTEM \"\"\"
You are a Semantic Assertion Record (SAR) translator. Convert reports into
frames using this exact fixed procedure every time:

Available SAR frame types:

CargoLoading: {{ FrameName: CargoLoading, Cargo(R), Vessel(R), Port(R), Time(R), Agent(O), SealStatus(O), Manifest(O) }}
Movement: {{ FrameName: Movement, Entity(R), From(O), To(O), DepartTime(O), ArriveTime(O), Mode(O), Source(O) }}
FlaggedEntity: {{ FrameName: FlaggedEntity, Entity(R), FlaggedBy(R), Reason(R), Confidence(R), Date(O) }}
RiskAssessment: {{ FrameName: RiskAssessment, Subject(R), RiskLevel(R), Basis(R), Analyst(R), Confidence(O), Date(O) }}
RetailPurchase: {{ FrameName: RetailPurchase, Buyer(R), Store(R), Time(R), Item1(O), Item2(O), Item3(O), PaymentCard(O) }}

(R) = required slot, must be filled. (O) = optional slot, omit if not in the report.

Always respond with the frame in this exact form, filling slots in this
fixed order: FrameName first, then required slots in the order listed above,
then optional slots in the order listed above. If a required slot is missing,
write INCOMPLETE: <missing slot name>.

EXAMPLE: {{ FrameName: CargoLoading, Cargo: X, Vessel: Y, Port: Z, Time: T }}
\"\"\"
"""

MODELFILE_A = """FROM {base_model}
SYSTEM \"\"\"
You are a helpful assistant. When given a report, summarize the key facts
in a structured way. Use your best judgment about format.
\"\"\"
"""
