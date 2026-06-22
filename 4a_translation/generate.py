"""
Experiment 4a: Case generator
Produces (natural language report, correct SAR) training and test pairs.
Condition B: varied, naturally-phrased reports.
Condition C: repetitive, templated reports (fixed slot order, fixed phrasing).
Condition A: same reports as B, no SAR examples shown to the model.
"""

import json, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (CONCEPTS, FRAMES, FRAME_KEYS, NEGATIVES,
                    DOMAINS, DOMAIN_KEYS, SEED_TRAIN, SEED_TEST,
                    TRANCHE_TRAIN, TRANCHE_TEST)

import anthropic
client = anthropic.Anthropic()


def render_correct_sar(frame_key, slot_values):
    """Render a dict of {slot: value} into the canonical SAR text form."""
    parts = [f"FrameName: {frame_key}"]
    for k, v in slot_values.items():
        parts.append(f"{k}: {v}")
    return "{ " + ", ".join(parts) + " }"


def generate_scenario_llm(frame_key, domain, condition, negative_type=None, seed=None):
    """
    Generate a single (report, correct SAR) pair using Claude as the sampler.
    condition: 'B' (varied phrasing), 'C' (templated/repetitive phrasing),
               'A' (same as B, used only for prompt diversity in control set)
    negative_type: if set, generate a report that should NOT yield a complete
                   SAR of this frame type.
    """
    frame = CONCEPTS[frame_key]
    domain_desc = DOMAINS[domain]
    required = frame["required"]
    optional = frame["optional"]

    if negative_type:
        neg_desc = {
            "wrong_frame":      f"the report actually describes a different kind of event, not a {frame_key}",
            "missing_required": f"the report omits information needed for at least one required slot ({', '.join(required)})",
            "ambiguous_entity": "the identity of a key participant is unclear or could refer to multiple entities",
            "non_event":        "the text is a general statement or policy, not a specific observed event",
            "contradictory":    "the report contains two statements that contradict each other about a slot value",
        }[negative_type]

        prompt = f"""Generate a realistic intelligence/maritime report for testing an
NL-to-SAR translation system.

Domain: {domain_desc}
Target frame (which should NOT be cleanly producible): {frame_key} -- {frame['desc']}
Required slots for this frame: {', '.join(required)}
Reason this report should NOT yield a complete {frame_key} frame: {neg_desc}

Write a short report (1-3 sentences) in the style of a {domain_desc} note,
such that {neg_desc}.

Then state what the correct system response should be: either "INCOMPLETE: <reason>"
naming the specific missing/unclear slot, or if wrong_frame, name the frame that
WOULD actually fit (briefly) and that {frame_key} does not apply.

Format exactly:
REPORT: <report text>
CORRECT_RESPONSE: <INCOMPLETE: ... or brief explanation of wrong frame>
FRAME_APPLIES: false
FRAME: {frame_key}
DOMAIN: {domain}
NEGATIVE_TYPE: {negative_type}"""

    else:
        # Build a plausible slot-value assignment for this frame
        rng = random.Random(seed or 0)
        sample_values = {
            "Cargo": "a numbered crate or container",
            "Vessel": "a named ship",
            "Port": "a named port city",
            "Time": "a specific time or date",
            "Agent": "a person or crew responsible",
            "SealStatus": "intact, tampered, or missing",
            "Manifest": "a manifest reference or declared contents",
            "Entity": "a named person, vessel, or vehicle",
            "From": "a named origin location",
            "To": "a named destination location",
            "DepartTime": "a departure time",
            "ArriveTime": "an arrival time",
            "Mode": "foot, vehicle, vessel, or air",
            "Source": "who observed or reported the movement",
            "FlaggedBy": "a named watch-list authority or agency",
            "Reason": "a brief reason for the flag",
            "Confidence": "a confidence level or estimate",
            "Date": "a date",
            "Subject": "a named person or entity",
            "RiskLevel": "low, medium, high, or critical",
            "Basis": "the reasoning behind the rating",
            "Analyst": "a named analyst or office",
            "Buyer": "a named buyer",
            "Store": "a named store or retailer",
            "Item1": "a purchased item",
            "Item2": "a second purchased item",
            "Item3": "a third purchased item",
            "PaymentCard": "a payment method detail",
        }
        slot_hints = "; ".join(f"{s} ({sample_values.get(s, 'a value')})" for s in required)
        opt_hints  = "; ".join(f"{s} ({sample_values.get(s, 'a value')})" for s in optional)

        if condition == "C":
            style_instruction = (
                "Use plain, direct, formulaic language. State each fact in the "
                "same fixed order every time: who/what, then location, then time, "
                "then any status. Do not vary sentence structure across examples."
            )
        else:
            style_instruction = (
                "Use varied, natural language: vary sentence structure, word choice, "
                "and the order in which facts are mentioned, as a real human reporter "
                "would write across different reports. Some optional slots may be present."
            )

        prompt = f"""Generate a realistic intelligence/maritime report for training an
NL-to-SAR translation system.

Domain: {domain_desc}
Target frame: {frame_key} -- {frame['desc']}
Required slots (must all be clearly present in the report): {slot_hints}
Optional slots (include 0-2 of these if natural): {opt_hints}

{style_instruction}

Write a short report (1-3 sentences) that clearly and unambiguously supports
extracting a complete {frame_key} frame with all required slots filled.

Then write the correct SAR frame as JSON-like key:value pairs, naming exactly
which slots you filled and their values (use short identifier-style values,
e.g. MV_Meridian not "the ship called the Meridian"), required slots first.

Format exactly:
REPORT: <report text>
SLOTS: <slot1>=<value1>; <slot2>=<value2>; ...
FRAME_APPLIES: true
FRAME: {frame_key}
DOMAIN: {domain}
NEGATIVE_TYPE: none"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def parse_case(text, case_id, frame_key):
    """Parse LLM output into a structured case dict with scenario/correct_response."""
    lines = text.strip().split('\n')
    case = {"id": case_id}
    current_key = None
    current_val = []

    keys = ["REPORT", "SLOTS", "CORRECT_RESPONSE", "FRAME_APPLIES",
            "FRAME", "DOMAIN", "NEGATIVE_TYPE"]

    def flush():
        if current_key and current_val:
            val = " ".join(current_val).strip()
            if current_key == "FRAME_APPLIES":
                val = val.lower().startswith("true")
            case[current_key.lower()] = val

    # Re-split the whole text on key markers wherever they appear (not just
    # at line start), so two fields accidentally placed on one physical
    # line are still correctly separated. Sort keys longest-first so
    # "FRAME_APPLIES" is matched before the shorter "FRAME".
    import re as _re
    pattern = r'(?:^|\s)(' + '|'.join(sorted(keys, key=len, reverse=True)) + r'):\s*'
    full_text = "\n".join(lines)
    parts = _re.split(pattern, full_text)
    # parts alternates: [pre-text, key, value, key, value, ...]
    if len(parts) > 1:
        for i in range(1, len(parts) - 1, 2):
            key = parts[i]
            val = parts[i + 1].strip()
            if current_key:
                flush()
            current_key = key
            current_val = [val]
            flush()
            current_key, current_val = None, []
    else:
        # Fallback to original line-based parsing if no keys found at all
        for line in lines:
            matched = False
            for key in keys:
                if line.startswith(f"{key}:"):
                    flush()
                    current_key = key
                    current_val = [line[len(key)+1:].strip()]
                    matched = True
                    break
            if not matched and current_key:
                current_val.append(line.strip())
        flush()

    # Build scenario (the report text) and correct_response (the target SAR
    # or INCOMPLETE message) used by build_models.py / evaluate.py.
    case["scenario"] = case.get("report", text[:400].strip())

    if case.get("frame_applies", True) and "slots" in case:
        # Parse "slot1=val1; slot2=val2" into the canonical SAR string
        slot_pairs = {}
        for piece in case["slots"].split(";"):
            piece = piece.strip()
            if "=" in piece:
                k, v = piece.split("=", 1)
                slot_pairs[k.strip()] = v.strip()
        case["correct_response"] = render_correct_sar(frame_key, slot_pairs)
        case["target_slots"] = slot_pairs
    else:
        case["correct_response"] = case.get("correct_response", "INCOMPLETE: unspecified")
        case["target_slots"] = {}

    case.setdefault("frame_applies", True)
    case.setdefault("frame", frame_key)
    case.setdefault("domain", DOMAIN_KEYS[0])
    case.setdefault("negative_type", "none")

    # Rename to match evaluate.py's expected field names (concept/concept_applies)
    case["concept"] = case["frame"]
    case["concept_applies"] = case["frame_applies"]

    return case


def generate_tranche(tranche_num, split, condition, rng, existing_ids=None):
    n_total = TRANCHE_TRAIN if split == 'train' else TRANCHE_TEST
    n_neg   = n_total // 4
    n_pos   = n_total - n_neg

    cases = []
    frame_pool  = FRAME_KEYS.copy()
    domain_pool = DOMAIN_KEYS.copy()

    def next_id():
        return f"T{tranche_num}_{split}_{condition}_{len(cases)+1:03d}"

    for i in range(n_pos):
        frame_key = frame_pool[i % len(frame_pool)]
        domain    = rng.choice(domain_pool)
        seed_i    = rng.randint(0, 9999)
        text = generate_scenario_llm(frame_key, domain, condition, seed=seed_i)
        case = parse_case(text, next_id(), frame_key)
        case["tranche"] = tranche_num
        case["split"]   = split
        case["condition"] = condition
        cases.append(case)

    for j in range(n_neg):
        frame_key  = frame_pool[j % len(frame_pool)]
        neg_types  = NEGATIVES[frame_key]
        neg_type   = neg_types[j % len(neg_types)]
        domain     = rng.choice(domain_pool)
        text = generate_scenario_llm(frame_key, domain, condition,
                                     negative_type=neg_type)
        case = parse_case(text, next_id(), frame_key)
        case["tranche"]   = tranche_num
        case["split"]     = split
        case["condition"] = condition
        cases.append(case)

    return cases


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tranche", type=int, required=True)
    parser.add_argument("--split",   choices=["train","test"], required=True)
    parser.add_argument("--condition", choices=["A","B","C"], required=True)
    parser.add_argument("--out",    required=True)
    parser.add_argument("--existing", default=None)
    args = parser.parse_args()

    base_seed = SEED_TRAIN if args.split == "train" else SEED_TEST
    rng = random.Random(base_seed + args.tranche * 100 + ord(args.condition))

    print(f"Generating tranche {args.tranche} {args.split} condition {args.condition}...")
    cases = generate_tranche(args.tranche, args.split, args.condition, rng)

    with open(args.out, "w") as f:
        json.dump(cases, f, indent=2)

    pos = sum(1 for c in cases if c.get("frame_applies"))
    neg = len(cases) - pos
    print(f"Generated {len(cases)} cases: {pos} positive, {neg} negative")
    print(f"Saved to {args.out}")
