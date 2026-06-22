"""
Experiment 4b: Case generator
Produces multi-report cases: one [CANDIDATE PAIR] (ground truth MERGE or
NO_MERGE) embedded among N distractor reports drawn from HAYSTACK_DENSITIES.
"""

import json, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (CONCEPTS, FRAME_KEYS, DOMAINS, DOMAIN_KEYS,
                    SEED_TRAIN, SEED_TEST, TRANCHE_TRAIN, TRANCHE_TEST,
                    HAYSTACK_DENSITIES, DISTRACTOR_FIRST_NAMES,
                    DISTRACTOR_LAST_NAMES, DISTRACTOR_VESSELS,
                    DISTRACTOR_PORTS, DISTRACTOR_TEMPLATES, DISTRACTOR_ITEMS)

import anthropic
client = anthropic.Anthropic()


def make_distractor(rng):
    """Generate one short, plausible, irrelevant distractor report."""
    template = rng.choice(DISTRACTOR_TEMPLATES)
    name = f"{rng.choice(DISTRACTOR_FIRST_NAMES)} {rng.choice(DISTRACTOR_LAST_NAMES)}"
    vessel = rng.choice(DISTRACTOR_VESSELS)
    port = rng.choice(DISTRACTOR_PORTS)
    item = rng.choice(DISTRACTOR_ITEMS)
    n = rng.randint(12, 28)
    return template.format(name=name, vessel=vessel, port=port, item=item, n=n)


def generate_pair_llm(scenario_key, domain, condition, seed=None):
    """
    Generate the two CANDIDATE PAIR reports (and ground truth) using Claude
    as the sampler. scenario_key determines whether ground truth is MERGE
    or NO_MERGE.
    """
    concept = CONCEPTS[scenario_key]
    domain_desc = DOMAINS[domain]
    ground_truth = concept["ground_truth"]

    if condition == "C":
        style_instruction = (
            "Use plain, formulaic, near-identical phrasing for any shared "
            "identifying details across the two reports -- minimize lexical "
            "variation."
        )
    else:
        style_instruction = (
            "Use varied, natural phrasing: the two reports should come from "
            "different sources with different writing styles, different "
            "partial details, and realistic gaps -- as real cross-agency "
            "reports would."
        )

    prompt = f"""Generate TWO short intelligence/maritime reports (1-2
sentences each) for an entity identity resolution test.

Domain: {domain_desc}
Scenario type: {concept['desc']}
Ground truth: the two reports {"DO" if ground_truth == "MERGE" else "DO NOT"} describe the same real-world entity.

{style_instruction}

{"If MERGE: include ONE specific, non-superficial linking detail across the two reports (an explicit alias statement, a matching partial document/registry number, a distinctive unusual fact mentioned in both) that justifies merging them. Do not rely on name match alone -- vary surface form (alias, role reference, partial name, descriptive reference)." if ground_truth == "MERGE" else "If NO_MERGE: the two reports should share superficial similarity (same nationality, similar name, same vessel class/flag) that could tempt an incorrect merge, but must NOT share any specific distinguishing linking detail."}

Format exactly:
REPORT_1: <report text>
REPORT_2: <report text>
GROUND_TRUTH: {ground_truth}
LINKING_DETAIL: <the specific detail that justifies the ground truth decision, or 'none, superficial similarity only' if NO_MERGE>
SCENARIO: {scenario_key}
DOMAIN: {domain}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def parse_pair(text, case_id, scenario_key):
    lines = text.strip().split('\n')
    case = {"id": case_id}
    current_key = None
    current_val = []
    keys = ["REPORT_1", "REPORT_2", "GROUND_TRUTH", "LINKING_DETAIL",
            "SCENARIO", "DOMAIN"]

    for line in lines:
        matched = False
        for key in keys:
            if line.startswith(f"{key}:"):
                if current_key and current_val:
                    case[current_key.lower()] = " ".join(current_val).strip()
                current_key = key
                current_val = [line[len(key)+1:].strip()]
                matched = True
                break
        if not matched and current_key:
            current_val.append(line.strip())
    if current_key and current_val:
        case[current_key.lower()] = " ".join(current_val).strip()

    case.setdefault("report_1", "")
    case.setdefault("report_2", "")
    case.setdefault("ground_truth", CONCEPTS[scenario_key]["ground_truth"])
    case.setdefault("linking_detail", "")
    case.setdefault("scenario", scenario_key)
    return case


def build_full_case(pair_case, haystack_density, rng, case_id, condition, domain):
    """Embed the candidate pair among `haystack_density` distractor reports,
    shuffled, with the pair tagged [CANDIDATE PAIR]."""
    r1 = ">>CANDIDATE<< " + pair_case["report_1"]
    r2 = ">>CANDIDATE<< " + pair_case["report_2"]

    distractors = [make_distractor(rng) for _ in range(haystack_density)]
    all_reports = distractors + [r1, r2]
    rng.shuffle(all_reports)

    scenario_text = "\n\n".join(f"REPORT: {r}" for r in all_reports)

    ground_truth = pair_case["ground_truth"]
    if ground_truth == "MERGE":
        correct_response = (
            f"DECISION: MERGE\nENTITY: <unified identity>\n"
            f"EVIDENCE: {pair_case.get('linking_detail', '')}"
        )
    else:
        correct_response = (
            "DECISION: NO_MERGE\nENTITY: N/A\n"
            "EVIDENCE: superficial similarity only, no distinguishing linking detail"
        )

    return {
        "id": case_id,
        "scenario": scenario_text,
        "correct_response": correct_response,
        "ground_truth": ground_truth,
        "haystack_density": haystack_density,
        "frame": pair_case["scenario"],
        "domain": domain,
        "condition": condition,
        # alias fields for evaluate.py / scaffold compatibility
        "concept": pair_case["scenario"],
        "concept_applies": (ground_truth == "MERGE"),
        "frame_applies": (ground_truth == "MERGE"),
        "negative_type": "none" if ground_truth == "MERGE" else "no_merge_distractor_similarity",
    }


def generate_tranche(tranche_num, split, condition, rng):
    n_total = TRANCHE_TRAIN if split == 'train' else TRANCHE_TEST
    # Half MERGE, half NO_MERGE ground truth, swept across haystack densities
    scenario_pool = FRAME_KEYS.copy()
    domain_pool = DOMAIN_KEYS.copy()
    density_pool = HAYSTACK_DENSITIES.copy()

    cases = []
    for i in range(n_total):
        scenario_key = scenario_pool[i % len(scenario_pool)]
        domain = rng.choice(domain_pool)
        density = density_pool[i % len(density_pool)]
        seed_i = rng.randint(0, 9999)

        text = generate_pair_llm(scenario_key, domain, condition, seed=seed_i)
        pair_case = parse_pair(text, f"T{tranche_num}_{split}_{condition}_{i+1:03d}_pair", scenario_key)
        full_case = build_full_case(pair_case, density, rng,
                                    f"T{tranche_num}_{split}_{condition}_{i+1:03d}",
                                    condition, domain)
        cases.append(full_case)

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

    merge = sum(1 for c in cases if c["ground_truth"] == "MERGE")
    no_merge = len(cases) - merge
    densities = sorted(set(c["haystack_density"] for c in cases))
    print(f"Generated {len(cases)} cases: {merge} MERGE, {no_merge} NO_MERGE")
    print(f"Haystack densities used: {densities}")
    print(f"Saved to {args.out}")
