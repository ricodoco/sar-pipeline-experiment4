"""
Experiment 4c: Case generator
Produces accumulated-SAR-set cases: a >>CLUSTER<< of 2-4 frames
(ground truth THREAT or NO_THREAT) embedded among N distractor frames drawn
from HAYSTACK_DENSITIES, mirroring the paper's two worked examples
(pre-attack indicators, maritime cargo incident) at small scale.
"""

import json, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (CONCEPTS, FRAME_KEYS, DOMAINS, DOMAIN_KEYS,
                    SEED_TRAIN, SEED_TEST, TRANCHE_TRAIN, TRANCHE_TEST,
                    HAYSTACK_DENSITIES, DISTRACTOR_FRAME_TYPES,
                    DISTRACTOR_FIRST_NAMES, DISTRACTOR_LAST_NAMES,
                    DISTRACTOR_VESSELS, DISTRACTOR_PORTS, DISTRACTOR_ITEMS,
                    DISTRACTOR_TEMPLATES)

import anthropic
client = anthropic.Anthropic()


def make_distractor_frame(rng, idx):
    template = rng.choice(DISTRACTOR_TEMPLATES)
    name = f"{rng.choice(DISTRACTOR_FIRST_NAMES)}_{rng.choice(DISTRACTOR_LAST_NAMES)}"
    vessel = rng.choice(DISTRACTOR_VESSELS).replace(" ", "_")
    port = rng.choice(DISTRACTOR_PORTS)
    n = idx
    return template.format(name=name, vessel=vessel, port=port, n=n)


def generate_cluster_llm(scenario_key, domain, condition, seed=None):
    """Generate the 2-4 >>CLUSTER<< SAR frames + ground truth using
    Claude as the sampler."""
    concept = CONCEPTS[scenario_key]
    domain_desc = DOMAINS[domain]
    ground_truth = concept["ground_truth"]

    if condition == "C":
        style_instruction = (
            "Use a fixed, formulaic slot order and consistent terse phrasing "
            "across all frames in the cluster."
        )
    else:
        style_instruction = (
            "Use varied frame types and natural variation in how slots are "
            "named and filled across the cluster, as real accumulated "
            "multi-source reporting would look."
        )

    prompt = f"""Generate 2-4 SAR (Semantic Assertion Record) frames forming
a single evidentiary cluster for a threat-detection test.

Domain: {domain_desc}
Pattern type: {concept['desc']}
Ground truth: this cluster of frames, taken JOINTLY, {"DOES" if ground_truth == "THREAT" else "DOES NOT"} support a genuine threat hypothesis.

{style_instruction}

Each frame should use the form: {{ FrameName: <Type>, Slot1: value1, Slot2: value2, ... }}
Use short identifier-style values (e.g. Atta_M, MV_Meridian), not full sentences.

{"If THREAT: design the cluster so that NO SINGLE frame alone is alarming, but the JOINT co-occurrence across all frames in the cluster is a strong, specific threat indicator (mirroring: partial flight training avoiding landing + extremist association + fuel quantity query + visa anomaly, OR: tampered cargo seal + missing crewman matched to flagged alias + precursor purchase)." if ground_truth == "THREAT" else "If NO_THREAT: design the cluster so individual frames may look superficially concerning (foreign national, cargo discrepancy, unusual purchase) but the joint pattern has an innocuous explanation and lacks the specific disqualifying co-occurrence that would indicate genuine threat."}

Then write a one-sentence HYPOTHESIS (if THREAT) and EVIDENCE explaining the
joint reasoning.

Format exactly:
FRAME_1: <frame>
FRAME_2: <frame>
FRAME_3: <frame, or 'none' if only 2 frames>
FRAME_4: <frame, or 'none' if only 3 frames>
GROUND_TRUTH: {ground_truth}
HYPOTHESIS: <one sentence, or 'none' if NO_THREAT>
EVIDENCE: <one sentence joint-reasoning explanation>
SCENARIO: {scenario_key}
DOMAIN: {domain}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def parse_cluster(text, case_id, scenario_key):
    lines = text.strip().split('\n')
    case = {"id": case_id}
    current_key = None
    current_val = []
    keys = ["FRAME_1", "FRAME_2", "FRAME_3", "FRAME_4", "GROUND_TRUTH",
            "HYPOTHESIS", "EVIDENCE", "SCENARIO", "DOMAIN"]

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

    frames = []
    for k in ["frame_1", "frame_2", "frame_3", "frame_4"]:
        v = case.get(k, "none").strip()
        if v and v.lower() != "none" and v.startswith("{"):
            frames.append(v)
    case["cluster_frames"] = frames
    case.setdefault("ground_truth", CONCEPTS[scenario_key]["ground_truth"])
    case.setdefault("hypothesis", "none")
    case.setdefault("evidence", "")
    case.setdefault("scenario", scenario_key)
    return case


def build_full_case(cluster_case, haystack_density, rng, case_id, condition, domain):
    """Embed the relevant cluster among `haystack_density` distractor SAR
    frames, shuffled, with the cluster tagged >>CLUSTER<<."""
    tagged_cluster = [">>CLUSTER<< " + f for f in cluster_case["cluster_frames"]]
    distractors = [make_distractor_frame(rng, i) for i in range(haystack_density)]
    all_frames = distractors + tagged_cluster
    rng.shuffle(all_frames)

    scenario_text = "\n".join(f"FRAME: {fr}" for fr in all_frames)

    ground_truth = cluster_case["ground_truth"]
    frame_names = []
    for f in cluster_case["cluster_frames"]:
        if "FrameName:" in f:
            fn = f.split("FrameName:")[1].split(",")[0].strip()
            frame_names.append(fn)

    if ground_truth == "THREAT":
        correct_response = (
            f"DECISION: THREAT\nHYPOTHESIS: {cluster_case.get('hypothesis','')}\n"
            f"SUPPORTING_FRAMES: {', '.join(frame_names)}\n"
            f"EVIDENCE: {cluster_case.get('evidence','')}"
        )
    else:
        correct_response = (
            f"DECISION: NO_THREAT\nHYPOTHESIS: none\n"
            f"SUPPORTING_FRAMES: {', '.join(frame_names)}\n"
            f"EVIDENCE: {cluster_case.get('evidence','')}"
        )

    return {
        "id": case_id,
        "scenario": scenario_text,
        "correct_response": correct_response,
        "ground_truth": ground_truth,
        "haystack_density": haystack_density,
        "frame": cluster_case["scenario"],
        "domain": domain,
        "condition": condition,
        "supporting_frame_names": frame_names,
        # alias fields for scaffold compatibility
        "concept": cluster_case["scenario"],
        "concept_applies": (ground_truth == "THREAT"),
        "frame_applies": (ground_truth == "THREAT"),
        "negative_type": "none" if ground_truth == "THREAT" else "superficial_concern_no_joint_pattern",
    }


def generate_tranche(tranche_num, split, condition, rng):
    n_total = TRANCHE_TRAIN if split == 'train' else TRANCHE_TEST
    scenario_pool = FRAME_KEYS.copy()
    domain_pool = DOMAIN_KEYS.copy()
    density_pool = HAYSTACK_DENSITIES.copy()

    cases = []
    for i in range(n_total):
        scenario_key = scenario_pool[i % len(scenario_pool)]
        domain = rng.choice(domain_pool)
        density = density_pool[i % len(density_pool)]
        seed_i = rng.randint(0, 9999)

        text = generate_cluster_llm(scenario_key, domain, condition, seed=seed_i)
        cluster_case = parse_cluster(text, f"T{tranche_num}_{split}_{condition}_{i+1:03d}_cluster", scenario_key)
        full_case = build_full_case(cluster_case, density, rng,
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

    threat = sum(1 for c in cases if c["ground_truth"] == "THREAT")
    no_threat = len(cases) - threat
    densities = sorted(set(c["haystack_density"] for c in cases))
    print(f"Generated {len(cases)} cases: {threat} THREAT, {no_threat} NO_THREAT")
    print(f"Haystack densities used: {densities}")
    print(f"Saved to {args.out}")
