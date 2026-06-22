#!/usr/bin/env python3
"""
Experiment 4c: Evaluation
Run haystacked accumulated-SAR-set cases against Ollama models, score
THREAT / NO_THREAT decisions AND explanation quality with Claude Haiku.
Reports using signal detection theory (hit rate, false alarm rate,
d-prime), the same framework as 4b, plus a CITATION_QUALITY dimension
testing the paper's claim that "the LLM can cite the specific SAR chains
that support each hypothesis."
"""

import json, os, sys, argparse, math, subprocess, re, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
OLLAMA_URL = "http://localhost:11434/api/generate"

import anthropic
client = anthropic.Anthropic()

WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def query_ollama(model_name, prompt, timeout=120, retries=2):
    payload = json.dumps({"model": model_name, "prompt": prompt, "stream": False}).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                OLLAMA_URL, data=payload,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode()).get("response", "").strip()
        except Exception as e:
            if attempt < retries:
                print(f"retry ({attempt+1}/{retries})...", end=" ")
            else:
                return f"ERROR: {str(e)[:200]}"
    return "ERROR: max retries exceeded"


JUDGE_PROMPT = """You are evaluating an AI intelligence-fusion assistant's
threat-detection decision over an accumulated set of SAR (Semantic
Assertion Record) frames, most of which are irrelevant distractors. A
>>CLUSTER<< of frames (listed below) jointly determines the ground
truth.

GROUND_TRUTH: {ground_truth}
RELEVANT CLUSTER FRAME NAMES: {supporting_frame_names}
REFERENCE HYPOTHESIS/EVIDENCE: {correct_response}

ACTUAL MODEL OUTPUT: {actual_response}

Determine THREE things:

1. MODEL_DECISION: THREAT, NO_THREAT, or UNCLEAR (from the model's output)

2. CITATION_QUALITY: Did the model's SUPPORTING_FRAMES (or equivalent
   citation) correctly identify frame types from the relevant cluster
   ({supporting_frame_names}) rather than citing distractor frames or
   citing nothing specific? YES/NO. (If ground truth is NO_THREAT, YES
   means the model correctly referenced the relevant cluster frames as
   the basis for its NO_THREAT call, not that it found a threat.)

3. REASONING_QUALITY: Did the model's evidence/explanation reflect JOINT
   reasoning across multiple frames (not just one frame in isolation) and
   avoid being swayed by superficial concern alone? YES/NO.

Respond in this exact format:
MODEL_DECISION: THREAT/NO_THREAT/UNCLEAR
CITATION_QUALITY: YES/NO
REASONING_QUALITY: YES/NO
NOTES: <one sentence>"""


def judge_response(case, actual_response):
    prompt = JUDGE_PROMPT.format(
        ground_truth=case.get("ground_truth", "NO_THREAT"),
        supporting_frame_names=", ".join(case.get("supporting_frame_names", [])),
        correct_response=case.get("correct_response", ""),
        actual_response=actual_response,
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=180,
        messages=[{"role": "user", "content": prompt}]
    )
    text = resp.content[0].text
    decision = "UNCLEAR"
    for d in ["THREAT", "NO_THREAT", "UNCLEAR"]:
        if f"MODEL_DECISION: {d}" in text:
            decision = d
            break
    citation = "YES" in text.split("CITATION_QUALITY:")[-1].split("\n")[0]
    reasoning = "YES" in text.split("REASONING_QUALITY:")[-1].split("\n")[0]
    return {
        "model_decision": decision,
        "citation_quality": citation,
        "reasoning_quality": reasoning,
        "raw": text,
    }


def classify_sdt(ground_truth, model_decision):
    if ground_truth == "THREAT":
        return "hit" if model_decision == "THREAT" else "miss"
    else:
        return "false_alarm" if model_decision == "THREAT" else "correct_rejection"


def d_prime(hit_rate, fa_rate):
    def clamp(p):
        return min(max(p, 0.01), 0.99)
    hit_rate = clamp(hit_rate)
    fa_rate  = clamp(fa_rate)
    def inv_norm(p):
        a1,a2,a3 = 2.515517, 0.802853, 0.010328
        b1,b2,b3 = 1.432788, 0.189269, 0.001308
        pp = p if p < 0.5 else 1 - p
        t = math.sqrt(-2.0 * math.log(pp))
        z = t - (a1 + a2*t + a3*t*t) / (1 + b1*t + b2*t*t + b3*t*t*t)
        return -z if p < 0.5 else z
    return inv_norm(hit_rate) - inv_norm(fa_rate)


def wilson_lower(n, k, z=1.645):
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2*n)
    spread = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))
    return (center - spread) / denom


def evaluate_tranche(tranche, conditions=("A","B","C")):
    results = {}
    for cond in conditions:
        model_name = f"exp4c_cond{cond}_T{tranche}"
        test_path  = os.path.join(WORK_DIR, "data",
                                  f"test_cond{cond}_T{tranche}.json")
        if not os.path.exists(test_path):
            print(f"No test file for condition {cond} tranche {tranche}, skipping.")
            continue
        with open(test_path) as f:
            cases = json.load(f)

        print(f"\nEvaluating condition {cond} (model: {model_name}), "
              f"{len(cases)} cases...")

        cond_results = []
        for i, case in enumerate(cases):
            print(f"  Case {i+1}/{len(cases)}: {case['id']} "
                  f"(density={case.get('haystack_density')})...", end=" ")
            scenario = case.get("scenario", "").strip()
            if not scenario or len(scenario) < 10:
                print("SKIP (missing scenario)")
                continue
            actual = query_ollama(model_name, scenario)
            judgment = judge_response(case, actual)
            sdt_class = classify_sdt(case["ground_truth"], judgment["model_decision"])
            record = {
                "case_id":          case["id"],
                "ground_truth":     case["ground_truth"],
                "haystack_density": case.get("haystack_density", 0),
                "frame":            case.get("frame"),
                "domain":           case.get("domain"),
                "actual_response":  actual,
                "judgment":         judgment,
                "sdt_class":        sdt_class,
            }
            cond_results.append(record)
            print(f"{sdt_class.upper()} (cite={judgment['citation_quality']}, "
                  f"reason={judgment['reasoning_quality']})")

        results[cond] = cond_results
        out_path = os.path.join(WORK_DIR, "data",
                                f"results_cond{cond}_T{tranche}.json")
        with open(out_path, "w") as f:
            json.dump(cond_results, f, indent=2)
        print(f"  Saved: {out_path}")

    return results


def report_cumulative(up_to_tranche, conditions=("A","B","C")):
    print(f"\n{'='*60}")
    print(f"CUMULATIVE SIGNAL DETECTION RESULTS through tranche {up_to_tranche}")
    print(f"{'='*60}")

    for cond in conditions:
        all_results = []
        for t in range(1, up_to_tranche + 1):
            path = os.path.join(WORK_DIR, "data",
                                f"results_cond{cond}_T{t}.json")
            if os.path.exists(path):
                with open(path) as f:
                    all_results.extend(json.load(f))
        if not all_results:
            continue

        n  = len(all_results)
        hits   = sum(1 for r in all_results if r["sdt_class"] == "hit")
        misses = sum(1 for r in all_results if r["sdt_class"] == "miss")
        fas    = sum(1 for r in all_results if r["sdt_class"] == "false_alarm")
        crs    = sum(1 for r in all_results if r["sdt_class"] == "correct_rejection")

        n_threat_truth    = hits + misses
        n_nothreat_truth  = fas + crs
        hit_rate = hits / n_threat_truth if n_threat_truth else 0.0
        fa_rate  = fas / n_nothreat_truth if n_nothreat_truth else 0.0
        dprime   = d_prime(hit_rate, fa_rate)
        accuracy = (hits + crs) / n if n else 0.0
        acc_lb   = wilson_lower(n, hits + crs)

        cite_ok = sum(1 for r in all_results if r["judgment"]["citation_quality"])
        reason_ok = sum(1 for r in all_results if r["judgment"]["reasoning_quality"])

        print(f"\nCondition {cond}: n={n}")
        print(f"  Hits={hits}/{n_threat_truth} (HR={hit_rate:.3f})   "
              f"FalseAlarms={fas}/{n_nothreat_truth} (FAR={fa_rate:.3f})")
        print(f"  Misses={misses}   CorrectRejections={crs}")
        print(f"  d-prime = {dprime:.3f}")
        print(f"  Overall accuracy = {accuracy:.3f} (Wilson LB={acc_lb:.3f})")
        print(f"  Citation quality: {cite_ok}/{n} ({cite_ok/n*100:.1f}%)")
        print(f"  Reasoning quality (joint, not superficial): {reason_ok}/{n} ({reason_ok/n*100:.1f}%)")

        by_density = {}
        for r in all_results:
            d = r.get("haystack_density", 0)
            by_density.setdefault(d, []).append(r)
        print(f"  By haystack density (distractor frame count):")
        for density, recs in sorted(by_density.items()):
            dn = len(recs)
            dh = sum(1 for r in recs if r["sdt_class"] == "hit")
            dm = sum(1 for r in recs if r["sdt_class"] == "miss")
            df = sum(1 for r in recs if r["sdt_class"] == "false_alarm")
            dc = sum(1 for r in recs if r["sdt_class"] == "correct_rejection")
            d_threat_truth = dh + dm
            d_nothreat_truth = df + dc
            d_hr = dh / d_threat_truth if d_threat_truth else 0.0
            d_far = df / d_nothreat_truth if d_nothreat_truth else 0.0
            d_dp = d_prime(d_hr, d_far)
            d_acc = (dh + dc) / dn if dn else 0.0
            print(f"    density={density:3d}: n={dn:3d}  acc={d_acc:.3f}  "
                  f"d'={d_dp:.3f}  (HR={d_hr:.2f}, FAR={d_far:.2f})")

    print(f"\n{'─'*60}")
    print("STOPPING RULE: d-prime >= 2.0 AND accuracy >= 0.90 at the highest")
    print("haystack density, sustained for two consecutive tranches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tranche",    type=int, required=True)
    parser.add_argument("--conditions", default="A,B,C")
    parser.add_argument("--report-only",action="store_true")
    args = parser.parse_args()

    conds = tuple(c.strip().upper() for c in args.conditions.split(","))

    if not args.report_only:
        evaluate_tranche(args.tranche, conds)

    report_cumulative(args.tranche, conds)
