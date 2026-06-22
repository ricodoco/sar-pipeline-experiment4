#!/usr/bin/env python3
"""
Experiment 4a: Evaluation
Run test reports against Ollama models, score SAR correctness with Claude
Haiku, report results. Reuses Experiment 3's tranche/Wilson-LB machinery.
"""

import json, os, sys, argparse, math, subprocess, re, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from config import CONCEPTS

import anthropic
client = anthropic.Anthropic()

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
OLLAMA_URL = "http://localhost:11434/api/generate"


def query_ollama(model_name, prompt, timeout=90, retries=2):
    """Query Ollama via REST API (not CLI) to avoid terminal escape code
    corruption that occurs when using 'ollama run' with subprocess."""
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }).encode()

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                return data.get("response", "").strip()
        except Exception as e:
            if attempt < retries:
                print(f"retry ({attempt+1}/{retries})...", end=" ")
            else:
                return f"ERROR: {str(e)[:200]}"
    return "ERROR: max retries exceeded"


JUDGE_PROMPT = """You are evaluating an AI system's translation of a natural
language report into a Semantic Assertion Record (SAR) -- a named case frame
with named slot:value pairs.

TARGET FRAME: {frame_name}
FRAME REQUIRED SLOTS: {required_slots}
REPORT: {scenario}
FRAME_SHOULD_APPLY: {applies}
REFERENCE CORRECT OUTPUT: {correct_response}
ACTUAL MODEL OUTPUT: {actual_response}

Score the actual output on TWO dimensions:

D1_CORRECT_STRUCTURE: 
- If FRAME_SHOULD_APPLY=true: did the model output a frame with the correct
  FrameName ({frame_name}) AND all required slots ({required_slots}) present
  with plausible values consistent with the report? Order of slots does not
  matter. Minor wording differences in values are fine if the meaning matches.
  YES/NO.
- If FRAME_SHOULD_APPLY=false: did the model correctly decline to produce a
  complete frame (output something like "INCOMPLETE" or correctly identify
  a different frame applies), rather than fabricating values for missing
  information? YES/NO.

D2_NO_FABRICATION: Did the model avoid inventing specific factual details
that are neither stated nor clearly implied in the report? Score YES if the
model's slot values are all grounded in or reasonably inferred from the
report text. Score NO only for clear fabrications: specific names, dates,
numbers, or facts that have no basis in the report at all.

IMPORTANT -- these are NOT fabrication and should score YES on D2:
- Reasonable format normalization (e.g. "14:30" for a reported time of
  "1430 hours", or "65%" for "moderate confidence around 65 percent")
- Minor abbreviation or identifier-style reformatting of values
- Selecting one end of a reported range for a slot value when the slot
  expects a single value (e.g. using 1825 from "between 1815 and 1825")
- Omitting optional slots that were not mentioned in the report

These ARE fabrication and should score NO on D2:
- Inventing a specific person's name not mentioned in the report
- Inventing a specific date, time, or number with no basis in the report
- Adding slot values for facts that contradict or go beyond the report

OVERALL: PASS if both D1 and D2 are YES. FAIL otherwise.

Respond in this exact format:
D1_CORRECT_STRUCTURE: YES/NO
D2_NO_FABRICATION: YES/NO
OVERALL: PASS/FAIL
NOTES: <one sentence>"""


def judge_response(case, actual_response):
    frame_key = case.get("frame", case.get("concept", "CargoLoading"))
    frame = CONCEPTS.get(frame_key, {})
    applies = case.get("frame_applies", case.get("concept_applies", True))

    prompt = JUDGE_PROMPT.format(
        frame_name=frame_key,
        required_slots=", ".join(frame.get("required", [])),
        scenario=case.get("scenario", ""),
        applies=applies,
        correct_response=case.get("correct_response", ""),
        actual_response=actual_response,
    )

    scores = []
    for _ in range(2):
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text
        overall = "PASS" if "OVERALL: PASS" in text else "FAIL"
        d1 = "YES" in text.split("D1_CORRECT_STRUCTURE:")[-1].split("\n")[0]
        d2 = "YES" in text.split("D2_NO_FABRICATION:")[-1].split("\n")[0]
        scores.append({"d1": d1, "d2": d2, "overall": overall, "raw": text})

    agreed = scores[0]["overall"] == scores[1]["overall"]
    final  = scores[0]["overall"] if agreed else "DISAGREEMENT"
    return {
        "rating_1": scores[0],
        "rating_2": scores[1],
        "agreed":   agreed,
        "final":    final,
    }


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
        model_name = f"exp4a_cond{cond}_T{tranche}"
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
            print(f"  Case {i+1}/{len(cases)}: {case['id']}...", end=" ")
            scenario = case.get("scenario", "").strip()
            if not scenario or len(scenario) < 10:
                print("SKIP (missing scenario)")
                continue
            actual = query_ollama(model_name, scenario)
            judgment = judge_response(case, actual)
            record = {
                "case_id":         case["id"],
                "frame":           case.get("frame"),
                "domain":          case.get("domain"),
                "frame_applies":   case.get("frame_applies"),
                "negative_type":   case.get("negative_type"),
                "actual_response": actual,
                "judgment":        judgment,
                "pass":            judgment["final"] == "PASS",
            }
            cond_results.append(record)
            status = "PASS" if record["pass"] else ("DISAGREE" if judgment["final"] == "DISAGREEMENT" else "FAIL")
            print(status)

        results[cond] = cond_results
        out_path = os.path.join(WORK_DIR, "data",
                                f"results_cond{cond}_T{tranche}.json")
        with open(out_path, "w") as f:
            json.dump(cond_results, f, indent=2)
        print(f"  Saved: {out_path}")

    return results


def report_cumulative(up_to_tranche, conditions=("A","B","C")):
    print(f"\n{'='*60}")
    print(f"CUMULATIVE RESULTS through tranche {up_to_tranche}")
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

        n        = len(all_results)
        passes   = sum(1 for r in all_results if r.get("pass"))
        p_hat    = passes / n
        lb       = wilson_lower(n, passes)
        disagree = sum(1 for r in all_results
                       if r["judgment"]["final"] == "DISAGREEMENT")

        pos      = [r for r in all_results if r.get("frame_applies")]
        neg      = [r for r in all_results if not r.get("frame_applies")]
        pos_pass = sum(1 for r in pos if r.get("pass"))
        neg_pass = sum(1 for r in neg if r.get("pass"))

        by_frame = {}
        for r in all_results:
            fr = r.get("frame", "unknown")
            by_frame.setdefault(fr, []).append(r.get("pass", False))

        stop = "*** CRITERION MET ***" if p_hat >= 0.95 and lb > 0.90 else ""
        print(f"\nCondition {cond}: n={n}, pass={passes}, "
              f"P_hat={p_hat:.3f}, Wilson_LB={lb:.3f}  {stop}")
        if pos:
            print(f"  Positive (frame applies):  {pos_pass}/{len(pos)} ({pos_pass/len(pos)*100:.1f}%)")
        if neg:
            print(f"  Negative (frame declined): {neg_pass}/{len(neg)} ({neg_pass/len(neg)*100:.1f}%)")
        print(f"  Disagreements: {disagree}/{n}")
        print("  By frame type:")
        for fr, passes_list in sorted(by_frame.items()):
            dp = sum(passes_list)
            dn = len(passes_list)
            print(f"    {fr:15s}: {dp}/{dn} ({dp/dn*100:.0f}%)")

    print(f"\n{'─'*60}")
    print("STOPPING RULE: Two consecutive tranches with P_hat >= 0.95")


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
