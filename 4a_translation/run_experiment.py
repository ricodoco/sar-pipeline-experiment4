#!/usr/bin/env python3
"""
Experiment 3: Auto-running experiment loop.
Usage: python run_experiment.py [--from-tranche N]

Runs tranches automatically until stopping criterion is met
(P_hat >= 0.95 for condition B on two consecutive tranches)
or MAX_TRANCHES is reached.
"""

import subprocess, sys, os, json, math, argparse

WORK_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(WORK_DIR, "data")
MAX_TRANCHES = 5


def wilson_lower(n, k, z=1.645):
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2*n)
    spread = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))
    return (center - spread) / denom


def get_p_hat(condition, up_to_tranche):
    """Compute P_hat for a condition across all tranches up to N."""
    all_results = []
    for t in range(1, up_to_tranche + 1):
        path = os.path.join(DATA_DIR, f"results_cond{condition}_T{t}.json")
        if os.path.exists(path):
            with open(path) as f:
                all_results.extend(json.load(f))
    if not all_results:
        return None, 0, 0
    n = len(all_results)
    passes = sum(1 for r in all_results if r.get("pass"))
    return passes / n, passes, n


def criterion_met(tranche):
    """Check if B >= 0.95 on this tranche AND previous tranche."""
    if tranche < 2:
        return False
    p1, _, _ = get_p_hat("B", tranche - 1)
    p2, _, _ = get_p_hat("B", tranche)
    if p1 is None or p2 is None:
        return False
    # Use per-tranche pass rate (not cumulative) for consecutive check
    def tranche_p(t):
        path = os.path.join(DATA_DIR, f"results_condB_T{t}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            results = json.load(f)
        passes = sum(1 for r in results if r.get("pass"))
        return passes / len(results) if results else None
    tp1 = tranche_p(tranche - 1)
    tp2 = tranche_p(tranche)
    return tp1 is not None and tp2 is not None and tp1 >= 0.95 and tp2 >= 0.95


def run_tranche(t):
    print(f"\n{'#'*60}")
    print(f"# TRANCHE {t}")
    print(f"{'#'*60}")
    result = subprocess.run(
        [sys.executable, "run_tranche.py", "--tranche", str(t)],
        cwd=WORK_DIR
    )
    if result.returncode != 0:
        print(f"Tranche {t} failed. Stopping.")
        sys.exit(1)


def print_summary(up_to_tranche):
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for cond in ("A", "B", "C"):
        p_hat, passes, n = get_p_hat(cond, up_to_tranche)
        if p_hat is None:
            continue
        lb = wilson_lower(n, passes)
        print(f"  Condition {cond}: {passes}/{n} = {p_hat:.3f}  Wilson LB={lb:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-tranche", type=int, default=1,
                        help="Start from this tranche (default 1)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    consecutive_met = 0

    for t in range(args.from_tranche, MAX_TRANCHES + 1):
        run_tranche(t)
        print_summary(t)

        if criterion_met(t):
            print(f"\n*** STOPPING CRITERION MET at tranche {t} ***")
            print("Condition B achieved P_hat >= 0.95 on two consecutive tranches.")
            break
        elif t == MAX_TRANCHES:
            print(f"\nReached maximum of {MAX_TRANCHES} tranches without meeting criterion.")
            print("Review learning curves and discuss next steps.")
        else:
            p_hat, _, _ = get_p_hat("B", t)
            print(f"\nCriterion not yet met (B P_hat={p_hat:.3f}). Running tranche {t+1}...")
