#!/usr/bin/env python3
"""
Experiment 4b: Auto-running experiment loop.
Usage: python run_experiment.py [--from-tranche N]

Runs tranches automatically until stopping criterion is met
(d-prime >= 2.0 and accuracy >= 0.90 at the hardest haystack density,
for condition B, on two consecutive tranches) or MAX_TRANCHES is reached.
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


def get_p_hat(condition, up_to_tranche):
    """For 4b: compute d-prime and accuracy at the HIGHEST haystack density
    for a condition, using only results from tranche `up_to_tranche` (not
    cumulative), to match the per-tranche consecutive-criterion check."""
    path = os.path.join(DATA_DIR, f"results_cond{condition}_T{up_to_tranche}.json")
    if not os.path.exists(path):
        return None, 0, 0
    with open(path) as f:
        results = json.load(f)
    if not results:
        return None, 0, 0
    max_density = max(r.get("haystack_density", 0) for r in results)
    hard = [r for r in results if r.get("haystack_density", 0) == max_density]
    n = len(hard)
    hits = sum(1 for r in hard if r["sdt_class"] == "hit")
    crs  = sum(1 for r in hard if r["sdt_class"] == "correct_rejection")
    misses = sum(1 for r in hard if r["sdt_class"] == "miss")
    fas    = sum(1 for r in hard if r["sdt_class"] == "false_alarm")
    n_merge = hits + misses
    n_nomerge = fas + crs
    hr = hits / n_merge if n_merge else 0.0
    far = fas / n_nomerge if n_nomerge else 0.0
    dp = d_prime(hr, far)
    acc = (hits + crs) / n if n else 0.0
    return acc, dp, n


def criterion_met(tranche):
    """Check if condition B reaches d-prime >= 2.0 AND accuracy >= 0.90 at
    the hardest haystack density, on this tranche AND the previous tranche."""
    if tranche < 2:
        return False
    acc1, dp1, n1 = get_p_hat("B", tranche - 1)
    acc2, dp2, n2 = get_p_hat("B", tranche)
    if acc1 is None or acc2 is None:
        return False
    return (dp1 >= 2.0 and acc1 >= 0.90 and
            dp2 >= 2.0 and acc2 >= 0.90)


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
    print("EXPERIMENT SUMMARY (hardest haystack density this tranche)")
    print(f"{'='*60}")
    for cond in ("A", "B", "C"):
        acc, dp, n = get_p_hat(cond, up_to_tranche)
        if acc is None:
            continue
        print(f"  Condition {cond}: n={n}  accuracy={acc:.3f}  d-prime={dp:.3f}")


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
            print("Condition B achieved d-prime >= 2.0 and accuracy >= 0.90")
            print("at the hardest haystack density, on two consecutive tranches.")
            break
        elif t == MAX_TRANCHES:
            print(f"\nReached maximum of {MAX_TRANCHES} tranches without meeting criterion.")
            print("Review learning curves and discuss next steps.")
        else:
            acc, dp, _ = get_p_hat("B", t)
            print(f"\nCriterion not yet met (B accuracy={acc:.3f}, d-prime={dp:.3f}). "
                  f"Running tranche {t+1}...")
