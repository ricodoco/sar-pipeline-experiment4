#!/usr/bin/env python3
"""
Experiment 4c: Master tranche runner.
Usage: python run_tranche.py --tranche N

For each tranche N:
1. Generate training cases (conditions B and C)
2. Build/update Ollama models
3. Generate test cases (all conditions)
4. Evaluate test cases
5. Report cumulative results and stopping check
"""

import subprocess, sys, os, argparse

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(WORK_DIR, "data")


def run(cmd, desc):
    print(f"\n── {desc} ──")
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=WORK_DIR
    )
    if result.returncode != 0:
        print(f"FAILED: {desc}")
        sys.exit(1)


def data_path(name):
    return os.path.join(DATA_DIR, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tranche", type=int, required=True)
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip generation (cases already exist)")
    parser.add_argument("--skip-build",    action="store_true",
                        help="Skip model build (models already exist)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    T = args.tranche

    # Collect existing case IDs for dedup
    existing_train = []
    for t in range(1, T):
        for cond in ("B", "C"):
            p = data_path(f"train_cond{cond}_T{t}.json")
            if os.path.exists(p):
                existing_train.append(p)

    existing_test = []
    for t in range(1, T):
        for cond in ("B", "C"):
            p = data_path(f"test_cond{cond}_T{t}.json")
            if os.path.exists(p):
                existing_test.append(p)

    if not args.skip_generate:
        # Generate training cases for B and C
        for cond in ("B", "C"):
            out = data_path(f"train_cond{cond}_T{T}.json")
            cmd = [
                "generate.py",
                "--tranche",   str(T),
                "--split",     "train",
                "--condition", cond,
                "--out",       out,
            ]
            if existing_train:
                # Pass first existing file for ID dedup (simplified)
                cmd += ["--existing", existing_train[0]]
            run(cmd, f"Generate train cases: condition {cond}, tranche {T}")

        # Generate test cases for A, B, C
        for cond in ("B", "C"):
            out = data_path(f"test_cond{cond}_T{T}.json")
            cmd = [
                "generate.py",
                "--tranche",   str(T),
                "--split",     "test",
                "--condition", cond,
                "--out",       out,
            ]
            run(cmd, f"Generate test cases: condition {cond}, tranche {T}")

    if not args.skip_build:
        # Build models (A once at T=1, B and C updated every tranche)
        if T == 1:
            run(["build_models.py", "--tranche", str(T), "--conditions", "B,C"],
                "Build all models at tranche 1")
        else:
            run(["build_models.py", "--tranche", str(T), "--conditions", "B,C"],
                f"Update models B and C at tranche {T}")

    # Evaluate
    run(["evaluate.py", "--tranche", str(T)],
        f"Evaluate tranche {T}")

    print(f"\n{'='*60}")
    print(f"Tranche {T} complete.")
    print(f"Check d-prime and accuracy at the hardest haystack density above.")
    print(f"If d-prime >= 2.0 and accuracy >= 0.90 for two consecutive tranches, criterion met.")
    print(f"Otherwise run: python run_tranche.py --tranche {T+1}")
    print(f"{'='*60}")
