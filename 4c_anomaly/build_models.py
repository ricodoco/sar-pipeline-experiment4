#!/usr/bin/env python3
"""
Experiment 3: Build Ollama Modelfiles and register models.
Run after each training tranche to update conditions B and C.
Condition A (control) is built once and never updated.
"""

import subprocess, json, os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
from config import MODELFILE_A, MODELFILE_B, MODELFILE_C, MODEL

WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def load_train_cases(condition, up_to_tranche):
    """Load all training cases for a condition up to and including tranche N."""
    cases = []
    for t in range(1, up_to_tranche + 1):
        path = os.path.join(WORK_DIR, "data",
                            f"train_cond{condition}_T{t}.json")
        if os.path.exists(path):
            with open(path) as f:
                cases.extend(json.load(f))
    return cases


def format_training_messages(cases, condition):
    """
    Convert cases to chat message pairs for Modelfile TEMPLATE injection.
    For conditions B and C: scenario -> correct_response pairs.
    For condition A: scenario -> neutral helpful response (no concept invoked).
    """
    messages = []
    for c in cases:
        scenario = c.get("scenario", "")
        response = c.get("correct_response", "")
        if scenario and response:
            messages.append((scenario, response))
    return messages


def build_modelfile(condition, tranche, base_model=MODEL):
    """Build a Modelfile string for the given condition and tranche."""
    if condition == "A":
        template = MODELFILE_A
    elif condition == "B":
        template = MODELFILE_B
    elif condition == "C":
        template = MODELFILE_C
    else:
        raise ValueError(f"Unknown condition: {condition}")

    modelfile = template.format(base_model=base_model)

    # Append training examples as MESSAGE pairs
    if condition in ("B", "C") and tranche > 0:
        cases = load_train_cases(condition, tranche)
        messages = format_training_messages(cases, condition)
        if messages:
            modelfile += "\n# Training examples (tranche 1 through {tranche})\n"
            for scenario, response in messages:
                # Escape quotes for Modelfile syntax
                s = scenario.replace('"', '\\"')
                r = response.replace('"', '\\"')
                modelfile += f'MESSAGE user "{s}"\n'
                modelfile += f'MESSAGE assistant "{r}"\n'

    return modelfile


def register_model(condition, tranche):
    """Write Modelfile and run ollama create."""
    model_name = f"exp4c_cond{condition}_T{tranche}"
    modelfile_path = os.path.join(WORK_DIR, f"Modelfile_{condition}_T{tranche}")

    content = build_modelfile(condition, tranche)
    with open(modelfile_path, "w") as f:
        f.write(content)

    print(f"Creating Ollama model: {model_name}")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR creating {model_name}:\n{result.stderr}")
        return None
    print(f"Created: {model_name}")
    return model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tranche", type=int, required=True,
                        help="Tranche number just completed (0 for initial build)")
    parser.add_argument("--conditions", default="A,B,C",
                        help="Comma-separated conditions to build")
    args = parser.parse_args()

    os.makedirs(os.path.join(WORK_DIR, "data"), exist_ok=True)

    for cond in args.conditions.split(","):
        cond = cond.strip().upper()
        name = register_model(cond, args.tranche)
        if name:
            print(f"  Ready: {name}")
