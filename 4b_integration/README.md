# Experiment 4b: Multi-Report Identity Integration (Signal Detection)

**Research question:** Given multiple reports describing entities under
varied names, aliases, or partial descriptions, embedded among many
irrelevant distractor reports, can a small local LLM correctly decide
which pairs of references describe the SAME real-world entity (merge) and
which describe DIFFERENT entities (do not merge) -- without being misled
by superficial similarity (same nationality, similar name) or distracted
by haystack clutter?

This directly tests **Claim 2** of the MIEM/LLM paper (Section 2.3):
probabilistic entity identity resolution across multi-source reports.

## Why signal detection theory, not pass/fail

A **false merge** (incorrectly binding two different entities into one) and
a **missed merge** (incorrectly failing to bind the same entity together)
are the same underlying error class: the analyst ends up with the wrong
picture of who is who. There is no principled reason to weight one as worse
than the other a priori. This experiment scores every case as one of:

- **Hit** -- ground truth MERGE, model said MERGE
- **Miss** -- ground truth MERGE, model said NO_MERGE
- **False alarm** -- ground truth NO_MERGE, model said MERGE
- **Correct rejection** -- ground truth NO_MERGE, model said NO_MERGE

and reports **d-prime** (sensitivity, independent of the model's bias
toward merging or splitting) alongside raw accuracy.

## The haystack manipulation

Each test case embeds exactly one [CANDIDATE PAIR] of reports (whose
ground-truth merge status is known) among a swept number of distractor
reports about unrelated entities:

```
HAYSTACK_DENSITIES = [0, 5, 15, 30]
```

Performance is reported broken out by density. A system that only works at
density=0 (clean pair, no clutter) does not support the paper's operational
claim; the claim requires sustained d-prime and accuracy as density grows.

## Design

| | |
|---|---|
| Model | llama3.2 (3B, local via Ollama) |
| Conditions | A (control), B (varied phrasing), C (templated phrasing) |
| Scenario types | same_person_alias, same_vessel_variant_spelling (ground truth MERGE); different_person_similar_features, different_vessel_similar_class (ground truth NO_MERGE, designed to tempt false merges) |
| Haystack densities | 0, 5, 15, 30 distractor reports, swept across test cases |
| Tranche size | 20 cases per split |
| Stopping rule | d-prime >= 2.0 AND accuracy >= 0.90 at the HARDEST density, two consecutive tranches |
| Judge | Claude Haiku, classifies model's MERGE/NO_MERGE decision |

**NO_MERGE scenarios are deliberately adversarial**: they share superficial
similarity (nationality, name, vessel class) precisely to test whether the
model over-merges on weak cues -- the false-alarm side of the ledger.

## Running

```bash
cd 4b_integration
python run_tranche.py --tranche 1
python run_tranche.py --tranche 2
```

## Predicted outcome

- Condition B should sustain d-prime >= 2.0 even at density=30, because the
  model has learned to require a specific, non-superficial linking detail
  rather than pattern-matching on surface similarity.
- Condition C may show degraded d-prime at high density: rigid pattern
  matching on a fixed template is more easily confused when distractors
  share the same templated surface features.
- Condition A should show near-chance d-prime (~0), establishing that
  some exposure to the merge-judgment task is necessary.

## Relationship to the paper

Operationalizes Section 2.3: "the LLM is asked whether two entity
descriptions likely refer to the same individual...Uncertain identity is
represented explicitly as a probabilistic merge rather than a forced
binary decision." The "needle in haystack" requirement reflects the
operational reality that real intelligence fusion never presents only the
relevant evidence in isolation.
