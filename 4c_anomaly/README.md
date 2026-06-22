# Experiment 4c: Anomaly Detection and Explanation from Accumulated SARs

**Research question:** Given an accumulated set of SAR frames from
multiple sources -- most of them irrelevant distractors -- can a small
local LLM correctly detect when a small cluster of frames jointly
constitutes a genuine threat pattern, correctly decline to manufacture a
threat from superficially concerning but benign frames, and cite the
SPECIFIC frames supporting its decision?

This directly tests **Claim 3** of the MIEM/LLM paper (Section 2.4):
LLM inference and hypothesis ranking over the accumulated knowledge
graph, with the paper's explicit claim that "the LLM can cite the
specific SAR chains that support each hypothesis."

## Why signal detection theory, not pass/fail

A **false alarm** (flagging a benign pattern as a threat) and a **miss**
(failing to flag a genuine threat) are both "wrong picture" errors with
real operational cost -- alert fatigue in the first case, a missed attack
in the second. Neither is treated as a priori worse; both feed the same
**d-prime** sensitivity statistic used in Experiment 4b.

## The haystack manipulation

Each test case embeds a [RELEVANT CLUSTER] of 2-4 SAR frames (whose joint
ground truth, THREAT or NO_THREAT, is known) among a swept number of
benign distractor frames:

```
HAYSTACK_DENSITIES = [0, 5, 15, 30]
```

THREAT scenarios are modeled directly on the paper's two worked examples
(pre-attack indicator cluster, cargo IED cluster): no single frame alone
is alarming, only the joint co-occurrence. NO_THREAT scenarios are
deliberately adversarial -- individual frames look superficially
concerning (foreign national, cargo discrepancy) but the joint pattern
has an innocuous explanation.

## Design

| | |
|---|---|
| Model | llama3.2 (3B, local via Ollama) |
| Conditions | A (control), B (varied frame phrasing), C (templated frame phrasing) |
| Threat scenarios | preattack_indicator_cluster, cargo_ied_cluster (ground truth THREAT) |
| Benign scenarios | benign_training_pattern, benign_cargo_pattern (ground truth NO_THREAT, superficially concerning) |
| Haystack densities | 0, 5, 15, 30 distractor frames |
| Stopping rule | d-prime >= 2.0 AND accuracy >= 0.90 at the hardest density, two consecutive tranches |
| Judge | Claude Haiku: decision classification + citation quality + reasoning quality |

## Judge dimensions

- **SDT classification:** hit / miss / false alarm / correct rejection
  on the THREAT/NO_THREAT decision.
- **Citation quality:** did the model's cited supporting frames actually
  come from the relevant cluster, not the distractor haystack?
- **Reasoning quality:** did the explanation reflect joint, cross-frame
  reasoning rather than single-frame or superficial pattern matching?

## Running

```bash
cd 4c_anomaly
python run_tranche.py --tranche 1
python run_tranche.py --tranche 2
```

## Predicted outcome

- Condition B should sustain d-prime >= 2.0 at high density and high
  citation quality, because varied training exposes the model to the
  *pattern* of joint co-occurrence rather than memorized frame phrasing.
- Condition C may show degraded citation quality at high density: a
  rigid checklist procedure is easily confused by distractors phrased
  similarly to the fixed template.
- Condition A should show near-chance d-prime, confirming the
  hypothesis-ranking task requires some exposure to worked examples.

## Relationship to the paper

Operationalizes Section 2.4 ("the LLM flags SAR patterns that are
statistically unusual relative to its training distribution") and the
worked-example claim that explanations cite specific supporting SAR
chains rather than functioning as an unexplainable black box. Together
with 4a and 4b, this completes the empirical test of all three
operational claims in the paper's four-stage pipeline.
