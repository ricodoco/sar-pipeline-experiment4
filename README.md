# SAR Pipeline Experiment 4: Empirical Evaluation

**Repository for:** "Empirical Evaluation of the SAR Pipeline: NL Translation,
Identity Integration, and Anomaly Detection in a Small Local Language Model"
(Roth, 2026)

**Companion paper (architecture):** "Natural Language Input, Semantic Track
Representation, and LLM Inference: Making the Maritime Information Exchange
Model Tractable" — available at SSRN (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6880161).
**This paper (experimental results):** SSRN abstract 6978398 (https://ssrn.com/abstract=6978398).

**All materials are released to the public domain. No patents. No licensing.**

---

## What This Is

Three sub-experiments that empirically test the three operational claims of
the SAR pipeline architecture:

| Sub-experiment | Claim tested | Key result |
|---|---|---|
| `4a_translation` | LLM translates NL reports to SAR frames | 70% accuracy from 20 examples (templated training) |
| `4b_integration` | LLM resolves entity identity across sources | Pretrained capability; d'=2.33 untrained; cliff above 5 distractors |
| `4c_anomaly` | LLM detects threat patterns and cites frames | d'=2.41, 85% citation quality, 77% joint reasoning (varied training, 60 examples) |

## Requirements

- **Ollama** installed, `llama3.2` pulled: `ollama pull llama3.2`
- **Python 3.9+**
- **anthropic** package: `pip install anthropic`
- **ANTHROPIC_API_KEY** set in environment (used for case generation and judging via Claude Haiku)

## Quick Start

```bash
# Run Experiment 4a (NL to SAR translation)
cd 4a_translation
python run_experiment.py

# Run Experiment 4b (identity integration)
cd ../4b_integration
python run_experiment.py

# Run Experiment 4c (anomaly detection)
cd ../4c_anomaly
python run_experiment.py
```

Each experiment runs up to 5 tranches of 20 cases (100 total train, 100 total
test) and stops early if the stopping criterion is met on two consecutive
tranches. See each sub-directory's README for details.

## Reproduce From Any Tranche

```bash
# Skip generation and model building if already done
python run_tranche.py --tranche N --skip-generate --skip-build

# Report only (no new scoring)
python evaluate.py --tranche N --report-only

# Score specific conditions only
python evaluate.py --tranche N --conditions B,C
```

## Cost

All generation and judging uses Claude Haiku at approximately $0.02 per case.
Full 5-tranche run of all three experiments: approximately $12-15 total.
Local Ollama inference is free.

## Design Details

Each experiment uses three conditions:

- **Condition A** — untrained baseline (no task-specific examples)
- **Condition B** — varied, naturally-phrased training examples
- **Condition C** — templated, formulaic training examples

Experiments 4b and 4c use signal detection theory scoring (d-prime, hit rate,
false alarm rate) with a haystack density sweep (0, 5, 15, 30 distractor
items) to test performance under realistic information-dense conditions.

## Citation

```
Roth, F. (2026). Empirical evaluation of the SAR pipeline: NL translation,
identity integration, and anomaly detection in a small local language model.
SSRN Working Paper. [URL]
```

## License

Public domain. No restrictions on use, modification, or redistribution.
