# Experiment 4a: Natural Language to SAR Translation

**Research question:** Can a small local LLM be trained, from a small
number of examples, to translate natural-language intelligence/maritime
reports into well-formed Semantic Assertion Records (SARs) -- the named
n-ary case frames proposed in the MIEM/LLM paper -- including correctly
declining to produce a frame when the report is incomplete or ambiguous?

This directly tests **Claim 1** of the paper (Section 2.2): *"A first LLM
pass translates each natural language input into one or more Semantic
Assertion Records (SARs)."*

## Design

Reuses the Experiment 3 tranche/build/evaluate scaffold unchanged in
structure; only `config.py`, `generate.py`, and `evaluate.py` are adapted
to the SAR-translation task.

| | |
|---|---|
| Model | llama3.2 (3B, local via Ollama) |
| Conditions | A (control, no SAR examples), B (varied phrasing), C (templated/repetitive phrasing) |
| SAR frame types | CargoLoading, Movement, FlaggedEntity, RiskAssessment, RetailPurchase (from the paper's Section 3 vocabulary) |
| Domains | customs, coastguard, intel, retail_sec, port_auth, interpol |
| Tranche size | 20 training cases (15 pos / 5 neg) + 20 test cases |
| Max tranches | 5 (100 training, 100 test cases) |
| Stopping rule | P_hat >= 0.95 on two consecutive tranches |
| Judge | Claude Haiku, double-rated, two dimensions: correct structure, no fabrication |

**Condition B** trains on reports with varied natural phrasing, varied slot
order, and 0-2 optional slots present, mirroring how real operators would
actually write reports.

**Condition C** trains on reports with fixed, formulaic phrasing and a
fixed slot presentation order -- the modal-token analog for this task.

**Negative cases (25%):** reports that should NOT yield a complete frame:
wrong_frame, missing_required, ambiguous_entity, non_event, contradictory.
Correct response is either "INCOMPLETE: <reason>" or correctly naming a
different applicable frame.

## Judge dimensions

- **D1 Correct structure:** right FrameName, all required slots present
  with plausible correct values (order-independent), OR correctly declined.
- **D2 No fabrication:** model did not invent slot values absent from the
  report.

## Running

```bash
cd 4a_translation
python run_tranche.py --tranche 1
python run_tranche.py --tranche 2
# ... continue until criterion met or tranche 5 reached
```

Resume / report-only flags are identical to Experiment 3 -- see
`run_tranche.py --help` and `evaluate.py --report-only`.

## Predicted outcome

- Condition B reaches P >= 0.95 within 2-3 tranches (40-60 examples),
  because the model can generalize the slot-naming pattern across varied
  phrasing -- this is the paper's core claim that named slots are learned
  as manifold dimensions, not memorized surface strings.
- Condition C may plateau lower on held-out reports with phrasing that
  diverges from the fixed training template, particularly on negative
  cases requiring judgment about what's missing.
- Condition A (no SAR examples) should perform poorly or inconsistently,
  establishing that the SAR format itself is not "free" -- the model needs
  some exposure to the target structure, even if very little.

## Relationship to the paper

This experiment operationalizes the claim in Section 2.2 that "the LLM is
prompted with these definitions and a few-shot examples; it learns to
produce well-formed SARs from natural language inputs," and tests it as a
falsifiable, measured claim rather than an assertion.
