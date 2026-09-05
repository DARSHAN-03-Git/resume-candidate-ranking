# Phase 3: Baseline Structured Extraction

## Concept

Document extraction produces text; structured extraction turns that text into fields that downstream ranking can inspect. The baseline uses transparent regular expressions and section boundaries. This is intentionally easy to audit and will later provide weak labels for NER experiments.

## Basic implementation

`baseline_extractor.py` emits a versioned JSON-compatible record containing candidate contact fields, experience, education, explicit skills, inferred skills, evidence, and warnings. It never populates inferred skills and never invents a missing value.

Run:

```powershell
.\.venv\Scripts\python.exe scripts/phase3_smoke_test.py
```

Expected output:

```text
Phase 3 smoke test passed: 4 structured resume records created.
```

## Intermediate/advanced evolution

The schema is the contract. A spaCy pipeline can be added behind the same extractor interface for entity candidates, while rules remain a fallback for high-precision fields such as email and phone. Later, the bootstrapped labels will train and evaluate a transformer NER model. Those model results will be reported only after a held-out evaluation.

## Limitations

The baseline recognizes a deliberately small skill vocabulary, simple section headings, and compact date formats. It is not a production parser and has not been evaluated on a public dataset yet. This limitation is recorded rather than hidden behind broad entity guesses.
