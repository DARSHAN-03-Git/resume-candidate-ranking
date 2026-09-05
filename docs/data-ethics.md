# Data, Privacy, and Evaluation Contract

## Allowed starting data

- Self-created synthetic resumes, marked `synthetic` in metadata and UI.
- An anonymized public resume dataset, with its license and source recorded before use.
- Job descriptions created for testing or supplied by the user with permission.

No private third-party resume is stored or processed without consent.

## PII rules

- Raw uploads stay local during development and are excluded from logs.
- API logs use request IDs and aggregate timings, not resume text.
- Structured records separate contact fields from ranking features.
- UI views mask email addresses, phone numbers, and full street addresses by default.
- Temporary files are deleted after processing unless retention is explicitly enabled.
- Secrets belong in environment variables, never source files or reports.

## Fairness rules

The project will report measured behavior, not a claim that the system is unbiased. A planned test swaps a name or gender-coded term while holding the rest of a resume constant, then compares extracted fields and ranking outputs. The report will state exactly what changed, what did not change, and what this narrow test cannot prove.

Protected attributes will not be used as ranking features. Correlated proxies and distribution shift remain risks that require broader evaluation.

## Honesty rules

- A pretrained model is not described as fine-tuned.
- Metrics are written only after an executable evaluation produces them.
- Small self-made training data is described as small and self-made.
- Explanations and outreach drafts can cite only extracted or explicitly labeled inferred facts.
- Missing evidence is represented as missing, never filled with plausible wording.
