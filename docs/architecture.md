# System Architecture

## Plain-language model

The platform turns unstructured hiring documents into a traceable candidate comparison. Parsing answers "what does the document say?" Normalization answers "which canonical skill does that phrase represent?" Retrieval finds plausible candidates quickly. Reranking and weighted scoring compare candidates against the job description. Explainability returns the exact extracted evidence behind each score.

The system must preserve the distinction between:

- **Explicit evidence:** text directly found in a resume.
- **Inferred evidence:** a relationship suggested by the skill graph or rules.
- **Generated phrasing:** optional wording assembled from verified evidence.

Only explicit and clearly labeled inferred evidence may influence a ranking. Generated phrasing never creates evidence.

## Request flow

```text
Recruiter
  -> FastAPI upload and job-description endpoints
  -> Document parser (PDF/DOCX; OCR only when a scanned file is verified)
  -> Structured extraction (rules/spaCy baseline, then fine-tuned NER if evaluation supports it)
  -> PII sanitization and provenance records
  -> Skill normalization and Neo4j-backed relationships
  -> JD requirements extraction
  -> Sentence-transformer embedding
  -> Vector retrieval (FAISS baseline, then persistent Weaviate)
  -> Cross-encoder reranking
  -> Configurable score: semantic + experience + skill overlap
  -> Evidence-grounded explanation, comparison, look-alikes, outreach draft
  -> PostgreSQL persistence and recruiter UI
```

## Ownership boundaries

| Area | Initial owner | Durable record |
|---|---|---|
| Uploads and request state | FastAPI | PostgreSQL |
| Extracted candidate/JD schema | Pydantic service layer | PostgreSQL + source provenance |
| Skill aliases and relationships | Normalization service | Neo4j |
| Candidate vectors | Retrieval service | Weaviate; FAISS during Phase 8 baseline |
| Batch jobs | Celery workers | PostgreSQL job state |
| Ranking evidence | Scoring service | PostgreSQL JSON records |
| UI display | Recruiter web client | API responses only |

## Deliberate architecture adjustments

1. **Weaviate is the selected persistent vector database.** The implementation will confirm local resource usage before enabling it; FAISS remains the educational and low-resource fallback.
2. **OCR is conditional, not a default dependency.** It will be added only if an actual sample resume is scanned. This avoids silently treating OCR output as equally reliable text.
3. **An LLM is optional and downstream.** The first explanation system is deterministic and evidence-grounded. LLM phrasing, if added, receives verified facts only and is never part of extraction or ranking correctness.
4. **The public deployment is intentionally smaller than local production-like infrastructure.** PostgreSQL, Neo4j, RabbitMQ, and Weaviate are demonstrated locally; the public service hosts the core flow only unless a free host can support more without misrepresenting availability.

## Phase gates

Each phase must provide a command, test, or measured artifact before the next phase begins. A phase checkpoint records what works, what is deferred, known limitations, and the exact next approval.
