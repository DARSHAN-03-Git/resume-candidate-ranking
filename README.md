# Resume Parsing and Intelligent Candidate Ranking

A local-first recruiter workflow for parsing resumes, matching candidates to job descriptions, ranking them with evidence, and making the result inspectable.

## Current Status

Phases 1-18 are implemented at baseline level. Phase 20 has a reproducible synthetic evaluation script. Phase 19 public deployment is not claimed because it requires a hosting account and user-controlled credentials. Advanced transformer NER, Sentence Transformer embeddings, FAISS/Weaviate persistence, Neo4j inference storage, and Docker startup require their respective runtime/data verification; no results are fabricated.

## Quick Start

```powershell
.\.venv\Scripts\python.exe scripts/phase2_smoke_test.py
.\.venv\Scripts\python.exe scripts/phase3_smoke_test.py
.\.venv\Scripts\python.exe scripts/phase4_13_smoke_test.py
.\.venv\Scripts\python.exe scripts/phase14_smoke_test.py
.\.venv\Scripts\python.exe scripts/phase20_evaluate.py
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/` for the recruiter UI.

## What This System Does

A recruiter provides a job description and resumes in PDF or DOCX format. The system will:

1. Extract text and preserve document provenance.
2. Convert resumes and job descriptions into structured, auditable JSON.
3. Normalize explicit skills and separately label inferred skills.
4. Retrieve and rerank candidates using semantic and evidence-based signals.
5. Explain every ranking with facts from the source resume.
6. Support look-alike search, comparison, and evidence-grounded outreach drafts.

This is decision support, not an autonomous hiring decision. Recruiters remain responsible for reviewing source evidence and applying consistent hiring policy.

## Architecture

See [docs/architecture.md](docs/architecture.md).

## Data and Ethics Contract

See [docs/data-ethics.md](docs/data-ethics.md). Synthetic data is visibly labeled, third-party PII is not accepted without consent, and every metric is generated from an actual evaluation run.

## Phase 1 Verification

From this directory, run:

```powershell
python scripts/phase1_check.py
```

Expected result:

```text
Phase 1 check passed: architecture and data contract are present.
```

## Planned Technology Stack

- Python 3.11 or 3.12, FastAPI, Pydantic, SQLAlchemy/Alembic, PostgreSQL
- pdfminer.six, python-docx, optional Tesseract OCR only for verified scanned documents
- spaCy baseline and Hugging Face Transformers/Datasets for attempted DistilBERT NER fine-tuning
- sentence-transformers and a cross-encoder reranker
- FAISS for the learning baseline; Weaviate for persistent local vector retrieval
- Neo4j for skill relationships and inference provenance
- RabbitMQ and Celery for asynchronous batch processing
- Docker Compose for the real local stack; kind or minikube for Kubernetes demonstration
- A lightweight FastAPI/UI deployment for the public core flow only

Heavy dependencies and infrastructure are intentionally deferred until the phase that needs and verifies them.

## Development Rule

No phase is treated as complete until its smallest runnable layer produces real output. Metrics, model-training claims, latency, and deployment claims will only be recorded after they are measured.
