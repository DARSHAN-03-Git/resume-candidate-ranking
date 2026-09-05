# Phases 14-18: Platform

## Phase 14: FastAPI and PostgreSQL

FastAPI exposes health, upload/parse, candidate listing, ranking, look-alike, and outreach endpoints. SQLAlchemy persists structured JSON records. SQLite is the verified zero-setup development default; `DATABASE_URL` selects PostgreSQL in the Compose deployment.

## Phase 15: Async processing

`async_tasks.py` defines a Celery task boundary using RabbitMQ as broker. The synchronous parser remains available for local development. A worker is not claimed as running until Docker and RabbitMQ are available.

## Phase 16: Recruiter UI

The static UI supports resume upload, job-description editing, parsing, and ranking. It is intentionally small but calls real API endpoints rather than using fake result data.

## Phase 17: Outreach

Outreach is deterministic and uses only the candidate name and explicit extracted skills. It does not infer accomplishments or invent employers.

## Phase 18: Docker and Kubernetes

`docker-compose.yml` defines API, PostgreSQL, Neo4j, RabbitMQ, and Weaviate. `k8s/api-deployment.yaml` provides the local image deployment starting point. Docker is not available in the current environment, so container startup and Kubernetes rollout remain unverified.
