# Public Deployment Fallback

The standalone API is designed for a single-container deployment such as Render's free tier.

- SQLite is used when `DATABASE_URL` is unset.
- Auto retrieval prefers Weaviate, then falls back to in-process FAISS when Weaviate is unreachable.
- Neo4j-backed skill normalization falls back to the canonical rule map when Neo4j is unreachable.
- Rule-based extraction and explanations remain available without external services.
- The Docker command listens on `${PORT:-8000}`, so Render can provide its assigned port.

PostgreSQL, Neo4j, RabbitMQ, and Weaviate remain optional external services for deployments that provide them. The public single-container path does not require those services to start successfully.
