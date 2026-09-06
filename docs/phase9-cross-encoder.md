# Phase 9: Cross-Encoder Reranking

Candidate retrieval and final scoring are separate interfaces so a cross-encoder can rerank a small shortlist using the JD and candidate text together.

### Model Variants & Memory Footprints
- **Public Cloud Deployment (Render 512MB limit)**:
  - Model: `cross-encoder/ms-marco-TinyBERT-L-2-v2`
  - Parameters: 4.4M (2 layers, 128 hidden dim)
  - Model weights: ~17.5 MB
  - Loaded RAM footprint: ~40 MB
  - Lazy-loaded: Model weights are not loaded until the `/rank` endpoint is triggered, leaving baseline container memory at ~75–85MB.
- **Local Development & Full Docker Stack**:
  - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Parameters: 22.7M (6 layers, 384 hidden dim)
  - Model weights: ~90 MB
  - Loaded RAM footprint: ~240 MB
  - Configurable via `RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`.

