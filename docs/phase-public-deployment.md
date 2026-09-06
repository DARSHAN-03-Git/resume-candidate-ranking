# Public Deployment Fallback

The standalone API is designed for a single-container deployment such as Render's 512MB free tier.

- **Memory-Conservative Torch Settings**:
  - `torch.set_num_threads(1)` and `set_num_interop_threads(1)` prevent thread-pool memory amplification.
  - Environment variables set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`.
  - All inference runs within `torch.inference_mode()` with immediate tensor deallocation and garbage collection (`gc.collect()`).
- **Model Loading & Selection**:
  - **Singletons**: All models are instantiated at most once per process and held as singletons.
  - **Lazy-Loading**: Neither the embedder nor the cross-encoder is loaded at process startup. Container startup memory stays at ~75–85MB RSS, allowing instant health checks.
  - **Public Deployment Cross-Encoder**: Defaults to `cross-encoder/ms-marco-TinyBERT-L-2-v2` (~17MB weights, ~40MB RAM) to operate comfortably within Render's 512MB limit.
  - **Full-Stack / Local Cross-Encoder**: The full 6-layer `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB weights, ~240MB RAM) can be used by setting `RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - **Embedder**: Uses `sentence-transformers/all-MiniLM-L6-v2` (~80MB weights, ~120MB RAM).
- **Diagnostics Endpoint**:
  - Real-time resident memory (VmRSS) and model status can be inspected at `/health` and `/system/memory`.
- **Database & Services**:
  - SQLite is used when `DATABASE_URL` is unset.
  - Auto retrieval prefers Weaviate, then falls back to in-process FAISS when Weaviate is unreachable.
  - Neo4j-backed skill normalization falls back to the canonical rule map when Neo4j is unreachable.
  - Rule-based extraction and explanations remain available without external services.
  - The Docker command listens on `${PORT:-8000}`, so Render can provide its assigned port.

PostgreSQL, Neo4j, RabbitMQ, and Weaviate remain optional external services for deployments that provide them. The public single-container path does not require those services to start successfully.

