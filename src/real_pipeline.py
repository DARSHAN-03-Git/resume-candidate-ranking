"""Verified transformer retrieval pipeline: embeddings, FAISS, reranking, and Weaviate."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

# Ensure single-threaded execution to minimize OpenMP/MKL thread pool memory overhead
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _init_torch_runtime() -> None:
    """Configure PyTorch in memory-conservative single-thread mode."""
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except ImportError:
        pass


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Default to distilled TinyBERT cross-encoder (~17MB weights, ~40MB RAM) for memory-constrained
# environments like Render 512MB free tier. Full 6-layer model ("cross-encoder/ms-marco-MiniLM-L-6-v2",
# ~90MB weights, ~240MB RAM) can be specified via RERANKER_MODEL for full-memory local/Docker stacks.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-TinyBERT-L-2-v2")


class RealMatchingPipeline:
    """Owns retrieval and reranking components with lazy loading and singleton caching."""

    def __init__(self, embedding_model: str = EMBEDDING_MODEL, reranker_model: str = RERANKER_MODEL) -> None:
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.weaviate_host = os.getenv("WEAVIATE_HOST")
        self.weaviate_port = int(os.getenv("WEAVIATE_PORT", "8080"))
        self.retrieval_backend = "uninitialized"
        self._embedder = None
        self._reranker = None

    @property
    def embedder(self) -> Any:
        """Lazy-load the embedding model once on first vector encoding."""
        if self._embedder is None:
            _init_torch_runtime()
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    @property
    def reranker(self) -> Any:
        """Lazy-load the cross-encoder once on first rerank call (/rank)."""
        if self._reranker is None:
            _init_torch_runtime()
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        try:
            import torch

            ctx = torch.inference_mode()
        except (ImportError, AttributeError):
            from contextlib import nullcontext

            ctx = nullcontext()

        with ctx:
            vectors = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(vectors, dtype="float32")

    def build_faiss_index(self, records: list[dict[str, Any]]) -> tuple[Any, Any]:
        import faiss

        vectors = self.encode([record_text(record) for record in records])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index, vectors

    def retrieve_faiss(self, records: list[dict[str, Any]], query: str, limit: int = 20) -> list[tuple[dict[str, Any], float]]:
        index, _ = self.build_faiss_index(records)
        scores, positions = index.search(self.encode([query]), min(limit, len(records)))
        return [(records[int(position)], float(score)) for score, position in zip(scores[0], positions[0]) if position >= 0]

    def retrieve_weaviate(self, records: list[dict[str, Any]], query: str, limit: int = 20) -> list[tuple[dict[str, Any], float]]:
        self.upsert_weaviate(records)
        results = self.query_weaviate(query, limit=limit)
        by_name = {record.get("candidate", {}).get("name"): record for record in records}
        return [
            (by_name[item["candidate_name"]], 1.0 - float(item["distance"]))
            for item in results
            if item.get("candidate_name") in by_name
        ]

    def retrieve(self, records: list[dict[str, Any]], query: str, limit: int = 20) -> list[tuple[dict[str, Any], float]]:
        """Prefer Weaviate, but use the local FAISS fallback when unavailable."""
        if not self.weaviate_host:
            self.retrieval_backend = "faiss-fallback"
            return self.retrieve_faiss(records, query, limit=limit)
        try:
            result = self.retrieve_weaviate(records, query, limit=limit)
            self.retrieval_backend = "weaviate"
            return result
        except Exception:
            self.retrieval_backend = "faiss-fallback"
            return self.retrieve_faiss(records, query, limit=limit)

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[tuple[dict[str, Any], float]]:
        import gc

        try:
            import torch

            ctx = torch.inference_mode()
        except (ImportError, AttributeError):
            from contextlib import nullcontext

            ctx = nullcontext()

        pairs = [(query, record_text(candidate)) for candidate in candidates]
        with ctx:
            scores = self.reranker.predict(pairs, show_progress_bar=False)

        results = sorted(zip(candidates, (float(score) for score in scores)), key=lambda item: item[1], reverse=True)
        del pairs
        del scores
        gc.collect()
        return results

    def upsert_weaviate(self, records: list[dict[str, Any]], host: str | None = None, port: int | None = None) -> int:
        import weaviate
        from weaviate.classes.config import Configure, DataType, Property

        client = weaviate.connect_to_local(host=host or self.weaviate_host, port=port or self.weaviate_port)
        try:
            name = "ResumeCandidate"
            if not client.collections.exists(name):
                client.collections.create(
                    name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(name="candidate_name", data_type=DataType.TEXT),
                        Property(name="source_text", data_type=DataType.TEXT),
                    ],
                )
            collection = client.collections.get(name)
            vectors = self.encode([record_text(record) for record in records])
            for record, vector in zip(records, vectors):
                candidate_name = record.get("candidate", {}).get("name") or "Unknown"
                object_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"resume-candidate:{candidate_name}"))
                properties = {"candidate_name": candidate_name, "source_text": record_text(record)}
                try:
                    collection.data.insert(uuid=object_uuid, properties=properties, vector=vector.tolist())
                except Exception:
                    collection.data.replace(uuid=object_uuid, properties=properties, vector=vector.tolist())
            return len(records)
        finally:
            client.close()

    def query_weaviate(self, query: str, limit: int = 10, host: str | None = None, port: int | None = None) -> list[dict[str, Any]]:
        import weaviate
        from weaviate.classes.query import MetadataQuery

        client = weaviate.connect_to_local(host=host or self.weaviate_host, port=port or self.weaviate_port)
        try:
            collection = client.collections.get("ResumeCandidate")
            result = collection.query.near_vector(
                self.encode([query])[0].tolist(),
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
            )
            return [item.properties | {"distance": item.metadata.distance} for item in result.objects]
        finally:
            client.close()


def record_text(record: dict[str, Any]) -> str:
    candidate = record.get("candidate", {})
    experience = " ".join(item.get("evidence", "") for item in record.get("experience", []))
    education = " ".join(item.get("evidence", "") for item in record.get("education", []))
    skills = " ".join(record.get("skills", {}).get("explicit", []))
    return " ".join(filter(None, [candidate.get("summary"), experience, education, skills]))


_pipeline: RealMatchingPipeline | None = None


def get_pipeline() -> RealMatchingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RealMatchingPipeline()
    return _pipeline


def get_memory_diagnostics() -> dict[str, Any]:
    """Inspect current process resident memory (VmRSS) and model loading state."""
    rss_mb = 0.0
    try:
        with open("/proc/self/status", "r") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    rss_mb = round(int(line.split()[1]) / 1024.0, 2)
                    break
    except Exception:
        import resource

        rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)

    torch_threads = 1
    try:
        import torch

        torch_threads = torch.get_num_threads()
    except Exception:
        pass

    pipeline = _pipeline
    return {
        "memory_rss_mb": rss_mb,
        "embedder_loaded": pipeline._embedder is not None if pipeline else False,
        "reranker_loaded": pipeline._reranker is not None if pipeline else False,
        "embedding_model": pipeline.embedding_model_name if pipeline else EMBEDDING_MODEL,
        "reranker_model": pipeline.reranker_model_name if pipeline else RERANKER_MODEL,
        "torch_num_threads": torch_threads,
    }

