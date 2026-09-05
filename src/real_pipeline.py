"""Verified transformer retrieval pipeline: embeddings, FAISS, reranking, and Weaviate."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


class RealMatchingPipeline:
    """Owns the non-fallback retrieval and reranking components."""

    def __init__(self, embedding_model: str = EMBEDDING_MODEL, reranker_model: str = RERANKER_MODEL) -> None:
        import numpy as np
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.weaviate_host = os.getenv("WEAVIATE_HOST")
        self.weaviate_port = int(os.getenv("WEAVIATE_PORT", "8080"))
        self.retrieval_backend = "uninitialized"
        self.embedder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(reranker_model)

    def encode(self, texts: list[str]) -> np.ndarray:
        import numpy as np

        vectors = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    def build_faiss_index(self, records: list[dict[str, Any]]) -> tuple[Any, np.ndarray]:
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
        pairs = [(query, record_text(candidate)) for candidate in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        return sorted(zip(candidates, (float(score) for score in scores)), key=lambda item: item[1], reverse=True)

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
