"""Verified ONNX retrieval pipeline: embeddings, FAISS, reranking, and Weaviate."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import uuid

# Ensure single-threaded execution to minimize OpenMP/MKL thread pool memory overhead
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX_ROOT = Path(os.getenv("ONNX_MODELS_DIR", ROOT / "onnx_models"))
DEFAULT_EMBEDDER_PATH = Path(os.getenv("EMBEDDER_ONNX_PATH", DEFAULT_ONNX_ROOT / "embedder"))
DEFAULT_RERANKER_PATH = Path(os.getenv("RERANKER_ONNX_PATH", DEFAULT_ONNX_ROOT / "reranker"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-TinyBERT-L-2-v2")


class OnnxEmbedder:
    """Lightweight ONNX Runtime embedder with Hugging Face tokenizers."""

    def __init__(self, model_dir: Path | str, model_filename: str | None = None) -> None:
        model_path = Path(model_dir)
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if model_filename:
            model_file = model_path / model_filename
        else:
            model_file = model_path / "model.onnx"
            if not model_file.exists():
                model_file = model_path / "model_quantized.onnx"
        if not model_file.exists():
            raise FileNotFoundError(
                f"No ONNX model file found in {model_path}. Run scripts/export_onnx_models.py first."
            )

        tokenizer_file = model_path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"No tokenizer.json found in {model_path}. Run scripts/export_onnx_models.py first."
            )

        self.session = ort.InferenceSession(str(model_file), sess_options=opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, texts: list[str], normalize_embeddings: bool = True, **kwargs: Any) -> Any:
        import numpy as np

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)

        outputs = self.session.run(None, feed)
        token_embeddings = outputs[0]

        # Mean pooling taking attention mask into account
        mask_expanded = np.broadcast_to(
            np.expand_dims(attention_mask, -1),
            token_embeddings.shape,
        ).astype(np.float32)

        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            embeddings = embeddings / norms

        return embeddings.astype(np.float32)


class OnnxReranker:
    """Lightweight ONNX Runtime cross-encoder reranker with Hugging Face tokenizers."""

    def __init__(self, model_dir: Path | str, model_filename: str | None = None) -> None:
        model_path = Path(model_dir)
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if model_filename:
            model_file = model_path / model_filename
        else:
            model_file = model_path / "model_quantized.onnx"
            if not model_file.exists():
                model_file = model_path / "model.onnx"
        if not model_file.exists():
            raise FileNotFoundError(
                f"No ONNX model file found in {model_path}. Run scripts/export_onnx_models.py first."
            )

        tokenizer_file = model_path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"No tokenizer.json found in {model_path}. Run scripts/export_onnx_models.py first."
            )

        self.session = ort.InferenceSession(str(model_file), sess_options=opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._input_names = {inp.name for inp in self.session.get_inputs()}

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        if not pairs:
            return []

        import numpy as np

        encoded = self.tokenizer.encode_batch(pairs)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)

        outputs = self.session.run(None, feed)
        logits = outputs[0]

        if logits.ndim == 2 and logits.shape[1] == 1:
            scores = logits[:, 0]
        elif logits.ndim == 2 and logits.shape[1] > 1:
            scores = logits[:, 1]
        elif logits.ndim == 1:
            scores = logits
        else:
            scores = logits.squeeze()

        return [float(s) for s in scores]


class RealMatchingPipeline:
    """Owns retrieval and reranking components with lazy loading and singleton caching."""

    def __init__(
        self,
        embedder_path: Path | str = DEFAULT_EMBEDDER_PATH,
        reranker_path: Path | str = DEFAULT_RERANKER_PATH,
    ) -> None:
        self.embedder_path = Path(embedder_path)
        self.reranker_path = Path(reranker_path)
        self.embedding_model_name = EMBEDDING_MODEL
        self.reranker_model_name = RERANKER_MODEL
        self.weaviate_host = os.getenv("WEAVIATE_HOST")
        self.weaviate_port = int(os.getenv("WEAVIATE_PORT", "8080"))
        self.retrieval_backend = "uninitialized"
        self._embedder = None
        self._reranker = None

    @property
    def embedder(self) -> Any:
        """Lazy-load the ONNX embedding model once on first vector encoding."""
        if self._embedder is None:
            import os
            import psutil

            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            print(f"[MEMORY] before embedding model load: {mem_mb:.1f} MB")

            self._embedder = OnnxEmbedder(self.embedder_path)

            import os
            import psutil

            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            print(f"[MEMORY] after embedding model load: {mem_mb:.1f} MB")
        return self._embedder

    @property
    def reranker(self) -> Any:
        """Lazy-load the ONNX cross-encoder once on first rerank call (/rank)."""
        if self._reranker is None:
            import os
            import psutil

            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            print(f"[MEMORY] before cross-encoder load: {mem_mb:.1f} MB")

            self._reranker = OnnxReranker(self.reranker_path)

            import os
            import psutil

            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            print(f"[MEMORY] after cross-encoder load: {mem_mb:.1f} MB")
        return self._reranker

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        vectors = self.embedder.encode(texts, normalize_embeddings=True)
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

        pairs = [(query, record_text(candidate)) for candidate in candidates]
        scores = self.reranker.predict(pairs)

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

    pipeline = _pipeline
    return {
        "memory_rss_mb": rss_mb,
        "runtime": "onnxruntime",
        "embedder_loaded": pipeline._embedder is not None if pipeline else False,
        "reranker_loaded": pipeline._reranker is not None if pipeline else False,
        "embedding_model": pipeline.embedding_model_name if pipeline else EMBEDDING_MODEL,
        "reranker_model": pipeline.reranker_model_name if pipeline else RERANKER_MODEL,
    }
