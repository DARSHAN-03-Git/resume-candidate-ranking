#!/usr/bin/env python3
"""Verification script to test parity between PyTorch models and quantized ONNX models."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import numpy as np

# Ensure root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.real_pipeline import (
    DEFAULT_EMBEDDER_PATH,
    DEFAULT_RERANKER_PATH,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    OnnxEmbedder,
    OnnxReranker,
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two 1D vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Calculate Spearman's rank correlation coefficient between two lists of scores."""
    n = len(x)
    if n <= 1:
        return 1.0

    def get_ranks(vals: list[float]) -> list[float]:
        sorted_indices = sorted(range(n), key=lambda k: vals[k])
        ranks = [0.0] * n
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = float(rank + 1)
        return ranks

    rank_x = get_ranks(x)
    rank_y = get_ranks(y)
    d_sq = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def main() -> None:
    print("=" * 70)
    print("ONNX vs. PyTorch Model Parity Verification")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Sample Texts & Pairs
    # -------------------------------------------------------------
    sample_texts = [
        (
            "Job Spec",
            "Senior Python Backend Engineer with 5+ years of experience in FastAPI, "
            "PostgreSQL, microservices architecture, and Celery asynchronous task queues."
        ),
        (
            "Strong Match",
            "Lead Software Developer with 6 years building distributed Python backend APIs "
            "using FastAPI, Docker, and PostgreSQL. Experienced with asynchronous pipelines and Redis."
        ),
        (
            "Moderate Match",
            "Frontend React Engineer with 3 years building web UIs using TypeScript, Tailwind CSS, "
            "and Next.js, with basic knowledge of Node.js APIs and Python scripting."
        ),
        (
            "Unrelated Match",
            "Digital Marketing Specialist with 4 years managing Google Ads, paid social media campaigns, "
            "SEO optimization strategies, and brand copywriting."
        ),
    ]

    # -------------------------------------------------------------
    # 2. Verify Embedder Parity (FP32 & INT8)
    # -------------------------------------------------------------
    print("\n--- [1/2] EMBEDDER PARITY (SentenceTransformer vs. OnnxEmbedder) ---")
    print(f"Base PyTorch Model : {EMBEDDING_MODEL}")
    print(f"ONNX Model Path    : {DEFAULT_EMBEDDER_PATH}")

    from sentence_transformers import SentenceTransformer

    print("Loading PyTorch SentenceTransformer...")
    pt_embedder = SentenceTransformer(EMBEDDING_MODEL)

    raw_texts = [text for _, text in sample_texts]

    # Compute PyTorch embeddings
    pt_vecs = pt_embedder.encode(raw_texts, normalize_embeddings=True, convert_to_numpy=True)

    # Check for FP32 model
    fp32_model_path = DEFAULT_EMBEDDER_PATH / "model.onnx"
    has_fp32 = fp32_model_path.exists()
    fp32_vecs = None
    if has_fp32:
        print("Loading FP32 OnnxEmbedder (unquantized)...")
        onnx_fp32 = OnnxEmbedder(DEFAULT_EMBEDDER_PATH, model_filename="model.onnx")
        fp32_vecs = onnx_fp32.encode(raw_texts, normalize_embeddings=True)

    # Check for INT8 model
    int8_model_path = DEFAULT_EMBEDDER_PATH / "model_quantized.onnx"
    has_int8 = int8_model_path.exists()
    int8_vecs = None
    if has_int8:
        print("Loading INT8 OnnxEmbedder (quantized)...")
        onnx_int8 = OnnxEmbedder(DEFAULT_EMBEDDER_PATH, model_filename="model_quantized.onnx")
        int8_vecs = onnx_int8.encode(raw_texts, normalize_embeddings=True)

    print("\nEmbedder Results:")
    header = f"{'Label':<18} {'PyTorch vs FP32':<20} {'PyTorch vs INT8':<20} {'Vector Dim':<12}"
    print(header)
    print("-" * len(header))

    fp32_sims = []
    int8_sims = []
    for i, (label, _) in enumerate(sample_texts):
        pt_v = pt_vecs[i]
        sim_fp32_str = "N/A"
        sim_int8_str = "N/A"

        if fp32_vecs is not None:
            sim_fp32 = cosine_similarity(pt_v, fp32_vecs[i])
            fp32_sims.append(sim_fp32)
            sim_fp32_str = f"{sim_fp32:.6f}"

        if int8_vecs is not None:
            sim_int8 = cosine_similarity(pt_v, int8_vecs[i])
            int8_sims.append(sim_int8)
            sim_int8_str = f"{sim_int8:.6f}"

        print(f"{label:<18} {sim_fp32_str:<20} {sim_int8_str:<20} {str(pt_v.shape):<12}")

    print("-" * len(header))
    if fp32_sims:
        print(f"{'Mean FP32':<18} {np.mean(fp32_sims):.6f}")
        print(f"{'Min FP32':<18} {np.min(fp32_sims):.6f}")
    if int8_sims:
        print(f"{'Mean INT8':<18} {'':<20} {np.mean(int8_sims):.6f}")
        print(f"{'Min INT8':<18} {'':<20} {np.min(int8_sims):.6f}")

    # -------------------------------------------------------------
    # 3. Verify Reranker Parity
    # -------------------------------------------------------------
    print("\n--- [2/2] RERANKER PARITY (CrossEncoder vs. OnnxReranker) ---")
    print(f"Base PyTorch Model : {RERANKER_MODEL}")
    print(f"ONNX Model Path    : {DEFAULT_RERANKER_PATH}")

    from sentence_transformers import CrossEncoder

    print("Loading PyTorch CrossEncoder...")
    pt_reranker = CrossEncoder(RERANKER_MODEL)

    print("Loading OnnxReranker...")
    onnx_reranker = OnnxReranker(DEFAULT_RERANKER_PATH)

    query = sample_texts[0][1]  # The Job Spec
    rerank_candidates = sample_texts[1:]  # Strong, Moderate, Unrelated

    pairs = [(query, text) for _, text in rerank_candidates]

    pt_scores = [float(s) for s in pt_reranker.predict(pairs)]
    onnx_scores = onnx_reranker.predict(pairs)

    print("\nReranker Results:")
    print(f"{'Candidate':<18} {'PyTorch Score':<16} {'ONNX Score':<16} {'Diff (ONNX - PT)':<18}")
    print("-" * 68)

    for (label, _), pt_score, onnx_score in zip(rerank_candidates, pt_scores, onnx_scores):
        diff = onnx_score - pt_score
        print(f"{label:<18} {pt_score:<16.4f} {onnx_score:<16.4f} {diff:<+18.4f}")

    pt_rank_order = [rerank_candidates[idx][0] for idx in np.argsort(pt_scores)[::-1]]
    onnx_rank_order = [rerank_candidates[idx][0] for idx in np.argsort(onnx_scores)[::-1]]
    spearman_rho = spearman_rank_correlation(pt_scores, onnx_scores)

    print("-" * 68)
    print(f"PyTorch Ranking Order : {' > '.join(pt_rank_order)}")
    print(f"ONNX Ranking Order    : {' > '.join(onnx_rank_order)}")
    print(f"Rank Order Match      : {'EXACT MATCH' if pt_rank_order == onnx_rank_order else 'MISMATCH'}")
    print(f"Spearman Correlation  : {spearman_rho:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
