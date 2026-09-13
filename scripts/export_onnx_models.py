#!/usr/bin/env python3
"""Export embedder and cross-encoder models to ONNX with INT8 dynamic quantization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import time

# Default model identifiers
DEFAULT_EMBEDDER_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-TinyBERT-L-2-v2")


def print_dir_sizes(directory: Path) -> None:
    """Print size of files within directory."""
    if not directory.exists():
        print(f"Directory {directory} does not exist.")
        return

    print(f"\n--- Files in {directory} ---")
    total_bytes = 0
    for item in sorted(directory.rglob("*")):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            total_bytes += item.stat().st_size
            rel_path = item.relative_to(directory)
            print(f"  {str(rel_path):<40} {size_mb:>8.2f} MB")
    print(f"Total directory size: {total_bytes / (1024 * 1024):.2f} MB\n")


def export_embedder(model_id: str, output_dir: Path) -> None:
    """Export SentenceTransformer embedding model to ONNX + dynamic INT8 quantization."""
    print(f"\n[1/2] Exporting embedder model '{model_id}' to ONNX...")
    t0 = time.time()

    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Export FP32 ONNX model & save tokenizer / config
    print("  - Exporting base ONNX model (FP32)...")
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ort_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Step 2: Apply dynamic INT8 quantization
    print("  - Applying INT8 dynamic quantization...")
    quantizer = ORTQuantizer.from_pretrained(output_dir, file_name="model.onnx")
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)

    print(f"  ✔ Embedder export and quantization completed in {time.time() - t0:.1f}s")


def export_reranker(model_id: str, output_dir: Path) -> None:
    """Export Cross-Encoder sequence classification model to ONNX + dynamic INT8 quantization."""
    print(f"\n[2/2] Exporting reranker model '{model_id}' to ONNX...")
    t0 = time.time()

    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Export FP32 ONNX model & save tokenizer / config
    print("  - Exporting base ONNX model (FP32)...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ort_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Step 2: Apply dynamic INT8 quantization
    print("  - Applying INT8 dynamic quantization...")
    quantizer = ORTQuantizer.from_pretrained(output_dir, file_name="model.onnx")
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)

    print(f"  ✔ Reranker export and quantization completed in {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export transformer models to INT8 ONNX format.")
    parser.add_argument(
        "--embedder",
        default=DEFAULT_EMBEDDER_MODEL,
        help=f"Hugging Face ID for embedder (default: {DEFAULT_EMBEDDER_MODEL})",
    )
    parser.add_argument(
        "--reranker",
        default=DEFAULT_RERANKER_MODEL,
        help=f"Hugging Face ID for cross-encoder reranker (default: {DEFAULT_RERANKER_MODEL})",
    )
    parser.add_argument(
        "--output-root",
        default="onnx_models",
        help="Root directory to store exported models (default: onnx_models)",
    )
    args = parser.parse_args()

    root_dir = Path(args.output_root).resolve()
    embedder_dir = root_dir / "embedder"
    reranker_dir = root_dir / "reranker"

    print("=" * 60)
    print("Transformer to INT8 ONNX Model Exporter")
    print(f"Embedder model : {args.embedder}")
    print(f"Reranker model : {args.reranker}")
    print(f"Output folder  : {root_dir}")
    print("=" * 60)

    export_embedder(args.embedder, embedder_dir)
    print_dir_sizes(embedder_dir)

    export_reranker(args.reranker, reranker_dir)
    print_dir_sizes(reranker_dir)

    print("=" * 60)
    print("SUMMARY OF ALL EXPORTED ONNX ARTIFACTS:")
    print_dir_sizes(root_dir)
    print("Export script finished successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
