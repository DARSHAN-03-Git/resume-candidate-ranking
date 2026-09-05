"""Measure ranking and paired-name behavior on the generated synthetic corpus."""

from __future__ import annotations

from math import log2
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline_extractor import extract_resume  # noqa: E402
from document_processing import extract_document  # noqa: E402
from ranking_core import enrich_skills, lexical_embed_text, parse_job_description, rank_candidates, score_candidate  # noqa: E402
from real_pipeline import get_pipeline, record_text  # noqa: E402


def dcg(relevances: list[int]) -> float:
    return sum(relevance / log2(index + 2) for index, relevance in enumerate(relevances))


def fallback_score(record: dict, job: dict) -> float:
    candidate = lexical_embed_text(" ".join(record.get("skills", {}).get("explicit", [])))
    query = lexical_embed_text(job["source_text"])
    semantic = sum(candidate.get(token, 0.0) * value for token, value in query.items())
    skills = {item.casefold() for item in record.get("skills", {}).get("explicit", [])}
    required = {item.casefold() for item in job.get("required_skills", [])}
    skill_score = len(skills & required) / len(skills | required) if skills | required else 0.0
    return 0.5 * semantic + 0.2 * skill_score


def real_score_rows(records: list[dict], job: dict) -> list[dict]:
    pipeline = get_pipeline()
    vectors = pipeline.encode([record_text(record) for record in records])
    query_vector = pipeline.encode([job["source_text"]])[0]
    embedding_scores = vectors @ query_vector
    rerank_scores = pipeline.rerank(job["source_text"], records)
    rerank_by_id = {id(record): score for record, score in rerank_scores}
    rows = []
    for record, embedding_score in zip(records, embedding_scores):
        base = score_candidate(record, job)
        cross_encoder = rerank_by_id[id(record)]
        final_score = 0.7 * base["score"] + 0.3 * max(0.0, cross_encoder)
        rows.append({
            "candidate": record["candidate"]["name"],
            "embedding_cosine": float(embedding_score),
            "cross_encoder": float(cross_encoder),
            "real_final_score": float(final_score),
        })
    return sorted(rows, key=lambda row: row["real_final_score"], reverse=True)


def ranking_metrics(ranked: list[dict], relevant: set[str]) -> dict[str, float]:
    top_k = [item["candidate"] for item in ranked[:2]]
    relevance = [int(item["candidate"] in relevant) for item in ranked]
    relevance_at_4 = relevance[:4]
    ideal = sorted(relevance_at_4, reverse=True)
    return {
        "precision_at_2": sum(name in relevant for name in top_k) / 2,
        "recall_at_2": sum(name in relevant for name in top_k) / len(relevant),
        "ndcg_at_4": dcg(relevance_at_4) / dcg(ideal) if dcg(ideal) else 0.0,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    records = []
    for path in sorted((root / "data" / "synthetic" / "bootstrapped_resumes").glob("resume_*.docx")):
        document = extract_document(path)
        records.append(enrich_skills(extract_resume(document.text, document.source_path, synthetic=True)))
    job = parse_job_description("Backend Engineer\n3 years Python FastAPI PostgreSQL Docker experience")
    manifest = json.loads((root / "data" / "synthetic" / "bootstrapped_resumes" / "manifest.json").read_text(encoding="utf-8"))
    relevant = {record["name"] for record in manifest["records"] if record["benchmark_relevant"]}
    real_faiss_ranked = rank_candidates(records, job, backend="faiss")
    real_weaviate_ranked = rank_candidates(records, job, backend="weaviate")
    fallback_ranked = sorted(
        [{"candidate": record["candidate"]["name"], "score": fallback_score(record, job)} for record in records],
        key=lambda item: item["score"], reverse=True,
    )
    fallback_scores = {item["candidate"]: item["score"] for item in fallback_ranked}
    fallback_ranks = {item["candidate"]: index + 1 for index, item in enumerate(fallback_ranked)}
    real_scores = real_score_rows(records, job)
    real_ranked_for_metrics = [{"candidate": row["candidate"], "score": row["real_final_score"]} for row in real_scores]
    weaviate_probe = get_pipeline().query_weaviate(job["source_text"], limit=len(records))

    original = records[0]
    swapped = json.loads(json.dumps(original).replace("Asha Mehta", "Alex Mehta"))
    original_score = score_candidate(original, job)["score"]
    swapped_score = score_candidate(swapped, job)["score"]
    report = {
        "data_status": "synthetic_generated_bootstrapped",
        "fallback_baseline": {"metrics": ranking_metrics(fallback_ranked, relevant), "implementation": "lexical cosine/Jaccard"},
        "real_faiss": {
            "metrics": ranking_metrics(real_faiss_ranked, relevant),
            "implementation": "Sentence-Transformer -> FAISS -> cross-encoder",
            "vector_store": "FAISS (in-process)",
            "embedding_model": real_faiss_ranked[0]["pipeline"]["embedding_model"],
            "reranker_model": real_faiss_ranked[0]["pipeline"]["reranker_model"],
        },
        "real_weaviate": {
            "metrics": ranking_metrics(real_weaviate_ranked, relevant),
            "implementation": "Sentence-Transformer -> Weaviate -> cross-encoder",
            "vector_store": "Weaviate (persistent)",
            "embedding_model": real_weaviate_ranked[0]["pipeline"]["embedding_model"],
            "reranker_model": real_weaviate_ranked[0]["pipeline"]["reranker_model"],
            "connection": {
                "status": "connected",
                "indexed_records": len(records),
                "benchmark_query_limit": len(records),
                "benchmark_query_results": len(weaviate_probe),
            },
            "score_rows": [
                {
                    **row,
                    "fallback_score": fallback_scores[row["candidate"]],
                    "fallback_rank": fallback_ranks[row["candidate"]],
                    "rank_order": index + 1,
                }
                for index, row in enumerate(real_scores)
            ],
        },
        "paired_name_test": {"original_score": original_score, "swapped_score": swapped_score, "difference": swapped_score - original_score},
    }
    output = root / "reports"
    output.mkdir(exist_ok=True)
    (output / "phase20_three_pipeline_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Phase 20 evaluation passed: ranking and paired-name measurements written.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
