"""Verify the dependency-free core for Phases 4 through 13."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline_extractor import extract_resume  # noqa: E402
from document_processing import extract_document  # noqa: E402
from ranking_core import (  # noqa: E402
    cosine_similarity,
    draft_outreach,
    lexical_embed_text,
    enrich_skills,
    explain_candidate,
    find_lookalikes,
    normalize_skills,
    parse_job_description,
    rank_candidates,
    sanitize_text,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "data" / "synthetic" / "resumes").glob("resume_*.docx"))
    records = []
    for path in paths:
        document = extract_document(path)
        records.append(enrich_skills(extract_resume(document.text, document.source_path, synthetic=True)))

    job = parse_job_description("Backend Engineer\n3 years Python FastAPI PostgreSQL Docker experience")
    ranked = rank_candidates(records, job)
    assert len(ranked) == 4
    assert ranked[0]["score"] >= ranked[-1]["score"]
    assert normalize_skills(["K8s", "Python"]) == ["Kubernetes", "Python"]
    assert cosine_similarity(lexical_embed_text("Python API"), lexical_embed_text("Python")) > 0
    assert explain_candidate(records[0], ranked[0], job)["generated_phrasing"]
    assert find_lookalikes(records, records[0]["candidate"]["name"])
    assert "[EMAIL_REDACTED]" in sanitize_text("a@example.com")
    assert records[0]["skills"]["inferred"][0]["type"] == "inferred"
    assert records[0]["skills"]["inferred"][0]["basis"] == "FastAPI + Python"
    assert draft_outreach(records[0], job).startswith("Hello")
    print("Phases 4-13 smoke test passed: normalize, match, rank, explain, and privacy checks succeeded.")
    print(f"Ranked candidates: {len(ranked)}; top candidate: {ranked[0]['candidate']}")


if __name__ == "__main__":
    main()
