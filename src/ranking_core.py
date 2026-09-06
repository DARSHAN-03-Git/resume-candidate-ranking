"""Local-first matching, scoring, explanation, and privacy primitives."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

try:
    from real_pipeline import get_pipeline, record_text
    from skill_graph import SKILL_ALIASES, get_skill_graph
except ImportError:
    from src.real_pipeline import get_pipeline, record_text
    from src.skill_graph import SKILL_ALIASES, get_skill_graph
TOKEN_RE = re.compile(r"[a-z0-9+#./-]+", re.I)


def normalize_skills(skills: Iterable[str]) -> list[str]:
    graph = get_skill_graph()
    normalized = []
    seen: set[str] = set()
    for skill in skills:
        canonical = graph.canonicalize(skill)
        if canonical and canonical.casefold() not in seen:
            seen.add(canonical.casefold())
            normalized.append(canonical)
    return normalized


def infer_skills(explicit_skills: Iterable[str]) -> list[dict[str, str]]:
    explicit = set(normalize_skills(explicit_skills))
    return [item for item in get_skill_graph().infer(explicit) if item["skill"] not in explicit]


def enrich_skills(record: dict[str, Any]) -> dict[str, Any]:
    explicit = normalize_skills(record.get("skills", {}).get("explicit", []))
    record["skills"]["explicit"] = explicit
    record["skills"]["inferred"] = infer_skills(explicit)
    return record


def parse_job_description(text: str) -> dict[str, Any]:
    years = [int(value) for value in re.findall(r"\b(\d+)\+?\s+years?", text, re.I)]
    found = []
    lowered = text.casefold()
    for alias, canonical in SKILL_ALIASES.items():
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", lowered):
            if canonical not in found:
                found.append(canonical)
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return {
        "role": first_line,
        "required_skills": found,
        "minimum_years": max(years, default=0),
        "source_text": text,
    }


def lexical_embed_text(text: str) -> dict[str, float]:
    """Retained only for the explicitly labeled historical baseline."""
    from collections import Counter

    tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
    counts = Counter(tokens)
    magnitude = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    return {token: value / magnitude for token, value in counts.items()}


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    return sum(left.get(token, 0.0) * value for token, value in right.items())


def candidate_text(record: dict[str, Any]) -> str:
    candidate = record.get("candidate", {})
    experience = " ".join(item.get("evidence", "") for item in record.get("experience", []))
    education = " ".join(item.get("evidence", "") for item in record.get("education", []))
    explicit = " ".join(record.get("skills", {}).get("explicit", []))
    return " ".join(filter(None, [candidate.get("summary"), experience, education, explicit]))


def jaccard_similarity(candidate_skills: Iterable[str], required_skills: Iterable[str]) -> float:
    candidate = {skill.casefold() for skill in candidate_skills}
    required = {skill.casefold() for skill in required_skills}
    return len(candidate & required) / len(candidate | required) if candidate | required else 0.0


def score_candidate(
    record: dict[str, Any],
    job: dict[str, Any],
    weights: dict[str, float] | None = None,
    semantic_score: float | None = None,
) -> dict[str, Any]:
    weights = weights or {"semantic": 0.5, "experience": 0.3, "skills": 0.2}
    explicit = record.get("skills", {}).get("explicit", [])
    inferred = [item["skill"] for item in record.get("skills", {}).get("inferred", [])]
    pipeline = get_pipeline()
    if semantic_score is not None:
        semantic = float(semantic_score)
    else:
        encoded = pipeline.encode([candidate_text(record), job["source_text"]])
        semantic = float(encoded[0] @ encoded[1])
    end_years = [item.get("end_year") for item in record.get("experience", []) if item.get("end_year")]
    start_years = [item.get("start_year") for item in record.get("experience", []) if item.get("start_year")]
    experience_years = max(end_years, default=0) - min(start_years, default=0) if start_years else 0
    experience_score = min(experience_years / max(job.get("minimum_years", 1), 1), 1.0)
    skill_score = jaccard_similarity(explicit + inferred, job.get("required_skills", []))
    total = (
        weights["semantic"] * semantic
        + weights["experience"] * experience_score
        + weights["skills"] * skill_score
    )
    return {
        "candidate": record.get("candidate", {}).get("name"),
        "score": round(total, 6),
        "components": {
            "semantic": round(semantic, 6),
            "experience": round(experience_score, 6),
            "skills_jaccard": round(skill_score, 6),
        },
        "weights": weights,
        "pipeline": {
            "embedding_model": pipeline.embedding_model_name,
            "reranker_model": pipeline.reranker_model_name,
        },
    }


def rank_candidates(records: list[dict[str, Any]], job: dict[str, Any], backend: str = "auto") -> list[dict[str, Any]]:
    if not records:
        return []
    pipeline = get_pipeline()
    if backend == "faiss":
        retrieved = pipeline.retrieve_faiss(records, job["source_text"], limit=len(records))
    elif backend == "weaviate":
        retrieved = pipeline.retrieve(records, job["source_text"], limit=len(records))
    elif backend == "auto":
        retrieved = pipeline.retrieve(records, job["source_text"], limit=len(records))
    else:
        raise ValueError(f"Unsupported retrieval backend: {backend}")
    reranked = pipeline.rerank(job["source_text"], [record for record, _ in retrieved])
    scores = {record.get("candidate", {}).get("name"): score for record, score in reranked}

    # Batch encode candidate texts and job text once to prevent redundant memory and forward passes
    cand_texts = [candidate_text(r) for r in records]
    cand_vectors = pipeline.encode(cand_texts)
    job_vector = pipeline.encode([job["source_text"]])[0]
    semantic_scores = {
        records[i].get("candidate", {}).get("name"): float(cand_vectors[i] @ job_vector)
        for i in range(len(records))
    }

    ranked = []
    for record in records:
        candidate_name = record.get("candidate", {}).get("name")
        sem_score = semantic_scores.get(candidate_name)
        result = score_candidate(record, job, semantic_score=sem_score)
        if candidate_name not in scores:
            raise RuntimeError(f"Retrieval backend {backend!r} did not return candidate {candidate_name!r}")
        result["components"]["cross_encoder"] = round(scores[candidate_name], 6)
        result["score"] = round(0.7 * result["score"] + 0.3 * max(0.0, scores[candidate_name]), 6)
        ranked.append(result)
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def explain_candidate(record: dict[str, Any], score: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    required = {skill.casefold() for skill in job.get("required_skills", [])}
    matched = [skill for skill in record.get("skills", {}).get("explicit", []) if skill.casefold() in required]
    evidence = [item for item in record.get("evidence", []) if item["text"] in matched]
    return {
        "candidate": record.get("candidate", {}).get("name"),
        "score": score["score"],
        "evidence": evidence,
        "matched_explicit_skills": matched,
        "inferred_skills": record.get("skills", {}).get("inferred", []),
        "generated_phrasing": f"Matched verified skills: {', '.join(matched) or 'none'}.",
    }


def find_lookalikes(records: list[dict[str, Any]], candidate_name: str, limit: int = 3) -> list[dict[str, Any]]:
    target = next(record for record in records if record.get("candidate", {}).get("name") == candidate_name)
    pipeline = get_pipeline()
    target_vector = pipeline.encode([record_text(target)])[0]
    neighbors = []
    for record in records:
        if record is target:
            continue
        neighbors.append({
            "candidate": record.get("candidate", {}).get("name"),
            "similarity": round(float(target_vector @ pipeline.encode([record_text(record)])[0]), 6),
        })
    return sorted(neighbors, key=lambda item: item["similarity"], reverse=True)[:limit]


def sanitize_text(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[EMAIL_REDACTED]", text)
    text = re.sub(r"(?:\+?\d[\d ()-]{7,}\d)", "[PHONE_REDACTED]", text)
    return text


def draft_outreach(record: dict[str, Any], job: dict[str, Any]) -> str:
    name = record.get("candidate", {}).get("name") or "Candidate"
    skills = record.get("skills", {}).get("explicit", [])
    verified = ", ".join(skills[:3]) or "your experience"
    return f"Hello {name}, your resume shows verified experience with {verified}. We are reviewing candidates for {job.get('role', 'this role')}. Would you be open to a conversation?"


def compute_experience_years(record: Any) -> int:
    if hasattr(record, "experience"):
        exp_list = record.experience
    elif isinstance(record, dict):
        exp_list = record.get("experience", [])
    else:
        exp_list = []

    start_years = []
    end_years = []
    for item in exp_list:
        sy = getattr(item, "start_year", None) if hasattr(item, "start_year") else (item.get("start_year") if isinstance(item, dict) else None)
        ey = getattr(item, "end_year", None) if hasattr(item, "end_year") else (item.get("end_year") if isinstance(item, dict) else None)
        if sy:
            start_years.append(sy)
        if ey:
            end_years.append(ey)

    if start_years and end_years:
        return max(0, max(end_years) - min(start_years))
    return 0


def compute_experience_score(record: Any, minimum_years: int = 1) -> float:
    years = compute_experience_years(record)
    min_required = max(1, minimum_years)
    return min(1.0, years / min_required)

