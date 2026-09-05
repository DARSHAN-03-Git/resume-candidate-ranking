"""Auditable rule-based baseline extractor for resume text."""

from __future__ import annotations

import re
from typing import Any

from resume_schema import empty_resume


SECTION_NAMES = ("SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?:\+?\d[\d ()-]{7,}\d)")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
SKILL_RE = re.compile(
    r"(?i)\b(?:python|fastapi|postgresql|docker|rest apis?|sql|pandas|tableau|a/b testing|"
    r"scikit-learn|model evaluation|git|javascript|typescript|react|css|accessibility)\b"
)


def extract_resume(text: str, source_path: str = "<memory>", synthetic: bool = False) -> dict[str, Any]:
    """Extract conservative fields and evidence from normalized resume text."""
    result = empty_resume(source_path, synthetic)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections = _sections(lines)
    candidate = result["candidate"]
    candidate["name"] = _first_name(lines)
    candidate["email"] = _first_match(EMAIL_RE, text)
    candidate["phone"] = _first_match(PHONE_RE, text)
    candidate["location"] = _location(lines)
    candidate["summary"] = _first_content(sections.get("SUMMARY", []))

    for line in sections.get("EXPERIENCE", []):
        entry = _experience_entry(line)
        if entry:
            result["experience"].append(entry)
    for line in sections.get("EDUCATION", []):
        entry = _education_entry(line)
        if entry:
            result["education"].append(entry)

    skills = _unique(match.group(0) for match in SKILL_RE.finditer(" ".join(sections.get("SKILLS", []))))
    result["skills"]["explicit"] = skills
    result["evidence"] = _evidence(result, text)
    result["extraction"]["warnings"] = _warnings(result)
    return result


def _sections(lines: list[str]) -> dict[str, list[str]]:
    sections = {name: [] for name in SECTION_NAMES}
    current: str | None = None
    for line in lines:
        heading = line.rstrip(":").upper()
        if heading in sections:
            current = heading
        elif current:
            sections[current].append(line)
    return sections


def _first_name(lines: list[str]) -> str | None:
    for line in lines:
        upper = line.upper()
        if upper.startswith("SYNTHETIC RESUME"):
            continue
        if "|" in line and "@" in line:
            return line.split("|", 1)[0].strip() or None
        if not any(token in upper for token in ("@", "SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS")):
            return line
    return None


def _location(lines: list[str]) -> str | None:
    for line in lines[:4]:
        if "|" in line:
            fields = [field.strip() for field in line.split("|")]
            if len(fields) >= 4:
                return fields[-1]
    return None


def _first_content(lines: list[str]) -> str | None:
    return lines[0] if lines else None


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0).strip() if match else None


def _experience_entry(line: str) -> dict[str, Any] | None:
    years = [int(year) for year in YEAR_RE.findall(line)]
    if not years:
        return None
    title_company = line.split("|", 1)[0].strip()
    title, _, company = title_company.partition(",")
    return {
        "title": title.strip() or None,
        "company": company.strip() or None,
        "start_year": min(years),
        "end_year": max(years),
        "evidence": line,
    }


def _education_entry(line: str) -> dict[str, Any] | None:
    years = [int(year) for year in YEAR_RE.findall(line)]
    if not years:
        return None
    degree, _, institution = line.partition(",")
    return {
        "degree": degree.strip() or None,
        "institution": institution.rsplit("|", 1)[0].strip() or None,
        "year": max(years),
        "evidence": line,
    }


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _evidence(result: dict[str, Any], text: str) -> list[dict[str, str]]:
    evidence = []
    for field in ("name", "email", "phone", "location", "summary"):
        value = result["candidate"].get(field)
        if value:
            evidence.append({"field": f"candidate.{field}", "text": value, "source": "resume_text"})
    for skill in result["skills"]["explicit"]:
        evidence.append({"field": "skills.explicit", "text": skill, "source": "resume_text"})
    return evidence


def _warnings(result: dict[str, Any]) -> list[str]:
    warnings = []
    if not result["candidate"]["name"]:
        warnings.append("Name was not confidently extracted.")
    if not result["skills"]["explicit"]:
        warnings.append("No supported explicit skills were found.")
    return warnings
