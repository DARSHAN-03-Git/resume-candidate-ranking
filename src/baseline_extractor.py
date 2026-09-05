import re
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.resume_schema import (
        CandidateProfile,
        EducationItem,
        EvidenceItem,
        ExperienceItem,
        ExtractionMeta,
        ResumeRecord,
        ResumeSource,
        SkillsProfile,
    )
except ImportError:
    from resume_schema import (
        CandidateProfile,
        EducationItem,
        EvidenceItem,
        ExperienceItem,
        ExtractionMeta,
        ResumeRecord,
        ResumeSource,
        SkillsProfile,
    )

SECTION_NAMES = ["SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS"]
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]*)?(?:\(\d{2,5}\)|\b\d{2,5}\b)[\d\s().-]{5,}\d")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
SKILL_ALIASES = {
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "python": "Python",
    "fastapi": "FastAPI",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "sql": "SQL",
    "docker": "Docker",
    "git": "Git",
    "rabbitmq": "RabbitMQ",
    "celery": "Celery",
    "aws eks": "AWS EKS",
    "eks": "AWS EKS",
    "reactjs": "ReactJS",
    "react": "React",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "pandas": "pandas",
    "tableau": "Tableau",
    "a/b testing": "A/B testing",
    "scikit-learn": "scikit-learn",
    "model evaluation": "model evaluation",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "css": "CSS",
    "accessibility": "Accessibility",
}
SKILL_RE = re.compile(
    r"\b(?:"
    r"aws\s+eks|eks|kubernetes|k8s|rabbitmq|celery|pytorch|tensorflow|reactjs|react|"
    r"python|fastapi|postgresql|postgres|docker|git|"
    r"rest\s+apis?|sql|pandas|tableau|a/b\s+testing|scikit-learn|model\s+evaluation|"
    r"javascript|typescript|css|accessibility"
    r")\b",
    re.IGNORECASE,
)

CURRENT_YEAR = datetime.now().year

DATE_WORDS_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|present|current|now|graduated|class of|degree conferred|summer|winter|spring|fall|autumn|since|to|from|until|full-?time|part-?time|contract|intern(?:ship)?|remote|hybrid|onsite|expected)\b",
    re.IGNORECASE,
)

DEGREE_KW = re.compile(
    r"\b(?:b\.?s\.?|b\.?a\.?|b\.?tech|b\.?e\.?|bachelor|m\.?s\.?|m\.?a\.?|m\.?tech|master|ph\.?d|doctor|associate|degree|diploma)\b",
    re.IGNORECASE,
)


def has_date_indicator(text: str) -> bool:
    if not text:
        return False
    return bool(YEAR_RE.search(text) or re.search(r"\b(?:present|current|graduated|class of)\b", text, re.IGNORECASE))


def is_date_only_line(text: str) -> bool:
    if not text or not has_date_indicator(text):
        return False
    stripped = YEAR_RE.sub("", text)
    stripped = DATE_WORDS_RE.sub("", stripped)
    stripped = re.sub(r"[\d\s.,|—–\-/()#]", "", stripped)
    return len(stripped) <= 3


def parse_date_range(text: str) -> Tuple[Optional[int], Optional[int]]:
    years = [int(m.group(0)) for m in YEAR_RE.finditer(text)]
    has_present = bool(re.search(r"\b(?:present|current|now|ongoing)\b", text, re.IGNORECASE))

    if len(years) >= 2:
        start_year = min(years)
        end_year = max(years)
        if has_present:
            end_year = max(end_year, CURRENT_YEAR)
        return start_year, end_year
    elif len(years) == 1:
        start_year = years[0]
        end_year = CURRENT_YEAR if has_present else years[0]
        return start_year, end_year
    elif has_present:
        return CURRENT_YEAR, CURRENT_YEAR
    return None, None


def parse_education_year(text: str) -> Optional[int]:
    years = [int(m.group(0)) for m in YEAR_RE.finditer(text)]
    if years:
        return max(years)
    if re.search(r"\b(?:present|current|now)\b", text, re.IGNORECASE):
        return CURRENT_YEAR
    return None


def parse_title_company(text: str) -> Tuple[Optional[str], Optional[str]]:
    cleaned = re.sub(r"^[\s•\-*]+", "", text.strip()).strip()
    parts = None

    if "|" in cleaned:
        parts = cleaned.split("|")
    elif any(d in cleaned for d in ["—", "–"]):
        parts = re.split(r"[—–]", cleaned)
    elif re.search(r"\s+-\s+", cleaned):
        parts = re.split(r"\s+-\s+", cleaned)
    elif re.search(r"\s+at\s+", cleaned, re.IGNORECASE):
        parts = re.split(r"\s+at\s+", cleaned, flags=re.IGNORECASE)
    elif "," in cleaned:
        parts = cleaned.split(",")

    if parts and len(parts) >= 2:
        title = parts[0].strip() or None
        company = " ".join(p.strip() for p in parts[1:]).strip() or None
        return title, company

    return cleaned or None, None


def parse_degree_institution(text: str) -> Tuple[Optional[str], Optional[str]]:
    cleaned = re.sub(r"^[\s•\-*]+", "", text.strip()).strip()
    parts = None

    if "|" in cleaned:
        parts = cleaned.split("|")
    elif any(d in cleaned for d in ["—", "–"]):
        parts = re.split(r"[—–]", cleaned)
    elif re.search(r"\s+-\s+", cleaned):
        parts = re.split(r"\s+-\s+", cleaned)
    elif re.search(r"\s+at\s+", cleaned, re.IGNORECASE):
        parts = re.split(r"\s+at\s+", cleaned, flags=re.IGNORECASE)
    elif re.search(r"\s+from\s+", cleaned, re.IGNORECASE):
        parts = re.split(r"\s+from\s+", cleaned, flags=re.IGNORECASE)
    elif "," in cleaned:
        parts = cleaned.split(",")

    if parts and len(parts) >= 2:
        degree = parts[0].strip() or None
        institution = " ".join(p.strip() for p in parts[1:]).strip() or None

        if institution and DEGREE_KW.search(institution) and not (degree and DEGREE_KW.search(degree)):
            degree, institution = institution, degree

        return degree, institution

    return cleaned or None, None


def parse_combined_experience_line(line: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
    start_year, end_year = parse_date_range(line)

    if "|" in line:
        segments = [s.strip() for s in line.split("|")]
        date_idx = next((idx for idx, s in enumerate(segments) if has_date_indicator(s)), -1)
        if date_idx != -1:
            non_date = [s for idx, s in enumerate(segments) if idx != date_idx]
            title, company = parse_title_company(" | ".join(non_date))
            return title, company, start_year, end_year

    paren_match = re.search(r"\(([^)]*(?:19|20)\d{2}[^)]*)\)", line)
    if paren_match:
        title_company_text = line.replace(paren_match.group(0), "").strip()
        title, company = parse_title_company(title_company_text)
        return title, company, start_year, end_year

    trailing = re.match(
        r"^(.*?)[,\s—–|]+((?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?(?:19|20)\d{2}.*)$",
        line,
        re.IGNORECASE,
    )
    if trailing:
        title, company = parse_title_company(trailing.group(1).strip())
        return title, company, start_year, end_year

    title, company = parse_title_company(line)
    return title, company, start_year, end_year


def parse_combined_education_line(line: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    year = parse_education_year(line)

    if "|" in line:
        segments = [s.strip() for s in line.split("|")]
        date_idx = next((idx for idx, s in enumerate(segments) if has_date_indicator(s)), -1)
        if date_idx != -1:
            non_date = [s for idx, s in enumerate(segments) if idx != date_idx]
            degree, institution = parse_degree_institution(" | ".join(non_date))
            return degree, institution, year

    paren_match = re.search(r"\(([^)]*(?:19|20)\d{2}[^)]*)\)", line)
    if paren_match:
        degree_inst_text = line.replace(paren_match.group(0), "").strip()
        degree, institution = parse_degree_institution(degree_inst_text)
        return degree, institution, year

    trailing = re.match(
        r"^(.*?)[,\s—–|]+((?:graduated\s+|class of\s+)?(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?(?:19|20)\d{2}.*)$",
        line,
        re.IGNORECASE,
    )
    if trailing:
        degree, institution = parse_degree_institution(trailing.group(1).strip())
        return degree, institution, year

    degree, institution = parse_degree_institution(line)
    return degree, institution, year


def extract_baseline_from_text(
    text: str,
    source_path: str = "<memory>",
    synthetic: bool = False,
) -> ResumeRecord:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections: Dict[str, List[str]] = {name: [] for name in SECTION_NAMES}
    current_section = None

    for line in lines:
        heading = line.rstrip(":").upper()
        if heading in SECTION_NAMES:
            current_section = heading
        elif current_section:
            sections[current_section].append(line)

    candidate_name = None
    for line in lines:
        upper = line.upper()
        if upper.startswith("SYNTHETIC RESUME"):
            continue
        if "|" in line and "@" in line:
            candidate_name = line.split("|")[0].strip() or None
            break
        if not any(token in upper for token in ["@", "SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS"]):
            candidate_name = line
            break

    location = None
    for line in lines[:4]:
        if "|" in line:
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 4:
                location = parts[-1]
                break

    email_match = EMAIL_RE.search(text)
    phone_match = PHONE_RE.search(text)

    # --- EXPERIENCE PARSING ---
    exp_lines = [l.strip() for l in sections["EXPERIENCE"] if l.strip()]
    experience: List[ExperienceItem] = []
    exp_idx = 0
    while exp_idx < len(exp_lines):
        line = exp_lines[exp_idx]
        if re.match(r"^[•\-*]\s+", line):
            exp_idx += 1
            continue

        next_line = exp_lines[exp_idx + 1] if exp_idx + 1 < len(exp_lines) else None

        # Layout B: Current line is title/company, next line is date range
        if next_line and is_date_only_line(next_line) and not is_date_only_line(line):
            title, company = parse_title_company(line)
            start_year, end_year = parse_date_range(next_line)
            experience.append(
                ExperienceItem(
                    title=title,
                    company=company,
                    start_year=start_year,
                    end_year=end_year,
                    evidence=f"{line}\n{next_line}",
                )
            )
            exp_idx += 2
            continue

        # Layout A: Current line contains date indicator
        if has_date_indicator(line):
            title, company, start_year, end_year = parse_combined_experience_line(line)
            experience.append(
                ExperienceItem(
                    title=title,
                    company=company,
                    start_year=start_year,
                    end_year=end_year,
                    evidence=line,
                )
            )
            exp_idx += 1
            continue

        # Fallback Layout B
        if next_line and has_date_indicator(next_line) and not has_date_indicator(line) and not re.match(r"^[•\-*]\s+", next_line):
            title, company = parse_title_company(line)
            start_year, end_year = parse_date_range(next_line)
            experience.append(
                ExperienceItem(
                    title=title,
                    company=company,
                    start_year=start_year,
                    end_year=end_year,
                    evidence=f"{line}\n{next_line}",
                )
            )
            exp_idx += 2
            continue

        exp_idx += 1

    # --- EDUCATION PARSING ---
    edu_lines = [l.strip() for l in sections["EDUCATION"] if l.strip()]
    education: List[EducationItem] = []
    edu_idx = 0
    while edu_idx < len(edu_lines):
        line = edu_lines[edu_idx]
        if re.match(r"^[•\-*]\s+", line):
            edu_idx += 1
            continue

        next_line = edu_lines[edu_idx + 1] if edu_idx + 1 < len(edu_lines) else None

        # Layout B: Current line is degree/institution, next line is date line
        if next_line and is_date_only_line(next_line) and not is_date_only_line(line):
            degree, institution = parse_degree_institution(line)
            year = parse_education_year(next_line)
            education.append(
                EducationItem(
                    degree=degree,
                    institution=institution,
                    year=year,
                    evidence=f"{line}\n{next_line}",
                )
            )
            edu_idx += 2
            continue

        # Layout A: Current line contains date indicator
        if has_date_indicator(line):
            degree, institution, year = parse_combined_education_line(line)
            education.append(
                EducationItem(
                    degree=degree,
                    institution=institution,
                    year=year,
                    evidence=line,
                )
            )
            edu_idx += 1
            continue

        # Fallback Layout B
        if next_line and has_date_indicator(next_line) and not has_date_indicator(line) and not re.match(r"^[•\-*]\s+", next_line):
            degree, institution = parse_degree_institution(line)
            year = parse_education_year(next_line)
            education.append(
                EducationItem(
                    degree=degree,
                    institution=institution,
                    year=year,
                    evidence=f"{line}\n{next_line}",
                )
            )
            edu_idx += 2
            continue

        edu_idx += 1

    skills_text = " ".join(sections["SKILLS"]) if sections["SKILLS"] else text
    explicit_skills = sorted({
        SKILL_ALIASES.get(match.group(0).strip().casefold(), match.group(0).strip())
        for match in SKILL_RE.finditer(skills_text)
    })

    warnings = []
    if not candidate_name:
        warnings.append("Name was not confidently extracted.")
    if not explicit_skills:
        warnings.append("No supported explicit skills were found.")

    evidence = []
    if candidate_name:
        evidence.append(EvidenceItem(field="candidate.name", text=candidate_name))
    if email_match:
        evidence.append(EvidenceItem(field="candidate.email", text=email_match.group(0)))
    if phone_match:
        evidence.append(EvidenceItem(field="candidate.phone", text=phone_match.group(0)))
    if location:
        evidence.append(EvidenceItem(field="candidate.location", text=location))
    if sections["SUMMARY"]:
        evidence.append(EvidenceItem(field="candidate.summary", text=sections["SUMMARY"][0]))
    for skill in explicit_skills:
        evidence.append(EvidenceItem(field="skills.explicit", text=skill))

    return ResumeRecord(
        source=ResumeSource(path=source_path, synthetic=synthetic),
        candidate=CandidateProfile(
            name=candidate_name,
            email=email_match.group(0) if email_match else None,
            phone=phone_match.group(0) if phone_match else None,
            location=location,
            summary=sections["SUMMARY"][0] if sections["SUMMARY"] else None,
        ),
        experience=experience,
        education=education,
        skills=SkillsProfile(explicit=explicit_skills),
        evidence=evidence,
        extraction=ExtractionMeta(method="rule_based", warnings=warnings),
    )


def extract_resume(
    text: str,
    source_path: str = "<memory>",
    synthetic: bool = False,
) -> dict[str, Any]:
    """Extract conservative fields and evidence from normalized resume text as a dictionary."""
    record = extract_baseline_from_text(text, source_path=source_path, synthetic=synthetic)
    return asdict(record)

