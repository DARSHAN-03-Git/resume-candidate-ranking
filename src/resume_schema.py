from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class CandidateProfile:
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None

@dataclass
class ExperienceItem:
    title: Optional[str] = None
    company: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    evidence: str = ""

@dataclass
class EducationItem:
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[int] = None
    evidence: str = ""

@dataclass
class SkillsProfile:
    explicit: List[str] = field(default_factory=list)
    inferred: List[dict] = field(default_factory=list)

@dataclass
class EvidenceItem:
    field: str
    text: str
    source: str = "resume_text"

@dataclass
class ExtractionMeta:
    method: str
    warnings: List[str] = field(default_factory=list)
    file_type: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None

@dataclass
class ResumeSource:
    path: str
    synthetic: bool = False

@dataclass
class ResumeRecord:
    schema_version: str = "1.0"
    source: ResumeSource = field(default_factory=lambda: ResumeSource(path="<memory>"))
    candidate: CandidateProfile = field(default_factory=CandidateProfile)
    experience: List[ExperienceItem] = field(default_factory=list)
    education: List[EducationItem] = field(default_factory=list)
    skills: SkillsProfile = field(default_factory=SkillsProfile)
    evidence: List[EvidenceItem] = field(default_factory=list)
    extraction: ExtractionMeta = field(default_factory=lambda: ExtractionMeta(method="rule_based"))
