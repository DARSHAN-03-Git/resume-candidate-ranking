from typing import List, Dict, Any, Optional
from src.resume_schema import ResumeRecord

def compute_experience_years(record: ResumeRecord) -> int:
    start_years = [e.start_year for e in record.experience if e.start_year]
    end_years = [e.end_year for e in record.experience if e.end_year]
    if start_years and end_years:
        return max(0, max(end_years) - min(start_years))
    return 0

def compute_experience_score(record: ResumeRecord, minimum_years: int = 1) -> float:
    experience_years = compute_experience_years(record)
    min_required = max(1, minimum_years)
    return min(1.0, experience_years / min_required)
