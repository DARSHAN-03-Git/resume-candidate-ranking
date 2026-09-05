#!/usr/bin/env python3
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baseline_extractor import extract_baseline_from_text
from src.ranking_core import compute_experience_years, compute_experience_score

def run_smoke_test():
    print("Running phase3 smoke test on resume extraction...")
    sample = """
EXPERIENCE
Backend Engineer — Northwind Data Systems
June 2021 – Present

EDUCATION
B.Tech in Computer Science — Riverdale Institute of Technology
Graduated 2019
"""
    record = extract_baseline_from_text(sample)
    assert len(record.experience) == 1
    assert record.experience[0].title == "Backend Engineer"
    assert record.experience[0].company == "Northwind Data Systems"
    assert record.experience[0].start_year == 2021
    assert record.experience[0].end_year >= 2021

    assert len(record.education) == 1
    assert record.education[0].degree == "B.Tech in Computer Science"
    assert record.education[0].institution == "Riverdale Institute of Technology"
    assert record.education[0].year == 2019

    exp_years = compute_experience_years(record)
    assert exp_years >= 5
    print("Phase 3 smoke test PASSED successfully.")

if __name__ == "__main__":
    run_smoke_test()
