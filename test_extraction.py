import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.baseline_extractor import extract_baseline_from_text, extract_resume
from src.ranking_core import compute_experience_years, compute_experience_score

print("=== RUNNING PYTHON RESUME EXTRACTION & SCORING TESTS ===\n")

# --- TEST 1: Layout B (Separate-line experience & education) ---
print("Test 1: Layout B (Separate-line experience & education)...")
resume_layout_b = """
SUMMARY
Experienced backend engineer building scalable systems.

EXPERIENCE
Backend Engineer — Northwind Data Systems
June 2021 – Present
- Built microservices in Python and FastAPI
- Optimized PostgreSQL queries

EDUCATION
B.Tech in Computer Science — Riverdale Institute of Technology
Graduated 2019

SKILLS
Python, FastAPI, PostgreSQL, Docker
"""

parsed_b = extract_baseline_from_text(resume_layout_b)

print(f"Parsed Experience (Layout B): {parsed_b.experience}")
print(f"Parsed Education (Layout B): {parsed_b.education}")

assert len(parsed_b.experience) == 1, "Should extract exactly 1 experience entry"
assert parsed_b.experience[0].title == "Backend Engineer", f"Expected 'Backend Engineer', got '{parsed_b.experience[0].title}'"
assert parsed_b.experience[0].company == "Northwind Data Systems", f"Expected 'Northwind Data Systems', got '{parsed_b.experience[0].company}'"
assert parsed_b.experience[0].start_year == 2021, f"Expected 2021, got {parsed_b.experience[0].start_year}"
assert parsed_b.experience[0].end_year >= 2021, f"Expected end_year >= 2021, got {parsed_b.experience[0].end_year}"

assert len(parsed_b.education) == 1, "Should extract exactly 1 education entry"
assert parsed_b.education[0].degree == "B.Tech in Computer Science", f"Expected 'B.Tech in Computer Science', got '{parsed_b.education[0].degree}'"
assert parsed_b.education[0].institution == "Riverdale Institute of Technology", f"Expected 'Riverdale Institute of Technology', got '{parsed_b.education[0].institution}'"
assert parsed_b.education[0].year == 2019, f"Expected 2019, got {parsed_b.education[0].year}"

print("✔ Test 1 PASSED: Layout B correctly extracted title, company, degree, institution, and dates.\n")


# --- TEST 2: Layout A (Same-line experience & education) ---
print("Test 2: Layout A (Same-line experience & education)...")
resume_layout_a = """
SUMMARY
Backend developer with production experience.

EXPERIENCE
Backend Engineer, Northwind Data Systems | 2021 - 2024
Backend Engineer — Northwind Data Systems | June 2021 – Present

EDUCATION
B.Tech in Computer Science — Riverdale Institute of Technology | Graduated 2019
B.S. Computer Science, State University | 2018

SKILLS
Python, FastAPI, PostgreSQL, Docker
"""

parsed_a = extract_baseline_from_text(resume_layout_a)

print(f"Parsed Experience (Layout A): {parsed_a.experience}")
print(f"Parsed Education (Layout A): {parsed_a.education}")

assert len(parsed_a.experience) == 2, "Should extract 2 experience entries"

# Entry 1
assert parsed_a.experience[0].title == "Backend Engineer"
assert parsed_a.experience[0].company == "Northwind Data Systems"
assert parsed_a.experience[0].start_year == 2021
assert parsed_a.experience[0].end_year == 2024

# Entry 2
assert parsed_a.experience[1].title == "Backend Engineer"
assert parsed_a.experience[1].company == "Northwind Data Systems"
assert parsed_a.experience[1].start_year == 2021
assert parsed_a.experience[1].end_year >= 2021

assert len(parsed_a.education) == 2, "Should extract 2 education entries"
assert parsed_a.education[0].degree == "B.Tech in Computer Science"
assert parsed_a.education[0].institution == "Riverdale Institute of Technology"
assert parsed_a.education[0].year == 2019

assert parsed_a.education[1].degree == "B.S. Computer Science"
assert parsed_a.education[1].institution == "State University"
assert parsed_a.education[1].year == 2018

print("✔ Test 2 PASSED: Layout A correctly extracted title, company, degree, institution, and dates.\n")


# --- TEST 3: Downstream Experience Scoring Verification ---
print("Test 3: Downstream experience score verification...")
exp_years = compute_experience_years(parsed_b)
exp_score = compute_experience_score(parsed_b, minimum_years=3)

print(f"Computed experience years: {exp_years}")
print(f"Computed experience score (req: 3 yrs): {exp_score}")

# For 2021 to Present (>= 2026), experience is at least 5 years. For min 3 years, score is 1.0
assert exp_years >= 5, f"Expected exp_years >= 5, got {exp_years}"
assert exp_score == 1.0, f"Expected exp_score == 1.0, got {exp_score}"

print("✔ Test 3 PASSED: Downstream experience score computed directly from start/end years.\n")

# --- TEST 4: Phone with parenthesis & All 12 Skills Extraction ---
print("Test 4: Phone with parenthesis & 12 skills extraction...")
resume_12_skills = """
Alex Morgan | (555) 019-4482 | alex.morgan@example.com | New York, NY

SUMMARY
Senior full-stack and cloud engineer with deep distributed systems experience.

EXPERIENCE
Lead Systems Engineer — Northwind Systems
2019 - Present
- Built distributed messaging pipelines with RabbitMQ and Celery
- Deployed microservices to AWS EKS with Kubernetes

EDUCATION
B.S. in Computer Science — State University
2015 - 2019

SKILLS
Docker, FastAPI, Git, PostgreSQL, Python, RabbitMQ, Celery, Kubernetes/K8s, AWS EKS, ReactJS, PyTorch, TensorFlow
"""

parsed_12 = extract_baseline_from_text(resume_12_skills)
dict_12 = extract_resume(resume_12_skills)

print(f"Extracted Candidate Phone: {parsed_12.candidate.phone}")
print(f"Extracted Explicit Skills ({len(parsed_12.skills.explicit)}): {parsed_12.skills.explicit}")

# Verify phone preserves opening parenthesis
assert parsed_12.candidate.phone == "(555) 019-4482", f"Expected '(555) 019-4482', got '{parsed_12.candidate.phone}'"
assert dict_12["candidate"]["phone"] == "(555) 019-4482", f"Expected dict phone '(555) 019-4482', got '{dict_12['candidate']['phone']}'"

# Verify all 12 skills are present in parsed_12.skills.explicit
expected_12_skills = [
    "AWS EKS",
    "Celery",
    "Docker",
    "FastAPI",
    "Git",
    "Kubernetes",
    "PostgreSQL",
    "PyTorch",
    "Python",
    "RabbitMQ",
    "ReactJS",
    "TensorFlow",
]

assert len(parsed_12.skills.explicit) == 12, f"Expected exactly 12 skills, got {len(parsed_12.skills.explicit)}: {parsed_12.skills.explicit}"
assert parsed_12.skills.explicit == expected_12_skills, f"Skills mismatch: expected {expected_12_skills}, got {parsed_12.skills.explicit}"
assert dict_12["skills"]["explicit"] == expected_12_skills, f"Dict skills mismatch: {dict_12['skills']['explicit']}"

print("✔ Test 4 PASSED: (555) 019-4482 retains opening parenthesis and all 12 skills extracted into skills.explicit.\n")

print("ALL PYTHON TESTS COMPLETED SUCCESSFULLY! 🎉")
