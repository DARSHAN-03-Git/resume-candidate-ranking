import sys
from src.baseline_extractor import extract_baseline_from_text
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

print("ALL PYTHON TESTS COMPLETED SUCCESSFULLY! 🎉")
