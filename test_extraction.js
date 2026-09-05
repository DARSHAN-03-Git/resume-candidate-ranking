import assert from "assert";
import { extractResume, scoreCandidate, parseJobDescription } from "./server.js";

console.log("=== RUNNING RESUME EXTRACTION & SCORING TESTS ===\n");

// --- TEST 1: Layout B (Title/Company on one line, Date on very next line) ---
console.log("Test 1: Layout B (Separate-line experience & education)...");
const resumeLayoutB = `
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
`;

const parsedB = extractResume(resumeLayoutB);

console.log("Parsed Experience (Layout B):", parsedB.experience);
console.log("Parsed Education (Layout B):", parsedB.education);

assert.strictEqual(parsedB.experience.length, 1, "Should extract exactly 1 experience entry");
assert.strictEqual(parsedB.experience[0].title, "Backend Engineer", "Title must be 'Backend Engineer'");
assert.strictEqual(parsedB.experience[0].company, "Northwind Data Systems", "Company must be 'Northwind Data Systems'");
assert.strictEqual(parsedB.experience[0].start_year, 2021, "Start year must be 2021");
assert(parsedB.experience[0].end_year >= 2021, "End year must represent present/current year (>= 2021)");

assert.strictEqual(parsedB.education.length, 1, "Should extract exactly 1 education entry");
assert.strictEqual(parsedB.education[0].degree, "B.Tech in Computer Science", "Degree must be 'B.Tech in Computer Science'");
assert.strictEqual(parsedB.education[0].institution, "Riverdale Institute of Technology", "Institution must be 'Riverdale Institute of Technology'");
assert.strictEqual(parsedB.education[0].year, 2019, "Graduation year must be 2019");

console.log("✔ Test 1 PASSED: Layout B correctly extracted title, company, degree, institution, and dates.\n");


// --- TEST 2: Layout A (Title/Company and Date on the same line) ---
console.log("Test 2: Layout A (Same-line experience & education)...");
const resumeLayoutA = `
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
`;

const parsedA = extractResume(resumeLayoutA);

console.log("Parsed Experience (Layout A):", parsedA.experience);
console.log("Parsed Education (Layout A):", parsedA.education);

assert.strictEqual(parsedA.experience.length, 2, "Should extract 2 experience entries");

// Entry 1
assert.strictEqual(parsedA.experience[0].title, "Backend Engineer");
assert.strictEqual(parsedA.experience[0].company, "Northwind Data Systems");
assert.strictEqual(parsedA.experience[0].start_year, 2021);
assert.strictEqual(parsedA.experience[0].end_year, 2024);

// Entry 2
assert.strictEqual(parsedA.experience[1].title, "Backend Engineer");
assert.strictEqual(parsedA.experience[1].company, "Northwind Data Systems");
assert.strictEqual(parsedA.experience[1].start_year, 2021);
assert(parsedA.experience[1].end_year >= 2021);

assert.strictEqual(parsedA.education.length, 2, "Should extract 2 education entries");
assert.strictEqual(parsedA.education[0].degree, "B.Tech in Computer Science");
assert.strictEqual(parsedA.education[0].institution, "Riverdale Institute of Technology");
assert.strictEqual(parsedA.education[0].year, 2019);

assert.strictEqual(parsedA.education[1].degree, "B.S. Computer Science");
assert.strictEqual(parsedA.education[1].institution, "State University");
assert.strictEqual(parsedA.education[1].year, 2018);

console.log("✔ Test 2 PASSED: Layout A correctly extracted title, company, degree, institution, and dates.\n");


// --- TEST 3: Downstream Experience Scoring Verification ---
console.log("Test 3: Downstream experience score verification...");
const job = parseJobDescription("Backend Engineer\n3 years Python FastAPI PostgreSQL Docker experience");

const scoreResultB = scoreCandidate(parsedB, job);
console.log("Scored Candidate components:", scoreResultB.components);

// For 2021 to Present (2026), experience is 5 years. Job requires 3 years.
// Experience score should be min(5/3, 1.0) = 1.0!
assert.strictEqual(scoreResultB.components.experience, 1.0, "Experience score should be 1.0 for 5 years with 3 required");
assert(scoreResultB.score > 0, "Overall score should be computed properly");

console.log("✔ Test 3 PASSED: Downstream experience score computed directly from start/end years.\n");

console.log("ALL TESTS COMPLETED SUCCESSFULLY! 🎉");
