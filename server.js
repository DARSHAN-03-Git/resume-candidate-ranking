import express from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import mammoth from "mammoth";
import pdfParse from "pdf-parse";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;
const upload = multer({ storage: multer.memoryStorage() });

app.use(express.json());
app.use(express.static(path.join(__dirname, "static")));

// --- Skill normalization, inference & matching rules ---
const SKILL_ALIASES = {
  k8s: "Kubernetes",
  kubernetes: "Kubernetes",
  python: "Python",
  fastapi: "FastAPI",
  postgresql: "PostgreSQL",
  postgres: "PostgreSQL",
  sql: "SQL",
  docker: "Docker",
  git: "Git",
  rabbitmq: "RabbitMQ",
  celery: "Celery",
  "aws eks": "AWS EKS",
  eks: "AWS EKS",
  reactjs: "ReactJS",
  react: "React",
  pytorch: "PyTorch",
  tensorflow: "TensorFlow",
  "rest api": "REST APIs",
  "rest apis": "REST APIs",
  pandas: "pandas",
  tableau: "Tableau",
  "a/b testing": "A/B testing",
  "scikit-learn": "scikit-learn",
  "model evaluation": "model evaluation",
  javascript: "JavaScript",
  typescript: "TypeScript",
  css: "CSS",
  accessibility: "Accessibility",
};

const INFERENCE_RULES = [
  { skills: ["PyTorch", "TensorFlow"], inferred: "Machine Learning", rule: "PyTorch + TensorFlow" },
  { skills: ["FastAPI", "Python"], inferred: "Python Web Services", rule: "FastAPI + Python" },
  { skills: ["Docker", "Kubernetes"], inferred: "Container Orchestration", rule: "Docker + Kubernetes" },
];

function canonicalizeSkill(skill) {
  const lowered = skill.trim().toLowerCase();
  return SKILL_ALIASES[lowered] || skill.trim();
}

function normalizeSkills(skills) {
  const seen = new Set();
  const normalized = [];
  for (const skill of skills) {
    const canonical = canonicalizeSkill(skill);
    const key = canonical.toLowerCase();
    if (canonical && !seen.has(key)) {
      seen.add(key);
      normalized.push(canonical);
    }
  }
  return normalized;
}

function inferSkills(explicitSkills) {
  const explicitSet = new Set(normalizeSkills(explicitSkills).map((s) => s.toLowerCase()));
  const inferred = [];
  for (const rule of INFERENCE_RULES) {
    const hasAll = rule.skills.every((s) => explicitSet.has(s.toLowerCase()));
    if (hasAll && !explicitSet.has(rule.inferred.toLowerCase())) {
      inferred.push({ skill: rule.inferred, rule: rule.rule });
    }
  }
  return inferred;
}

function enrichSkills(record) {
  const explicit = normalizeSkills(record.skills?.explicit || []);
  record.skills = record.skills || {};
  record.skills.explicit = explicit;
  record.skills.inferred = inferSkills(explicit);
  return record;
}

export function parseJobDescription(text) {
  const yearMatches = [...text.matchAll(/\b(\d+)\+?\s+years?/gi)].map((m) => parseInt(m[1], 10));
  const minimumYears = yearMatches.length > 0 ? Math.max(...yearMatches) : 0;
  const found = [];
  const lowered = text.toLowerCase();

  for (const [alias, canonical] of Object.entries(SKILL_ALIASES)) {
    const escaped = alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`(?<!\\w)${escaped}(?!\\w)`, "i");
    if (regex.test(lowered)) {
      if (!found.includes(canonical)) {
        found.push(canonical);
      }
    }
  }

  const firstLine = text.split("\n").map((l) => l.trim()).find((l) => l.length > 0) || "";

  return {
    role: firstLine,
    required_skills: found,
    minimum_years: minimumYears,
    source_text: text,
  };
}

const TOKEN_RE = /[a-z0-9+#./-]+/gi;

function tokenize(text) {
  return (text.match(TOKEN_RE) || []).map((t) => t.toLowerCase());
}

function lexicalEmbedText(text) {
  const tokens = tokenize(text);
  const counts = {};
  for (const token of tokens) {
    counts[token] = (counts[token] || 0) + 1;
  }
  let sumSq = 0;
  for (const val of Object.values(counts)) {
    sumSq += val * val;
  }
  const magnitude = Math.sqrt(sumSq) || 1.0;
  const result = {};
  for (const [token, val] of Object.entries(counts)) {
    result[token] = val / magnitude;
  }
  return result;
}

function cosineSimilarity(left, right) {
  let sum = 0;
  for (const [token, val] of Object.entries(left)) {
    if (right[token]) {
      sum += val * right[token];
    }
  }
  return sum;
}

function candidateText(record) {
  const candidate = record.candidate || {};
  const experience = (record.experience || []).map((e) => e.evidence || "").join(" ");
  const education = (record.education || []).map((e) => e.evidence || "").join(" ");
  const explicit = (record.skills?.explicit || []).join(" ");
  return [candidate.summary, experience, education, explicit].filter(Boolean).join(" ");
}

function jaccardSimilarity(candidateSkills, requiredSkills) {
  const candidate = new Set((candidateSkills || []).map((s) => s.toLowerCase()));
  const required = new Set((requiredSkills || []).map((s) => s.toLowerCase()));
  if (candidate.size === 0 && required.size === 0) return 0.0;
  let intersection = 0;
  for (const skill of candidate) {
    if (required.has(skill)) intersection++;
  }
  const union = new Set([...candidate, ...required]).size;
  return union > 0 ? intersection / union : 0.0;
}

export function scoreCandidate(record, job, weights = { semantic: 0.5, experience: 0.3, skills: 0.2 }) {
  const cText = candidateText(record);
  const candVector = lexicalEmbedText(cText);
  const jobVector = lexicalEmbedText(job.source_text || "");
  const semantic = cosineSimilarity(candVector, jobVector);

  const startYears = (record.experience || []).map((e) => e.start_year).filter(Boolean);
  const endYears = (record.experience || []).map((e) => e.end_year).filter(Boolean);
  const minStart = startYears.length > 0 ? Math.min(...startYears) : 0;
  const maxEnd = endYears.length > 0 ? Math.max(...endYears) : 0;
  const experienceYears = minStart > 0 ? Math.max(0, maxEnd - minStart) : 0;

  const minRequiredYears = Math.max(job.minimum_years || 1, 1);
  const experienceScore = Math.min(experienceYears / minRequiredYears, 1.0);

  const explicit = record.skills?.explicit || [];
  const inferred = (record.skills?.inferred || []).map((i) => i.skill);
  const skillScore = jaccardSimilarity([...explicit, ...inferred], job.required_skills || []);

  const total = weights.semantic * semantic + weights.experience * experienceScore + weights.skills * skillScore;
  const crossEncoderScore = Math.min(1.0, Math.max(0.0, semantic * 1.05));
  const finalScore = Math.round((0.7 * total + 0.3 * Math.max(0.0, crossEncoderScore)) * 1000000) / 1000000;

  return {
    candidate: record.candidate?.name || "Unknown",
    score: finalScore,
    components: {
      semantic: Math.round(semantic * 1000000) / 1000000,
      experience: Math.round(experienceScore * 1000000) / 1000000,
      skills_jaccard: Math.round(skillScore * 1000000) / 1000000,
      cross_encoder: Math.round(crossEncoderScore * 1000000) / 1000000,
    },
    weights,
    pipeline: {
      embedding_model: "lexical-cosine-embedder",
      reranker_model: "heuristic-cross-encoder",
    },
  };
}

export function rankCandidates(records, job) {
  if (!records || records.length === 0) return [];
  const scored = records.map((record) => scoreCandidate(record, job));
  return scored.sort((a, b) => b.score - a.score);
}

function findLookalikes(records, candidateName, limit = 3) {
  const target = records.find((r) => r.candidate?.name === candidateName);
  if (!target) {
    const err = new Error("Candidate not found");
    err.statusCode = 404;
    throw err;
  }
  const targetVector = lexicalEmbedText(candidateText(target));
  const neighbors = [];
  for (const record of records) {
    if (record === target || record.candidate?.name === candidateName) continue;
    const sim = cosineSimilarity(targetVector, lexicalEmbedText(candidateText(record)));
    neighbors.push({
      candidate: record.candidate?.name,
      similarity: Math.round(sim * 1000000) / 1000000,
    });
  }
  return neighbors.sort((a, b) => b.similarity - a.similarity).slice(0, limit);
}

function draftOutreach(record, job) {
  const name = record.candidate?.name || "Candidate";
  const skills = record.skills?.explicit || [];
  const verified = skills.slice(0, 3).join(", ") || "your experience";
  const role = job.role || "this role";
  return `Hello ${name}, your resume shows verified experience with ${verified}. We are reviewing candidates for ${role}. Would you be open to a conversation?`;
}

// --- Rule-based resume text extraction ---
const SECTION_NAMES = ["SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS"];
const EMAIL_RE = /[\w.+-]+@[\w-]+(?:\.[\w-]+)+/;
const PHONE_RE = /(?:\+?\d{1,3}[\s.-]*)?(?:\(\d{2,5}\)|\b\d{2,5}\b)[\d\s().-]{5,}\d/;
const YEAR_RE = /\b(?:19|20)\d{2}\b/g;
const SKILL_RE = /\b(?:aws\s+eks|eks|kubernetes|k8s|rabbitmq|celery|pytorch|tensorflow|reactjs|react|python|fastapi|postgresql|postgres|docker|git|rest\s+apis?|sql|pandas|tableau|a\/b\s+testing|scikit-learn|model\s+evaluation|javascript|typescript|css|accessibility)\b/gi;

const CURRENT_YEAR = new Date().getFullYear();

const DATE_WORDS_RE = /\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|present|current|now|graduated|class of|degree conferred|summer|winter|spring|fall|autumn|since|to|from|until|full-?time|part-?time|contract|intern(?:ship)?|remote|hybrid|onsite|expected)\b/gi;

export function hasDateIndicator(text) {
  if (!text) return false;
  return /\b(?:19|20)\d{2}\b/.test(text) || /\b(?:present|current|graduated|class of)\b/i.test(text);
}

export function isDateOnlyLine(text) {
  if (!text || !hasDateIndicator(text)) return false;
  const stripped = text
    .replace(/\b(?:19|20)\d{2}\b/g, "")
    .replace(DATE_WORDS_RE, "")
    .replace(/[\d\s.,|—–\-/()#]/g, "");
  return stripped.length <= 3;
}

export function parseDateRange(text) {
  const years = [...text.matchAll(YEAR_RE)].map((m) => parseInt(m[0], 10));
  const hasPresent = /\b(?:present|current|now|ongoing)\b/i.test(text);

  let startYear = null;
  let endYear = null;

  if (years.length >= 2) {
    startYear = Math.min(...years);
    endYear = Math.max(...years);
    if (hasPresent) {
      endYear = Math.max(endYear, CURRENT_YEAR);
    }
  } else if (years.length === 1) {
    startYear = years[0];
    endYear = hasPresent ? CURRENT_YEAR : years[0];
  } else if (hasPresent) {
    startYear = CURRENT_YEAR;
    endYear = CURRENT_YEAR;
  }

  return { startYear, endYear };
}

export function parseEducationYear(text) {
  const years = [...text.matchAll(YEAR_RE)].map((m) => parseInt(m[0], 10));
  if (years.length > 0) {
    return Math.max(...years);
  }
  if (/\b(?:present|current|now)\b/i.test(text)) {
    return CURRENT_YEAR;
  }
  return null;
}

export function parseTitleCompany(text) {
  let cleaned = text.trim().replace(/^[\s•\-*]+/, "").trim();
  let parts = null;

  if (cleaned.includes("|")) {
    parts = cleaned.split("|");
  } else if (/[—–]/.test(cleaned)) {
    parts = cleaned.split(/[—–]/);
  } else if (/\s+-\s+/.test(cleaned)) {
    parts = cleaned.split(/\s+-\s+/);
  } else if (/\s+at\s+/i.test(cleaned)) {
    parts = cleaned.split(/\s+at\s+/i);
  } else if (cleaned.includes(",")) {
    parts = cleaned.split(",");
  }

  if (parts && parts.length >= 2) {
    const title = parts[0].trim() || null;
    const company = parts.slice(1).join(" ").trim() || null;
    return { title, company };
  }

  return {
    title: cleaned || null,
    company: null,
  };
}

export function parseDegreeInstitution(text) {
  let cleaned = text.trim().replace(/^[\s•\-*]+/, "").trim();
  let parts = null;

  if (cleaned.includes("|")) {
    parts = cleaned.split("|");
  } else if (/[—–]/.test(cleaned)) {
    parts = cleaned.split(/[—–]/);
  } else if (/\s+-\s+/.test(cleaned)) {
    parts = cleaned.split(/\s+-\s+/);
  } else if (/\s+at\s+/i.test(cleaned)) {
    parts = cleaned.split(/\s+at\s+/i);
  } else if (/\s+from\s+/i.test(cleaned)) {
    parts = cleaned.split(/\s+from\s+/i);
  } else if (cleaned.includes(",")) {
    parts = cleaned.split(",");
  }

  if (parts && parts.length >= 2) {
    let degree = parts[0].trim() || null;
    let institution = parts.slice(1).join(" ").trim() || null;

    const DEGREE_KW = /\b(?:b\.?s\.?|b\.?a\.?|b\.?tech|b\.?e\.?|bachelor|m\.?s\.?|m\.?a\.?|m\.?tech|master|ph\.?d|doctor|associate|degree|diploma)\b/i;
    if (institution && DEGREE_KW.test(institution) && !DEGREE_KW.test(degree)) {
      const tmp = degree;
      degree = institution;
      institution = tmp;
    }

    return { degree, institution };
  }

  return {
    degree: cleaned || null,
    institution: null,
  };
}

export function parseCombinedExperienceLine(line) {
  const dateRange = parseDateRange(line);

  if (line.includes("|")) {
    const segments = line.split("|").map((s) => s.trim());
    const dateIdx = segments.findIndex((s) => hasDateIndicator(s));
    if (dateIdx !== -1) {
      const nonDateSegments = segments.filter((_, idx) => idx !== dateIdx);
      const titleCompanyText = nonDateSegments.join(" | ");
      const { title, company } = parseTitleCompany(titleCompanyText);
      return {
        title,
        company,
        start_year: dateRange.startYear,
        end_year: dateRange.endYear,
      };
    }
  }

  const parenMatch = line.match(/\(([^)]*(?:19|20)\d{2}[^)]*)\)/);
  if (parenMatch) {
    const titleCompanyText = line.replace(parenMatch[0], "").trim();
    const { title, company } = parseTitleCompany(titleCompanyText);
    return {
      title,
      company,
      start_year: dateRange.startYear,
      end_year: dateRange.endYear,
    };
  }

  const trailingMatch = line.match(/^(.*?)[,\s—–|]+((?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?(?:19|20)\d{2}.*)$/i);
  if (trailingMatch) {
    const titleCompanyText = trailingMatch[1].trim();
    const { title, company } = parseTitleCompany(titleCompanyText);
    return {
      title,
      company,
      start_year: dateRange.startYear,
      end_year: dateRange.endYear,
    };
  }

  const { title, company } = parseTitleCompany(line);
  return {
    title,
    company,
    start_year: dateRange.startYear,
    end_year: dateRange.endYear,
  };
}

export function parseCombinedEducationLine(line) {
  const year = parseEducationYear(line);

  if (line.includes("|")) {
    const segments = line.split("|").map((s) => s.trim());
    const dateIdx = segments.findIndex((s) => hasDateIndicator(s));
    if (dateIdx !== -1) {
      const nonDateSegments = segments.filter((_, idx) => idx !== dateIdx);
      const degreeInstText = nonDateSegments.join(" | ");
      const { degree, institution } = parseDegreeInstitution(degreeInstText);
      return { degree, institution, year };
    }
  }

  const parenMatch = line.match(/\(([^)]*(?:19|20)\d{2}[^)]*)\)/);
  if (parenMatch) {
    const degreeInstText = line.replace(parenMatch[0], "").trim();
    const { degree, institution } = parseDegreeInstitution(degreeInstText);
    return { degree, institution, year };
  }

  const trailingMatch = line.match(/^(.*?)[,\s—–|]+((?:graduated\s+|class of\s+)?(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?(?:19|20)\d{2}.*)$/i);
  if (trailingMatch) {
    const degreeInstText = trailingMatch[1].trim();
    const { degree, institution } = parseDegreeInstitution(degreeInstText);
    return { degree, institution, year };
  }

  const { degree, institution } = parseDegreeInstitution(line);
  return { degree, institution, year };
}

export function extractResume(text, sourcePath = "<memory>", synthetic = false) {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
  const sections = { SUMMARY: [], EXPERIENCE: [], EDUCATION: [], SKILLS: [] };
  let currentSection = null;

  for (const line of lines) {
    const heading = line.replace(/:$/, "").toUpperCase();
    if (SECTION_NAMES.includes(heading)) {
      currentSection = heading;
    } else if (currentSection) {
      sections[currentSection].push(line);
    }
  }

  let candidateName = null;
  for (const line of lines) {
    const upper = line.toUpperCase();
    if (upper.startsWith("SYNTHETIC RESUME")) continue;
    if (line.includes("|") && line.includes("@")) {
      candidateName = line.split("|")[0].trim() || null;
      break;
    }
    if (!["@", "SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS"].some((token) => upper.includes(token))) {
      candidateName = line;
      break;
    }
  }

  let location = null;
  for (const line of lines.slice(0, 4)) {
    if (line.includes("|")) {
      const parts = line.split("|").map((p) => p.trim());
      if (parts.length >= 4) {
        location = parts[parts.length - 1];
        break;
      }
    }
  }

  const emailMatch = text.match(EMAIL_RE);
  const phoneMatch = text.match(PHONE_RE);

  const expLines = sections.EXPERIENCE.map((l) => l.trim()).filter(Boolean);
  const experience = [];
  let expIdx = 0;
  while (expIdx < expLines.length) {
    const line = expLines[expIdx];
    if (/^[•\-*]\s+/.test(line)) {
      expIdx++;
      continue;
    }

    const nextLine = expIdx + 1 < expLines.length ? expLines[expIdx + 1] : null;

    // Layout B: Current line is title/company, next line is date-only line
    if (nextLine && isDateOnlyLine(nextLine) && !isDateOnlyLine(line)) {
      const { title, company } = parseTitleCompany(line);
      const { startYear, endYear } = parseDateRange(nextLine);
      experience.push({
        title,
        company,
        start_year: startYear,
        end_year: endYear,
        evidence: `${line}\n${nextLine}`,
      });
      expIdx += 2;
      continue;
    }

    // Layout A: Current line has date indicator and is combined with title/company
    if (hasDateIndicator(line)) {
      const parsed = parseCombinedExperienceLine(line);
      experience.push({
        title: parsed.title,
        company: parsed.company,
        start_year: parsed.start_year,
        end_year: parsed.end_year,
        evidence: line,
      });
      expIdx++;
      continue;
    }

    // Fallback Layout B: Current line has no date, but next line contains date
    if (nextLine && hasDateIndicator(nextLine) && !hasDateIndicator(line) && !/^[•\-*]\s+/.test(nextLine)) {
      const { title, company } = parseTitleCompany(line);
      const { startYear, endYear } = parseDateRange(nextLine);
      experience.push({
        title,
        company,
        start_year: startYear,
        end_year: endYear,
        evidence: `${line}\n${nextLine}`,
      });
      expIdx += 2;
      continue;
    }

    expIdx++;
  }

  const eduLines = sections.EDUCATION.map((l) => l.trim()).filter(Boolean);
  const education = [];
  let eduIdx = 0;
  while (eduIdx < eduLines.length) {
    const line = eduLines[eduIdx];
    if (/^[•\-*]\s+/.test(line)) {
      eduIdx++;
      continue;
    }

    const nextLine = eduIdx + 1 < eduLines.length ? eduLines[eduIdx + 1] : null;

    // Layout B: Current line is degree/institution, next line is date-only line
    if (nextLine && isDateOnlyLine(nextLine) && !isDateOnlyLine(line)) {
      const { degree, institution } = parseDegreeInstitution(line);
      const year = parseEducationYear(nextLine);
      education.push({
        degree,
        institution,
        year,
        evidence: `${line}\n${nextLine}`,
      });
      eduIdx += 2;
      continue;
    }

    // Layout A: Current line has date indicator and is combined
    if (hasDateIndicator(line)) {
      const parsed = parseCombinedEducationLine(line);
      education.push({
        degree: parsed.degree,
        institution: parsed.institution,
        year: parsed.year,
        evidence: line,
      });
      eduIdx++;
      continue;
    }

    // Fallback Layout B
    if (nextLine && hasDateIndicator(nextLine) && !hasDateIndicator(line) && !/^[•\-*]\s+/.test(nextLine)) {
      const { degree, institution } = parseDegreeInstitution(line);
      const year = parseEducationYear(nextLine);
      education.push({
        degree,
        institution,
        year,
        evidence: `${line}\n${nextLine}`,
      });
      eduIdx += 2;
      continue;
    }

    eduIdx++;
  }

  const skillsText = sections.SKILLS.join(" ");
  const skillMatches = [...skillsText.matchAll(SKILL_RE)].map((m) => m[0]);
  const explicitSkills = [];
  const seenSkill = new Set();
  for (const s of skillMatches) {
    const canonical = canonicalizeSkill(s);
    const key = canonical.toLowerCase();
    if (!seenSkill.has(key)) {
      seenSkill.add(key);
      explicitSkills.push(canonical);
    }
  }

  const warnings = [];
  if (!candidateName) warnings.push("Name was not confidently extracted.");
  if (explicitSkills.length === 0) warnings.push("No supported explicit skills were found.");

  const evidence = [];
  if (candidateName) evidence.push({ field: "candidate.name", text: candidateName, source: "resume_text" });
  if (emailMatch) evidence.push({ field: "candidate.email", text: emailMatch[0], source: "resume_text" });
  if (phoneMatch) evidence.push({ field: "candidate.phone", text: phoneMatch[0], source: "resume_text" });
  if (location) evidence.push({ field: "candidate.location", text: location, source: "resume_text" });
  if (sections.SUMMARY[0]) evidence.push({ field: "candidate.summary", text: sections.SUMMARY[0], source: "resume_text" });
  for (const s of explicitSkills) {
    evidence.push({ field: "skills.explicit", text: s, source: "resume_text" });
  }

  return {
    schema_version: "1.0",
    source: { path: sourcePath, synthetic },
    candidate: {
      name: candidateName,
      email: emailMatch ? emailMatch[0] : null,
      phone: phoneMatch ? phoneMatch[0] : null,
      location,
      summary: sections.SUMMARY[0] || null,
    },
    experience,
    education,
    skills: { explicit: explicitSkills, inferred: [] },
    evidence,
    extraction: { method: "rule_based", warnings },
  };
}

// --- In-Memory Candidate Store & Pre-seeding ---
const candidatesStore = [];
let nextCandidateId = 1;

function seedCandidates() {
  const FIRST_NAMES = [
    "Asha", "Daniel", "Leila", "Mateo", "Nora", "Priya", "Jonas", "Mina", "Owen", "Rina",
    "Kai", "Sofia", "Ethan", "Amara", "Luca", "Ivy", "Noah", "Zara", "Milo", "Anika",
  ];
  const LAST_NAMES = [
    "Mehta", "Brooks", "Haddad", "Silva", "Chen", "Patel", "Fischer", "Okafor", "Nguyen", "Rossi",
  ];
  const COMPANIES = ["Northstar Labs", "Blue Oak Systems", "Cedar Health", "Orbit Commerce", "Maple Analytics"];
  const LOCATIONS = ["Austin, TX", "Boston, MA", "Chicago, IL", "Denver, CO", "Seattle, WA"];
  const BACKEND_SKILLS = ["Python", "FastAPI", "PostgreSQL", "Docker", "REST APIs", "SQL", "Git"];
  const DATA_SKILLS = ["Python", "Pandas", "SQL", "Tableau", "Git"];
  const FRONTEND_SKILLS = ["JavaScript", "TypeScript", "React", "CSS", "Accessibility", "Git"];

  for (let index = 0; index < 50; index++) {
    const first = FIRST_NAMES[index % FIRST_NAMES.length];
    const last = LAST_NAMES[(index * 3) % LAST_NAMES.length];
    const name = `${first} ${last} ${String(index + 1).padStart(2, "0")}`;
    const location = LOCATIONS[index % LOCATIONS.length];
    const company = COMPANIES[index % COMPANIES.length];
    const startYear = 2017 + (index % 6);
    const endYear = startYear + 3 + (index % 3);

    let title;
    let skills;
    let summary;
    if (index % 2 === 0) {
      title = ["Backend Engineer", "Platform Engineer", "API Engineer"][index % 3];
      skills = BACKEND_SKILLS.slice(0, 4 + (index % 4));
      summary = `Backend engineer building reliable Python services and APIs for ${company}.`;
    } else if (index % 3 === 0) {
      title = "Data Analyst";
      skills = DATA_SKILLS.slice(0, 3 + (index % 3));
      summary = `Data analyst turning operational data into decisions at ${company}.`;
    } else {
      title = "Frontend Developer";
      skills = FRONTEND_SKILLS.slice(0, 3 + (index % 4));
      summary = `Frontend developer delivering accessible web experiences for ${company}.`;
    }

    const email = `${name.toLowerCase().replace(/ /g, ".")}@example.com`;
    const phone = `+1 555 010 ${String(index).padStart(4, "0")}`;
    const explicit = normalizeSkills(skills);

    const record = {
      id: nextCandidateId++,
      schema_version: "1.0",
      source: { path: `resume_${String(index + 1).padStart(3, "0")}_${first.toLowerCase()}_${last.toLowerCase()}.docx`, synthetic: true },
      candidate: {
        name,
        email,
        phone,
        location,
        summary,
      },
      experience: [
        {
          title,
          company,
          start_year: startYear,
          end_year: endYear,
          evidence: `${title}, ${company} | ${startYear} - ${endYear}`,
        },
      ],
      education: [
        {
          degree: "B.S. Computer Science",
          institution: ["State University", "Metro Institute", "Lakeside College"][index % 3],
          year: startYear - 1,
          evidence: `B.S. Computer Science, ${["State University", "Metro Institute", "Lakeside College"][index % 3]} | ${startYear - 1}`,
        },
      ],
      skills: { explicit, inferred: inferSkills(explicit) },
      evidence: [
        { field: "candidate.name", text: name, source: "resume_text" },
        { field: "candidate.email", text: email, source: "resume_text" },
        { field: "candidate.phone", text: phone, source: "resume_text" },
        { field: "candidate.location", text: location, source: "resume_text" },
        { field: "candidate.summary", text: summary, source: "resume_text" },
        ...explicit.map((s) => ({ field: "skills.explicit", text: s, source: "resume_text" })),
      ],
      extraction: {
        method: "rule_based",
        warnings: [],
        file_type: "docx",
        page_count: 1,
        word_count: 85,
      },
    };

    candidatesStore.push(record);
  }
}

seedCandidates();

// --- HTTP Routes ---
app.get("/health", (req, res) => {
  res.json({ status: "ok", service: "resume-candidate-ranking" });
});

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "static", "index.html"));
});

app.post("/candidates/parse", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ detail: "No file uploaded" });
    }

    const originalName = req.file.originalname || "upload";
    const ext = path.extname(originalName).toLowerCase();
    if (![".pdf", ".docx", ".txt"].includes(ext)) {
      return res.status(400).json({ detail: "Only PDF and DOCX files are supported" });
    }

    let rawText = "";
    let fileType = ext.replace(/^\./, "");
    let pageCount = 1;

    if (ext === ".docx") {
      const parsed = await mammoth.extractRawText({ buffer: req.file.buffer });
      rawText = parsed.value;
    } else if (ext === ".pdf") {
      const parsed = await pdfParse(req.file.buffer);
      rawText = parsed.text;
      pageCount = parsed.numpages || 1;
    } else {
      rawText = req.file.buffer.toString("utf-8");
    }

    const payload = enrichSkills(extractResume(rawText, originalName, false));
    payload.extraction = payload.extraction || {};
    payload.extraction.file_type = fileType;
    payload.extraction.page_count = pageCount;
    payload.extraction.word_count = rawText.split(/\s+/).filter(Boolean).length;
    payload.source = { path: originalName, synthetic: false };

    payload.id = nextCandidateId++;
    candidatesStore.push(payload);

    res.json(payload);
  } catch (err) {
    console.error("Failed to parse resume:", err);
    res.status(500).json({ detail: err.message || "Failed to parse document" });
  }
});

app.get("/candidates", (req, res) => {
  res.json(candidatesStore);
});

app.post("/rank", (req, res) => {
  const text = req.body?.text || "";
  const parsedJob = parseJobDescription(text);
  const ranked = rankCandidates(candidatesStore, parsedJob);
  res.json({ job: parsedJob, ranked });
});

app.post("/lookalikes/:candidateName", (req, res) => {
  try {
    const candidateName = req.params.candidateName;
    const limit = parseInt(req.query.limit || req.body?.limit || 3, 10);
    const results = findLookalikes(candidatesStore, candidateName, limit);
    res.json(results);
  } catch (err) {
    res.status(err.statusCode || 500).json({ detail: err.message });
  }
});

app.post("/outreach/:candidateName", (req, res) => {
  const candidateName = req.params.candidateName;
  const record = candidatesStore.find((r) => r.candidate?.name === candidateName);
  if (!record) {
    return res.status(404).json({ detail: "Candidate not found" });
  }
  const job = parseJobDescription(req.body?.text || "");
  res.json({ candidate: candidateName, draft: draftOutreach(record, job) });
});

if (process.env.NODE_ENV !== "test") {
  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

export default app;
