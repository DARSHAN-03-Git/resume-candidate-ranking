"""Neo4j-backed canonical skill normalization and adjacency queries."""

from __future__ import annotations

import os
from typing import Iterable

SKILL_ALIASES = {
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "python": "Python",
    "fastapi": "FastAPI",
    "postgresql": "PostgreSQL",
    "sql": "SQL",
    "docker": "Docker",
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "pandas": "pandas",
    "tableau": "Tableau",
    "a/b testing": "A/B testing",
    "scikit-learn": "scikit-learn",
    "model evaluation": "model evaluation",
    "git": "Git",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "react": "React",
    "css": "CSS",
    "accessibility": "Accessibility",
}
INFERENCE_RULES = (
    ({"PyTorch", "TensorFlow"}, "Machine Learning", "PyTorch + TensorFlow"),
    ({"FastAPI", "Python"}, "Python Web Services", "FastAPI + Python"),
    ({"Docker", "Kubernetes"}, "Container Orchestration", "Docker + Kubernetes"),
)


class SkillGraph:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None) -> None:
        self.driver = None
        configured_uri = uri or os.getenv("NEO4J_URI")
        self._fallback = not configured_uri
        try:
            if self._fallback:
                raise RuntimeError("NEO4J_URI is not configured")
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                configured_uri,
                auth=(user or os.getenv("NEO4J_USER", "neo4j"), password or os.getenv("NEO4J_PASSWORD", "resume-dev-only")),
            )
        except Exception:
            self._fallback = True
        self._seeded = False

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

    def seed(self) -> dict[str, int]:
        canonical_names = sorted(
            set(SKILL_ALIASES.values())
            | {alias for alias in SKILL_ALIASES if alias != SKILL_ALIASES[alias].casefold()}
            | {required_skill for required, _, _ in INFERENCE_RULES for required_skill in required}
            | {item[1] for item in INFERENCE_RULES}
        )
        aliases = [(alias, canonical) for alias, canonical in SKILL_ALIASES.items() if alias != canonical.casefold()]
        edges = [(source, target, basis) for required, target, basis in INFERENCE_RULES for source in required]
        if not self._fallback:
            try:
                with self.driver.session() as session:
                    session.run("CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (skill:Skill) REQUIRE skill.name IS UNIQUE").consume()
                    session.run("UNWIND $names AS name MERGE (:Skill {name: name})", names=canonical_names).consume()
                    session.run(
                        "UNWIND $aliases AS item MATCH (alias:Skill {name: item.alias}), (canonical:Skill {name: item.canonical}) "
                        "MERGE (alias)-[:ALIAS_OF]->(canonical)",
                        aliases=[{"alias": alias, "canonical": canonical} for alias, canonical in aliases],
                    ).consume()
                    session.run(
                        "UNWIND $edges AS item MATCH (source:Skill {name: item.source}), (target:Skill {name: item.target}) "
                        "MERGE (source)-[edge:IMPLIES {basis: item.basis}]->(target)",
                        edges=[{"source": source, "target": target, "basis": basis} for source, target, basis in edges],
                    ).consume()
            except Exception:
                self._fallback = True
        self._seeded = True
        return {
            "skills": len(canonical_names),
            "aliases": len(aliases),
            "adjacency_edges": len(edges),
            "backend": "in-memory-fallback" if self._fallback else "neo4j",
        }

    def _ensure_seeded(self) -> None:
        if not self._seeded:
            self.seed()

    def canonicalize(self, skill: str) -> str:
        self._ensure_seeded()
        if self._fallback:
            return SKILL_ALIASES.get(skill.casefold().strip(), skill.strip())
        with self.driver.session() as session:
            record = session.run(
                "MATCH (input:Skill) WHERE toLower(input.name) = toLower($skill) "
                "OPTIONAL MATCH (input)-[:ALIAS_OF]->(canonical:Skill) "
                "RETURN coalesce(canonical.name, input.name) AS canonical LIMIT 1",
                skill=skill.strip(),
            ).single()
        return record["canonical"] if record else skill.strip()

    def infer(self, explicit_skills: Iterable[str]) -> list[dict[str, str]]:
        self._ensure_seeded()
        canonical = [self.canonicalize(skill) for skill in explicit_skills]
        if self._fallback:
            explicit = set(canonical)
            return [
                {"skill": target, "basis": basis, "type": "inferred"}
                for required, target, basis in INFERENCE_RULES
                if required.issubset(explicit) and target not in explicit
            ]
        with self.driver.session() as session:
            rows = session.run(
                "MATCH (source:Skill)-[edge:IMPLIES]->(target:Skill) "
                "WHERE source.name IN $skills "
                "WITH target, collect(DISTINCT source.name) AS matched, collect(DISTINCT edge.basis)[0] AS basis "
                "WHERE size(matched) = size([(target)<-[:IMPLIES]-() | 1]) "
                "RETURN target.name AS skill, basis ORDER BY skill",
                skills=canonical,
            )
            return [{"skill": row["skill"], "basis": row["basis"], "type": "inferred"} for row in rows]

    def adjacency(self, skill: str) -> list[str]:
        self._ensure_seeded()
        canonical = self.canonicalize(skill)
        if self._fallback:
            neighbors = []
            for required, target, _ in INFERENCE_RULES:
                if canonical in required:
                    neighbors.append(target)
            return sorted(set(neighbors))
        with self.driver.session() as session:
            rows = session.run(
                "MATCH (source:Skill {name: $skill})-[edge:IMPLIES|ALIAS_OF]-(neighbor:Skill) "
                "RETURN DISTINCT neighbor.name AS name ORDER BY name",
                skill=canonical,
            )
            return [row["name"] for row in rows]


_graph: SkillGraph | None = None


def get_skill_graph() -> SkillGraph:
    global _graph
    if _graph is None:
        _graph = SkillGraph()
    return _graph
