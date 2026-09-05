"""Stable JSON-compatible schema for extracted resume data."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "1.0"


def empty_resume(source_path: str, synthetic: bool = False) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"path": source_path, "synthetic": synthetic},
        "candidate": {
            "name": None,
            "email": None,
            "phone": None,
            "location": None,
            "summary": None,
        },
        "experience": [],
        "education": [],
        "skills": {"explicit": [], "inferred": []},
        "evidence": [],
        "extraction": {"method": "rule_based", "warnings": []},
    }
