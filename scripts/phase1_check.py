"""Verify the Phase 1 documentation contract without external dependencies."""

from pathlib import Path


REQUIRED_FILES = (
    Path("README.md"),
    Path("docs/architecture.md"),
    Path("docs/data-ethics.md"),
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    missing = [path for path in REQUIRED_FILES if not (project_root / path).is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Phase 1 check failed; missing: {missing_text}")

    architecture = (project_root / "docs/architecture.md").read_text(encoding="utf-8")
    ethics = (project_root / "docs/data-ethics.md").read_text(encoding="utf-8")
    required_terms = (
        "FastAPI",
        "Weaviate",
        "Neo4j",
        "RabbitMQ",
        "Explicit evidence",
        "Synthetic",
    )
    combined_text = (architecture + ethics).lower()
    missing_terms = [term for term in required_terms if term.lower() not in combined_text]
    if missing_terms:
        missing_text = ", ".join(missing_terms)
        raise SystemExit(f"Phase 1 check failed; missing terms: {missing_text}")

    print("Phase 1 check passed: architecture and data contract are present.")


if __name__ == "__main__":
    main()
