"""Verify baseline structured extraction on synthetic Phase 2 resumes."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline_extractor import extract_resume  # noqa: E402
from document_processing import extract_document  # noqa: E402


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = sorted((project_root / "data" / "synthetic" / "resumes").glob("resume_*.docx"))
    assert len(paths) == 4
    extracted = []
    for path in paths:
        document = extract_document(path)
        record = extract_resume(document.text, document.source_path, synthetic=True)
        assert record["schema_version"] == "1.0"
        assert record["source"]["synthetic"] is True
        assert record["candidate"]["name"]
        assert record["candidate"]["email"].endswith("example.invalid")
        assert record["experience"]
        assert record["education"]
        assert record["skills"]["explicit"]
        assert record["skills"]["inferred"] == []
        assert record["evidence"]
        extracted.append(record)
    print(f"Phase 3 smoke test passed: {len(extracted)} structured resume records created.")
    print(f"First candidate: {extracted[0]['candidate']['name']}; skills: {len(extracted[0]['skills']['explicit'])}")


if __name__ == "__main__":
    main()
