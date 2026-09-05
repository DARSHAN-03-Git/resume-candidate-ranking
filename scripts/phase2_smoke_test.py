"""Run Phase 2 extraction against persisted synthetic resume fixtures."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from document_processing import extract_document  # noqa: E402


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    fixture_root = project_root / "data" / "synthetic" / "resumes"
    resume_paths = sorted(fixture_root.glob("resume_*.docx"))
    assert len(resume_paths) == 4, "Expected four persisted synthetic resume fixtures"
    assert (fixture_root / "scanned_like.pdf").is_file()

    results = [extract_document(path) for path in resume_paths]
    scanned_result = extract_document(fixture_root / "scanned_like.pdf")

    for result in results:
        assert result.file_type == "docx"
        assert result.page_count == 1
        assert result.table_count == 1
        assert "SYNTHETIC RESUME" in result.text
        for section in ("EXPERIENCE", "EDUCATION", "SKILLS"):
            assert section in result.text
        assert result.word_count > 20

    assert scanned_result.file_type == "pdf"
    assert scanned_result.likely_scanned
    assert scanned_result.warnings
    print("Phase 2 smoke test passed: four synthetic resumes and a sparse PDF extracted.")
    print(f"Resume fixtures: {len(results)}; scanned-like warnings: {len(scanned_result.warnings)}")


if __name__ == "__main__":
    main()
