# Phase 2: Document Processing

## Basic layer

The basic layer supports text-based PDF and DOCX files. `pdfminer.six` reads PDF text and counts pages. `python-docx` reads paragraphs and table rows. Both formats return the same `ExtractionResult`, which includes the absolute source path, text, counts, and warnings. The scanned signal uses the named `SCANNED_TEXT_THRESHOLD` constant, currently 40 extracted characters.

The parser normalizes line endings and repeated spaces, but it does not discard the original source path or pretend that layout has been perfectly reconstructed.

Run the basic verification from the project directory:

```powershell
.\.venv\Scripts\python.exe scripts/phase2_smoke_test.py
```

The persisted fixture set contains four resume-shaped synthetic DOCX files with contact, experience, education, and skills sections, plus one sparse PDF used to exercise the scanned signal. Expected output includes:

```text
Phase 2 smoke test passed: four synthetic resumes and a sparse PDF extracted.
```

## Advanced decision gate

A PDF with pages but almost no extracted characters is marked `likely_scanned`. The current implementation reports that condition and does not invoke OCR. This is intentional: OCR introduces a new error source and dependency, so it should be enabled after a real scanned sample is inspected.

Multi-column PDFs can produce reading-order artifacts because PDF text stores positioned fragments rather than semantic paragraphs. After real sample resumes are supplied, we will inspect extraction output. If columns are misordered, the next increment will group text by layout coordinates and add a fixture-based regression test. OCR will be added only if a scanned fixture exists.

## Known limitations

- PDF tables and complex columns are not yet reconstructed semantically.
- Headers and footers may repeat in extracted PDF text.
- DOCX headers, footers, and text boxes are not yet included.
- OCR is not installed or claimed to work.
- Synthetic fixtures prove the code path, not production parsing accuracy. They are labeled and use reserved placeholder contact details.

## Common errors

- `FileNotFoundError`: pass a real path or place a fixture under the project data directory later.
- Unsupported extension: only `.pdf` and `.docx` are accepted.
- Empty PDF text: inspect whether the source is scanned before adding OCR.
- Garbled reading order: preserve the source and inspect layout before changing normalization.
