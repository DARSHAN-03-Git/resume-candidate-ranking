"""FastAPI entry point for the lightweight recruiter workflow."""

from __future__ import annotations

import gc
from pathlib import Path
import sys
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from baseline_extractor import extract_resume  # noqa: E402
from document_processing import extract_document  # noqa: E402
from ranking_core import (  # noqa: E402
    draft_outreach,
    enrich_skills,
    explain_candidate,
    find_lookalikes,
    parse_job_description,
    rank_candidates,
)
from real_pipeline import _init_torch_runtime, get_memory_diagnostics  # noqa: E402
from storage import initialize_database, list_candidates, save_candidate  # noqa: E402


app = FastAPI(title="Resume Candidate Ranking", version="0.1.0")


class JobDescription(BaseModel):
    text: str


@app.on_event("startup")
def startup() -> None:
    _init_torch_runtime()
    initialize_database()


@app.get("/health")
def health() -> dict:
    diag = get_memory_diagnostics()
    return {
        "status": "ok",
        "service": "resume-candidate-ranking",
        "memory_rss_mb": diag.get("memory_rss_mb"),
        "models": {
            "embedder_loaded": diag.get("embedder_loaded"),
            "reranker_loaded": diag.get("reranker_loaded"),
            "reranker_model": diag.get("reranker_model"),
        },
    }


@app.get("/system/memory")
def system_memory() -> dict:
    return get_memory_diagnostics()


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


@app.post("/candidates/parse")
async def parse_candidate(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
        temporary.write(await file.read())
        temporary_path = Path(temporary.name)
    try:
        document = extract_document(temporary_path)
        payload = enrich_skills(extract_resume(document.text, file.filename or "upload", synthetic=False))
        payload["extraction"].update({
            "file_type": document.file_type,
            "page_count": document.page_count,
            "word_count": document.word_count,
            "warnings": list(document.warnings) + payload["extraction"]["warnings"],
        })
        payload["source"]["path"] = file.filename or "upload"
        payload["id"] = save_candidate(payload)
        return payload
    finally:
        temporary_path.unlink(missing_ok=True)


@app.get("/candidates")
def candidates() -> list[dict]:
    return list_candidates()


@app.post("/rank")
def rank(job: JobDescription) -> dict:
    parsed_job = parse_job_description(job.text)
    records = list_candidates()
    ranked = rank_candidates(records, parsed_job)
    gc.collect()
    return {"job": parsed_job, "ranked": ranked}


@app.post("/lookalikes/{candidate_name}")
def lookalikes(candidate_name: str, limit: int = 3) -> list[dict]:
    records = list_candidates()
    try:
        results = find_lookalikes(records, candidate_name, limit)
        gc.collect()
        return results
    except StopIteration as error:
        raise HTTPException(status_code=404, detail="Candidate not found") from error


@app.post("/outreach/{candidate_name}")
def outreach(candidate_name: str, job: JobDescription) -> dict[str, str]:
    records = list_candidates()
    record = next((item for item in records if item.get("candidate", {}).get("name") == candidate_name), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return {"candidate": candidate_name, "draft": draft_outreach(record, parse_job_description(job.text))}

