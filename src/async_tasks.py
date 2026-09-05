"""Celery task boundary for asynchronous batch parsing.

The task is importable without Celery so the lightweight local workflow remains
usable; workers require the optional Celery dependency and a RabbitMQ broker.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

try:
    from celery import Celery
except ImportError:  # pragma: no cover - exercised only without optional runtime
    Celery = None  # type: ignore[assignment]

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from baseline_extractor import extract_resume  # noqa: E402
from document_processing import extract_document  # noqa: E402
from ranking_core import enrich_skills  # noqa: E402


BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
celery_app = Celery("resume_tasks", broker=BROKER_URL) if Celery else None


if celery_app:
    @celery_app.task
    def parse_resume_file(path: str) -> dict:
        document = extract_document(path)
        return enrich_skills(extract_resume(document.text, str(path)))
else:
    def parse_resume_file(path: str) -> dict:
        document = extract_document(path)
        return enrich_skills(extract_resume(document.text, str(path)))
