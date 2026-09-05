"""SQLAlchemy persistence with SQLite by default and PostgreSQL via DATABASE_URL."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'data' / 'candidate_ranking.db'}")
Path(PROJECT_ROOT / "data").mkdir(exist_ok=True)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})


class Base(DeclarativeBase):
    pass


class Candidate(Base):
    __tablename__ = "candidates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    source_path: Mapped[str] = mapped_column(String(500))
    payload: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


def initialize_database() -> None:
    Base.metadata.create_all(engine)


def save_candidate(payload: dict[str, Any]) -> int:
    initialize_database()
    candidate = Candidate(
        name=payload.get("candidate", {}).get("name") or "Unknown candidate",
        source_path=payload.get("source", {}).get("path", ""),
        payload=json.dumps(payload),
    )
    with Session(engine) as session:
        session.add(candidate)
        session.commit()
        session.refresh(candidate)
        return candidate.id


def list_candidates() -> list[dict[str, Any]]:
    initialize_database()
    with Session(engine) as session:
        rows = session.scalars(select(Candidate).order_by(Candidate.id)).all()
        return [{"id": row.id, **json.loads(row.payload)} for row in rows]
