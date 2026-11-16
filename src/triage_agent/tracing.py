"""Observability: every plan decision, tool call and verdict becomes one JSON line.

Traces go to runs/<run_id>.jsonl and to the `triage` logger. The format is flat JSON
so it can be shipped to any log stack (ELK, Datadog, CloudWatch) without changes.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("triage")


def digest(value: Any, max_len: int = 400) -> str:
    """Short, single-line preview of a tool result for the log."""
    text = json.dumps(value, default=str, sort_keys=True)
    return text if len(text) <= max_len else text[:max_len] + "...(truncated)"


class Tracer:
    def __init__(self, runs_dir: Path | None = None, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.events: list[dict[str, Any]] = []
        self._path: Path | None = None
        if runs_dir is not None:
            runs_dir.mkdir(parents=True, exist_ok=True)
            self._path = runs_dir / f"{self.run_id}.jsonl"

    @property
    def path(self) -> Path | None:
        return self._path

    def event(self, kind: str, **fields: Any) -> None:
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "event": kind, **fields}
        self.events.append(record)
        line = json.dumps(record, default=str)
        logger.info(line)
        if self._path is not None:
            with self._path.open("a") as fh:
                fh.write(line + "\n")
