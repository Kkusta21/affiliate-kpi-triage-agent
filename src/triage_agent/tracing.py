"""Observability: every plan decision, tool call, model call and verdict becomes one JSON line.

Traces go to runs/<run_id>.jsonl and to the `triage` logger. The format is flat JSON
so it can be shipped to any log stack (ELK, Datadog, CloudWatch) without changes.

Every event carries the run id and, once known, the campaign id. Values are redacted
before they are written: anything under a secret-looking key, and anything that looks
like an API key or bearer token inside a string.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("triage")

_SECRET_KEY_RE = re.compile(r"(api[_-]?key|authorization|token|secret|password|cookie)", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(r"(sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9._-]{8,})")
REDACTED = "[redacted]"


def redact(value: Any, key: str = "") -> Any:
    """Recursively remove secrets from a value that is about to be logged."""
    if key and _SECRET_KEY_RE.search(key) and not key.endswith("_tokens"):
        return REDACTED
    if isinstance(value, dict):
        return {k: redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _SECRET_VALUE_RE.sub(REDACTED, value)
    return value


def digest(value: Any, max_len: int = 400) -> str:
    """Short, single-line preview of a tool result for the log."""
    text = json.dumps(redact(value), default=str, sort_keys=True)
    return text if len(text) <= max_len else text[:max_len] + "...(truncated)"


class Tracer:
    def __init__(self, runs_dir: Path | None = None, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.events: list[dict[str, Any]] = []
        self._context: dict[str, Any] = {}
        self._path: Path | None = None
        if runs_dir is not None:
            runs_dir.mkdir(parents=True, exist_ok=True)
            self._path = runs_dir / f"{self.run_id}.jsonl"

    @property
    def path(self) -> Path | None:
        return self._path

    def bind(self, **fields: Any) -> None:
        """Attach fields (for example campaign_id) to every later event of this run."""
        self._context.update(fields)

    def event(self, kind: str, **fields: Any) -> None:
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "event": kind, **self._context, **redact(fields)}
        self.events.append(record)
        line = json.dumps(record, default=str)
        logger.info(line)
        if self._path is not None:
            with self._path.open("a") as fh:
                fh.write(line + "\n")
