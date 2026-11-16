"""Input validation, refusal rules and prompt injection defences.

Three ideas keep the agent safe:
1. Nothing from outside (alert note, page title, runbook text) is ever treated as an
   instruction. Text that looks like an instruction is removed before a model sees it.
2. The agent is read-only plus "draft". Requests to change campaigns are refused and
   routed to a human owner.
3. The model never supplies tool arguments such as URLs or campaign ids. Code does.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from .models import Alert

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)",
    r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)",
    r"\bsystem\s*(prompt|message)?\s*:",
    r"\b(developer|assistant)\s+message\b",
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\b",
    r"</?\s*(system|assistant|tool|instructions?)\s*>",
    r"reveal\s+.{0,40}(prompt|api\s*key|secret|credentials?)",
    r"\bjailbreak\b",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

# "pause the campaign", "block sub_778", "raise the payout" and similar
_MUTATION_RE = re.compile(
    r"\b(pause|stop|disable|block|ban|blacklist|delete|remove|change|increase|decrease|raise|lower|"
    r"update|edit|approve|reject)\b[^.!?\n]{0,60}?\b(campaigns?|offers?|payouts?|affiliates?|sources?|"
    r"sub[-_ ]?ids?|sub_\w+|caps?|budgets?|conversions?)\b",
    re.IGNORECASE,
)

REMOVED = "[removed: suspected prompt injection]"


def looks_like_injection(text: str | None) -> bool:
    return bool(text) and bool(_INJECTION_RE.search(text or ""))


def sanitize_untrusted(text: str | None, max_len: int = 200) -> tuple[str, bool]:
    """Return (safe_text, was_flagged). Flagged text is replaced, not passed through."""
    if not text:
        return "", False
    if looks_like_injection(text):
        return REMOVED, True
    cleaned = re.sub(r"[\x00-\x1f<>{}`]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_len], False


def find_refused_requests(note: str | None) -> list[dict[str, str]]:
    """Requests to change something. The agent diagnoses and drafts, it does not act."""
    if not note:
        return []
    refused = []
    for match in _MUTATION_RE.finditer(note):
        refused.append(
            {
                "request": match.group(0)[:120],
                "reason": "This agent is read-only. Changes to campaigns, offers, payouts or traffic "
                "sources need a human with the right permissions.",
            }
        )
    return refused


def validate_alert(raw: Any) -> tuple[Alert | None, dict[str, Any] | None]:
    """Return (alert, None) or (None, rejection)."""
    if not isinstance(raw, dict):
        return None, {"code": "invalid_input", "message": "Alert must be a JSON object.", "errors": []}
    try:
        return Alert.model_validate(raw), None
    except ValidationError as exc:
        errors = [{"field": ".".join(str(p) for p in e["loc"]) or "(root)", "problem": e["msg"]} for e in exc.errors()]
        return None, {"code": "invalid_input", "message": "Alert failed validation.", "errors": errors}
