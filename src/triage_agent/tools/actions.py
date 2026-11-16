"""The only two tools with side effects: the incident store (CRUD) and the alert drafter.

Neither touches a campaign. Incidents are internal records. Alerts are drafts that a
human approves and sends.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..models import AlertDraft, IncidentRef, RootCause, Verdict

CHANNEL = {
    "tech_oncall": "pager + #ops-alerts",
    "tracking_team": "#tracking-support",
    "affiliate_manager": "direct message",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class IncidentStore:
    """Small SQLite CRUD store. One open incident per campaign and root cause."""

    def __init__(self, db_path: str = ":memory:", clock: Callable[[], str] = _utcnow) -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._clock = clock
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS incidents (
                   seq INTEGER PRIMARY KEY AUTOINCREMENT,
                   campaign_id TEXT NOT NULL,
                   root_cause TEXT NOT NULL,
                   severity TEXT NOT NULL,
                   status TEXT NOT NULL DEFAULT 'open',
                   occurrences INTEGER NOT NULL DEFAULT 1,
                   summary TEXT NOT NULL,
                   last_alert_id TEXT NOT NULL,
                   created_at TEXT NOT NULL,
                   updated_at TEXT NOT NULL)"""
        )
        # At most one OPEN incident per campaign and root cause, enforced by the database, so two
        # workers that triage the same problem at the same time cannot both create one.
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS one_open_incident_per_cause"
            " ON incidents (campaign_id, root_cause) WHERE status = 'open'"
        )
        self._conn.commit()

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["incident_id"] = f"INC-{data.pop('seq'):04d}"
        return data

    def create(self, campaign_id: str, root_cause: str, severity: str, summary: str, alert_id: str) -> dict[str, Any]:
        """Raises sqlite3.IntegrityError if an open incident for the same campaign and cause exists."""
        now = self._clock()
        try:
            cur = self._conn.execute(
                "INSERT INTO incidents (campaign_id, root_cause, severity, summary, last_alert_id, created_at,"
                " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (campaign_id, root_cause, severity, summary, alert_id, now, now),
            )
        except sqlite3.IntegrityError:
            self._conn.rollback()
            raise
        self._conn.commit()
        return self.get(f"INC-{cur.lastrowid:04d}")  # type: ignore[return-value]

    def get(self, incident_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM incidents WHERE seq = ?", (int(incident_id.split("-")[1]),)).fetchone()
        return self._to_dict(row) if row else None

    def find_open(self, campaign_id: str, root_cause: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM incidents WHERE campaign_id = ? AND root_cause = ? AND status = 'open'",
            (campaign_id, root_cause),
        ).fetchone()
        return self._to_dict(row) if row else None

    def record_repeat(self, incident_id: str, severity: str, alert_id: str) -> dict[str, Any]:
        self._conn.execute(
            "UPDATE incidents SET occurrences = occurrences + 1, severity = ?, last_alert_id = ?, updated_at = ?"
            " WHERE seq = ?",
            (severity, alert_id, self._clock(), int(incident_id.split("-")[1])),
        )
        self._conn.commit()
        return self.get(incident_id)  # type: ignore[return-value]

    def close(self, incident_id: str) -> dict[str, Any] | None:
        self._conn.execute(
            "UPDATE incidents SET status = 'closed', updated_at = ? WHERE seq = ?",
            (self._clock(), int(incident_id.split("-")[1])),
        )
        self._conn.commit()
        return self.get(incident_id)

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._conn.execute("SELECT * FROM incidents WHERE status = ? ORDER BY seq", (status,)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM incidents ORDER BY seq").fetchall()
        return [self._to_dict(r) for r in rows]


def upsert_incident(store: IncidentStore, campaign_id: str, alert_id: str, verdict: Verdict) -> IncidentRef:
    """Create an incident, or bump the open one so repeated alerts do not spam the queue.

    Check-then-insert is racy across workers. The unique index closes the race: the loser of
    a concurrent insert gets an IntegrityError and records a repeat on the winner's incident.
    """
    cause = verdict.root_cause.value

    def bump(existing: dict[str, Any]) -> IncidentRef:
        updated = store.record_repeat(existing["incident_id"], verdict.severity.value, alert_id)
        return IncidentRef(action="updated", incident_id=updated["incident_id"], occurrences=updated["occurrences"])

    existing = store.find_open(campaign_id, cause)
    if existing:
        return bump(existing)
    try:
        created = store.create(campaign_id, cause, verdict.severity.value, " ".join(verdict.reasons), alert_id)
    except sqlite3.IntegrityError:
        existing = store.find_open(campaign_id, cause)
        if existing is None:  # not the duplicate case, so do not hide it
            raise
        return bump(existing)
    return IncidentRef(action="created", incident_id=created["incident_id"], occurrences=1)


def draft_alert(
    campaign: dict[str, Any], kpi: dict[str, Any], verdict: Verdict, incident: IncidentRef, analyst_note: str | None = None
) -> AlertDraft:
    """Template-based so the facts in the message always match the verdict."""
    role = verdict.owner_role or "affiliate_manager"
    title = verdict.root_cause.value.replace("_", " ")
    subject = f"[{verdict.severity.value.upper()}] {campaign['id']} {campaign['name']}: {title}"
    lines = [
        f"Campaign: {campaign['id']} {campaign['name']} (affiliate {campaign['affiliate_name']}, "
        f"manager {campaign['manager']})",
        f"Incident: {incident.incident_id} ({incident.action}, seen {incident.occurrences}x)",
        f"What happened: {kpi.get('metric', 'KPI')} changed {kpi.get('metric_change_pct', 'n/a')}% "
        f"vs baseline. "
        + (
            f"Estimated loss ${verdict.est_revenue_loss_per_hour_usd:.0f}/hour."
            if verdict.est_revenue_loss_per_hour_usd > 0
            else "No direct revenue loss measured in this window."
        ),
        "Why: " + " ".join(verdict.reasons),
        *[f"Also found: {finding}" for finding in verdict.other_findings],
        "Suggested next steps:",
        *[f"  {i}. {action}" for i, action in enumerate(verdict.recommended_actions, 1)],
    ]
    if verdict.root_cause is RootCause.unknown or verdict.needs_human_review:
        lines.append("Note: automated triage is not conclusive here, please review the evidence.")
    if analyst_note:
        lines.append(f"Analyst note (AI written): {analyst_note}")
    return AlertDraft(to_role=role, channel=CHANNEL.get(role, "direct message"), subject=subject, body="\n".join(lines))
