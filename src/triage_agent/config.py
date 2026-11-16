"""Thresholds and paths in one place so the business rules are easy to review and tune."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # KPI windows (hours)
    baseline_hours: int = 21
    current_hours: int = 3
    # A drop only counts when there is enough traffic to trust it
    min_clicks_for_signal: int = 100
    drop_threshold_pct: float = -30.0
    # Root cause thresholds
    bot_score_threshold: float = 0.7
    bad_source_share_threshold: float = 0.3
    disallowed_geo_share_threshold: float = 0.3
    postback_ratio_threshold: float = 0.5
    page_latency_limit_ms: int = 10_000
    # Agent budgets
    max_planner_steps: int = 8
    max_tool_calls: int = 12
    tool_max_attempts: int = 3
    tool_backoff_seconds: float = 0.5
    llm_max_attempts: int = 2
    # Severity
    critical_loss_per_hour_usd: float = 200.0
    high_loss_per_hour_usd: float = 50.0
    # Paths
    world_path: Path = REPO_ROOT / "data" / "world.json"
    runbooks_dir: Path = REPO_ROOT / "runbooks"
    db_path: str = os.getenv("TRIAGE_DB", str(REPO_ROOT / "var" / "incidents.db"))
    runs_dir: Path = Path(os.getenv("TRIAGE_RUNS_DIR", str(REPO_ROOT / "runs")))


SETTINGS = Settings()
