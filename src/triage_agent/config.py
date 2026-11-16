"""Thresholds and paths in one place so the business rules are easy to review and tune.

Every threshold can be overridden with an environment variable (TRIAGE_<NAME>), read once
at import. Tests build their own `Settings(...)` instead of touching the environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(f"TRIAGE_{name}", "").strip()
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(f"TRIAGE_{name}", "").strip()
    return int(raw) if raw else default


def _env_hosts(name: str) -> tuple[str, ...]:
    return tuple(h.strip().lower() for h in os.getenv(f"TRIAGE_{name}", "").split(",") if h.strip())


@dataclass(frozen=True)
class Settings:
    # KPI windows (hours)
    baseline_hours: int = 21
    current_hours: int = 3
    # KPI data quality: below these the drop can neither be confirmed nor ruled out
    min_current_hours: int = 2
    min_baseline_hours: int = 6
    # Seasonal baseline: same hours of the day on earlier days of the same type (weekday or weekend)
    seasonal_min_days: int = 3
    # A drop only counts when there is enough traffic to trust it and it is not statistical noise
    min_clicks_for_signal: int = _env_int("MIN_CLICKS_FOR_SIGNAL", 100)
    drop_threshold_pct: float = _env_float("DROP_THRESHOLD_PCT", -30.0)
    z_threshold: float = _env_float("Z_THRESHOLD", 2.58)           # about 99% for a two-sided test
    spike_threshold_pct: float = _env_float("SPIKE_THRESHOLD_PCT", 100.0)
    # Root cause thresholds
    bot_score_threshold: float = _env_float("BOT_SCORE_THRESHOLD", 0.7)
    bad_source_share_threshold: float = _env_float("BAD_SOURCE_SHARE_THRESHOLD", 0.3)
    disallowed_geo_share_threshold: float = _env_float("DISALLOWED_GEO_SHARE_THRESHOLD", 0.3)
    postback_ratio_threshold: float = _env_float("POSTBACK_RATIO_THRESHOLD", 0.5)
    page_latency_limit_ms: int = _env_int("PAGE_LATENCY_LIMIT_MS", 10_000)
    # A value within this factor of its threshold is reported as "suspected", not "confirmed"
    suspect_margin: float = 0.8
    # Agent budgets
    max_planner_steps: int = 8
    max_tool_calls: int = 12
    tool_max_attempts: int = 3
    tool_backoff_seconds: float = 0.5
    llm_max_attempts: int = 2
    planner_max_consecutive_failures: int = 2   # then the planner model is skipped for the rest of the run
    # Severity
    critical_loss_per_hour_usd: float = 200.0
    high_loss_per_hour_usd: float = 50.0
    # Landing page probe: "fixture" (mock, default) or "http" (real read-only HTTP requests)
    page_probe: str = os.getenv("TRIAGE_PAGE_PROBE", "fixture").strip().lower()
    page_allowed_hosts: tuple[str, ...] = _env_hosts("PAGE_ALLOWED_HOSTS")
    page_timeout_seconds: float = _env_float("PAGE_TIMEOUT_SECONDS", 5.0)
    page_max_bytes: int = _env_int("PAGE_MAX_BYTES", 512_000)
    page_pixel_marker: str = os.getenv("TRIAGE_PAGE_PIXEL_MARKER", "data-tracking-pixel")
    # Paths
    world_path: Path = REPO_ROOT / "data" / "world.json"
    runbooks_dir: Path = REPO_ROOT / "runbooks"
    db_path: str = os.getenv("TRIAGE_DB", str(REPO_ROOT / "var" / "incidents.db"))
    runs_dir: Path = Path(os.getenv("TRIAGE_RUNS_DIR", str(REPO_ROOT / "runs")))


SETTINGS = Settings()
