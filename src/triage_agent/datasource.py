"""Read-only access to campaign data.

`DataSource` is the seam between the agent and the outside world. Here it reads
data/world.json (a mock tracker, uptime prober and offer API). In production you
implement the same five methods against the real systems and nothing else changes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .config import SETTINGS, Settings


class TransientToolError(Exception):
    """A failure worth retrying (timeout, connection reset, 5xx from an internal API)."""


class PermanentToolError(Exception):
    """A failure that will not go away by retrying."""


class PageProbe(Protocol):
    """Anything that can fetch a landing page and reduce it to a few facts."""

    def fetch(self, url: str) -> dict[str, Any]: ...


class DataSource:
    def __init__(self, world_path: Path | None = None, page_probe: PageProbe | None = None) -> None:
        self._world: dict[str, Any] = json.loads((world_path or SETTINGS.world_path).read_text())
        self._page_attempts: dict[str, int] = {}
        self._page_probe = page_probe   # None = scripted fixture responses, otherwise a real probe

    def get_campaign(self, campaign_id: str) -> dict[str, Any] | None:
        return self._world["campaigns"].get(campaign_id)

    def get_offer(self, offer_id: str) -> dict[str, Any]:
        return self._world["offers"][offer_id]

    def get_hourly_kpi(self, campaign_id: str) -> list[dict[str, Any]]:
        return self._world["kpi"][campaign_id]["hourly"]

    def fetch_landing_page(self, url: str) -> dict[str, Any]:
        """Probe a landing page. With no probe configured this is a mock: scripted responses
        from the fixture file simulate flaky probes and outages. With a probe (see http_probe.py)
        it is a real HTTP request, and the probe enforces its own host allowlist."""
        if self._page_probe is not None:
            return self._page_probe.fetch(url)
        page = self._world["landing_pages"].get(url)
        if page is None:
            raise PermanentToolError(f"url is not a registered landing page: {url}")
        attempt = self._page_attempts.get(url, 0)
        self._page_attempts[url] = attempt + 1
        responses = page["responses"]
        response = responses[min(attempt, len(responses) - 1)]
        if "transient_error" in response:
            raise TransientToolError(response["transient_error"])
        return response

    def get_tracking_stats(self, campaign_id: str) -> dict[str, Any]:
        stats = self._world["tracking"][campaign_id]
        if "permanent_error" in stats:
            # The outage lasts the whole run, but each call looks like a retryable 502 to the caller.
            raise TransientToolError(stats["permanent_error"])
        return stats

    def get_traffic_breakdown(self, campaign_id: str) -> dict[str, Any]:
        return self._world["traffic"][campaign_id]


def make_datasource(settings: Settings = SETTINGS) -> DataSource:
    """Build the data source from config. `page_probe=http` swaps the mock page probe for real HTTP."""
    if settings.page_probe == "fixture":
        return DataSource(settings.world_path)
    if settings.page_probe == "http":
        from .http_probe import HttpLandingPageProbe
        probe = HttpLandingPageProbe(settings.page_allowed_hosts, settings.page_timeout_seconds,
                                     settings.page_max_bytes, settings.page_pixel_marker)
        return DataSource(settings.world_path, page_probe=probe)
    raise ValueError(f'TRIAGE_PAGE_PROBE must be "fixture" or "http", got {settings.page_probe!r}')
