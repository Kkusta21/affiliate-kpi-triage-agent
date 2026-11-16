"""Command line entry point.

    python -m triage_agent run samples/alerts/01_landing_page_down.json
    python -m triage_agent run samples/alerts/01_landing_page_down.json --json
    python -m triage_agent demo
    python -m triage_agent incidents
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .agent import run_triage
from .config import REPO_ROOT, SETTINGS
from .llm import make_llm
from .tools.actions import IncidentStore
from .tracing import Tracer


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        # Hand the broken payload to the agent so the rejection path is the same everywhere.
        return f"<unparseable JSON: {exc}>"


def _run_one(path: Path, llm_spec: str | None, as_json: bool) -> int:
    tracer = Tracer(SETTINGS.runs_dir)
    result = run_triage(_load(path), llm=make_llm(llm_spec), tracer=tracer)
    if as_json:
        print(result.model_dump_json(indent=2))
    else:
        print(result.summary)
        if result.alert_draft:
            print("\n--- Alert draft (not sent) ---")
            print(f"To: {result.alert_draft.to_role} via {result.alert_draft.channel}")
            print(f"Subject: {result.alert_draft.subject}")
            print(result.alert_draft.body)
        print(f"\nTrace: {tracer.path}")
    return 0 if result.status != "failed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage", description="Affiliate KPI triage agent")
    parser.add_argument("--llm", default=None, help='"offline" (default) or "<provider>:<model>"')
    parser.add_argument("-v", "--verbose", action="store_true", help="print trace events to stderr")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="triage one alert file")
    run.add_argument("alert", type=Path)
    run.add_argument("--json", action="store_true", help="print the structured JSON result")
    sub.add_parser("demo", help="run every sample alert")
    sub.add_parser("incidents", help="list incidents in the store")
    probe = sub.add_parser("probe", help="real HTTP probe of one landing page URL (needs network)")
    probe.add_argument("url")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")

    if args.command == "run":
        return _run_one(args.alert, args.llm, args.json)
    if args.command == "demo":
        code = 0
        for path in sorted((REPO_ROOT / "samples" / "alerts").glob("*.json")):
            print(f"\n{'=' * 78}\n{path.name}\n{'=' * 78}")
            code |= _run_one(path, args.llm, False)
        return code
    if args.command == "incidents":
        print(json.dumps(IncidentStore(SETTINGS.db_path).list(), indent=2))
        return 0
    if args.command == "probe":
        return _probe(args.url)
    return 2


def _probe(url: str) -> int:
    """Run the real HTTP landing page probe once. The operator names the URL on the command
    line, so its host is the allowlist for this call. Private addresses are still refused."""
    from urllib.parse import urlsplit

    from .datasource import PermanentToolError, TransientToolError
    from .guardrails import sanitize_untrusted
    from .http_probe import HttpLandingPageProbe
    from .tools.investigation import classify_status

    host = (urlsplit(url).hostname or "").lower()
    probe = HttpLandingPageProbe((host,), SETTINGS.page_timeout_seconds, SETTINGS.page_max_bytes,
                                 SETTINGS.page_pixel_marker)
    try:
        result = probe.fetch(url)
    except (PermanentToolError, TransientToolError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1
    finally:
        probe.close()
    title, flagged = sanitize_untrusted(result.pop("title"), max_len=80)
    print(json.dumps({"ok": True, "outcome": classify_status(result["status_code"]), **result,
                      "title": title, "untrusted_text_flagged": flagged}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
