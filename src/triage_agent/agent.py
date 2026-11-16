"""The triage workflow as a LangGraph state machine.

    guard -> plan <-> tool -> diagnose -> act -> finalize
               (loop)          |  ^
                               v  |
                           tool (forced, if a required check is missing)

- guard:     validate input, apply refusal rules, quarantine suspicious text
- plan:      the model picks the next investigation tool (allowlist, budgeted)
- tool:      run it with retries, store the result as evidence, trace it
- diagnose:  deterministic rules give the verdict; missing checks are forced first
- act:       create or update the incident, draft the alert (never sent automatically)
- finalize:  structured JSON result plus a human-readable summary
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from . import rules
from .config import SETTINGS, Settings
from .datasource import DataSource, TransientToolError, make_datasource
from .guardrails import find_refused_requests, sanitize_untrusted, validate_alert
from .llm import LLM, OfflineLLM, PlanContext, make_llm
from .models import (Alert, AlertDraft, IncidentRef, LLMUsage, PlanStep, RootCause, StepRecord, TriageResult,
                     Verdict)
from .tools import FINISH, INVESTIGATION_TOOLS, ToolContext
from .tools.actions import IncidentStore, draft_alert, upsert_incident
from .tracing import Tracer, digest


@dataclass
class Deps:
    """Everything the workflow talks to. Swap any of these in tests or in production."""

    ds: DataSource
    incidents: IncidentStore
    llm: LLM
    tracer: Tracer
    settings: Settings = SETTINGS
    sleep: Callable[[float], None] = time.sleep
    fallback: OfflineLLM = field(default_factory=OfflineLLM)


class State(TypedDict, total=False):
    raw_input: Any
    alert: dict[str, Any]
    campaign: dict[str, Any]
    note_facts: dict[str, Any]
    rejection: dict[str, Any]
    flags: list[str]
    refused: list[dict[str, str]]
    evidence: dict[str, Any]
    steps: list[dict[str, Any]]
    next_tool: str | None
    next_reason: str
    next_chosen_by: str
    runbook_query: str | None
    planner_steps: int
    planner_failures: int
    phase: str
    verdict: dict[str, Any]
    llm_hypothesis: str | None
    incident: dict[str, Any]
    alert_draft: dict[str, Any] | None
    result: dict[str, Any]


def _add_flag(state: State, flag: str) -> list[str]:
    flags = list(state.get("flags", []))
    if flag not in flags:
        flags.append(flag)
    return flags


def build_graph(deps: Deps):
    s = deps.settings
    tracer = deps.tracer

    # ------------------------------------------------------------------ guard
    def guard(state: State) -> State:
        alert, rejection = validate_alert(state.get("raw_input"))
        if alert is None:
            tracer.event("guardrail", outcome="rejected", rejection=rejection)
            return {"rejection": rejection, "flags": ["invalid_input"]}
        tracer.bind(campaign_id=alert.campaign_id, alert_id=alert.alert_id)
        campaign = deps.ds.get_campaign(alert.campaign_id)
        if campaign is None:
            rejection = {"code": "unknown_campaign",
                         "message": f"Campaign {alert.campaign_id} does not exist.", "errors": []}
            tracer.event("guardrail", outcome="rejected", rejection=rejection)
            return {"alert": alert.model_dump(mode="json"), "rejection": rejection, "flags": ["unknown_campaign"]}

        flags: list[str] = []
        refused = find_refused_requests(alert.note)
        if refused:
            flags.append("action_request_refused")
        # The note itself never reaches the model. Only these facts about it do.
        _, injected = sanitize_untrusted(alert.note, max_len=300)
        if injected:
            flags.append("prompt_injection_suspected_in_note")
        tracer.event("guardrail", outcome="accepted", flags=flags, refused_requests=refused)
        note_facts = {"note_present": bool(alert.note and alert.note.strip()), "note_flagged_injection": injected,
                      "refused_request_count": len(refused)}
        return {"alert": alert.model_dump(mode="json"), "campaign": campaign, "note_facts": note_facts,
                "flags": flags, "refused": refused, "evidence": {}, "steps": [], "planner_steps": 0,
                "planner_failures": 0, "phase": "investigate"}

    # ------------------------------------------------------------------- plan
    def _valid(step: PlanStep, state: State) -> str | None:
        """Return a reason if the model's choice must be overridden."""
        evidence = state["evidence"]
        if step.action == FINISH:
            return None
        if step.action not in INVESTIGATION_TOOLS:
            return f"'{step.action}' is not an allowed tool"
        if step.action in evidence:
            return f"'{step.action}' already ran"
        if "kpi_snapshot" not in evidence and step.action != "kpi_snapshot":
            return "kpi_snapshot must run first"
        return None

    def plan(state: State) -> State:
        evidence = state["evidence"]
        if state["planner_steps"] >= s.max_planner_steps or len(state["steps"]) >= s.max_tool_calls:
            tracer.event("plan", action=FINISH, chosen_by="budget", reason="step budget reached")
            return {"next_tool": None, "flags": _add_flag(state, "step_budget_reached")}

        ctx = PlanContext(
            metric=state["alert"]["metric"], campaign_id=state["campaign"]["id"], evidence=evidence,
            remaining_tools=[name for name in INVESTIGATION_TOOLS if name not in evidence],
            **state.get("note_facts", {}),
        )
        chosen_by = deps.llm.mode
        flags = state.get("flags", [])
        failures = state.get("planner_failures", 0)
        if failures >= s.planner_max_consecutive_failures:
            # Circuit breaker: the planner model failed several times in a row. Stop paying for
            # (and waiting on) a broken model and let the deterministic policy finish the run.
            if "planner_circuit_open" not in flags:
                tracer.event("planner_circuit_open", consecutive_failures=failures)
            flags = _add_flag({"flags": flags}, "planner_circuit_open")
            step = deps.fallback.plan(ctx)
            chosen_by = "policy_fallback"
        else:
            try:
                step = deps.llm.plan(ctx)
                problem = _valid(step, state)
            except Exception as exc:  # model or network failure
                step, problem = None, f"planner error: {type(exc).__name__}: {exc}"
            if problem:
                tracer.event("plan_override", problem=problem, proposed=step.action if step else None)
                step = deps.fallback.plan(ctx)
                chosen_by = "policy_fallback"
                flags = _add_flag(state, "planner_fallback_used")
                failures += 1
            else:
                failures = 0

        tracer.event("plan", action=step.action, chosen_by=chosen_by, reason=step.reason)
        query, _ = sanitize_untrusted(step.runbook_query, max_len=120)
        return {"next_tool": None if step.action == FINISH else step.action, "next_reason": step.reason,
                "next_chosen_by": chosen_by, "runbook_query": query or None,
                "planner_steps": state["planner_steps"] + 1, "planner_failures": failures, "flags": flags}

    # ------------------------------------------------------------------- tool
    def run_tool(state: State) -> State:
        name = state["next_tool"]
        assert name is not None
        spec = INVESTIGATION_TOOLS[name]
        ctx = ToolContext(alert=Alert.model_validate(state["alert"]), campaign=state["campaign"], ds=deps.ds,
                          settings=s, evidence=state["evidence"], runbook_query=state.get("runbook_query"))
        started = time.perf_counter()
        result: dict[str, Any]
        attempts = 0
        status = "failed"
        while attempts < s.tool_max_attempts:
            attempts += 1
            attempt_started = time.perf_counter()

            def elapsed() -> float:
                return round((time.perf_counter() - attempt_started) * 1000, 2)

            try:
                result = spec.fn(ctx)
                status = "ok"
                tracer.event("tool_call", tool=name, attempt=attempts, ok=True, duration_ms=elapsed(),
                             result=digest(result))
                break
            except TransientToolError as exc:
                result = {"error": str(exc), "retryable": True}
                tracer.event("tool_call", tool=name, attempt=attempts, ok=False, duration_ms=elapsed(),
                             error=str(exc), retryable=True)
                if attempts < s.tool_max_attempts:
                    # Exponential backoff, or the wait the remote side asked for (rate limit), capped.
                    wait = s.tool_backoff_seconds * 2 ** (attempts - 1)
                    deps.sleep(min(max(wait, getattr(exc, "retry_after", 0.0) or 0.0), 30.0))
            except Exception as exc:  # permanent: do not retry
                result = {"error": f"{type(exc).__name__}: {exc}", "retryable": False}
                tracer.event("tool_call", tool=name, attempt=attempts, ok=False, duration_ms=elapsed(),
                             error=result["error"], retryable=False)
                break

        flags = state.get("flags", [])
        if status == "failed":
            flags = _add_flag(state, f"tool_failed:{name}")
        elif result.get("untrusted_text_flagged") or any(
                hit.get("untrusted_text_flagged") for hit in result.get("hits", [])):
            flags = _add_flag(state, f"prompt_injection_suspected_in:{name}")

        record = StepRecord(step=len(state["steps"]) + 1, tool=name, status=status, attempts=attempts,
                            duration_ms=round((time.perf_counter() - started) * 1000, 2),
                            chosen_by=state.get("next_chosen_by", "rules"), reason=state.get("next_reason", ""))
        return {"evidence": {**state["evidence"], name: result}, "steps": state["steps"] + [record.model_dump()],
                "flags": flags, "next_tool": None}

    # --------------------------------------------------------------- diagnose
    def diagnose(state: State) -> State:
        evidence = state["evidence"]
        missing = rules.next_needed(evidence, s)
        if missing and len(state["steps"]) < s.max_tool_calls:
            # The planner stopped too early. The rules decide what is required, not the model.
            tracer.event("coverage_check", missing=missing, action="forcing required check")
            return {"phase": "verify", "next_tool": missing, "next_chosen_by": "rules_coverage",
                    "next_reason": "Required for a verdict but skipped by the planner.",
                    "flags": _add_flag(state, "coverage_check_forced_tool")}

        flags = state.get("flags", [])
        try:
            hypothesis = deps.llm.hypothesize(evidence)
        except Exception as exc:  # the verdict does not need the hypothesis, but the failure must be visible
            hypothesis = None
            flags = _llm_call_failed(state, "hypothesize", exc, "hypothesis_unavailable")
        verdict = rules.evaluate(evidence, s, llm_hypothesis=hypothesis)
        tracer.event("diagnosis", root_cause=verdict.root_cause.value, severity=verdict.severity.value,
                     confidence=verdict.confidence, needs_human_review=verdict.needs_human_review,
                     llm_hypothesis=hypothesis, reasons=verdict.reasons, stop_reason=verdict.stop_reason,
                     findings=[{"cause": f.cause.value, "status": f.status.value, "primary": f.primary,
                                "evidence_strength": f.evidence_strength} for f in verdict.findings])
        return {"verdict": verdict.model_dump(mode="json"), "llm_hypothesis": hypothesis, "next_tool": None,
                "flags": flags}

    def _llm_call_failed(state: State, call: str, exc: Exception, flag: str) -> list[str]:
        tracer.event("llm_call_failed", call=call, error_type=type(exc).__name__, error=str(exc)[:200])
        return _add_flag(state, flag)

    # -------------------------------------------------------------------- act
    def act(state: State) -> State:
        verdict = Verdict.model_validate(state["verdict"])
        if verdict.root_cause is RootCause.no_anomaly:
            tracer.event("action", incident="none", alert_draft=False, reason="no anomaly")
            return {"incident": IncidentRef(action="none").model_dump(), "alert_draft": None}
        started = time.perf_counter()
        incident = upsert_incident(deps.incidents, state["campaign"]["id"], state["alert"]["alert_id"], verdict)
        tracer.event("tool_call", tool="incident_store", attempt=1, ok=True,
                     duration_ms=round((time.perf_counter() - started) * 1000, 2), result=digest(incident.model_dump()))
        flags = state.get("flags", [])
        try:
            note = deps.llm.analyst_note(state["verdict"], state["evidence"])
        except Exception as exc:  # the draft is complete without the note, but the failure must be visible
            note = None
            flags = _llm_call_failed(state, "analyst_note", exc, "analyst_note_unavailable")
        started = time.perf_counter()
        draft = draft_alert(state["campaign"], state["evidence"].get("kpi_snapshot", {}), verdict, incident, note)
        tracer.event("tool_call", tool="draft_alert", attempt=1, ok=True,
                     duration_ms=round((time.perf_counter() - started) * 1000, 2),
                     result=digest({"to_role": draft.to_role, "subject": draft.subject, "status": draft.status}))
        return {"incident": incident.model_dump(), "alert_draft": draft.model_dump(), "flags": flags}

    # --------------------------------------------------------------- finalize
    def finalize(state: State) -> State:
        result = _build_result(state, deps)
        tracer.event("final", status=result.status, root_cause=result.root_cause, severity=result.severity,
                     incident=result.incident.incident_id, incident_action=result.incident.action,
                     approval_status=result.alert_draft.status if result.alert_draft else "no_alert_drafted",
                     needs_human_review=result.needs_human_review, flags=result.guardrail_flags,
                     tool_calls=len(result.steps), tool_retries=sum(st.attempts - 1 for st in result.steps),
                     llm_usage=result.llm_usage.model_dump())
        return {"result": result.model_dump(mode="json")}

    graph = StateGraph(State)
    graph.add_node("guard", guard)
    graph.add_node("plan", plan)
    graph.add_node("tool", run_tool)
    graph.add_node("diagnose", diagnose)
    graph.add_node("act", act)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", lambda st: "finalize" if st.get("rejection") else "plan",
                                {"finalize": "finalize", "plan": "plan"})
    graph.add_conditional_edges("plan", lambda st: "tool" if st.get("next_tool") else "diagnose",
                                {"tool": "tool", "diagnose": "diagnose"})
    graph.add_conditional_edges("tool", lambda st: "diagnose" if st.get("phase") == "verify" else "plan",
                                {"diagnose": "diagnose", "plan": "plan"})
    graph.add_conditional_edges("diagnose", lambda st: "tool" if st.get("next_tool") else "act",
                                {"tool": "tool", "act": "act"})
    graph.add_edge("act", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def _loss_text(loss: float) -> str:
    return f"Estimated loss: ${loss:.0f}/hour." if loss > 0 else "No direct revenue loss measured in this window."


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _summary(result: TriageResult) -> str:
    if result.status == "rejected":
        rejection = result.rejection or {}
        lines = [f"Triage rejected: {rejection.get('message', '')}"]
        lines += [f"  - {e['field']}: {e['problem']}" for e in rejection.get("errors", [])]
        return "\n".join(lines)

    kpi = result.kpi
    lines = [
        f"Triage {result.alert_id} | {result.campaign_id} {result.campaign_name}",
        f"Root cause: {result.root_cause.value} | severity: {result.severity.value} | "
        f"confidence: {result.confidence:.2f}" + (" | NEEDS HUMAN REVIEW" if result.needs_human_review else ""),
    ]
    if kpi and "error" not in kpi:
        change = "no baseline to compare" if kpi["metric_change_pct"] is None else f"{kpi['metric_change_pct']}% vs baseline"
        lines.append(
            f"KPI: {kpi['metric']} {change} "
            f"(conversion rate {_pct(kpi['baseline']['conversion_rate'])} -> {_pct(kpi['current']['conversion_rate'])}, "
            f"{kpi['current']['clicks_total']} clicks in the last {SETTINGS.current_hours}h). "
            + _loss_text(result.est_revenue_loss_per_hour_usd)
        )
        season = kpi.get("seasonal", {})
        quality = kpi.get("data_quality", {})
        if season.get("available"):
            lines.append(f"     Same hours on the previous {season['days_used']} {season['day_type'].replace('_', ' ')}: "
                         f"{season['metric_change_pct']}%. Signal strength: {kpi.get('signal_strength')}.")
        if quality.get("missing_current_hours") or quality.get("missing_baseline_hours"):
            lines.append(f"     Data gaps: {quality.get('missing_current_hours', 0)} current and "
                         f"{quality.get('missing_baseline_hours', 0)} baseline hours missing.")
    lines.append("Why:")
    lines += [f"  - {reason}" for reason in result.reasons]
    if result.other_findings:
        lines.append("Also found (lower priority, check after the main cause):")
        lines += [f"  - {finding}" for finding in result.other_findings]
    if result.recommended_actions:
        lines.append("Recommended actions (for a human to carry out):")
        lines += [f"  {i}. {a}" for i, a in enumerate(result.recommended_actions, 1)]
    if result.runbook_refs:
        lines.append("Runbooks: " + ", ".join(result.runbook_refs))
    if result.incident.action != "none":
        lines.append(f"Incident {result.incident.incident_id} {result.incident.action} "
                     f"(seen {result.incident.occurrences}x). "
                     f"Alert drafted for {result.alert_draft.to_role}, waiting for approval.")
    else:
        lines.append("No incident opened and no alert drafted.")
    for refused in result.refused_requests:
        lines.append(f"Refused request: \"{refused['request']}\". {refused['reason']}")
    if result.guardrail_flags:
        lines.append("Flags: " + ", ".join(result.guardrail_flags))
    lines.append("Steps: " + " -> ".join(
        f"{st.tool}{'' if st.status == 'ok' else ' (failed)'}{f' x{st.attempts}' if st.attempts > 1 else ''}"
        for st in result.steps))
    return "\n".join(lines)


def _llm_usage(tracer: Tracer) -> LLMUsage:
    """Totals of the model calls of this run, read back from the trace."""
    calls = [e for e in tracer.events if e["event"] == "llm_call"]
    return LLMUsage(
        model=next((e["model"] for e in calls if e.get("model")), None),
        calls=len(calls), failed_calls=sum(not e["ok"] for e in calls),
        input_tokens=sum(e.get("input_tokens") or 0 for e in calls),
        output_tokens=sum(e.get("output_tokens") or 0 for e in calls),
        latency_ms=round(sum(e.get("latency_ms") or 0.0 for e in calls), 1),
    )


def _build_result(state: State, deps: Deps) -> TriageResult:
    alert = state.get("alert") or {}
    raw = state.get("raw_input") if isinstance(state.get("raw_input"), dict) else {}
    if state.get("rejection"):
        result = TriageResult(
            run_id=deps.tracer.run_id, status="rejected", llm_mode=deps.llm.mode,
            alert_id=alert.get("alert_id") or (raw.get("alert_id") if isinstance(raw.get("alert_id"), str) else None),
            campaign_id=alert.get("campaign_id"), rejection=state["rejection"],
            guardrail_flags=state.get("flags", []), stop_reason="input rejected before any tool ran",
        )
        result.summary = _summary(result)
        return result

    verdict = Verdict.model_validate(state["verdict"])
    evidence = state["evidence"]
    runbook = evidence.get("runbook_search", {})
    result = TriageResult(
        run_id=deps.tracer.run_id, status="completed", llm_mode=deps.llm.mode,
        alert_id=alert["alert_id"], campaign_id=alert["campaign_id"], campaign_name=state["campaign"]["name"],
        root_cause=verdict.root_cause, severity=verdict.severity, confidence=verdict.confidence,
        needs_human_review=verdict.needs_human_review, reasons=verdict.reasons,
        findings=verdict.findings, other_findings=verdict.other_findings, stop_reason=verdict.stop_reason,
        llm_usage=_llm_usage(deps.tracer), recommended_actions=verdict.recommended_actions,
        est_revenue_loss_per_hour_usd=verdict.est_revenue_loss_per_hour_usd,
        llm_hypothesis=state.get("llm_hypothesis"),
        kpi=evidence.get("kpi_snapshot", {}),
        evidence={k: v for k, v in evidence.items() if k != "kpi_snapshot"},
        runbook_refs=[f"{hit['source']}#{hit['heading']}" for hit in runbook.get("hits", [])],
        incident=IncidentRef.model_validate(state["incident"]),
        alert_draft=AlertDraft.model_validate(state["alert_draft"]) if state.get("alert_draft") else None,
        refused_requests=state.get("refused", []), guardrail_flags=state.get("flags", []),
        steps=[StepRecord.model_validate(step) for step in state["steps"]],
    )
    result.summary = _summary(result)
    return result


def run_triage(raw_alert: Any, *, llm: LLM | None = None, ds: DataSource | None = None,
               incidents: IncidentStore | None = None, tracer: Tracer | None = None,
               settings: Settings = SETTINGS, sleep: Callable[[float], None] = time.sleep) -> TriageResult:
    """Run one alert through the workflow. Never raises on bad input: it returns a rejected result."""
    deps = Deps(
        ds=ds or make_datasource(settings),
        incidents=incidents or IncidentStore(settings.db_path),
        llm=llm or make_llm(settings=settings),
        tracer=tracer or Tracer(settings.runs_dir),
        settings=settings,
        sleep=sleep,
        fallback=OfflineLLM(settings),
    )
    if hasattr(deps.llm, "observer"):
        tracer_ref = deps.tracer
        deps.llm.observer = lambda record: tracer_ref.event("llm_call", **record)
    deps.tracer.event("run_start", llm_mode=deps.llm.mode,
                      alert_id=raw_alert.get("alert_id") if isinstance(raw_alert, dict) else None)
    try:
        final = build_graph(deps).invoke({"raw_input": raw_alert}, {"recursion_limit": 60})
        return TriageResult.model_validate(final["result"])
    except Exception as exc:  # last line of defence: a structured failure, never a stack trace to the caller
        deps.tracer.event("run_failed", error=f"{type(exc).__name__}: {exc}")
        return TriageResult(run_id=deps.tracer.run_id, status="failed", llm_mode=deps.llm.mode,
                            rejection={"code": "internal_error", "message": f"{type(exc).__name__}: {exc}",
                                       "errors": []},
                            summary=f"Triage failed with an internal error: {type(exc).__name__}: {exc}")
