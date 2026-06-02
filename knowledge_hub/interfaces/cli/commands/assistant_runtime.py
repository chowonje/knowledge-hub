"""Shared assistant routing helpers for chat-like CLI surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from knowledge_hub.interfaces.cli.commands.search_cmd import (
    _ask_allow_external_default,
    _generate_answer_compat,
    _get_searcher,
    _graph_query_signal,
    _runtime_diagnostics,
    _selected_answer_route_fields,
)

CHAT_RESULT_SCHEMA = "knowledge-hub.chat.result.v1"

PAPER_TOP_K = 8
PAPER_RETRIEVAL_MODE = "hybrid"
PAPER_ALPHA = 0.7
PAPER_SOURCE_TYPE = "paper"
PAPER_MEMORY_ROUTE_MODE = "off"
PAPER_PAPER_MEMORY_MODE = "off"
PAPER_ANSWER_ROUTE = "auto"


@dataclass(frozen=True)
class PaperSlashCommand:
    """Parsed paper-evidence slash command."""

    command: str
    question: str


def parse_paper_slash(text: str, *, allow_ask_alias: bool = False) -> PaperSlashCommand | None:
    stripped = str(text or "").strip()
    lowered = stripped.lower()
    aliases = ("/paper", "/ask") if allow_ask_alias else ("/paper",)
    for alias in aliases:
        if lowered == alias:
            return PaperSlashCommand(command=alias, question="")
        if lowered.startswith(f"{alias} "):
            return PaperSlashCommand(command=alias, question=stripped[len(alias) :].strip())
    return None


def build_assist_usage(
    *,
    graph_query_signal: Any | None = None,
    paper_evidence_used: bool = False,
    enrich_recommended: bool = False,
) -> dict[str, Any]:
    """Summarize derivative-layer usage without upgrading signals to evidence."""

    return {
        "paperEvidence": "used" if paper_evidence_used else "not_used",
        "paperMemory": PAPER_PAPER_MEMORY_MODE,
        "documentMemory": "not_used",
        "claimCards": "not_used",
        "ontology": "signal_only" if bool(graph_query_signal) else "not_used",
        "cluster": "not_used",
        "enrichRecommended": bool(enrich_recommended),
    }


def _paper_base_payload(
    *,
    status: str,
    original_prompt: str,
    question: str,
    allow_external: bool,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": CHAT_RESULT_SCHEMA,
        "status": status,
        "mode": "paper",
        "route": "paper",
        "sourceType": PAPER_SOURCE_TYPE,
        "retrievalMode": PAPER_RETRIEVAL_MODE,
        "alpha": PAPER_ALPHA,
        "topK": PAPER_TOP_K,
        "memoryRouteMode": PAPER_MEMORY_ROUTE_MODE,
        "paperMemoryMode": PAPER_PAPER_MEMORY_MODE,
        "answerRouteRequested": PAPER_ANSWER_ROUTE,
        "allowExternal": bool(allow_external),
        "historyPersisted": False,
        "promptChars": len(original_prompt),
        "questionChars": len(question),
        "answer": "",
        "sources": [],
        "citations": [],
        "warnings": list(warnings or []),
        "assistUsage": build_assist_usage(enrich_recommended=status == "init_error"),
    }


def _policy_blocked_reason(error: Exception) -> str:
    decision = getattr(error, "decision", None)
    classification = str(getattr(decision, "classification", "") or "unknown")
    trace_id = str(getattr(decision, "trace_id", "") or "-")
    return f"outbound policy blocked: classification={classification} trace_id={trace_id}"


def generate_paper_answer_payload(
    khub: Any,
    *,
    original_prompt: str,
    question: str,
    allow_external_override: bool | None,
) -> dict[str, Any]:
    """Run the existing ask engine in paper-only mode and return a chat-shaped payload."""

    question = str(question or "").strip()
    if not question:
        return _paper_base_payload(
            status="blocked",
            original_prompt=original_prompt,
            question=question,
            allow_external=False if allow_external_override is None else bool(allow_external_override),
            warnings=["paper question required after /paper"],
        )

    try:
        searcher = _get_searcher(khub)
    except Exception as error:
        return _paper_base_payload(
            status="init_error",
            original_prompt=original_prompt,
            question=question,
            allow_external=False if allow_external_override is None else bool(allow_external_override),
            warnings=[f"searcher init failed: {error}"],
        )

    allow_external_effective = (
        _ask_allow_external_default(khub, searcher) if allow_external_override is None else bool(allow_external_override)
    )

    try:
        result = _generate_answer_compat(
            searcher,
            question,
            top_k=PAPER_TOP_K,
            source_type=PAPER_SOURCE_TYPE,
            retrieval_mode=PAPER_RETRIEVAL_MODE,
            alpha=PAPER_ALPHA,
            allow_external=allow_external_effective,
            memory_route_mode=PAPER_MEMORY_ROUTE_MODE,
            paper_memory_mode=PAPER_PAPER_MEMORY_MODE,
            answer_route_override=None,
        )
    except Exception as error:
        if getattr(error, "code", "") == "POLICY_BLOCKED_OUTBOUND":
            return _paper_base_payload(
                status="blocked",
                original_prompt=original_prompt,
                question=question,
                allow_external=allow_external_effective,
                warnings=[_policy_blocked_reason(error)],
            )
        raise

    payload = _paper_base_payload(
        status="ok",
        original_prompt=original_prompt,
        question=question,
        allow_external=allow_external_effective,
    )
    payload["answer"] = str((result or {}).get("answer") or "")
    payload["sources"] = list((result or {}).get("sources") or [])
    payload["citations"] = list((result or {}).get("citations") or [])
    payload["warnings"] = list((result or {}).get("warnings") or [])
    payload.update(_selected_answer_route_fields(result))

    for key in (
        "router",
        "answerGeneration",
        "answerVerification",
        "answerRewrite",
        "paperAnswerScope",
        "paperMemoryPrefilter",
        "memoryPrefilter",
        "memoryRoute",
    ):
        if key in (result or {}):
            payload[key] = (result or {}).get(key)

    runtime_diagnostics = _runtime_diagnostics(searcher)
    graph_query_signal = _graph_query_signal(searcher, question)
    payload["runtimeDiagnostics"] = runtime_diagnostics
    payload["graphQuerySignal"] = graph_query_signal
    payload["graph_query_signal"] = graph_query_signal
    payload["assistUsage"] = build_assist_usage(
        graph_query_signal=graph_query_signal,
        paper_evidence_used=bool(payload["sources"] or payload["citations"]),
        enrich_recommended=not bool(payload["sources"] or payload["citations"]),
    )
    return payload


def compact_source_label(source: Any) -> str:
    if isinstance(source, dict):
        for key in ("title", "paperTitle", "paper_title", "source", "sourceId", "id", "doc_id"):
            value = source.get(key)
            if value:
                return str(value)
        return str(source)
    return str(source)


def paper_payload_text_lines(payload: dict[str, Any], *, source_limit: int = 5) -> list[str]:
    lines: list[str] = []
    answer = str(payload.get("answer") or "").strip()
    if answer:
        lines.append(answer)

    warnings = [str(item) for item in list(payload.get("warnings") or []) if str(item)]
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings[:source_limit])

    sources = list(payload.get("sources") or [])
    if sources:
        lines.append("")
        lines.append("Sources:")
        for source in sources[:source_limit]:
            lines.append(f"- {compact_source_label(source)}")

    route_bits = [
        str(payload.get("answerRouteApplied") or "").strip(),
        str(payload.get("answerProviderApplied") or "").strip(),
        str(payload.get("answerModelApplied") or "").strip(),
    ]
    if any(route_bits):
        lines.append("")
        lines.append(f"route={route_bits[0] or 'unknown'} provider={route_bits[1] or '-'} model={route_bits[2] or '-'}")

    if not lines:
        lines.append("No paper answer was generated.")
    return lines
