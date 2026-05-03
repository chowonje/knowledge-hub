#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import signal
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from knowledge_hub.ai.ask_v2 import AskV2Service
from knowledge_hub.application.context import AppContextFactory


PAPER_DEFAULT_EVAL_FIELDNAMES = [
    "query",
    "source",
    "expected_family",
    "expected_top1_or_set",
    "expected_answer_mode",
    "allowed_fallback",
    "actual_family",
    "family_match",
    "query_frame_family",
    "actual_answer_mode",
    "answer_mode_match",
    "query_frame_answer_mode",
    "evidence_policy_key",
    "actual_representative_paper_id",
    "actual_representative_paper_title",
    "actual_representative_selection_score",
    "actual_representative_title_hits",
    "actual_representative_selection_reason",
    "representative_match",
    "actual_runtime_used",
    "actual_fallback_reason",
    "no_result",
    "planner_attempted",
    "planner_used",
    "planner_reason",
    "pred_label",
    "pred_reason",
    "citation_count",
    "citation_support_match",
    "top_source_titles",
    "latency_ms",
    "top_k",
    "retrieval_mode",
    "gate_mode",
    "timeout_flag",
    "notes",
    "final_label",
    "final_notes",
]
_LIVE_SMOKE_QUERIES = (
    "CNN을 쉽게 설명해줘",
    "AlexNet 논문 요약해줘",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _read_queries(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    items: list[dict[str, str]] = []
    for row in rows:
        query = _clean_text(row.get("query"))
        if not query:
            continue
        items.append({str(key): _clean_text(value) for key, value in row.items()})
    return items


def _normalize_eval_token(value: Any) -> str:
    token = _clean_text(value).casefold()
    token = " ".join(token.split())
    return token


def _expected_set(raw_value: Any) -> list[str]:
    return [item for item in (_normalize_eval_token(part) for part in str(raw_value or "").split("|")) if item]


def _representative_match(
    *,
    expected_top1_or_set: str,
    actual_paper_id: str,
    actual_title: str,
) -> str:
    expected = _expected_set(expected_top1_or_set)
    if not expected:
        return ""
    normalized_id = _normalize_eval_token(actual_paper_id)
    normalized_title = _normalize_eval_token(actual_title)
    if not normalized_id and not normalized_title:
        return "0"
    for item in expected:
        if item == normalized_id or item == normalized_title:
            return "1"
        if normalized_title and (item in normalized_title or normalized_title in item):
            return "1"
    return "0"


def _machine_judgment(
    *,
    expected_family: str,
    actual_family: str,
    expected_answer_mode: str,
    actual_answer_mode: str,
    allowed_fallback: str,
    no_result: bool,
    representative_match: str,
) -> tuple[str, str]:
    if _normalize_eval_token(expected_family) != _normalize_eval_token(actual_family):
        return "bad", "family_mismatch"
    if no_result:
        allowed = _normalize_eval_token(allowed_fallback)
        # Keep in sync with collect_paper_regression_eval._machine_judgment: compare rows
        # use allowed_fallback=need_multiple_papers for orchestrator no_result.
        if any(
            token in allowed
            for token in ("no_result", "planner_retry", "need_multiple_papers")
        ):
            return "partial", "allowed_no_result_fallback"
        return "bad", "unexpected_no_result"
    if expected_answer_mode and _normalize_eval_token(expected_answer_mode) != _normalize_eval_token(actual_answer_mode):
        return "partial", "answer_mode_mismatch"
    if _normalize_eval_token(expected_family) in {"concept_explainer", "paper_lookup"} and representative_match == "0":
        return "partial", "representative_mismatch"
    return "good", "family_and_mode_match"


def _citation_support_match(payload: dict[str, Any]) -> str:
    citations = [dict(item or {}) for item in list(payload.get("citations") or [])]
    if not citations:
        return ""
    sources = [dict(item or {}) for item in list(payload.get("sources") or [])]
    supported_targets = {
        _normalize_eval_token(
            item.get("citation_target")
            or item.get("arxiv_id")
            or item.get("file_path")
            or item.get("source_url")
            or item.get("title")
        )
        for item in sources
        if _normalize_eval_token(
            item.get("citation_target")
            or item.get("arxiv_id")
            or item.get("file_path")
            or item.get("source_url")
            or item.get("title")
        )
    }
    if not supported_targets:
        return "0"
    for citation in citations:
        target = _normalize_eval_token(citation.get("target"))
        if target and target not in supported_targets:
            return "0"
    return "1"


def _select_queries_for_gate(
    rows: list[dict[str, str]],
    *,
    gate_mode: str,
    family_filter: str = "",
) -> list[dict[str, str]]:
    selected = rows
    normalized_families = {
        _normalize_eval_token(item)
        for item in str(family_filter or "").split(",")
        if _normalize_eval_token(item)
    }
    if normalized_families:
        selected = [
            row
            for row in selected
            if _normalize_eval_token(row.get("expected_family")) in normalized_families
        ]
    normalized_mode = _clean_text(gate_mode) or "standard"
    if normalized_mode != "live_smoke":
        return selected
    smoke_set = {_clean_text(item) for item in _LIVE_SMOKE_QUERIES}
    return [row for row in selected if _clean_text(row.get("query")) in smoke_set]


def _gate_mode_defaults(*, gate_mode: str, stub_llm: bool, timeout_seconds: int) -> tuple[bool, int]:
    normalized_mode = _clean_text(gate_mode) or "standard"
    if normalized_mode == "stub_hard":
        return True, timeout_seconds or 20
    if normalized_mode == "live_smoke":
        return False, timeout_seconds or 60
    return bool(stub_llm), int(timeout_seconds or 0)


def _run_with_timeout(timeout_seconds: int, fn, *args, **kwargs):  # noqa: ANN001
    if timeout_seconds <= 0:
        return fn(*args, **kwargs)

    def _handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"collector timeout after {timeout_seconds}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.alarm(int(timeout_seconds))
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def _readonly_ask_v2_runtime():
    original_paper = AskV2Service._ensure_paper_card
    original_web = AskV2Service._ensure_web_card
    original_vault = AskV2Service._ensure_vault_card
    try:
        AskV2Service._ensure_paper_card = lambda self, paper_id: self.sqlite_db.get_paper_card_v2(paper_id)  # type: ignore[method-assign]
        AskV2Service._ensure_web_card = lambda self, url: self.sqlite_db.get_web_card_v2_by_url(url)  # type: ignore[method-assign]
        AskV2Service._ensure_vault_card = lambda self, note_id: self.sqlite_db.get_vault_card_v2(note_id)  # type: ignore[method-assign]
        yield
    finally:
        AskV2Service._ensure_paper_card = original_paper  # type: ignore[method-assign]
        AskV2Service._ensure_web_card = original_web  # type: ignore[method-assign]
        AskV2Service._ensure_vault_card = original_vault  # type: ignore[method-assign]


class _CollectorStubLLM:
    def generate(self, prompt: str, context: str = "") -> str:  # noqa: ARG002
        return "collector stub answer"

    def stream_generate(self, prompt: str, context: str = ""):
        _ = (prompt, context)
        yield "collector stub answer"


@contextmanager
def _stubbed_answer_runtime(searcher: Any):
    llm = _CollectorStubLLM()
    original_llm = getattr(searcher, "llm", None)
    original_resolve = getattr(searcher, "_resolve_llm_for_request")
    original_verify = getattr(searcher, "_verify_answer")
    original_rewrite = getattr(searcher, "_rewrite_answer")
    original_fallback = getattr(searcher, "_apply_conservative_fallback_if_needed")
    original_record = getattr(searcher, "_record_answer_log")
    try:
        searcher.llm = llm
        searcher._resolve_llm_for_request = lambda **kwargs: (  # type: ignore[method-assign]
            llm,
            {"route": "collector_stub", "provider": "stub", "model": "collector-stub"},
            [],
        )
        searcher._verify_answer = lambda **kwargs: {  # type: ignore[method-assign]
            "status": "verified",
            "supportedClaimCount": 1,
            "unsupportedClaimCount": 0,
            "uncertainClaimCount": 0,
            "conflictMentioned": False,
            "needsCaution": False,
            "warnings": [],
        }
        searcher._rewrite_answer = lambda **kwargs: (  # type: ignore[method-assign]
            kwargs["answer"],
            {"attempted": False, "applied": False, "finalAnswerSource": "original", "warnings": []},
        )
        searcher._apply_conservative_fallback_if_needed = lambda **kwargs: (  # type: ignore[method-assign]
            kwargs["answer"],
            kwargs["rewrite_meta"],
            kwargs["verification"],
        )
        searcher._record_answer_log = lambda **kwargs: None  # type: ignore[method-assign]
        yield
    finally:
        searcher.llm = original_llm
        searcher._resolve_llm_for_request = original_resolve  # type: ignore[method-assign]
        searcher._verify_answer = original_verify  # type: ignore[method-assign]
        searcher._rewrite_answer = original_rewrite  # type: ignore[method-assign]
        searcher._apply_conservative_fallback_if_needed = original_fallback  # type: ignore[method-assign]
        searcher._record_answer_log = original_record  # type: ignore[method-assign]


def _serialize_row(
    query_row: dict[str, str],
    result: dict[str, Any],
    *,
    top_k: int,
    retrieval_mode: str,
    latency_ms: float,
    gate_mode: str = "standard",
    timeout_flag: bool = False,
) -> dict[str, str]:
    payload = dict(result or {})
    query_plan = dict(payload.get("queryPlan") or {})
    query_frame = dict(payload.get("queryFrame") or {})
    planner = dict(payload.get("plannerFallback") or {})
    representative = dict(payload.get("representativePaper") or {})
    answer_signals = dict(payload.get("answerSignals") or {})
    representative_selection = dict(answer_signals.get("representative_selection") or {})
    evidence_policy = dict(payload.get("evidencePolicy") or {})
    family_diagnostics = dict(payload.get("familyRouteDiagnostics") or {})
    runtime_execution = dict(dict(payload.get("v2") or {}).get("runtimeExecution") or {})
    sources = [dict(item or {}) for item in list(payload.get("sources") or [])[:3]]
    expected_family = _clean_text(query_row.get("expected_family"))
    actual_family = _clean_text(payload.get("paperFamily") or query_frame.get("family"))
    expected_answer_mode = _clean_text(query_row.get("expected_answer_mode"))
    actual_answer_mode = _clean_text(family_diagnostics.get("answerMode") or query_frame.get("answer_mode"))
    actual_representative_paper_id = _clean_text(representative.get("paperId"))
    actual_representative_paper_title = _clean_text(representative.get("title"))
    representative_match = _representative_match(
        expected_top1_or_set=_clean_text(query_row.get("expected_top1_or_set")),
        actual_paper_id=actual_representative_paper_id,
        actual_title=actual_representative_paper_title,
    )
    no_result = _clean_text(payload.get("status")).lower() == "no_result"
    pred_label, pred_reason = _machine_judgment(
        expected_family=expected_family,
        actual_family=actual_family,
        expected_answer_mode=expected_answer_mode,
        actual_answer_mode=actual_answer_mode,
        allowed_fallback=_clean_text(query_row.get("allowed_fallback")),
        no_result=no_result,
        representative_match=representative_match,
    )
    citations = [dict(item or {}) for item in list(payload.get("citations") or [])]
    citation_support_match = _citation_support_match(payload)

    return {
        "query": _clean_text(query_row.get("query")),
        "source": _clean_text(query_row.get("source")),
        "expected_family": expected_family,
        "expected_top1_or_set": _clean_text(query_row.get("expected_top1_or_set")),
        "expected_answer_mode": expected_answer_mode,
        "allowed_fallback": _clean_text(query_row.get("allowed_fallback")),
        "actual_family": actual_family,
        "family_match": "1" if _normalize_eval_token(expected_family) == _normalize_eval_token(actual_family) else "0",
        "query_frame_family": _clean_text(query_frame.get("family")),
        "actual_answer_mode": actual_answer_mode,
        "answer_mode_match": "1" if not expected_answer_mode or _normalize_eval_token(expected_answer_mode) == _normalize_eval_token(actual_answer_mode) else "0",
        "query_frame_answer_mode": _clean_text(query_frame.get("answer_mode")),
        "evidence_policy_key": _clean_text(evidence_policy.get("policyKey")),
        "actual_representative_paper_id": actual_representative_paper_id,
        "actual_representative_paper_title": actual_representative_paper_title,
        "actual_representative_selection_score": _clean_text(representative_selection.get("score")),
        "actual_representative_title_hits": _clean_text(representative_selection.get("titleHits")),
        "actual_representative_selection_reason": _clean_text(representative_selection.get("reason")),
        "representative_match": representative_match,
        "actual_runtime_used": _clean_text(runtime_execution.get("used")),
        "actual_fallback_reason": _clean_text(runtime_execution.get("fallbackReason")),
        "no_result": "1" if no_result else "0",
        "planner_attempted": "1" if bool(planner.get("attempted")) else "0",
        "planner_used": "1" if bool(planner.get("used")) else "0",
        "planner_reason": _clean_text(
            planner.get("reason")
            or query_plan.get("plannerReason")
            or query_plan.get("planner_reason")
            or query_frame.get("planner_reason")
        ),
        "pred_label": pred_label,
        "pred_reason": pred_reason,
        "citation_count": str(len(citations)),
        "citation_support_match": citation_support_match,
        "top_source_titles": " | ".join(_clean_text(item.get("title")) for item in sources if _clean_text(item.get("title"))),
        "latency_ms": str(round(float(latency_ms or 0.0), 3)),
        "top_k": str(max(1, int(top_k))),
        "retrieval_mode": _clean_text(retrieval_mode),
        "gate_mode": _clean_text(gate_mode) or "standard",
        "timeout_flag": "1" if bool(timeout_flag) else "0",
        "notes": "",
        "final_label": "",
        "final_notes": "",
    }


def _error_row(
    query_row: dict[str, str],
    *,
    top_k: int,
    retrieval_mode: str,
    latency_ms: float,
    error: Exception,
    gate_mode: str = "standard",
) -> dict[str, str]:
    timeout_flag = isinstance(error, TimeoutError)
    row = _serialize_row(
        query_row,
        {
            "status": "error",
            "paperFamily": "",
            "queryPlan": {},
            "queryFrame": {},
            "representativePaper": {},
            "answerSignals": {},
            "evidencePolicy": {},
            "plannerFallback": {},
            "familyRouteDiagnostics": {},
            "sources": [],
        },
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        latency_ms=latency_ms,
        gate_mode=gate_mode,
        timeout_flag=timeout_flag,
    )
    row["notes"] = f"collector_error={type(error).__name__}: {error}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect manual eval rows for the default paper ask contract.")
    parser.add_argument("--config", default=None, help="Optional config path")
    parser.add_argument(
        "--queries",
        default="eval/knowledgeos/queries/paper_default_eval_queries_v1.csv",
        help="CSV path with paper default contract queries",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--family-filter",
        default="",
        help="Optional comma-separated expected_family subset filter (for example: concept_explainer,paper_discover)",
    )
    parser.add_argument("--top-k", type=int, default=6, help="Ask retrieval top-k")
    parser.add_argument("--mode", default="hybrid", choices=["semantic", "keyword", "hybrid"], help="Retrieval mode")
    parser.add_argument("--alpha", type=float, default=0.7, help="Hybrid alpha")
    parser.add_argument(
        "--gate-mode",
        default="standard",
        choices=["standard", "stub_hard", "live_smoke"],
        help="Collection profile: keep current behavior, run full-sheet stub hard gate, or run the tiny live smoke gate.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=0,
        help="Per-query timeout. Gate modes can supply a default when this is zero.",
    )
    parser.add_argument(
        "--stub-llm",
        action="store_true",
        help="Use a local stub answer runtime so eval collects route/retrieval diagnostics without live generation latency.",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stub_llm, timeout_seconds = _gate_mode_defaults(
        gate_mode=str(args.gate_mode),
        stub_llm=bool(args.stub_llm),
        timeout_seconds=int(args.timeout_seconds),
    )

    factory = AppContextFactory(config_path=args.config)
    searcher = factory.get_searcher()
    selected_queries = _select_queries_for_gate(
        _read_queries(queries_path),
        gate_mode=str(args.gate_mode),
        family_filter=str(args.family_filter),
    )

    rows: list[dict[str, str]] = []
    with _readonly_ask_v2_runtime(), (_stubbed_answer_runtime(searcher) if bool(stub_llm) else nullcontext()):
        for item in selected_queries:
            query = _clean_text(item.get("query"))
            source_type = _clean_text(item.get("source")) or None
            started = time.perf_counter()
            try:
                result = _run_with_timeout(
                    int(timeout_seconds),
                    searcher.generate_answer,
                    query,
                    top_k=max(1, int(args.top_k)),
                    source_type=source_type,
                    retrieval_mode=str(args.mode),
                    alpha=float(args.alpha),
                    allow_external=False,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                row = _serialize_row(
                    item,
                    result,
                    top_k=max(1, int(args.top_k)),
                    retrieval_mode=str(args.mode),
                    latency_ms=latency_ms,
                    gate_mode=str(args.gate_mode),
                )
            except Exception as error:  # pragma: no cover - collector should preserve failures in CSV
                latency_ms = (time.perf_counter() - started) * 1000.0
                row = _error_row(
                    item,
                    top_k=max(1, int(args.top_k)),
                    retrieval_mode=str(args.mode),
                    latency_ms=latency_ms,
                    error=error,
                    gate_mode=str(args.gate_mode),
                )
            rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_DEFAULT_EVAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote paper default eval sheet: {out_path} "
        f"({len(rows)} queries, mode={str(args.mode)}, top_k={max(1, int(args.top_k))}, "
        f"stub_llm={bool(stub_llm)}, gate_mode={str(args.gate_mode)}, timeout={int(timeout_seconds)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
