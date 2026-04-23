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


YOUTUBE_DEFAULT_EVAL_FIELDNAMES = [
    "query",
    "source",
    "query_type",
    "expected_primary_source",
    "expected_answer_style",
    "difficulty",
    "regression_bucket",
    "expected_family",
    "expected_answer_mode",
    "actual_family",
    "family_match",
    "query_frame_family",
    "actual_answer_mode",
    "answer_mode_match",
    "query_frame_answer_mode",
    "evidence_policy_key",
    "actual_runtime_used",
    "no_result",
    "video_scope_applied",
    "chapter_scope_applied",
    "pred_label",
    "pred_reason",
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
    "https://www.youtube.com/watch?v=s4xnZMiEIJc 영상 요약해줘",
    "https://www.youtube.com/watch?v=s4xnZMiEIJc 에서 메모리는 언제 언급돼?",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _read_queries(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    items: list[dict[str, str]] = []
    for row in rows:
        query = _clean_text(row.get("query"))
        source = _clean_text(row.get("source"))
        if not query or source != "youtube":
            continue
        items.append({str(key): _clean_text(value) for key, value in row.items()})
    return items


def _normalize_token(value: Any) -> str:
    return " ".join(_clean_text(value).casefold().split())


def _expected_family(row: dict[str, str]) -> str:
    query_type = _normalize_token(row.get("query_type"))
    if query_type == "section_lookup":
        return "section_lookup"
    if query_type == "timestamp_lookup":
        return "timestamp_lookup"
    if query_type == "video_lookup":
        return "video_lookup"
    return "video_explainer"


def _expected_answer_mode(row: dict[str, str]) -> str:
    explicit = _clean_text(row.get("expected_answer_style"))
    return explicit or "concise_summary"


def _machine_judgment(
    *,
    expected_family: str,
    actual_family: str,
    expected_answer_mode: str,
    actual_answer_mode: str,
    no_result: bool,
) -> tuple[str, str]:
    if _normalize_token(expected_family) != _normalize_token(actual_family):
        return "bad", "family_mismatch"
    if no_result:
        return "partial", "route_match_no_result"
    if expected_answer_mode and _normalize_token(expected_answer_mode) != _normalize_token(actual_answer_mode):
        return "partial", "answer_mode_mismatch"
    return "good", "family_and_mode_match"


def _select_queries_for_gate(rows: list[dict[str, str]], *, gate_mode: str) -> list[dict[str, str]]:
    normalized_mode = _clean_text(gate_mode) or "standard"
    if normalized_mode != "live_smoke":
        return rows
    smoke_set = {_clean_text(item) for item in _LIVE_SMOKE_QUERIES}
    return [row for row in rows if _clean_text(row.get("query")) in smoke_set]


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
    original_vault = AskV2Service._ensure_vault_card
    try:
        AskV2Service._ensure_paper_card = lambda self, paper_id: self.sqlite_db.get_paper_card_v2(paper_id)  # type: ignore[method-assign]
        AskV2Service._ensure_vault_card = lambda self, note_id: self.sqlite_db.get_vault_card_v2(note_id)  # type: ignore[method-assign]
        yield
    finally:
        AskV2Service._ensure_paper_card = original_paper  # type: ignore[method-assign]
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
    query_frame = dict(payload.get("queryFrame") or {})
    family_diagnostics = dict(payload.get("familyRouteDiagnostics") or {})
    evidence_policy = dict(payload.get("evidencePolicy") or {})
    sources = [dict(item or {}) for item in list(payload.get("sources") or [])[:3]]
    expected_family = _expected_family(query_row)
    expected_answer_mode = _expected_answer_mode(query_row)
    actual_family = _clean_text(query_frame.get("family"))
    actual_answer_mode = _clean_text(family_diagnostics.get("answerMode") or query_frame.get("answer_mode"))
    no_result = _normalize_token(payload.get("status")) == "no_result"
    pred_label, pred_reason = _machine_judgment(
        expected_family=expected_family,
        actual_family=actual_family,
        expected_answer_mode=expected_answer_mode,
        actual_answer_mode=actual_answer_mode,
        no_result=no_result,
    )
    return {
        "query": _clean_text(query_row.get("query")),
        "source": "youtube",
        "query_type": _clean_text(query_row.get("query_type")),
        "expected_primary_source": _clean_text(query_row.get("expected_primary_source")),
        "expected_answer_style": _clean_text(query_row.get("expected_answer_style")),
        "difficulty": _clean_text(query_row.get("difficulty")),
        "regression_bucket": _clean_text(query_row.get("regression_bucket")),
        "expected_family": expected_family,
        "expected_answer_mode": expected_answer_mode,
        "actual_family": actual_family,
        "family_match": "1" if _normalize_token(expected_family) == _normalize_token(actual_family) else "0",
        "query_frame_family": _clean_text(query_frame.get("family")),
        "actual_answer_mode": actual_answer_mode,
        "answer_mode_match": "1" if _normalize_token(expected_answer_mode) == _normalize_token(actual_answer_mode) else "0",
        "query_frame_answer_mode": _clean_text(query_frame.get("answer_mode")),
        "evidence_policy_key": _clean_text(evidence_policy.get("policyKey")),
        "actual_runtime_used": _clean_text(family_diagnostics.get("runtimeUsed")),
        "no_result": "1" if no_result else "0",
        "video_scope_applied": "1" if bool(family_diagnostics.get("videoScopeApplied")) else "0",
        "chapter_scope_applied": "1" if bool(family_diagnostics.get("chapterScopeApplied")) else "0",
        "pred_label": pred_label,
        "pred_reason": pred_reason,
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
            "queryFrame": {},
            "evidencePolicy": {},
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
    parser = argparse.ArgumentParser(description="Collect manual eval rows for the default youtube ask contract.")
    parser.add_argument("--config", default=None, help="Optional config path")
    parser.add_argument(
        "--queries",
        default="eval/knowledgeos/queries/youtube_default_eval_queries_v1.csv",
        help="CSV path with youtube evaluation queries",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=6, help="Ask retrieval top-k")
    parser.add_argument("--mode", default="hybrid", choices=["semantic", "keyword", "hybrid"], help="Retrieval mode")
    parser.add_argument("--alpha", type=float, default=0.7, help="Hybrid alpha")
    parser.add_argument(
        "--gate-mode",
        default="standard",
        choices=["standard", "stub_hard", "live_smoke"],
        help="Collection profile: keep current behavior, run full-sheet stub hard gate, or run the tiny live smoke gate.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=0, help="Per-query timeout.")
    parser.add_argument("--stub-llm", action="store_true", help="Use a stub answer runtime for route/retrieval-only collection.")
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
    selected_queries = _select_queries_for_gate(_read_queries(queries_path), gate_mode=str(args.gate_mode))

    rows: list[dict[str, str]] = []
    with _readonly_ask_v2_runtime(), (_stubbed_answer_runtime(searcher) if bool(stub_llm) else nullcontext()):
        for item in selected_queries:
            query = _clean_text(item.get("query"))
            started = time.perf_counter()
            try:
                result = _run_with_timeout(
                    int(timeout_seconds),
                    searcher.generate_answer,
                    query,
                    top_k=max(1, int(args.top_k)),
                    source_type="youtube",
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
            except Exception as error:  # pragma: no cover
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
        writer = csv.DictWriter(handle, fieldnames=YOUTUBE_DEFAULT_EVAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote youtube default eval sheet: {out_path} "
        f"({len(rows)} queries, mode={str(args.mode)}, top_k={max(1, int(args.top_k))}, gate={str(args.gate_mode)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
