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


VAULT_DEFAULT_EVAL_FIELDNAMES = [
    "query",
    "source",
    "query_type",
    "expected_primary_source",
    "expected_family",
    "expected_answer_mode",
    "difficulty",
    "regression_bucket",
    "actual_family",
    "family_match",
    "query_frame_family",
    "actual_answer_mode",
    "answer_mode_match",
    "query_frame_answer_mode",
    "evidence_policy_key",
    "actual_runtime_used",
    "actual_fallback_reason",
    "no_result",
    "vault_scope_applied",
    "temporal_signals_applied",
    "pred_label",
    "pred_reason",
    "citation_count",
    "stale_citation_count",
    "stale_citation_rate",
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
    "RAG 검색 품질을 떨어뜨리는 가장 흔한 원인은 무엇인가?",
    "최근 retrieval pipeline에서 temporal route가 추가된 이유는 무엇인가?",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _read_queries(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    items: list[dict[str, str]] = []
    for row in rows:
        query = _clean_text(row.get("query"))
        source = _clean_text(row.get("source"))
        if not query or source != "vault":
            continue
        items.append({str(key): _clean_text(value) for key, value in row.items()})
    return items


def _normalize_token(value: Any) -> str:
    return " ".join(_clean_text(value).casefold().split())


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


def _resolve_vault_source_path(vault_root: Path | None, source: dict[str, Any]) -> Path | None:
    raw_path = _clean_text(source.get("file_path"))
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if vault_root is None:
        return None
    return (vault_root / raw_path).resolve()


def _vault_stale_citation_stats(payload: dict[str, Any], *, vault_root: Path | None) -> tuple[int, int, str]:
    sources = [dict(item or {}) for item in list(payload.get("sources") or [])]
    vault_sources = [
        item
        for item in sources
        if _normalize_token(item.get("normalized_source_type") or item.get("source_type")) == "vault"
    ]
    citation_count = len(vault_sources)
    if citation_count == 0:
        return 0, 0, ""
    stale_count = 0
    for source in vault_sources:
        resolved = _resolve_vault_source_path(vault_root, source)
        if resolved is None or not resolved.exists():
            stale_count += 1
    return citation_count, stale_count, f"{(stale_count / citation_count):.6f}"


def _select_queries_for_gate(rows: list[dict[str, str]], *, gate_mode: str) -> list[dict[str, str]]:
    if _clean_text(gate_mode) != "live_smoke":
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
    vault_root: Path | None,
    top_k: int,
    retrieval_mode: str,
    latency_ms: float,
    gate_mode: str = "standard",
    timeout_flag: bool = False,
) -> dict[str, str]:
    payload = dict(result or {})
    query_frame = dict(payload.get("queryFrame") or {})
    family_diagnostics = dict(payload.get("familyRouteDiagnostics") or {})
    runtime_execution = dict(dict(payload.get("v2") or {}).get("runtimeExecution") or {})
    evidence_policy = dict(payload.get("evidencePolicy") or {})
    sources = [dict(item or {}) for item in list(payload.get("sources") or [])[:3]]
    expected_family = _clean_text(query_row.get("expected_family"))
    expected_answer_mode = _clean_text(query_row.get("expected_answer_mode"))
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
    citation_count, stale_citation_count, stale_citation_rate = _vault_stale_citation_stats(
        payload,
        vault_root=vault_root,
    )
    return {
        "query": _clean_text(query_row.get("query")),
        "source": "vault",
        "query_type": _clean_text(query_row.get("query_type")),
        "expected_primary_source": _clean_text(query_row.get("expected_primary_source")),
        "expected_family": expected_family,
        "expected_answer_mode": expected_answer_mode,
        "difficulty": _clean_text(query_row.get("difficulty")),
        "regression_bucket": _clean_text(query_row.get("regression_bucket")),
        "actual_family": actual_family,
        "family_match": "1" if _normalize_token(expected_family) == _normalize_token(actual_family) else "0",
        "query_frame_family": actual_family,
        "actual_answer_mode": actual_answer_mode,
        "answer_mode_match": "1" if _normalize_token(expected_answer_mode) == _normalize_token(actual_answer_mode) else "0",
        "query_frame_answer_mode": _clean_text(query_frame.get("answer_mode")),
        "evidence_policy_key": _clean_text(evidence_policy.get("policyKey")),
        "actual_runtime_used": _clean_text(runtime_execution.get("used")),
        "actual_fallback_reason": _clean_text(runtime_execution.get("fallbackReason")),
        "no_result": "1" if no_result else "0",
        "vault_scope_applied": "1" if family_diagnostics.get("vaultScopeApplied") else "0",
        "temporal_signals_applied": "1" if family_diagnostics.get("temporalSignalsApplied") else "0",
        "pred_label": pred_label,
        "pred_reason": pred_reason,
        "citation_count": str(citation_count),
        "stale_citation_count": str(stale_citation_count),
        "stale_citation_rate": stale_citation_rate,
        "top_source_titles": " | ".join(_clean_text(item.get("title")) for item in sources if _clean_text(item.get("title"))),
        "latency_ms": f"{float(latency_ms):.3f}",
        "top_k": str(int(top_k)),
        "retrieval_mode": _clean_text(retrieval_mode),
        "gate_mode": _clean_text(gate_mode or "standard"),
        "timeout_flag": "1" if timeout_flag else "0",
        "notes": "",
        "final_label": "",
        "final_notes": "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect default vault-family eval rows.")
    parser.add_argument("--queries", default="eval/knowledgeos/queries/vault_default_eval_queries_v1.csv")
    parser.add_argument("--out", default="eval/knowledgeos/runs/vault_default_eval.csv")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--gate-mode", default="standard", choices=["standard", "stub_hard", "live_smoke"])
    parser.add_argument("--stub-llm", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=0)
    args = parser.parse_args(argv)

    queries_path = Path(args.queries)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected_queries = _select_queries_for_gate(_read_queries(queries_path), gate_mode=str(args.gate_mode))
    stub_llm, timeout_seconds = _gate_mode_defaults(
        gate_mode=str(args.gate_mode),
        stub_llm=bool(args.stub_llm),
        timeout_seconds=int(args.timeout_seconds or 0),
    )

    app = AppContextFactory().build(require_search=True)
    searcher = app.searcher
    vault_root = Path(str(getattr(app.config, "vault_path", "") or "")).expanduser().resolve() if getattr(app.config, "vault_path", "") else None
    runtime_cm = _stubbed_answer_runtime(searcher) if stub_llm else nullcontext()

    rows: list[dict[str, str]] = []
    with runtime_cm:
        for item in selected_queries:
            started = time.perf_counter()
            timeout_flag = False
            try:
                result = _run_with_timeout(
                    timeout_seconds,
                    searcher.generate_answer,
                    query=item["query"],
                    source_type="vault",
                    top_k=int(args.top_k),
                    retrieval_mode=str(args.mode),
                )
            except TimeoutError:
                timeout_flag = True
                result = {
                    "status": "no_result",
                    "queryFrame": {"family": "", "answer_mode": ""},
                    "familyRouteDiagnostics": {"answerMode": "", "runtimeUsed": ""},
                    "evidencePolicy": {"policyKey": ""},
                    "sources": [],
                }
            latency_ms = (time.perf_counter() - started) * 1000.0
            rows.append(
                _serialize_row(
                    item,
                    result,
                    vault_root=vault_root,
                    top_k=int(args.top_k),
                    retrieval_mode=str(args.mode),
                    latency_ms=latency_ms,
                    gate_mode=str(args.gate_mode),
                    timeout_flag=timeout_flag,
                )
            )

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VAULT_DEFAULT_EVAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote vault default eval sheet: {out_path} ({len(rows)} queries, mode={args.mode}, "
        f"top_k={args.top_k}, gate={args.gate_mode})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
