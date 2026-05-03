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


PAPER_REGRESSION_EVAL_FIELDNAMES = [
    "query",
    "source",
    "eval_bucket",
    "expected_family",
    "expected_top1_or_set",
    "expected_answer_mode",
    "expected_match_count",
    "expected_scope_applied",
    "allowed_fallback",
    "actual_family",
    "family_match",
    "query_frame_family",
    "actual_answer_mode",
    "answer_mode_match",
    "evidence_policy_key",
    "actual_representative_paper_id",
    "actual_representative_paper_title",
    "actual_runtime_used",
    "matched_expected_count",
    "source_match",
    "actual_paper_scoped",
    "actual_paper_scope_applied",
    "actual_paper_scope_fallback_used",
    "actual_paper_scope_reason",
    "answer_provenance_mode",
    "no_result",
    "needs_caution",
    "unsupported_claim_count",
    "rewrite_applied",
    "top_source_titles",
    "answer_preview",
    "latency_ms",
    "top_k",
    "retrieval_mode",
    "gate_mode",
    "timeout_flag",
    "pred_label",
    "pred_reason",
    "notes",
    "final_label",
    "final_notes",
]

_LIVE_SMOKE_QUERIES = (
    "Deep Residual Learning 논문 설명해줘",
    "Deep Residual Learning 논문의 방법을 설명해줘",
    "Batch Normalization 논문의 핵심 결과를 설명해줘",
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks와 Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering을 비교해줘",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _preview_text(value: Any, *, limit: int = 220) -> str:
    text = " ".join(_clean_text(value).split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)].rstrip()}…"


def _normalize_eval_token(value: Any) -> str:
    token = _clean_text(value).casefold()
    return " ".join(token.split())


def _read_queries(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    items: list[dict[str, str]] = []
    for row in rows:
        query = _clean_text(row.get("query"))
        if not query:
            continue
        items.append({str(key): _clean_text(value) for key, value in row.items()})
    return items


def _expected_set(raw_value: Any) -> list[str]:
    return [item for item in (_normalize_eval_token(part) for part in str(raw_value or "").split("|")) if item]


def _selected_queries(rows: list[dict[str, str]], *, gate_mode: str) -> list[dict[str, str]]:
    normalized = _clean_text(gate_mode) or "standard"
    if normalized != "live_smoke":
        return rows
    smoke_set = {_clean_text(item) for item in _LIVE_SMOKE_QUERIES}
    return [row for row in rows if _clean_text(row.get("query")) in smoke_set]


def _gate_mode_defaults(*, gate_mode: str, stub_llm: bool, timeout_seconds: int) -> tuple[bool, int]:
    normalized = _clean_text(gate_mode) or "standard"
    if normalized == "stub_hard":
        return True, timeout_seconds or 20
    if normalized == "live_smoke":
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


def _actual_source_tokens(payload: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    representative = dict(payload.get("representativePaper") or {})
    for key in ("paperId", "title"):
        token = _normalize_eval_token(representative.get(key))
        if token:
            candidates.append(token)
    for item in list(payload.get("sources") or []):
        source = dict(item or {})
        for key in ("paper_id", "arxiv_id", "citation_target", "title"):
            token = _normalize_eval_token(source.get(key))
            if token:
                candidates.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _count_expected_matches(expected_raw: str, actual_tokens: list[str]) -> int:
    expected = _expected_set(expected_raw)
    if not expected or not actual_tokens:
        return 0
    matched = 0
    remaining_actual = list(actual_tokens)
    for token in expected:
        found_index = -1
        for index, actual in enumerate(remaining_actual):
            if token == actual or token in actual or actual in token:
                found_index = index
                break
        if found_index < 0:
            continue
        matched += 1
        remaining_actual.pop(found_index)
    return matched


def _machine_judgment(
    *,
    eval_bucket: str,
    expected_family: str,
    actual_family: str,
    expected_answer_mode: str,
    actual_answer_mode: str,
    expected_match_count: int,
    matched_expected_count: int,
    expected_scope_applied: bool,
    actual_scope_applied: bool,
    no_result: bool,
    allowed_fallback: str,
    needs_caution: bool,
    unsupported_claim_count: int,
) -> tuple[str, str]:
    if _normalize_eval_token(expected_family) != _normalize_eval_token(actual_family):
        return "bad", "family_mismatch"
    if no_result:
        allowed = _normalize_eval_token(allowed_fallback)
        if any(token in allowed for token in ("no_result", "planner_retry", "need_multiple_papers")):
            return "partial", "allowed_no_result_fallback"
        return "bad", "unexpected_no_result"
    if expected_answer_mode and _normalize_eval_token(expected_answer_mode) != _normalize_eval_token(actual_answer_mode):
        return "partial", "answer_mode_mismatch"
    if matched_expected_count < max(1, int(expected_match_count or 1)):
        if _normalize_eval_token(eval_bucket) == "compare":
            return "partial", "source_undercoverage"
        return "bad", "source_mismatch"
    if expected_scope_applied and not actual_scope_applied:
        return "partial", "paper_scope_not_applied"
    if unsupported_claim_count > 0:
        return "partial", "unsupported_claims"
    if needs_caution:
        return "partial", "needs_caution"
    return "good", "family_scope_and_source_match"


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
    representative = dict(payload.get("representativePaper") or {})
    paper_scope = dict(payload.get("paperAnswerScope") or {})
    answer_verification = dict(payload.get("answerVerification") or {})
    answer_rewrite = dict(payload.get("answerRewrite") or {})
    answer_provenance = dict(payload.get("answerProvenance") or {})
    actual_family = _clean_text(payload.get("paperFamily") or query_frame.get("family"))
    actual_answer_mode = _clean_text(family_diagnostics.get("answerMode") or query_frame.get("answer_mode"))
    actual_tokens = _actual_source_tokens(payload)
    matched_expected_count = _count_expected_matches(_clean_text(query_row.get("expected_top1_or_set")), actual_tokens)
    expected_match_count = max(1, int(_clean_text(query_row.get("expected_match_count")) or "1"))
    expected_scope_applied = _clean_text(query_row.get("expected_scope_applied")) in {"1", "true", "yes"}
    actual_scope_applied = bool(paper_scope.get("applied")) or bool(family_diagnostics.get("resolvedSourceScopeApplied"))
    no_result = _clean_text(payload.get("status")).lower() == "no_result"
    needs_caution = bool(answer_verification.get("needsCaution"))
    unsupported_claim_count = int(
        answer_verification.get("unsupportedClaimCount")
        or payload.get("unsupportedClaimCount")
        or 0
    )
    pred_label, pred_reason = _machine_judgment(
        eval_bucket=_clean_text(query_row.get("eval_bucket")),
        expected_family=_clean_text(query_row.get("expected_family")),
        actual_family=actual_family,
        expected_answer_mode=_clean_text(query_row.get("expected_answer_mode")),
        actual_answer_mode=actual_answer_mode,
        expected_match_count=expected_match_count,
        matched_expected_count=matched_expected_count,
        expected_scope_applied=expected_scope_applied,
        actual_scope_applied=actual_scope_applied,
        no_result=no_result,
        allowed_fallback=_clean_text(query_row.get("allowed_fallback")),
        needs_caution=needs_caution,
        unsupported_claim_count=unsupported_claim_count,
    )

    return {
        "query": _clean_text(query_row.get("query")),
        "source": _clean_text(query_row.get("source")),
        "eval_bucket": _clean_text(query_row.get("eval_bucket")),
        "expected_family": _clean_text(query_row.get("expected_family")),
        "expected_top1_or_set": _clean_text(query_row.get("expected_top1_or_set")),
        "expected_answer_mode": _clean_text(query_row.get("expected_answer_mode")),
        "expected_match_count": str(expected_match_count),
        "expected_scope_applied": "1" if expected_scope_applied else "0",
        "allowed_fallback": _clean_text(query_row.get("allowed_fallback")),
        "actual_family": actual_family,
        "family_match": "1" if _normalize_eval_token(_clean_text(query_row.get("expected_family"))) == _normalize_eval_token(actual_family) else "0",
        "query_frame_family": _clean_text(query_frame.get("family")),
        "actual_answer_mode": actual_answer_mode,
        "answer_mode_match": "1" if not _clean_text(query_row.get("expected_answer_mode")) or _normalize_eval_token(_clean_text(query_row.get("expected_answer_mode"))) == _normalize_eval_token(actual_answer_mode) else "0",
        "evidence_policy_key": _clean_text(evidence_policy.get("policyKey")),
        "actual_representative_paper_id": _clean_text(representative.get("paperId")),
        "actual_representative_paper_title": _clean_text(representative.get("title")),
        "actual_runtime_used": _clean_text(family_diagnostics.get("runtimeUsed")),
        "matched_expected_count": str(matched_expected_count),
        "source_match": "1" if matched_expected_count >= expected_match_count else "0",
        "actual_paper_scoped": "1" if bool(paper_scope.get("paperScoped")) else "0",
        "actual_paper_scope_applied": "1" if actual_scope_applied else "0",
        "actual_paper_scope_fallback_used": "1" if bool(paper_scope.get("fallbackUsed")) else "0",
        "actual_paper_scope_reason": _clean_text(paper_scope.get("reason")),
        "answer_provenance_mode": _clean_text(answer_provenance.get("mode")),
        "no_result": "1" if no_result else "0",
        "needs_caution": "1" if needs_caution else "0",
        "unsupported_claim_count": str(unsupported_claim_count),
        "rewrite_applied": "1" if bool(answer_rewrite.get("applied")) else "0",
        "top_source_titles": " | ".join(_clean_text(dict(item or {}).get("title")) for item in list(payload.get("sources") or []) if _clean_text(dict(item or {}).get("title"))),
        "answer_preview": _preview_text(payload.get("answer")),
        "latency_ms": str(round(float(latency_ms or 0.0), 3)),
        "top_k": str(max(1, int(top_k))),
        "retrieval_mode": _clean_text(retrieval_mode),
        "gate_mode": _clean_text(gate_mode) or "standard",
        "timeout_flag": "1" if bool(timeout_flag) else "0",
        "pred_label": pred_label,
        "pred_reason": pred_reason,
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
            "familyRouteDiagnostics": {},
            "evidencePolicy": {},
            "representativePaper": {},
            "paperAnswerScope": {},
            "answerVerification": {},
            "answerRewrite": {},
            "answerProvenance": {},
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
    parser = argparse.ArgumentParser(description="Collect manual regression eval rows for the paper ask contract.")
    parser.add_argument("--config", default=None, help="Optional config path")
    parser.add_argument(
        "--queries",
        default="eval/knowledgeos/queries/paper_regression_eval_queries_v1.csv",
        help="CSV path with paper regression queries",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=6, help="Ask retrieval top-k")
    parser.add_argument("--mode", default="hybrid", choices=["semantic", "keyword", "hybrid"], help="Retrieval mode")
    parser.add_argument("--alpha", type=float, default=0.7, help="Hybrid alpha")
    parser.add_argument(
        "--gate-mode",
        default="standard",
        choices=["standard", "stub_hard", "live_smoke"],
        help="Collection profile: standard, full-sheet stub hard gate, or a tiny live smoke subset.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=0, help="Per-query timeout; gate modes can supply defaults.")
    parser.add_argument("--stub-llm", action="store_true", help="Use a local stub answer runtime to avoid live generation latency.")
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
    selected_queries = _selected_queries(_read_queries(queries_path), gate_mode=str(args.gate_mode))

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
        writer = csv.DictWriter(handle, fieldnames=PAPER_REGRESSION_EVAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote paper regression eval sheet: {out_path} "
        f"({len(rows)} queries, mode={str(args.mode)}, top_k={max(1, int(args.top_k))}, "
        f"stub_llm={bool(stub_llm)}, gate_mode={str(args.gate_mode)}, timeout={int(timeout_seconds)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
