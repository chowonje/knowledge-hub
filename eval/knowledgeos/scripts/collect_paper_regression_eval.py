#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from knowledge_hub.ai.ask_v2 import AskV2Service
from knowledge_hub.application.context import AppContextFactory


ASK_V2_DIAGNOSTIC_FIELDNAMES = [
    "v2_verification_status",
    "ask_v2_hard_gate",
    "answerable_decision_reason",
    "preHardGateAnswerable",
    "preHardGateReason",
    "v2ConsensusUnsupportedClaimCount",
    "v2ConsensusWeakClaimCount",
    "v2ConsensusSupportedClaimCount",
    "v2ClaimReasonSummary",
    "askV2OriginalHardGateReason",
    "claimCardGateRelaxed",
    "claimCardGateRelaxationReason",
    "unsupported_fields",
    "weak_claim_count",
    "selected_card_count",
    "filtered_evidence_count",
    "compare_unique_paper_count",
    "compare_anchor_coverage",
    "hard_gate_reason",
    "v2_diagnostics_keys",
    "resolved_paper_ids",
    "selected_card_ids",
    "selected_card_titles",
    "selected_card_paper_ids",
    "selected_card_stage",
    "selection_reason",
    "candidate_count_before_rerank",
    "candidate_count_after_rerank",
    "resolved_pair_preserved",
]

LIVE_GENERATION_DIAGNOSTIC_FIELDNAMES = [
    "answerProvider",
    "answerModel",
    "answerRoute",
    "finalAnswerSource",
    "answerGenerationFallbackUsed",
    "answerGenerationErrorType",
    "answerGenerationWarning",
    "generationFallbackUsed",
    "generationFallbackReason",
    "conservativeFallbackApplied",
    "answerVerificationStatus",
    "modelCallMs",
    "promptChars",
    "contextChars",
    "unsupportedClaimCount",
    "uncertainClaimCount",
    "needsCaution",
    "citationTracePresent",
    "citationCount",
    "sourceTitleTracePresent",
    "latencyMs",
    "timeoutFlag",
]

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
] + LIVE_GENERATION_DIAGNOSTIC_FIELDNAMES + ASK_V2_DIAGNOSTIC_FIELDNAMES

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


def _row_flag(row: dict[str, str], key: str) -> bool:
    return _clean_text(row.get(key)).casefold() in {"1", "true", "yes", "y", "on"}


def _float_or_zero(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, ((95 * len(ordered) + 99) // 100) - 1))
    return ordered[index]


def _route_source_preserved(row: dict[str, str]) -> bool:
    return _clean_text(row.get("resolved_pair_preserved")) not in {"0", "false", "False"}


def _build_live_smoke_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    row_count = len(rows)
    generation_fallback_count = sum(
        1
        for row in rows
        if _row_flag(row, "answerGenerationFallbackUsed") or _row_flag(row, "generationFallbackUsed")
    )
    conservative_fallback_count = sum(1 for row in rows if _row_flag(row, "conservativeFallbackApplied"))
    read_timeout_count = sum(
        1
        for row in rows
        if "readtimeout" in _clean_text(row.get("answerGenerationErrorType")).casefold()
        or "readtimeout" in _clean_text(row.get("answerGenerationWarning")).casefold()
        or "readtimeout" in _clean_text(row.get("generationFallbackReason")).casefold()
    )
    timeout_count = sum(1 for row in rows if _row_flag(row, "timeoutFlag") or _row_flag(row, "timeout_flag"))
    latencies = [_float_or_zero(row.get("latencyMs") or row.get("latency_ms")) for row in rows]
    p95_latency_ms = _p95(latencies)
    max_latency_ms = max(latencies) if latencies else 0.0
    citation_trace_present_count = sum(1 for row in rows if _row_flag(row, "citationTracePresent"))
    unsupported_claims_judgment_count = sum(
        1 for row in rows if _clean_text(row.get("pred_reason")) == "unsupported_claims"
    )
    structural_route_ready = bool(row_count) and all(
        not _row_flag(row, "no_result")
        and not _row_flag(row, "ask_v2_hard_gate")
        and _route_source_preserved(row)
        for row in rows
    )
    route_acceptance_ready = structural_route_ready and all(
        _clean_text(row.get("pred_label")) == "good" for row in rows
    )
    answer_acceptance_blockers: list[str] = []
    if not structural_route_ready:
        answer_acceptance_blockers.append("structural_route_not_ready")
    if unsupported_claims_judgment_count:
        answer_acceptance_blockers.append("unsupported_claims")
    if generation_fallback_count:
        answer_acceptance_blockers.append("generation_fallback")
    if read_timeout_count:
        answer_acceptance_blockers.append("read_timeout")
    if timeout_count:
        answer_acceptance_blockers.append("timeout")
    if p95_latency_ms > 45000:
        answer_acceptance_blockers.append("p95_latency_gt_45000")
    if citation_trace_present_count < row_count:
        answer_acceptance_blockers.append("citation_trace_missing")
    return {
        "rowCount": row_count,
        "generationFallbackCount": generation_fallback_count,
        "generationFallbackRate": round(generation_fallback_count / row_count, 6) if row_count else 0.0,
        "conservativeFallbackCount": conservative_fallback_count,
        "readTimeoutCount": read_timeout_count,
        "timeoutCount": timeout_count,
        "p95LatencyMs": round(p95_latency_ms, 3),
        "maxLatencyMs": round(max_latency_ms, 3),
        "citationTracePresentCount": citation_trace_present_count,
        "unsupportedClaimsJudgmentCount": unsupported_claims_judgment_count,
        "structuralRouteReady": structural_route_ready,
        "routeAcceptanceReady": route_acceptance_ready,
        "answerAcceptanceReady": structural_route_ready and not answer_acceptance_blockers,
        "answerAcceptanceBlockers": answer_acceptance_blockers,
    }


def _live_smoke_summary_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}.summary.json")


def _write_live_smoke_summary(rows: list[dict[str, str]], out_path: Path) -> Path:
    summary_path = _live_smoke_summary_path(out_path)
    summary_path.write_text(
        json.dumps(_build_live_smoke_summary(rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path

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


def _int_or_zero(*values: Any) -> int:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _join_values(values: Any) -> str:
    if isinstance(values, (str, bytes)):
        return _clean_text(values)
    return " | ".join(_clean_text(item) for item in list(values or []) if _clean_text(item))


def _optional_bool_flag(value: Any) -> str:
    if value in (None, ""):
        return ""
    normalized = _clean_text(value).casefold()
    if normalized in {"1", "true", "yes"}:
        return "1"
    if normalized in {"0", "false", "no"}:
        return "0"
    return "1" if bool(value) else "0"


def _bool_flag(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    token = _clean_text(value).casefold()
    if token in {"1", "true", "yes", "y", "on"}:
        return "1"
    if token in {"0", "false", "no", "n", "off", ""}:
        return "0"
    return "1"


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return dict(value)
    return {}


def _first_text(*values: Any) -> str:
    for value in values:
        token = _clean_text(value)
        if token:
            return token
    return ""




def _first_warning(*values: Any) -> str:
    for value in values:
        if isinstance(value, (str, bytes)):
            token = _clean_text(value)
            if token:
                return token
            continue
        for item in list(value or []):
            token = _clean_text(item)
            if token:
                return token
    return ""


def _numeric_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        token = _clean_text(value)
        if token:
            return token
    return ""


def _generation_fallback_reason(answer_generation: dict[str, Any], *, final_answer_source: str) -> str:
    fallback_used = bool(answer_generation.get("fallbackUsed")) or _clean_text(answer_generation.get("status")).casefold() == "fallback"
    if not fallback_used:
        if _clean_text(final_answer_source).casefold() == "generation_fallback":
            return "finalAnswerSource:generation_fallback"
        return ""
    stage = _clean_text(answer_generation.get("stage"))
    error_type = _clean_text(answer_generation.get("errorType"))
    error_message = _clean_text(answer_generation.get("errorMessage"))
    return ":".join(part for part in (stage, error_type, error_message) if part)


def _live_generation_diagnostic_columns(
    payload: dict[str, Any],
    *,
    citation_count: int,
    top_source_titles: str,
    latency_ms: float,
    timeout_flag: bool,
) -> dict[str, str]:
    answer_verification = dict(payload.get("answerVerification") or {})
    answer_rewrite = dict(payload.get("answerRewrite") or {})
    answer_generation = dict(payload.get("answerGeneration") or {})
    answer_contract = dict(payload.get("answerContract") or {})
    router = dict(payload.get("router") or {})
    route = _first_mapping(
        router.get("selected"),
        answer_generation.get("route"),
        answer_rewrite.get("route"),
    )
    final_answer_source = _first_text(
        answer_rewrite.get("finalAnswerSource"),
        dict(answer_contract.get("rewrite") or {}).get("finalAnswerSource"),
    )
    generation_fallback_used = bool(answer_generation.get("fallbackUsed")) or _clean_text(answer_generation.get("status")).casefold() == "fallback"
    if not generation_fallback_used and final_answer_source == "generation_fallback":
        generation_fallback_used = True
    citations = list(payload.get("citations") or [])
    contract_citations = list(answer_contract.get("citations") or [])
    context_stats = dict(payload.get("contextStats") or {})
    evidence = list(payload.get("evidence") or [])
    source_title_trace_present = bool(_clean_text(top_source_titles)) or any(
        _clean_text(dict(item or {}).get("title"))
        for item in [*list(payload.get("sources") or []), *evidence]
    )
    citation_trace_present = bool(citations or contract_citations or source_title_trace_present or evidence)
    generation_error_type = _clean_text(answer_generation.get("errorType"))
    generation_warning = _first_warning(
        answer_generation.get("warnings"),
        payload.get("warnings"),
        answer_verification.get("warnings"),
        answer_rewrite.get("warnings"),
    )
    fallback_reason = _generation_fallback_reason(answer_generation, final_answer_source=final_answer_source)
    return {
        "answerProvider": _clean_text(route.get("provider")),
        "answerModel": _clean_text(route.get("model") or route.get("model_id") or route.get("modelId")),
        "answerRoute": _clean_text(route.get("route")),
        "finalAnswerSource": final_answer_source,
        "answerGenerationFallbackUsed": "1" if generation_fallback_used else "0",
        "answerGenerationErrorType": generation_error_type,
        "answerGenerationWarning": generation_warning,
        "generationFallbackUsed": "1" if generation_fallback_used else "0",
        "generationFallbackReason": fallback_reason,
        "conservativeFallbackApplied": "1" if final_answer_source == "conservative_fallback" else "0",
        "answerVerificationStatus": _clean_text(answer_verification.get("status")),
        "modelCallMs": _numeric_text(
            answer_generation.get("modelCallMs"),
            answer_generation.get("model_call_ms"),
            answer_generation.get("durationMs"),
            answer_generation.get("duration_ms"),
            answer_generation.get("latencyMs"),
            answer_generation.get("latency_ms"),
        ),
        "promptChars": _numeric_text(answer_generation.get("promptChars"), answer_generation.get("prompt_chars")),
        "contextChars": _numeric_text(
            answer_generation.get("contextChars"),
            answer_generation.get("context_chars"),
            context_stats.get("contextChars"),
            context_stats.get("context_chars"),
        ),
        "unsupportedClaimCount": str(_int_or_zero(answer_verification.get("unsupportedClaimCount"), payload.get("unsupportedClaimCount"))),
        "uncertainClaimCount": str(_int_or_zero(answer_verification.get("uncertainClaimCount"), payload.get("uncertainClaimCount"))),
        "needsCaution": _bool_flag(answer_verification.get("needsCaution")),
        "citationTracePresent": "1" if citation_trace_present else "0",
        "citationCount": str(max(int(citation_count or 0), len(contract_citations))),
        "sourceTitleTracePresent": "1" if source_title_trace_present else "0",
        "latencyMs": str(round(float(latency_ms or 0.0), 3)),
        "timeoutFlag": "1" if bool(timeout_flag) else "0",
    }


def _claim_reason_summary(values: Any, *, limit: int = 6, max_length: int = 360) -> str:
    reason_counts: dict[str, int] = {}
    for item in list(values or []):
        claim = dict(item or {})
        status = _clean_text(claim.get("status")) or "unknown"
        reasons = [_clean_text(reason) for reason in list(claim.get("reasons") or []) if _clean_text(reason)]
        if not reasons:
            reasons = [_clean_text(claim.get("verdict")) or "no_reason"]
        for reason in reasons:
            key = f"{status}:{reason}"
            reason_counts[key] = reason_counts.get(key, 0) + 1
    parts = [
        f"{reason}={count}"
        for reason, count in sorted(reason_counts.items(), key=lambda entry: (-entry[1], entry[0]))[: max(1, int(limit))]
    ]
    summary = " | ".join(parts)
    if len(summary) <= max_length:
        return summary
    return f"{summary[: max(0, max_length - 1)].rstrip()}..."


def _unique_values(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = _clean_text(value)
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _resolved_paper_ids_from_payload(payload: dict[str, Any]) -> list[str]:
    query_frame = dict(payload.get("queryFrame") or {})
    query_plan = dict(payload.get("queryPlan") or {})
    return _unique_values(
        [
            *list(query_frame.get("resolved_source_ids") or []),
            *list(query_plan.get("resolvedPaperIds") or []),
            *list(query_plan.get("resolved_paper_ids") or []),
            *list(query_plan.get("resolvedSourceIds") or []),
        ]
    )


def _selected_card_ids(*, routing: dict[str, Any], selected_cards: list[dict[str, Any]]) -> list[str]:
    return _unique_values(
        [
            *list(routing.get("selected_card_ids") or []),
            *[card.get("cardId") for card in selected_cards],
        ]
    )


def _selected_card_paper_ids(selected_cards: list[dict[str, Any]]) -> list[str]:
    return _unique_values([card.get("paperId") or card.get("sourceId") for card in selected_cards])


def _card_selection_diagnostic_columns(
    *,
    payload: dict[str, Any],
    routing: dict[str, Any],
    card_selection: dict[str, Any],
    selected_cards: list[dict[str, Any]],
) -> dict[str, str]:
    resolved_paper_ids = _unique_values(
        [
            *list(card_selection.get("resolvedPaperIds") or []),
            *_resolved_paper_ids_from_payload(payload),
        ]
    )
    selected_card_paper_ids = _selected_card_paper_ids(selected_cards)
    selected_card_stage = _clean_text(card_selection.get("selectionStage") or card_selection.get("stage")) or _join_values(
        _unique_values([card.get("selectionStage") for card in selected_cards])
    )
    selection_reason = _clean_text(card_selection.get("selectionReason") or card_selection.get("reason")) or _join_values(
        _unique_values([card.get("selectionReason") for card in selected_cards])
    )
    resolved_pair_preserved = ""
    if len(resolved_paper_ids) >= 2:
        pair = resolved_paper_ids[:2]
        selected_set = set(selected_card_paper_ids)
        preserved = all(paper_id in selected_set for paper_id in pair)
        if "resolvedPairPreserved" in card_selection:
            preserved = _optional_bool_flag(card_selection.get("resolvedPairPreserved")) == "1"
        resolved_pair_preserved = "1" if preserved else "0"

    return {
        "resolved_paper_ids": _join_values(resolved_paper_ids),
        "selected_card_ids": _join_values(_selected_card_ids(routing=routing, selected_cards=selected_cards)),
        "selected_card_titles": _join_values(_unique_values([card.get("title") for card in selected_cards])),
        "selected_card_paper_ids": _join_values(selected_card_paper_ids),
        "selected_card_stage": selected_card_stage,
        "selection_reason": selection_reason,
        "candidate_count_before_rerank": (
            "" if card_selection.get("candidateCountBeforeRerank") is None else str(card_selection.get("candidateCountBeforeRerank")).strip()
        ),
        "candidate_count_after_rerank": (
            "" if card_selection.get("candidateCountAfterRerank") is None else str(card_selection.get("candidateCountAfterRerank")).strip()
        ),
        "resolved_pair_preserved": resolved_pair_preserved,
    }


def _unique_paper_ids_from_payload(payload: dict[str, Any], v2: dict[str, Any]) -> set[str]:
    paper_ids: set[str] = set()
    for item in list(payload.get("evidence") or payload.get("sources") or []):
        source = dict(item or {})
        for key in ("paper_id", "arxiv_id", "citation_target"):
            token = _clean_text(source.get(key))
            if token:
                paper_ids.add(token)
    card_selection = dict(v2.get("cardSelection") or {})
    for item in list(card_selection.get("selected") or []):
        card = dict(item or {})
        token = _clean_text(card.get("sourceId"))
        if token:
            paper_ids.add(token)
    section_selection = dict(v2.get("sectionSelection") or {})
    for item in list(section_selection.get("selected") or []):
        section = dict(item or {})
        token = _clean_text(section.get("paperId"))
        if token:
            paper_ids.add(token)
    claim_selection = dict(v2.get("claimSelection") or {})
    for item in list(claim_selection.get("selected") or []):
        claim = dict(item or {})
        if _clean_text(claim.get("sourceKind")).lower() not in {"", "paper"}:
            continue
        token = _clean_text(claim.get("sourceId"))
        if token:
            paper_ids.add(token)
    for item in list(v2.get("sectionCards") or []):
        section = dict(item or {})
        token = _clean_text(section.get("paperId"))
        if token:
            paper_ids.add(token)
    for item in list(v2.get("claimCards") or []):
        claim = dict(item or {})
        if _clean_text(claim.get("sourceKind")).lower() not in {"", "paper"}:
            continue
        token = _clean_text(claim.get("sourceId"))
        if token:
            paper_ids.add(token)
    return paper_ids


def _ask_v2_diagnostic_columns(payload: dict[str, Any], *, actual_family: str) -> dict[str, str]:
    v2 = dict(payload.get("v2") or {})
    verification = dict(v2.get("evidenceVerification") or {})
    consensus = dict(
        v2.get("consensus")
        or payload.get("claimConsensus")
        or payload.get("claim_consensus")
        or {}
    )
    evidence_packet = dict(payload.get("evidencePacket") or {})
    routing = dict(v2.get("routing") or {})
    card_selection = dict(v2.get("cardSelection") or {})
    selected_card_ids = [
        _clean_text(item)
        for item in list(routing.get("selected_card_ids") or [])
        if _clean_text(item)
    ]
    selected_cards = [dict(item or {}) for item in list(card_selection.get("selected") or [])]
    selected_card_count = len(selected_card_ids) or len(selected_cards)
    evidence_items = list(payload.get("evidence") or payload.get("sources") or [])
    unique_paper_ids = _unique_paper_ids_from_payload(payload, v2)
    unique_paper_count = _int_or_zero(evidence_packet.get("uniquePaperCount")) or len(unique_paper_ids)
    anchor_ids = list(verification.get("anchorIdsUsed") or [])
    answerable_decision_reason = _clean_text(evidence_packet.get("answerableDecisionReason"))
    ask_v2_hard_gate = bool(evidence_packet.get("askV2HardGate"))
    is_compare = _normalize_eval_token(actual_family) == "paper_compare"
    v2_consensus_unsupported = _int_or_zero(
        v2.get("v2ConsensusUnsupportedClaimCount"),
        consensus.get("unsupportedClaimCount"),
    )
    v2_consensus_weak = _int_or_zero(
        v2.get("v2ConsensusWeakClaimCount"),
        consensus.get("weakClaimCount"),
    )
    v2_consensus_supported = _int_or_zero(
        v2.get("v2ConsensusSupportedClaimCount"),
        consensus.get("supportedClaimCount"),
        consensus.get("supportCount"),
    )
    v2_claim_reason_summary = _clean_text(v2.get("v2ClaimReasonSummary")) or _claim_reason_summary(v2.get("claimVerification"))

    diagnostics = {
        "v2_verification_status": _clean_text(
            verification.get("verificationStatus")
            or evidence_packet.get("askV2VerificationStatus")
        ),
        "ask_v2_hard_gate": "1" if ask_v2_hard_gate else "0",
        "answerable_decision_reason": answerable_decision_reason,
        "preHardGateAnswerable": _optional_bool_flag(v2.get("preHardGateAnswerable")),
        "preHardGateReason": _clean_text(v2.get("preHardGateReason")),
        "v2ConsensusUnsupportedClaimCount": str(v2_consensus_unsupported),
        "v2ConsensusWeakClaimCount": str(v2_consensus_weak),
        "v2ConsensusSupportedClaimCount": str(v2_consensus_supported),
        "v2ClaimReasonSummary": v2_claim_reason_summary,
        "askV2OriginalHardGateReason": _clean_text(
            v2.get("askV2OriginalHardGateReason")
            or evidence_packet.get("askV2OriginalHardGateReason")
        ),
        "claimCardGateRelaxed": _optional_bool_flag(
            v2.get("claimCardGateRelaxed")
            if "claimCardGateRelaxed" in v2
            else evidence_packet.get("claimCardGateRelaxed")
        ),
        "claimCardGateRelaxationReason": _clean_text(
            v2.get("claimCardGateRelaxationReason")
            or evidence_packet.get("claimCardGateRelaxationReason")
        ),
        "unsupported_fields": _join_values(verification.get("unsupportedFields")),
        "weak_claim_count": str(_int_or_zero(consensus.get("weakClaimCount"))),
        "selected_card_count": str(selected_card_count),
        "filtered_evidence_count": str(len(evidence_items)),
        "compare_unique_paper_count": str(unique_paper_count) if is_compare or unique_paper_count else "",
        "compare_anchor_coverage": (
            f"unique_papers={unique_paper_count};anchors={len(anchor_ids)};selected_cards={selected_card_count}"
            if is_compare or unique_paper_count or anchor_ids
            else ""
        ),
        "hard_gate_reason": answerable_decision_reason if ask_v2_hard_gate else "",
        "v2_diagnostics_keys": " | ".join(sorted(str(key) for key in v2.keys())),
    }
    diagnostics.update(
        _card_selection_diagnostic_columns(
            payload=payload,
            routing=routing,
            card_selection=card_selection,
            selected_cards=selected_cards,
        )
    )
    return diagnostics


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
    citations = [dict(item or {}) for item in list(payload.get("citations") or [])]
    top_source_titles = " | ".join(
        _clean_text(dict(item or {}).get("title"))
        for item in list(payload.get("sources") or [])
        if _clean_text(dict(item or {}).get("title"))
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

    row = {
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
        "top_source_titles": top_source_titles,
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
    row.update(
        _live_generation_diagnostic_columns(
            payload,
            citation_count=len(citations),
            top_source_titles=top_source_titles,
            latency_ms=latency_ms,
            timeout_flag=timeout_flag,
        )
    )
    row.update(_ask_v2_diagnostic_columns(payload, actual_family=actual_family))
    return row


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
    summary_path = None
    if _clean_text(args.gate_mode) == "live_smoke":
        summary_path = _write_live_smoke_summary(rows, out_path)
    print(
        f"Wrote paper regression eval sheet: {out_path} "
        f"({len(rows)} queries, mode={str(args.mode)}, top_k={max(1, int(args.top_k))}, "
        f"stub_llm={bool(stub_llm)}, gate_mode={str(args.gate_mode)}, timeout={int(timeout_seconds)})"
    )
    if summary_path is not None:
        print(f"Wrote paper regression live smoke summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
