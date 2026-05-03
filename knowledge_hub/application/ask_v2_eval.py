"""Helpers for ask-v2 manual evaluation collection and reporting."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

POSITIVE_LABELS = {"good", "1", "pass", "relevant"}
PARTIAL_LABELS = {"partial"}
NEGATIVE_LABELS = {"bad", "0", "fail", "irrelevant"}

ASK_V2_EVAL_FIELDNAMES = [
    "query",
    "source",
    "query_type",
    "expected_primary_source",
    "expected_answer_style",
    "difficulty",
    "regression_bucket",
    "answer_status",
    "answer_preview",
    "no_result",
    "fallback_used",
    "weak_evidence",
    "needs_caution",
    "selected_source_kind",
    "selected_card_ids",
    "matched_entities",
    "routing_mode",
    "intent",
    "anchor_count",
    "unsupported_fields",
    "claim_verification_summary",
    "claim_conflict_count",
    "claim_weak_count",
    "claim_unsupported_count",
    "claim_card_count",
    "claim_alignment_group_count",
    "answer_provenance_mode",
    "latency_ms",
    "top_k",
    "retrieval_mode",
    "label",
    "wrong_source",
    "wrong_era",
    "should_abstain",
    "notes",
    "final_label",
    "final_wrong_source",
    "final_wrong_era",
    "final_should_abstain",
    "final_notes",
]

ASK_V2_MACHINE_REVIEW_FIELDNAMES = [
    *ASK_V2_EVAL_FIELDNAMES,
    "pred_label",
    "pred_wrong_source",
    "pred_wrong_era",
    "pred_should_abstain",
    "pred_confidence",
    "pred_reason",
]


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _preview_text(value: Any, *, limit: int = 220) -> str:
    text = " ".join(_clean_text(value).split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)].rstrip()}…"


def _normalize_label(value: Any) -> str:
    raw = _clean_text(value).lower()
    if raw in POSITIVE_LABELS:
        return "positive"
    if raw in PARTIAL_LABELS:
        return "partial"
    if raw in NEGATIVE_LABELS:
        return "negative"
    return ""


def _normalize_boolish(value: Any) -> bool:
    return _clean_text(value).lower() in {"1", "true", "yes", "y", "on"}


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _review_value(row: dict[str, str], column: str) -> str:
    final_column = f"final_{column}"
    if final_column in row and _clean_text(row.get(final_column)):
        return _clean_text(row.get(final_column))
    return _clean_text(row.get(column))


def _bucket_name(row: dict[str, str]) -> str:
    query_type = _clean_text(row.get("query_type")).lower()
    expected_style = _clean_text(row.get("expected_answer_style")).lower()
    source = _clean_text(row.get("source")).lower()
    regression_bucket = _clean_text(row.get("regression_bucket")).lower()
    if query_type == "temporal" or "temporal" in regression_bucket or "latest" in regression_bucket:
        return "temporal"
    if query_type == "abstention" or expected_style == "abstain":
        return "abstention"
    if source == "project" or "architecture" in regression_bucket:
        return "architecture-project"
    if query_type in {"implementation", "evaluation"}:
        return "implementation"
    if query_type in {"comparison", "relation", "definition"} or "compare" in expected_style:
        return "relation-comparison"
    return "general"


def serialize_ask_v2_eval_row(
    query_row: dict[str, Any],
    result: dict[str, Any],
    *,
    top_k: int,
    retrieval_mode: str,
    latency_ms: float,
) -> dict[str, str]:
    payload = dict(result or {})
    v2 = dict(payload.get("v2") or {})
    routing = dict(v2.get("routing") or {})
    evidence_verification = dict(v2.get("evidenceVerification") or {})
    claim_consensus = dict(payload.get("claimConsensus") or payload.get("claim_consensus") or v2.get("consensus") or {})
    claim_cards = list(payload.get("claimCards") or v2.get("claimCards") or [])
    claim_alignment = dict(payload.get("claimAlignment") or v2.get("claimAlignment") or {})
    answer_provenance = dict(payload.get("answerProvenance") or v2.get("answerProvenance") or {})
    answer_verification = dict(payload.get("answerVerification") or {})
    answer_rewrite = dict(payload.get("answerRewrite") or {})
    fallback = dict(v2.get("fallback") or {})

    selected_card_ids = [
        _clean_text(item)
        for item in list(routing.get("selected_card_ids") or [])
        if _clean_text(item)
    ]
    matched_entities = []
    for item in list(routing.get("matched_entities") or []):
        token = _clean_text(item.get("canonical_name") or item.get("entity_id"))
        if token:
            matched_entities.append(token)

    verification_status = _clean_text(evidence_verification.get("verificationStatus")).lower()
    weak_evidence = verification_status in {"weak", "missing"}
    fallback_used = bool(fallback.get("used"))
    if _clean_text(answer_rewrite.get("finalAnswerSource")) == "conservative_fallback":
        fallback_used = True

    return {
        "query": _clean_text(query_row.get("query")),
        "source": _clean_text(query_row.get("source")),
        "query_type": _clean_text(query_row.get("query_type")),
        "expected_primary_source": _clean_text(query_row.get("expected_primary_source")),
        "expected_answer_style": _clean_text(query_row.get("expected_answer_style")),
        "difficulty": _clean_text(query_row.get("difficulty")),
        "regression_bucket": _clean_text(query_row.get("regression_bucket")),
        "answer_status": _clean_text(payload.get("status") or "unknown"),
        "answer_preview": _preview_text(payload.get("answer")),
        "no_result": "1" if _clean_text(payload.get("status")).lower() == "no_result" else "0",
        "fallback_used": "1" if fallback_used else "0",
        "weak_evidence": "1" if weak_evidence else "0",
        "needs_caution": "1" if bool(answer_verification.get("needsCaution")) else "0",
        "selected_source_kind": _clean_text(routing.get("sourceKind")),
        "selected_card_ids": " | ".join(selected_card_ids),
        "matched_entities": " | ".join(matched_entities),
        "routing_mode": _clean_text(routing.get("mode")),
        "intent": _clean_text(routing.get("intent")),
        "anchor_count": str(len(list(evidence_verification.get("anchorIdsUsed") or []))),
        "unsupported_fields": " | ".join(_clean_text(item) for item in list(evidence_verification.get("unsupportedFields") or []) if _clean_text(item)),
        "claim_verification_summary": _clean_text(claim_consensus.get("claimVerificationSummary")),
        "claim_conflict_count": str(int(claim_consensus.get("conflictCount") or 0)),
        "claim_weak_count": str(int(claim_consensus.get("weakClaimCount") or 0)),
        "claim_unsupported_count": str(int(claim_consensus.get("unsupportedClaimCount") or 0)),
        "claim_card_count": str(len(claim_cards)),
        "claim_alignment_group_count": str(len(list(claim_alignment.get("groups") or []))),
        "answer_provenance_mode": _clean_text(answer_provenance.get("mode")),
        "latency_ms": str(round(float(latency_ms or 0.0), 3)),
        "top_k": str(max(1, int(top_k))),
        "retrieval_mode": _clean_text(retrieval_mode),
        "label": "",
        "wrong_source": "",
        "wrong_era": "",
        "should_abstain": "",
        "notes": "",
        "final_label": "",
        "final_wrong_source": "",
        "final_wrong_era": "",
        "final_should_abstain": "",
        "final_notes": "",
    }


def build_machine_review_template_rows() -> list[dict[str, str]]:
    return [{field: "" for field in ASK_V2_MACHINE_REVIEW_FIELDNAMES}]


def build_human_review_template_rows() -> list[dict[str, str]]:
    return [{field: "" for field in ASK_V2_EVAL_FIELDNAMES}]


def _scorecard(
    rows: list[dict[str, str]],
    *,
    label_col: str,
    wrong_source_col: str,
    wrong_era_col: str,
    should_abstain_col: str,
    no_result_col: str,
) -> dict[str, Any]:
    query_count = len(rows)
    label_values = [_normalize_label(_review_value(row, label_col)) for row in rows]
    labeled_rows = [row for row, label in zip(rows, label_values) if label]
    label_counter = Counter(label for label in label_values if label)

    wrong_source_rows = [row for row in rows if _clean_text(_review_value(row, wrong_source_col))]
    wrong_era_rows = [row for row in rows if _clean_text(_review_value(row, wrong_era_col))]
    should_abstain_rows = [row for row in rows if _clean_text(_review_value(row, should_abstain_col))]

    abstention_rows = [
        row
        for row in rows
        if _clean_text(row.get("query_type")).lower() == "abstention"
        or _clean_text(row.get("expected_answer_style")).lower() == "abstain"
    ]
    temporal_rows = [row for row in rows if _bucket_name(row) == "temporal"]

    correct_abstention = 0
    abstention_reviewed = 0
    for row in abstention_rows:
        review = _review_value(row, should_abstain_col)
        if not review:
            continue
        abstention_reviewed += 1
        if not _normalize_boolish(review):
            correct_abstention += 1

    metrics = {
        "queryCount": query_count,
        "labeledQueryCount": len(labeled_rows),
        "passRate": _rate(label_counter.get("positive", 0) + label_counter.get("partial", 0), len(labeled_rows)),
        "wrongSourceRate": _rate(sum(1 for row in wrong_source_rows if _normalize_boolish(_review_value(row, wrong_source_col))), len(wrong_source_rows)),
        "wrongEraRate": _rate(sum(1 for row in wrong_era_rows if _normalize_boolish(_review_value(row, wrong_era_col))), len(wrong_era_rows)),
        "noResultRate": _rate(sum(1 for row in rows if _normalize_boolish(row.get(no_result_col))), query_count),
        "fallbackRate": _rate(sum(1 for row in rows if _normalize_boolish(row.get("fallback_used"))), query_count),
        "weakEvidenceRate": _rate(sum(1 for row in rows if _normalize_boolish(row.get("weak_evidence"))), query_count),
        "needsCautionRate": _rate(sum(1 for row in rows if _normalize_boolish(row.get("needs_caution"))), query_count),
        "abstainCorrectRate": _rate(correct_abstention, abstention_reviewed),
        "weakEvidenceWithoutFallbackRate": _rate(
            sum(1 for row in rows if _normalize_boolish(row.get("weak_evidence")) and not _normalize_boolish(row.get("fallback_used"))),
            query_count,
        ),
        "temporalQueryCount": len(temporal_rows),
        "abstentionQueryCount": len(abstention_rows),
        "wrongSourceReviewedCount": len(wrong_source_rows),
        "wrongEraReviewedCount": len(wrong_era_rows),
        "abstentionReviewedCount": abstention_reviewed,
    }
    return {
        "metrics": metrics,
        "labelDistribution": dict(label_counter),
        "available": query_count > 0,
        "reviewCoverage": {
            "wrongSource": len(wrong_source_rows),
            "wrongEra": len(wrong_era_rows),
            "shouldAbstain": len(should_abstain_rows),
        },
    }


def build_ask_v2_eval_report(
    csv_path: str | Path,
    *,
    label_col: str = "label",
    wrong_source_col: str = "wrong_source",
    no_result_col: str = "no_result",
    should_abstain_col: str = "should_abstain",
    wrong_era_col: str = "wrong_era",
) -> dict[str, Any]:
    path = Path(csv_path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))

    warnings: list[str] = []
    query_count = len(rows)
    labeled_count = sum(1 for row in rows if _normalize_label(_review_value(row, label_col)))
    if labeled_count < query_count:
        warnings.append(f"only {labeled_count}/{query_count} ask-v2 rows have labels")

    by_source_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_bucket_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source_grouped[_clean_text(row.get("source")).lower() or "unknown"].append(row)
        by_bucket_grouped[_bucket_name(row)].append(row)

    by_source = {
        key: _scorecard(
            value,
            label_col=label_col,
            wrong_source_col=wrong_source_col,
            wrong_era_col=wrong_era_col,
            should_abstain_col=should_abstain_col,
            no_result_col=no_result_col,
        )
        for key, value in sorted(by_source_grouped.items())
    }
    by_bucket = {
        key: _scorecard(
            value,
            label_col=label_col,
            wrong_source_col=wrong_source_col,
            wrong_era_col=wrong_era_col,
            should_abstain_col=should_abstain_col,
            no_result_col=no_result_col,
        )
        for key, value in sorted(by_bucket_grouped.items())
    }
    overall = _scorecard(
        rows,
        label_col=label_col,
        wrong_source_col=wrong_source_col,
        wrong_era_col=wrong_era_col,
        should_abstain_col=should_abstain_col,
        no_result_col=no_result_col,
    )

    return {
        "schema": "knowledge-hub.ask-v2.eval.report.v1",
        "status": "ok" if labeled_count > 0 else "warning",
        "dataset": {
            "csvPath": str(path),
            "labelCol": label_col,
            "wrongSourceCol": wrong_source_col,
            "noResultCol": no_result_col,
            "shouldAbstainCol": should_abstain_col,
            "wrongEraCol": wrong_era_col,
        },
        "metrics": overall["metrics"],
        "labelDistribution": overall["labelDistribution"],
        "bySource": by_source,
        "byBucket": by_bucket,
        "warnings": warnings,
    }


__all__ = [
    "ASK_V2_EVAL_FIELDNAMES",
    "ASK_V2_MACHINE_REVIEW_FIELDNAMES",
    "build_ask_v2_eval_report",
    "build_human_review_template_rows",
    "build_machine_review_template_rows",
    "serialize_ask_v2_eval_row",
]
