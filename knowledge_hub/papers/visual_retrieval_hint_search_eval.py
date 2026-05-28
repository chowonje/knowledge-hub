"""Targeted report-only search eval for visual retrieval hints.

This helper turns validated visual retrieval hints into deterministic,
user-like search probes and compares in-memory text-only candidate retrieval
against an in-memory corpus augmented with visual hint text. It never queries
the operational search index, writes a candidate store, indexes vectors,
promotes evidence, or exposes hints at answer runtime.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.visual_retrieval_hint_usefulness_eval import (
    VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
    VISUAL_RETRIEVAL_HINT_USEFULNESS_EVAL_SCHEMA_ID,
    _hint_record,
    _preview_text,
    _rank_documents,
    _source_candidate_id,
    _text_context,
    _tokens,
    clear_rank_index_cache,
    load_json,
    normalize_text,
    sanitized_report_ref,
)


VISUAL_RETRIEVAL_HINT_SEARCH_EVAL_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-search-eval.v1"
)
VISUAL_RETRIEVAL_HINT_SEARCH_EVAL_QUERY_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-search-eval-query-row.v1"
)

READY_DECISION = "ready_for_limited_visual_retrieval_hint_candidate_store_apply_design"
BLOCKED_DECISION = "blocked"
NEXT_RECOMMENDED_TRANCHE = "limited_visual_retrieval_hint_candidate_store_apply_design"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)

TYPE_LABELS = {
    "figure_caption_region": "figure or caption",
    "table_region": "table",
    "equation_region": "equation or formula",
    "layout_region": "page layout or abstract area",
    "image_region": "image region",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _short_hash(value: str, *, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _dry_run_rows(dry_run_report: dict[str, Any], *, source_ref: str) -> list[dict[str, Any]]:
    rows = []
    for row in list(dry_run_report.get("dryRunRowsDetail") or []):
        if isinstance(row, dict):
            copied = dict(row)
            copied["_sourceDryRunReportRef"] = source_ref
            rows.append(copied)
    return rows


def _valid_dry_run_schema(schema: str) -> bool:
    return schema in {
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
    }


def _rank_value(rank: int | None, *, missing_rank: int) -> int:
    return rank if rank is not None else missing_rank


def _record_keywords(record: dict[str, Any], *, max_keywords: int = 5) -> list[str]:
    keywords = [normalize_text(item) for item in list(record.get("retrievalKeywords") or [])]
    return [item for item in keywords if item][:max_keywords]


def _keyword_query(record: dict[str, Any]) -> str:
    tokens: list[str] = []
    for keyword in _record_keywords(record, max_keywords=8):
        for token in _tokens(keyword):
            if token not in tokens:
                tokens.append(token)
    if len(tokens) < 4:
        for token in _tokens(normalize_text(record.get("derivedTextForRetrieval"))):
            if token not in tokens:
                tokens.append(token)
    return " ".join(tokens[:12])


def _natural_query(record: dict[str, Any]) -> str:
    paper_id = normalize_text(record.get("paperId")).replace("-", " ")
    candidate_type = TYPE_LABELS.get(normalize_text(record.get("candidateType")), "visual region")
    keywords = ", ".join(_record_keywords(record, max_keywords=4))
    if keywords:
        return f"find {paper_id} {candidate_type} about {keywords}"
    return f"find {paper_id} {candidate_type} described by visual annotation"


def _query_specs(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "queryKind": "keyword_lookup",
            "query": _keyword_query(record),
        },
        {
            "queryKind": "natural_lookup",
            "query": _natural_query(record),
        },
    ]


def _source_report_row(dry_run_report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(dry_run_report.get("counts") or {})
    return {
        "schema": normalize_text(dry_run_report.get("schema")),
        "status": normalize_text(dry_run_report.get("status")),
        "reportRef": normalize_text(report_ref),
        "dryRunRows": int(counts.get("dryRunRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
        "candidateStoreWriteRows": int(counts.get("candidateStoreWriteRows") or 0),
    }


def _source_usefulness_report_row(usefulness_report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = dict(usefulness_report.get("counts") or {})
    return {
        "schema": normalize_text(usefulness_report.get("schema")),
        "status": normalize_text(usefulness_report.get("status")),
        "reportRef": normalize_text(report_ref),
        "inputHintRows": int(counts.get("inputHintRows") or 0),
        "highUsefulnessRows": int(counts.get("highUsefulnessRows") or 0),
        "mediumUsefulnessRows": int(counts.get("mediumUsefulnessRows") or 0),
        "blockedRows": int(counts.get("blockedRows") or 0),
    }


def _scope(query_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "queryRows": int(query_rows),
        "candidateStoreWriteRows": 0,
        "vectorIndexing": False,
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidencePromotionRows": 0,
        "runtimeAnswerVisibleExposureRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "answerabilityGateBypassRows": 0,
        "operationalSearchIndexQueryRows": 0,
        "answerGenerationRows": 0,
    }


def _policy() -> dict[str, Any]:
    return {
        "evalKind": "targeted_in_memory_search_eval_only",
        "allowedUse": "retrieval_hint_search_utility_estimation_only",
        "doesNotUseOperationalIndex": True,
        "doesNotWriteCandidateStore": True,
        "doesNotPromoteEvidence": True,
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "runtimeVisible": False,
        "indexEligible": False,
    }


def _query_row(
    *,
    dry_row: dict[str, Any],
    query_kind: str,
    query: str,
    layout_row: dict[str, Any] | None,
    text_documents: dict[str, str],
    augmented_documents: dict[str, str],
    missing_rank: int,
) -> dict[str, Any]:
    record = _hint_record(dry_row)
    source_candidate_id = _source_candidate_id(dry_row)
    text_rank, text_score = _rank_documents(
        documents=text_documents,
        query=query,
        target_id=source_candidate_id,
    )
    augmented_rank, augmented_score = _rank_documents(
        documents=augmented_documents,
        query=query,
        target_id=source_candidate_id,
    )
    rank_delta = _rank_value(text_rank, missing_rank=missing_rank) - _rank_value(
        augmented_rank,
        missing_rank=missing_rank,
    )
    blocker_reasons: list[str] = []
    if layout_row is None:
        blocker_reasons.append("missing_layout_candidate_context")
    if not query:
        blocker_reasons.append("empty_query")
    if not record:
        blocker_reasons.append("missing_planned_record_preview")
    row_id_basis = "|".join(
        [
            source_candidate_id,
            normalize_text(dry_row.get("hintCandidateId")),
            query_kind,
            query,
        ]
    )
    return {
        "schema": VISUAL_RETRIEVAL_HINT_SEARCH_EVAL_QUERY_ROW_SCHEMA_ID,
        "queryRowId": "visual-retrieval-hint-search-query:" + _short_hash(row_id_basis),
        "queryKind": query_kind,
        "query": query,
        "hintCandidateId": normalize_text(dry_row.get("hintCandidateId")),
        "sourceCandidateId": source_candidate_id,
        "paperId": normalize_text(record.get("paperId") or (layout_row or {}).get("paperId")),
        "paperRef": normalize_text(record.get("paperRef") or (layout_row or {}).get("paperRef")),
        "sourceContentHash": normalize_text(record.get("sourceContentHash") or (layout_row or {}).get("sourceContentHash")),
        "page": int(record.get("page") or (layout_row or {}).get("page") or 0),
        "bbox": list(record.get("bbox") or (layout_row or {}).get("bbox") or []),
        "candidateType": normalize_text(record.get("candidateType") or (layout_row or {}).get("candidateType")),
        "sourceDryRunReportRef": normalize_text(dry_row.get("_sourceDryRunReportRef")),
        "expectedTarget": {
            "sourceCandidateId": source_candidate_id,
            "hintCandidateId": normalize_text(dry_row.get("hintCandidateId")),
        },
        "textOnlyResult": {
            "rank": text_rank,
            "score": round(float(text_score), 6),
            "hitAt1": bool(text_rank is not None and text_rank <= 1),
            "hitAt5": bool(text_rank is not None and text_rank <= 5),
            "hitAt10": bool(text_rank is not None and text_rank <= 10),
        },
        "visualHintAugmentedResult": {
            "rank": augmented_rank,
            "score": round(float(augmented_score), 6),
            "hitAt1": bool(augmented_rank is not None and augmented_rank <= 1),
            "hitAt5": bool(augmented_rank is not None and augmented_rank <= 5),
            "hitAt10": bool(augmented_rank is not None and augmented_rank <= 10),
        },
        "comparison": {
            "rankDelta": rank_delta,
            "improved": rank_delta > 0,
            "regressed": rank_delta < 0,
            "unchanged": rank_delta == 0,
            "liftBucket": _lift_bucket(text_rank=text_rank, augmented_rank=augmented_rank, rank_delta=rank_delta),
        },
        "policy": {
            "allowedUse": "retrieval_hint_search_utility_estimation_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
        "blockerReason": ";".join(blocker_reasons),
    }


def _lift_bucket(*, text_rank: int | None, augmented_rank: int | None, rank_delta: int) -> str:
    if augmented_rank is None:
        return "blocked"
    if augmented_rank <= 5 and (text_rank is None or text_rank > 10):
        return "strong_lift"
    if rank_delta > 0:
        return "moderate_lift"
    if rank_delta == 0:
        return "no_lift"
    return "regression"


def _mrr(rows: list[dict[str, Any]], *, result_key: str) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        rank = dict(row.get(result_key) or {}).get("rank")
        if isinstance(rank, int) and rank > 0:
            total += 1.0 / rank
    return round(total / len(rows), 6)


def _type_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_text(row.get("candidateType"))].append(row)
    output = []
    for candidate_type in sorted(grouped):
        items = grouped[candidate_type]
        output.append(
            {
                "candidateType": candidate_type,
                "queryRows": len(items),
                "textOnlyHitAt5Rows": sum(1 for row in items if row.get("textOnlyResult", {}).get("hitAt5")),
                "augmentedHitAt5Rows": sum(1 for row in items if row.get("visualHintAugmentedResult", {}).get("hitAt5")),
                "improvedRows": sum(1 for row in items if row.get("comparison", {}).get("improved")),
                "regressedRows": sum(1 for row in items if row.get("comparison", {}).get("regressed")),
            }
        )
    return output


def _counts(
    rows: list[dict[str, Any]],
    *,
    candidate_rows: int,
    layout_rows: int,
    source_dry_run_reports: int,
    private_path_leak_rows: int,
) -> dict[str, Any]:
    blocked_rows = sum(1 for row in rows if row.get("blockerReason"))
    return {
        "sourceDryRunReportRows": int(source_dry_run_reports),
        "layoutCandidateRows": int(layout_rows),
        "inputHintRows": int(candidate_rows),
        "queryRows": len(rows),
        "evaluatedQueryRows": len(rows) - blocked_rows,
        "textOnlyHitAt1Rows": sum(1 for row in rows if row.get("textOnlyResult", {}).get("hitAt1")),
        "textOnlyHitAt5Rows": sum(1 for row in rows if row.get("textOnlyResult", {}).get("hitAt5")),
        "textOnlyHitAt10Rows": sum(1 for row in rows if row.get("textOnlyResult", {}).get("hitAt10")),
        "augmentedHitAt1Rows": sum(1 for row in rows if row.get("visualHintAugmentedResult", {}).get("hitAt1")),
        "augmentedHitAt5Rows": sum(1 for row in rows if row.get("visualHintAugmentedResult", {}).get("hitAt5")),
        "augmentedHitAt10Rows": sum(1 for row in rows if row.get("visualHintAugmentedResult", {}).get("hitAt10")),
        "rankImprovedRows": sum(1 for row in rows if row.get("comparison", {}).get("improved")),
        "rankRegressedRows": sum(1 for row in rows if row.get("comparison", {}).get("regressed")),
        "rankUnchangedRows": sum(1 for row in rows if row.get("comparison", {}).get("unchanged")),
        "strongLiftRows": sum(1 for row in rows if row.get("comparison", {}).get("liftBucket") == "strong_lift"),
        "textOnlyMrr": _mrr(rows, result_key="textOnlyResult"),
        "augmentedMrr": _mrr(rows, result_key="visualHintAugmentedResult"),
        "blockedRows": int(blocked_rows),
        "candidateStoreWriteRows": 0,
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }


def build_visual_retrieval_hint_search_eval(
    layout_candidate_report: dict[str, Any],
    usefulness_report: dict[str, Any],
    dry_run_reports: list[tuple[str, dict[str, Any]]],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    clear_rank_index_cache()
    layout_rows = [
        row for row in list(layout_candidate_report.get("candidateRowsDetail") or []) if isinstance(row, dict)
    ]
    layout_by_id = {normalize_text(row.get("candidateId")): row for row in layout_rows}
    text_documents = {normalize_text(row.get("candidateId")): _text_context(row) for row in layout_rows}

    source_dry_run_reports = [
        _source_report_row(report, report_ref=report_ref) for report_ref, report in dry_run_reports
    ]
    source_blockers: list[str] = []
    dry_rows: list[dict[str, Any]] = []
    for report_ref, report in dry_run_reports:
        if not _valid_dry_run_schema(normalize_text(report.get("schema"))):
            source_blockers.append(f"invalid_dry_run_schema:{report_ref}")
        if report.get("status") != "ready":
            source_blockers.append(f"blocked_dry_run_report:{report_ref}")
        if int(dict(report.get("counts") or {}).get("candidateStoreWriteRows") or 0) != 0:
            source_blockers.append(f"dry_run_has_store_writes:{report_ref}")
        dry_rows.extend(_dry_run_rows(report, source_ref=report_ref))

    useful_ids = {
        normalize_text(row.get("sourceCandidateId"))
        for row in list(usefulness_report.get("evalRowsDetail") or [])
        if isinstance(row, dict)
        and dict(row.get("usefulness") or {}).get("tier") in {"high", "medium"}
        and not row.get("blockerReason")
    }
    selected_dry_rows = [row for row in dry_rows if _source_candidate_id(row) in useful_ids]
    augmented_documents = dict(text_documents)
    for dry_row in selected_dry_rows:
        source_id = _source_candidate_id(dry_row)
        augmented_documents[source_id] = normalize_text(
            " ".join([augmented_documents.get(source_id, ""), _preview_text(_hint_record(dry_row))])
        )

    missing_rank = len(text_documents) + 1
    query_rows: list[dict[str, Any]] = []
    for dry_row in selected_dry_rows:
        record = _hint_record(dry_row)
        for query_spec in _query_specs(record):
            query_rows.append(
                _query_row(
                    dry_row=dry_row,
                    query_kind=query_spec["queryKind"],
                    query=query_spec["query"],
                    layout_row=layout_by_id.get(_source_candidate_id(dry_row)),
                    text_documents=text_documents,
                    augmented_documents=augmented_documents,
                    missing_rank=missing_rank,
                )
            )

    private_path_leak_rows = 1 if _contains_private_path(query_rows) or _contains_private_path(source_dry_run_reports) else 0
    source_usefulness_ref = normalize_text(usefulness_report.get("_sourceReportRef"))
    report: dict[str, Any] = {
        "schema": VISUAL_RETRIEVAL_HINT_SEARCH_EVAL_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceLayoutCandidateReport": {
            "schema": normalize_text(layout_candidate_report.get("schema")),
            "status": normalize_text(layout_candidate_report.get("status")),
            "candidateRows": len(layout_rows),
        },
        "sourceUsefulnessEvalReport": _source_usefulness_report_row(
            usefulness_report,
            report_ref=source_usefulness_ref or "eval/knowledgeos/reports/visual_retrieval_hint_usefulness_eval.v1.json",
        ),
        "sourceDryRunReports": source_dry_run_reports,
        "scope": _scope(len(query_rows)),
        "policy": _policy(),
        "method": {
            "name": "targeted_in_memory_text_only_vs_visual_hint_search_eval_v1",
            "description": (
                "Generates deterministic keyword and natural-language lookup probes for high/medium "
                "visual hint candidates, then compares target candidate rank in text-only and visual-hint "
                "augmented in-memory corpora."
            ),
            "queryKinds": ["keyword_lookup", "natural_lookup"],
            "limitations": [
                "This is still an in-memory lexical eval, not an operational vector-index eval.",
                "Queries are deterministic probes derived from approved visual hint text, not user traffic.",
                "A later apply design may only write a limited candidate store; indexing remains a separate gate.",
            ],
        },
        "counts": {},
        "typeSummary": _type_summary(query_rows),
        "queryRowsDetail": query_rows,
        "warnings": [
            "This report measures retrieval-hint search utility only; it does not prove scientific answerability.",
            "Visual derived text remains non-evidence and must not become citation-grade evidence.",
            "The next step is limited apply design, not vector indexing or answer runtime exposure.",
        ],
    }
    counts = _counts(
        query_rows,
        candidate_rows=len(selected_dry_rows),
        layout_rows=len(layout_rows),
        source_dry_run_reports=len(source_dry_run_reports),
        private_path_leak_rows=private_path_leak_rows,
    )
    report["counts"] = counts
    if (
        layout_candidate_report.get("schema") != VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID
        or layout_candidate_report.get("status") != "ready"
        or usefulness_report.get("schema") != VISUAL_RETRIEVAL_HINT_USEFULNESS_EVAL_SCHEMA_ID
        or usefulness_report.get("status") != "ready"
        or int(dict(usefulness_report.get("counts") or {}).get("blockedRows") or 0)
        or source_blockers
        or private_path_leak_rows
        or counts["blockedRows"]
        or not query_rows
    ):
        report["status"] = "blocked"
        report["decision"] = BLOCKED_DECISION
        report["sourceBlockers"] = source_blockers
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    lines = [
        "# Visual Retrieval Hint Search Eval",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- inputHintRows: `{counts.get('inputHintRows')}`",
        f"- queryRows: `{counts.get('queryRows')}`",
        f"- textOnlyHitAt5Rows: `{counts.get('textOnlyHitAt5Rows')}`",
        f"- augmentedHitAt5Rows: `{counts.get('augmentedHitAt5Rows')}`",
        f"- rankImprovedRows: `{counts.get('rankImprovedRows')}`",
        f"- rankRegressedRows: `{counts.get('rankRegressedRows')}`",
        f"- textOnlyMrr: `{counts.get('textOnlyMrr')}`",
        f"- augmentedMrr: `{counts.get('augmentedMrr')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- candidateStoreWriteRows: `{scope.get('candidateStoreWriteRows')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- operationalSearchIndexQueryRows: `{scope.get('operationalSearchIndexQueryRows')}`",
        f"- answerGenerationRows: `{scope.get('answerGenerationRows')}`",
        f"- databaseMutationRows: `{scope.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{scope.get('indexMutationRows')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        "",
        "## Type Summary",
        "",
        "| type | queryRows | textHit@5 | augmentedHit@5 | improved | regressed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("typeSummary", []):
        lines.append(
            "| {candidateType} | {queryRows} | {textHit} | {augHit} | {improved} | {regressed} |".format(
                candidateType=row.get("candidateType"),
                queryRows=row.get("queryRows"),
                textHit=row.get("textOnlyHitAt5Rows"),
                augHit=row.get("augmentedHitAt5Rows"),
                improved=row.get("improvedRows"),
                regressed=row.get("regressedRows"),
            )
        )
    lines.extend(
        [
            "",
            "## Sample Query Rows",
            "",
            "| # | kind | paperId | type | textRank | augmentedRank | delta | sourceCandidateId |",
            "|---:|---|---|---|---:|---:|---:|---|",
        ]
    )
    sorted_rows = sorted(
        list(report.get("queryRowsDetail") or []),
        key=lambda row: (
            -(int(row.get("comparison", {}).get("rankDelta") or 0)),
            row.get("sourceCandidateId"),
            row.get("queryKind"),
        ),
    )
    for index, row in enumerate(sorted_rows[:24], start=1):
        lines.append(
            "| {index} | {kind} | {paperId} | {candidateType} | {textRank} | {augRank} | {delta} | `{sourceId}` |".format(
                index=index,
                kind=row.get("queryKind"),
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                textRank=row.get("textOnlyResult", {}).get("rank"),
                augRank=row.get("visualHintAugmentedResult", {}).get("rank"),
                delta=row.get("comparison", {}).get("rankDelta"),
                sourceId=row.get("sourceCandidateId"),
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_visual_retrieval_hint_search_eval(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "VISUAL_RETRIEVAL_HINT_SEARCH_EVAL_SCHEMA_ID",
    "build_visual_retrieval_hint_search_eval",
    "load_json",
    "sanitized_report_ref",
    "write_visual_retrieval_hint_search_eval",
]
