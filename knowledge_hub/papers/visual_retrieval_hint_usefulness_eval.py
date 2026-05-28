"""Report-only usefulness proxy for visual retrieval hints.

This helper compares text-only candidate context with an in-memory augmented
corpus that appends validated visual retrieval hints. It does not write a
candidate store, build vectors, mutate indexes, promote evidence, or expose
visual hints at answer runtime.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


VISUAL_RETRIEVAL_HINT_USEFULNESS_EVAL_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-usefulness-eval.v1"
)
VISUAL_RETRIEVAL_HINT_USEFULNESS_EVAL_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-usefulness-eval-row.v1"
)

VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID = (
    "knowledge-hub.paper.visual-layout-candidate-list-report.v1"
)
VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-dry-run.v1"
)
VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-dry-run.v1"
)

READY_DECISION = "ready_for_targeted_visual_retrieval_hint_search_eval"
BLOCKED_DECISION = "blocked"
NEXT_RECOMMENDED_TRANCHE = "targeted_visual_retrieval_hint_search_eval"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.-]*")
STOPWORDS = {
    "a",
    "about",
    "across",
    "after",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "caption",
    "category",
    "columns",
    "compared",
    "comparison",
    "contains",
    "crop",
    "cropped",
    "figure",
    "for",
    "from",
    "hint",
    "image",
    "include",
    "includes",
    "including",
    "is",
    "it",
    "labels",
    "left",
    "line",
    "model",
    "not",
    "of",
    "on",
    "only",
    "page",
    "paper",
    "partial",
    "partially",
    "plot",
    "region",
    "retrieval",
    "right",
    "row",
    "rows",
    "score",
    "section",
    "show",
    "showing",
    "table",
    "text",
    "the",
    "this",
    "to",
    "visible",
    "with",
}

_RANK_INDEX_CACHE: dict[
    int,
    tuple[tuple[str, ...], dict[str, Counter[str]], dict[str, float]],
] = {}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitized_report_ref(path: Path, *, project_root: Path | None = None) -> str:
    resolved = path.expanduser()
    if project_root is not None:
        try:
            rel = resolved.resolve().relative_to(project_root.resolve())
            return rel.as_posix()
        except Exception:
            pass
    return f"input_reports/{resolved.name}"


def _short_hash(value: str, *, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _tokens(value: str) -> list[str]:
    raw = TOKEN_RE.findall(normalize_text(value).lower())
    return [token for token in raw if len(token) >= 2 and token not in STOPWORDS]


def _text_context(row: dict[str, Any]) -> str:
    context = dict(row.get("textContext") or {})
    heading = " ".join(normalize_text(item) for item in list(context.get("headingPath") or []))
    return normalize_text(
        " ".join(
            [
                normalize_text(context.get("nearbyText")),
                normalize_text(context.get("captionText")),
                heading,
            ]
        )
    )


def _preview_text(record: dict[str, Any]) -> str:
    return normalize_text(
        " ".join(
            [
                normalize_text(record.get("derivedTextForRetrieval")),
                normalize_text(record.get("visibleText")),
                " ".join(normalize_text(item) for item in list(record.get("retrievalKeywords") or [])),
            ]
        )
    )


def _hint_record(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row.get("plannedJsonlRecordPreview") or {})


def _source_candidate_id(row: dict[str, Any]) -> str:
    return normalize_text(row.get("sourceCandidateId") or _hint_record(row).get("sourceCandidateId"))


def _dry_run_rows(dry_run_report: dict[str, Any], *, source_ref: str) -> list[dict[str, Any]]:
    rows = []
    for row in list(dry_run_report.get("dryRunRowsDetail") or []):
        if not isinstance(row, dict):
            continue
        copied = dict(row)
        copied["_sourceDryRunReportRef"] = source_ref
        rows.append(copied)
    return rows


def _valid_dry_run_schema(schema: str) -> bool:
    return schema in {
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_DRY_RUN_SCHEMA_ID,
        VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
    }


def _idf(documents: dict[str, str]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for text in documents.values():
        df.update(set(_tokens(text)))
    total = max(len(documents), 1)
    return {token: math.log((1 + total) / (1 + count)) + 1.0 for token, count in df.items()}


def _rank_index(documents: dict[str, str]) -> tuple[dict[str, Counter[str]], dict[str, float]]:
    keys = tuple(documents.keys())
    cache_key = id(documents)
    cached = _RANK_INDEX_CACHE.get(cache_key)
    if cached and cached[0] == keys:
        return cached[1], cached[2]
    token_counts: dict[str, Counter[str]] = {}
    df: Counter[str] = Counter()
    for doc_id, text in documents.items():
        counts = Counter(_tokens(text))
        token_counts[doc_id] = counts
        df.update(counts.keys())
    total = max(len(documents), 1)
    idf = {token: math.log((1 + total) / (1 + count)) + 1.0 for token, count in df.items()}
    _RANK_INDEX_CACHE[cache_key] = (keys, token_counts, idf)
    return token_counts, idf


def clear_rank_index_cache() -> None:
    _RANK_INDEX_CACHE.clear()


def _rank_documents(
    *,
    documents: dict[str, str],
    query: str,
    target_id: str,
) -> tuple[int | None, float]:
    query_terms = _tokens(query)
    if not query_terms:
        return None, 0.0
    token_counts, idf = _rank_index(documents)
    scores: list[tuple[float, str]] = []
    for doc_id, counts in token_counts.items():
        score = 0.0
        for term in query_terms:
            if counts.get(term):
                score += (1.0 + math.log(counts[term])) * idf.get(term, 1.0)
        scores.append((score, doc_id))
    scores.sort(key=lambda item: (-item[0], item[1]))
    target_score = 0.0
    for index, (score, doc_id) in enumerate(scores, start=1):
        if doc_id == target_id:
            target_score = score
            if score <= 0:
                return None, 0.0
            return index, target_score
    return None, target_score


def _novel_keyword_count(keywords: list[str], text_tokens: set[str]) -> int:
    count = 0
    for keyword in keywords:
        keyword_tokens = set(_tokens(keyword))
        if keyword_tokens and not keyword_tokens <= text_tokens:
            count += 1
    return count


def _probe_query(
    *,
    record: dict[str, Any],
    text_tokens: set[str],
    max_terms: int = 10,
) -> str:
    terms: list[str] = []
    for keyword in list(record.get("retrievalKeywords") or []):
        for token in _tokens(keyword):
            if token not in text_tokens and token not in terms:
                terms.append(token)
    for token in _tokens(normalize_text(record.get("derivedTextForRetrieval"))):
        if token not in text_tokens and token not in terms:
            terms.append(token)
    if len(terms) < 3:
        for keyword in list(record.get("retrievalKeywords") or []):
            for token in _tokens(keyword):
                if token not in terms:
                    terms.append(token)
    return " ".join(terms[:max_terms])


def _rank_value(rank: int | None, *, missing_rank: int) -> int:
    return rank if rank is not None else missing_rank


def _tier(
    *,
    text_rank: int | None,
    augmented_rank: int | None,
    rank_delta: int,
    novel_keyword_count: int,
    novel_visual_token_count: int,
) -> str:
    if augmented_rank is None:
        return "blocked"
    if augmented_rank <= 5 and (text_rank is None or text_rank > 10) and novel_keyword_count >= 2:
        return "high"
    if augmented_rank <= 10 and (rank_delta > 0 or novel_keyword_count >= 1 or novel_visual_token_count >= 8):
        return "medium"
    return "low"


def _recommendation(tier: str) -> str:
    if tier == "high":
        return "prioritize_for_targeted_search_eval"
    if tier == "medium":
        return "include_in_targeted_search_eval"
    if tier == "low":
        return "hold_until_more_search_signal"
    return "blocked"


def _scope(row_count: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "inputHintRows": int(row_count),
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
        "searchIndexQueryRows": 0,
        "answerGenerationRows": 0,
    }


def _policy() -> dict[str, Any]:
    return {
        "evalKind": "in_memory_proxy_only",
        "allowedUse": "retrieval_hint_utility_estimation_only",
        "doesNotUseOperationalIndex": True,
        "doesNotWriteCandidateStore": True,
        "doesNotPromoteEvidence": True,
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
        "runtimeVisible": False,
        "indexEligible": False,
    }


def _usefulness_row(
    *,
    dry_row: dict[str, Any],
    layout_row: dict[str, Any] | None,
    text_documents: dict[str, str],
    augmented_documents: dict[str, str],
    missing_rank: int,
) -> dict[str, Any]:
    record = _hint_record(dry_row)
    source_candidate_id = _source_candidate_id(dry_row)
    row_id_basis = "|".join(
        [
            source_candidate_id,
            normalize_text(dry_row.get("hintCandidateId")),
            normalize_text(dry_row.get("_sourceDryRunReportRef")),
        ]
    )
    blocker_reasons: list[str] = []
    if layout_row is None:
        blocker_reasons.append("missing_layout_candidate_context")
    if not record:
        blocker_reasons.append("missing_planned_record_preview")
    text_context = _text_context(layout_row or {})
    hint_text = _preview_text(record)
    text_token_set = set(_tokens(text_context))
    visual_token_set = set(_tokens(hint_text))
    novel_tokens = sorted(visual_token_set - text_token_set)
    keywords = [normalize_text(item) for item in list(record.get("retrievalKeywords") or []) if normalize_text(item)]
    probe_query = _probe_query(record=record, text_tokens=text_token_set)
    if not probe_query:
        blocker_reasons.append("empty_probe_query")
    text_rank, text_score = _rank_documents(
        documents=text_documents,
        query=probe_query,
        target_id=source_candidate_id,
    )
    augmented_rank, augmented_score = _rank_documents(
        documents=augmented_documents,
        query=probe_query,
        target_id=source_candidate_id,
    )
    rank_delta = _rank_value(text_rank, missing_rank=missing_rank) - _rank_value(
        augmented_rank,
        missing_rank=missing_rank,
    )
    novel_keyword_rows = _novel_keyword_count(keywords, text_token_set)
    tier = _tier(
        text_rank=text_rank,
        augmented_rank=augmented_rank,
        rank_delta=rank_delta,
        novel_keyword_count=novel_keyword_rows,
        novel_visual_token_count=len(novel_tokens),
    )
    if blocker_reasons:
        tier = "blocked"
    return {
        "schema": VISUAL_RETRIEVAL_HINT_USEFULNESS_EVAL_ROW_SCHEMA_ID,
        "evalRowId": "visual-retrieval-hint-usefulness:" + _short_hash(row_id_basis),
        "hintCandidateId": normalize_text(dry_row.get("hintCandidateId")),
        "sourceCandidateId": source_candidate_id,
        "paperId": normalize_text(record.get("paperId") or (layout_row or {}).get("paperId")),
        "paperRef": normalize_text(record.get("paperRef") or (layout_row or {}).get("paperRef")),
        "sourceContentHash": normalize_text(record.get("sourceContentHash") or (layout_row or {}).get("sourceContentHash")),
        "page": int(record.get("page") or (layout_row or {}).get("page") or 0),
        "bbox": list(record.get("bbox") or (layout_row or {}).get("bbox") or []),
        "candidateType": normalize_text(record.get("candidateType") or (layout_row or {}).get("candidateType")),
        "sourceDryRunReportRef": normalize_text(dry_row.get("_sourceDryRunReportRef")),
        "textOnlyContext": {
            "charCount": len(text_context),
            "tokenCount": len(text_token_set),
            "hasCaptionText": bool(normalize_text(dict((layout_row or {}).get("textContext") or {}).get("captionText"))),
        },
        "visualHint": {
            "derivedTextCharCount": len(normalize_text(record.get("derivedTextForRetrieval"))),
            "visibleTextCharCount": len(normalize_text(record.get("visibleText"))),
            "retrievalKeywordCount": len(keywords),
            "novelRetrievalKeywordCount": novel_keyword_rows,
            "novelVisualTokenCount": len(novel_tokens),
            "novelVisualTokenSample": novel_tokens[:12],
        },
        "proxyProbe": {
            "query": probe_query,
            "textOnlyRank": text_rank,
            "augmentedRank": augmented_rank,
            "rankDelta": rank_delta,
            "textOnlyScore": round(float(text_score), 6),
            "augmentedScore": round(float(augmented_score), 6),
            "textOnlyTop5Hit": bool(text_rank is not None and text_rank <= 5),
            "augmentedTop5Hit": bool(augmented_rank is not None and augmented_rank <= 5),
        },
        "usefulness": {
            "tier": tier,
            "recommendation": _recommendation(tier),
            "reason": _reason(tier, text_rank=text_rank, augmented_rank=augmented_rank, rank_delta=rank_delta, novel_keywords=novel_keyword_rows),
        },
        "policy": {
            "allowedUse": "retrieval_hint_utility_estimation_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
        "blockerReason": ";".join(blocker_reasons),
    }


def _reason(
    tier: str,
    *,
    text_rank: int | None,
    augmented_rank: int | None,
    rank_delta: int,
    novel_keywords: int,
) -> str:
    if tier == "high":
        return "visual hint adds novel query terms and moves the target into the top retrieval band"
    if tier == "medium":
        return "visual hint adds measurable lexical signal but should be checked in targeted search eval"
    if tier == "low":
        return "visual hint has limited proxy lift over nearby text context"
    return (
        "proxy eval blocked"
        if augmented_rank is None
        else f"rank_delta={rank_delta}, novel_keywords={novel_keywords}, text_rank={text_rank}, augmented_rank={augmented_rank}"
    )


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


def _counts(rows: list[dict[str, Any]], *, source_reports: list[dict[str, Any]], layout_rows: int, private_path_leak_rows: int) -> dict[str, int]:
    tier_counts = Counter(dict(row.get("usefulness") or {}).get("tier") for row in rows)
    source_ids = [normalize_text(row.get("sourceCandidateId")) for row in rows]
    duplicate_source_ids = {item for item, count in Counter(source_ids).items() if item and count > 1}
    text_top5 = sum(1 for row in rows if dict(row.get("proxyProbe") or {}).get("textOnlyTop5Hit"))
    augmented_top5 = sum(1 for row in rows if dict(row.get("proxyProbe") or {}).get("augmentedTop5Hit"))
    improved = sum(1 for row in rows if int(dict(row.get("proxyProbe") or {}).get("rankDelta") or 0) > 0)
    blocked = sum(1 for row in rows if row.get("blockerReason") or dict(row.get("usefulness") or {}).get("tier") == "blocked")
    return {
        "sourceDryRunReportRows": len(source_reports),
        "layoutCandidateRows": int(layout_rows),
        "inputHintRows": len(rows),
        "evaluatedRows": len(rows) - blocked,
        "highUsefulnessRows": int(tier_counts.get("high") or 0),
        "mediumUsefulnessRows": int(tier_counts.get("medium") or 0),
        "lowUsefulnessRows": int(tier_counts.get("low") or 0),
        "textOnlyTop5Rows": int(text_top5),
        "augmentedTop5Rows": int(augmented_top5),
        "rankImprovedRows": int(improved),
        "duplicateSourceCandidateIdRows": len(duplicate_source_ids),
        "blockedRows": int(blocked + len(duplicate_source_ids)),
        "candidateStoreWriteRows": 0,
        "indexEligibleRows": 0,
        "runtimeVisibleRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }


def _type_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_text(row.get("candidateType"))].append(row)
    summary = []
    for candidate_type in sorted(grouped):
        items = grouped[candidate_type]
        summary.append(
            {
                "candidateType": candidate_type,
                "rows": len(items),
                "highUsefulnessRows": sum(1 for row in items if row.get("usefulness", {}).get("tier") == "high"),
                "mediumUsefulnessRows": sum(1 for row in items if row.get("usefulness", {}).get("tier") == "medium"),
                "lowUsefulnessRows": sum(1 for row in items if row.get("usefulness", {}).get("tier") == "low"),
                "augmentedTop5Rows": sum(1 for row in items if row.get("proxyProbe", {}).get("augmentedTop5Hit")),
                "textOnlyTop5Rows": sum(1 for row in items if row.get("proxyProbe", {}).get("textOnlyTop5Hit")),
            }
        )
    return summary


def build_visual_retrieval_hint_usefulness_eval(
    layout_candidate_report: dict[str, Any],
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

    source_report_rows = [
        _source_report_row(report, report_ref=report_ref) for report_ref, report in dry_run_reports
    ]
    dry_rows: list[dict[str, Any]] = []
    source_blockers: list[str] = []
    for report_ref, report in dry_run_reports:
        if not _valid_dry_run_schema(normalize_text(report.get("schema"))):
            source_blockers.append(f"invalid_dry_run_schema:{report_ref}")
        if report.get("status") != "ready":
            source_blockers.append(f"blocked_dry_run_report:{report_ref}")
        if int(dict(report.get("counts") or {}).get("candidateStoreWriteRows") or 0) != 0:
            source_blockers.append(f"dry_run_has_store_writes:{report_ref}")
        dry_rows.extend(_dry_run_rows(report, source_ref=report_ref))

    augmented_documents = dict(text_documents)
    for dry_row in dry_rows:
        source_id = _source_candidate_id(dry_row)
        augmented_documents[source_id] = normalize_text(
            " ".join([augmented_documents.get(source_id, ""), _preview_text(_hint_record(dry_row))])
        )

    missing_rank = len(text_documents) + 1
    rows = [
        _usefulness_row(
            dry_row=dry_row,
            layout_row=layout_by_id.get(_source_candidate_id(dry_row)),
            text_documents=text_documents,
            augmented_documents=augmented_documents,
            missing_rank=missing_rank,
        )
        for dry_row in dry_rows
    ]

    private_path_leak_rows = 1 if _contains_private_path(rows) or _contains_private_path(source_report_rows) else 0
    report: dict[str, Any] = {
        "schema": VISUAL_RETRIEVAL_HINT_USEFULNESS_EVAL_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceLayoutCandidateReport": {
            "schema": normalize_text(layout_candidate_report.get("schema")),
            "status": normalize_text(layout_candidate_report.get("status")),
            "candidateRows": len(layout_rows),
        },
        "sourceDryRunReports": source_report_rows,
        "scope": _scope(len(rows)),
        "policy": _policy(),
        "method": {
            "name": "in_memory_text_only_vs_visual_hint_lexical_proxy_v1",
            "description": (
                "Builds an in-memory text-only corpus from layout candidate textContext and an augmented "
                "corpus that appends visual retrieval hints for validated hint rows, then compares target "
                "rank for visual-hint-derived probe queries."
            ),
            "limitations": [
                "This is not an operational vector-index or answer-quality evaluation.",
                "Probe queries are generated from visual hints, so results estimate potential recall lift rather than real user success.",
                "A later targeted search eval is required before scaling or indexing.",
            ],
        },
        "counts": {},
        "typeSummary": _type_summary(rows),
        "evalRowsDetail": rows,
        "warnings": [
            "This report estimates utility only; it does not prove answerability.",
            "Visual derived text remains retrieval-hint-only and non-evidence.",
            "Do not scale to all PDFs until targeted search eval confirms useful signal.",
        ],
    }
    counts = _counts(
        rows,
        source_reports=source_report_rows,
        layout_rows=len(layout_rows),
        private_path_leak_rows=private_path_leak_rows,
    )
    report["counts"] = counts
    if (
        layout_candidate_report.get("schema") != VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID
        or layout_candidate_report.get("status") != "ready"
        or source_blockers
        or private_path_leak_rows
        or counts["blockedRows"]
        or not rows
    ):
        report["status"] = "blocked"
        report["decision"] = BLOCKED_DECISION
        report["sourceBlockers"] = source_blockers
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    lines = [
        "# Visual Retrieval Hint Usefulness Eval",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- inputHintRows: `{counts.get('inputHintRows')}`",
        f"- evaluatedRows: `{counts.get('evaluatedRows')}`",
        f"- highUsefulnessRows: `{counts.get('highUsefulnessRows')}`",
        f"- mediumUsefulnessRows: `{counts.get('mediumUsefulnessRows')}`",
        f"- lowUsefulnessRows: `{counts.get('lowUsefulnessRows')}`",
        f"- textOnlyTop5Rows: `{counts.get('textOnlyTop5Rows')}`",
        f"- augmentedTop5Rows: `{counts.get('augmentedTop5Rows')}`",
        f"- rankImprovedRows: `{counts.get('rankImprovedRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- candidateStoreWriteRows: `{scope.get('candidateStoreWriteRows')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- searchIndexQueryRows: `{scope.get('searchIndexQueryRows')}`",
        f"- answerGenerationRows: `{scope.get('answerGenerationRows')}`",
        f"- databaseMutationRows: `{scope.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{scope.get('indexMutationRows')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        "",
        "## Type Summary",
        "",
        "| type | rows | high | medium | low | textTop5 | augmentedTop5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("typeSummary", []):
        lines.append(
            "| {candidateType} | {rows} | {high} | {medium} | {low} | {textTop5} | {augTop5} |".format(
                candidateType=row.get("candidateType"),
                rows=row.get("rows"),
                high=row.get("highUsefulnessRows"),
                medium=row.get("mediumUsefulnessRows"),
                low=row.get("lowUsefulnessRows"),
                textTop5=row.get("textOnlyTop5Rows"),
                augTop5=row.get("augmentedTop5Rows"),
            )
        )
    lines.extend(
        [
            "",
            "## Top Rows",
            "",
            "| # | tier | paperId | type | page | textRank | augmentedRank | delta | sourceCandidateId |",
            "|---:|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    tier_order = {"high": 0, "medium": 1, "low": 2, "blocked": 3}
    sorted_rows = sorted(
        list(report.get("evalRowsDetail") or []),
        key=lambda row: (
            tier_order.get(row.get("usefulness", {}).get("tier"), 9),
            -(int(row.get("proxyProbe", {}).get("rankDelta") or 0)),
            row.get("sourceCandidateId"),
        ),
    )
    for index, row in enumerate(sorted_rows[:24], start=1):
        probe = dict(row.get("proxyProbe") or {})
        lines.append(
            "| {index} | {tier} | {paperId} | {candidateType} | {page} | {textRank} | {augRank} | {delta} | `{sourceId}` |".format(
                index=index,
                tier=row.get("usefulness", {}).get("tier"),
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                textRank=probe.get("textOnlyRank"),
                augRank=probe.get("augmentedRank"),
                delta=probe.get("rankDelta"),
                sourceId=row.get("sourceCandidateId"),
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_visual_retrieval_hint_usefulness_eval(
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
    "VISUAL_RETRIEVAL_HINT_USEFULNESS_EVAL_SCHEMA_ID",
    "build_visual_retrieval_hint_usefulness_eval",
    "clear_rank_index_cache",
    "load_json",
    "sanitized_report_ref",
    "write_visual_retrieval_hint_usefulness_eval",
]
