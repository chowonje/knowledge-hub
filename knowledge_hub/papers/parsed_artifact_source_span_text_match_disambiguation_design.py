"""Report-only text-match disambiguation design for ambiguous SourceSpan rows.

This helper consumes strict-evidence design-review rows blocked for non-unique
text matches and attempts to narrow candidates using page/locator/block/bbox
context against canonical PDF-extracted text. It does not mutate SourceSpan rows,
create StrictEvidence, or change runtime/parser/answer/DB state.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_original_source_offset_authority_design import (
    _PaperSourceContext,
    _resolve_paper_source_context,
    CHARS_BASIS,
    CHARS_NORMALIZATION,
    DEFAULT_PAPERS_DIR,
)
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_design_review import (
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID,
    REVIEW_STATUS_BLOCKED_NON_UNIQUE,
)
from knowledge_hub.papers.sectionspan_pdf_offset_recovery_dry_run import (
    _exact_matches,
    _extract_pdf_pages,
    _normalized_matches,
    _with_offsets,
)


PARSED_ARTIFACT_SOURCE_SPAN_TEXT_MATCH_DISAMBIGUATION_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-text-match-disambiguation-design.v1"
)

DISAMBIGUATION_STATUS_CANDIDATE = "disambiguation_design_candidate_only"
DISAMBIGUATION_STATUS_BLOCKED_STILL_NON_UNIQUE = "blocked_still_non_unique_after_locator_context"
DISAMBIGUATION_STATUS_BLOCKED_MISSING_LOCATOR = "blocked_missing_locator_context"
DISAMBIGUATION_STATUS_BLOCKED_MISSING_MATCH_OFFSETS = "blocked_missing_candidate_match_offsets"
DISAMBIGUATION_STATUS_BLOCKED_SOURCE_TEXT_UNAVAILABLE = "blocked_source_text_unavailable"
DISAMBIGUATION_STATUS_BLOCKED_MANUAL_OR_LATER = "blocked_requires_manual_or_later_extractor_review"
DISAMBIGUATION_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

METHOD_PAGE_CONTEXT = "page_context_unique_match"
METHOD_BLOCK_CONTEXT = "block_context_unique_match"
METHOD_BBOX_ADJACENT = "bbox_adjacent_text_unique_match"

DEFAULT_DESIGN_REVIEW_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-strict-evidence-design-review"
    / "01-parsed-artifact-source-span-strict-evidence-design-review"
    / "parsed-artifact-source-span-strict-evidence-design-review.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-text-match-disambiguation-design"
    / "01-parsed-artifact-source-span-text-match-disambiguation-design"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = _safe_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _locator_dict(row: dict[str, Any]) -> dict[str, Any]:
    locator = row.get("locator")
    return locator if isinstance(locator, dict) else {}


def _substring_sha256(canonical_text: str, start: int, end: int) -> str:
    return hashlib.sha256(canonical_text[start:end].encode("utf-8")).hexdigest()


def _canonical_text_from_pages(pages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for page in pages:
        text = str(page.get("text") or "")
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _page_bounds(pages: list[dict[str, Any]], page: int) -> tuple[int, int] | None:
    for item in pages:
        if _safe_int(item.get("page")) == page:
            start = _safe_int(item.get("chars_start"))
            end = _safe_int(item.get("chars_end"))
            if start is not None and end is not None:
                return start, end
    return None


def _filter_matches_to_page(
    matches: list[dict[str, Any]],
    *,
    pages: list[dict[str, Any]],
    page: int,
) -> list[dict[str, Any]]:
    bounds = _page_bounds(pages, page)
    if not bounds:
        return [match for match in matches if _safe_int(match.get("page")) == page]
    page_start, page_end = bounds
    filtered: list[dict[str, Any]] = []
    for match in matches:
        start = _safe_int(match.get("chars_start"))
        if start is None:
            continue
        if page_start <= start < page_end:
            filtered.append(match)
    return filtered


def _parsed_document_elements(parsed_root: Path, paper_id: str) -> list[dict[str, Any]]:
    document_path = parsed_root / paper_id / "document.json"
    payload = _read_json(document_path)
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return []
    return [item for item in elements if isinstance(item, dict)]


def _filter_matches_to_blocks(
    matches: list[dict[str, Any]],
    *,
    pages: list[dict[str, Any]],
    elements: list[dict[str, Any]],
    page: int,
    block_indexes: list[int],
) -> list[dict[str, Any]]:
    if not block_indexes:
        return matches
    page_bounds = _page_bounds(pages, page)
    if not page_bounds:
        return []
    page_start, _page_end = page_bounds
    block_set = {int(item) for item in block_indexes}
    windows: list[tuple[int, int]] = []
    for element in elements:
        if _safe_int(element.get("page")) != page:
            continue
        order = _safe_int(element.get("reading_order"))
        if order is None or order not in block_set:
            continue
        text = _safe_text(element.get("text"))
        if not text:
            continue
        page_text = ""
        for page_item in pages:
            if _safe_int(page_item.get("page")) == page:
                page_text = str(page_item.get("text") or "")
                break
        if not page_text:
            continue
        local_start = page_text.find(text)
        if local_start < 0:
            continue
        global_start = page_start + local_start
        windows.append((global_start, global_start + len(text)))
    if not windows:
        return []
    filtered: list[dict[str, Any]] = []
    for match in matches:
        start = _safe_int(match.get("chars_start"))
        end = _safe_int(match.get("chars_end"))
        if start is None or end is None:
            continue
        if any(start >= window_start and end <= window_end for window_start, window_end in windows):
            filtered.append(match)
    return filtered


def _filter_matches_to_bbox(
    matches: list[dict[str, Any]],
    *,
    pages: list[dict[str, Any]],
    page: int,
    bbox: list[Any],
) -> list[dict[str, Any]]:
    if len(bbox) < 4:
        return matches
    page_bounds = _page_bounds(pages, page)
    if not page_bounds:
        return []
    page_start, page_end = page_bounds
    try:
        y_center = (float(bbox[1]) + float(bbox[3])) / 2.0
    except Exception:
        return matches
    page_text = ""
    for page_item in pages:
        if _safe_int(page_item.get("page")) == page:
            page_text = str(page_item.get("text") or "")
            break
    if not page_text:
        return []
    line_starts = [0]
    for index, char in enumerate(page_text):
        if char == "\n" and index + 1 < len(page_text):
            line_starts.append(index + 1)
    approx_line = max(0, min(len(line_starts) - 1, int(y_center / 12.0)))
    line_start = line_starts[approx_line]
    line_end = line_starts[approx_line + 1] if approx_line + 1 < len(line_starts) else len(page_text)
    global_line_start = page_start + line_start
    global_line_end = page_start + line_end
    filtered: list[dict[str, Any]] = []
    for match in matches:
        start = _safe_int(match.get("chars_start"))
        end = _safe_int(match.get("chars_end"))
        if start is None or end is None:
            continue
        if global_line_start <= start < global_line_end:
            filtered.append(match)
    return filtered


def _match_records(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chars_start": _safe_int(match.get("chars_start")),
            "chars_end": _safe_int(match.get("chars_end")),
            "page": _safe_int(match.get("page")),
            "match_method": _safe_text(match.get("match_method")),
            "match_confidence": match.get("match_confidence"),
        }
        for match in matches
    ]


def _collect_text_matches(
    pages: list[dict[str, Any]],
    text_surface: str,
) -> tuple[list[dict[str, Any]], str]:
    exact = _exact_matches(pages, text_surface)
    if exact:
        return exact, "exact"
    normalized = _normalized_matches(pages, text_surface)
    return normalized, "normalized_whitespace_case"


def _proposed_chars(
    *,
    canonical_text: str,
    match: dict[str, Any],
    source_hash: str,
    disambiguation_method: str,
    match_confidence: float,
) -> dict[str, Any]:
    start = _safe_int(match.get("chars_start"))
    end = _safe_int(match.get("chars_end"))
    if start is None or end is None or end <= start:
        return {}
    return {
        "start": start,
        "end": end,
        "basis": CHARS_BASIS,
        "normalization": CHARS_NORMALIZATION,
        "expectedSubstringSha256": _substring_sha256(canonical_text, start, end),
        "sourceContentHash": source_hash,
        "disambiguationMethod": disambiguation_method,
        "matchMethod": _safe_text(match.get("match_method")),
        "matchConfidence": match_confidence,
        "page": _safe_int(match.get("page")),
    }


def _prior_match_count_from_blockers(review_row: dict[str, Any]) -> int | None:
    for blocker in review_row.get("review_blockers") or []:
        text = _safe_text(blocker)
        if text.startswith("exact_match_count="):
            return _safe_int(text.split("=", 1)[1])
        if text.startswith("normalized_match_count="):
            return _safe_int(text.split("=", 1)[1])
    design_blockers = review_row.get("design_blockers") or []
    for blocker in design_blockers:
        text = _safe_text(blocker)
        if "match_count=" in text:
            return _safe_int(text.split("=", 1)[-1])
    return None


def _classify_ambiguous_row(
    review_row: dict[str, Any],
    *,
    paper_contexts: dict[str, _PaperSourceContext],
    parsed_root: Path,
    papers_dir: Path,
    page_loader: Callable[[str | Path], list[dict[str, Any]]],
) -> dict[str, Any]:
    paper_id = _safe_text(review_row.get("paper_id"))
    text_surface = _safe_text(review_row.get("text_surface"))
    source_hash = _safe_text(review_row.get("sourceContentHash"))
    locator = _locator_dict(review_row)
    page = _safe_int(locator.get("page"))
    bbox = _safe_list(locator.get("bbox"))
    block_indexes = [
        int(item)
        for item in _safe_list(locator.get("blockIndexes"))
        if _safe_int(item) is not None
    ]

    base = {
        "disambiguation_row_id": "",
        "review_row_id": _safe_text(review_row.get("review_row_id")),
        "design_row_id": _safe_text(review_row.get("design_row_id")),
        "sourceSpanId": _safe_text(review_row.get("sourceSpanId")),
        "candidateRecordId": _safe_text(review_row.get("candidateRecordId")),
        "paper_id": paper_id,
        "artifact_type": _safe_text(review_row.get("artifact_type")),
        "source_candidate_id": _safe_text(review_row.get("source_candidate_id")),
        "sourceContentHash": source_hash,
        "source_file": _safe_text(review_row.get("source_file")),
        "text_surface": text_surface,
        "locator": locator,
        "disambiguation_status": "",
        "disambiguation_blockers": [],
        "disambiguation_method": "",
        "candidate_match_offsets": [],
        "filtered_match_offsets": [],
        "proposed_chars": {},
        "recommended_action": "",
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "sourceSpanUpdatedRows": 0,
    }

    blockers = ["text_match_disambiguation_design_only"]

    if not text_surface:
        base["disambiguation_status"] = DISAMBIGUATION_STATUS_BLOCKED_MANUAL_OR_LATER
        base["disambiguation_blockers"] = _dedupe([*blockers, "text_surface_missing"])
        base["recommended_action"] = "recover_text_surface_before_disambiguation_design"
        return base

    context = paper_contexts.get(paper_id)
    if context is None:
        context = _resolve_paper_source_context(
            paper_id=paper_id,
            parsed_root=parsed_root,
            papers_dir=papers_dir,
            page_loader=page_loader,
            expected_hash=source_hash,
        )
        paper_contexts[paper_id] = context

    if context.status != "ok":
        status = (
            DISAMBIGUATION_STATUS_BLOCKED_SOURCE_TEXT_UNAVAILABLE
            if context.status
            in {
                "blocked_source_text_unavailable",
                "blocked_missing_source_file",
                "blocked_hash_basis_unavailable",
            }
            else DISAMBIGUATION_STATUS_BLOCKED_MANUAL_OR_LATER
        )
        base["disambiguation_status"] = status
        base["disambiguation_blockers"] = _dedupe([*blockers, context.status])
        base["recommended_action"] = "repair_source_context_before_disambiguation_design"
        return base

    pages = list(context.pages)
    canonical_text = context.canonical_text
    all_matches, _match_tier = _collect_text_matches(pages, text_surface)
    base["candidate_match_offsets"] = _match_records(all_matches)

    if not all_matches:
        base["disambiguation_status"] = DISAMBIGUATION_STATUS_BLOCKED_MISSING_MATCH_OFFSETS
        base["disambiguation_blockers"] = _dedupe([*blockers, "candidate_match_offsets_empty"])
        base["recommended_action"] = "manual_or_later_extractor_review_required_for_text_match"
        return base

    prior_count = _prior_match_count_from_blockers(review_row)
    if prior_count is not None and prior_count <= 0:
        base["disambiguation_status"] = DISAMBIGUATION_STATUS_BLOCKED_MISSING_MATCH_OFFSETS
        base["disambiguation_blockers"] = _dedupe([*blockers, "prior_match_count_zero"])
        base["recommended_action"] = "repair_offset_authority_design_match_metadata"
        return base

    if len(all_matches) == 1:
        match = all_matches[0]
        base["disambiguation_status"] = DISAMBIGUATION_STATUS_CANDIDATE
        base["disambiguation_method"] = METHOD_PAGE_CONTEXT
        base["filtered_match_offsets"] = _match_records([match])
        base["proposed_chars"] = _proposed_chars(
            canonical_text=canonical_text,
            match=match,
            source_hash=context.source_content_hash,
            disambiguation_method=METHOD_PAGE_CONTEXT,
            match_confidence=float(match.get("match_confidence") or 0.0),
        )
        base["disambiguation_blockers"] = _dedupe(
            [
                *blockers,
                "single_global_match_without_additional_locator_filtering",
            ]
        )
        base["recommended_action"] = "queue_for_strict_evidence_design_review_after_disambiguation_design"
        return base

    if page is None and not bbox and not block_indexes:
        base["disambiguation_status"] = DISAMBIGUATION_STATUS_BLOCKED_MISSING_LOCATOR
        base["disambiguation_blockers"] = _dedupe(
            [*blockers, "locator_page_bbox_blockIndexes_missing"]
        )
        base["recommended_action"] = "recover_locator_context_before_disambiguation_design"
        return base

    filtered = list(all_matches)
    method = ""

    if page is not None:
        page_filtered = _filter_matches_to_page(filtered, pages=pages, page=page)
        if page_filtered:
            filtered = page_filtered
            method = METHOD_PAGE_CONTEXT

    if len(filtered) > 1 and block_indexes:
        block_filtered = _filter_matches_to_blocks(
            filtered,
            pages=pages,
            elements=_parsed_document_elements(parsed_root, paper_id),
            page=page or 0,
            block_indexes=block_indexes,
        )
        if block_filtered:
            filtered = block_filtered
            method = METHOD_BLOCK_CONTEXT

    if len(filtered) > 1 and len(bbox) >= 4 and page is not None:
        bbox_filtered = _filter_matches_to_bbox(
            filtered,
            pages=pages,
            page=page,
            bbox=bbox,
        )
        if bbox_filtered:
            filtered = bbox_filtered
            method = METHOD_BBOX_ADJACENT

    base["filtered_match_offsets"] = _match_records(filtered)

    if len(filtered) == 1:
        match = filtered[0]
        base["disambiguation_status"] = DISAMBIGUATION_STATUS_CANDIDATE
        base["disambiguation_method"] = method or METHOD_PAGE_CONTEXT
        base["proposed_chars"] = _proposed_chars(
            canonical_text=canonical_text,
            match=match,
            source_hash=context.source_content_hash,
            disambiguation_method=base["disambiguation_method"],
            match_confidence=float(match.get("match_confidence") or 0.0),
        )
        base["disambiguation_blockers"] = _dedupe(
            [
                *blockers,
                "disambiguation_design_only_not_applied_to_source_span_store",
            ]
        )
        base["recommended_action"] = "queue_for_strict_evidence_design_review_after_disambiguation_design"
        return base

    if len(filtered) > 1:
        base["disambiguation_status"] = DISAMBIGUATION_STATUS_BLOCKED_STILL_NON_UNIQUE
        base["disambiguation_blockers"] = _dedupe(
            [
                *blockers,
                f"filtered_match_count={len(filtered)}",
                "locator_context_insufficient_for_unique_match",
            ]
        )
        base["recommended_action"] = "manual_or_later_extractor_review_for_ambiguous_text_match"
        return base

    base["disambiguation_status"] = DISAMBIGUATION_STATUS_BLOCKED_MANUAL_OR_LATER
    base["disambiguation_blockers"] = _dedupe([*blockers, "no_filtered_match_after_locator_context"])
    base["recommended_action"] = "manual_or_later_extractor_review_for_ambiguous_text_match"
    return base


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    input_rows: int,
    target_rows: int,
    input_schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": input_rows,
        "targetRows": target_rows,
        "disambiguationDesignCandidateOnlyRows": sum(
            1 for row in rows if row.get("disambiguation_status") == DISAMBIGUATION_STATUS_CANDIDATE
        ),
        "blockedStillNonUniqueAfterLocatorContextRows": sum(
            1
            for row in rows
            if row.get("disambiguation_status") == DISAMBIGUATION_STATUS_BLOCKED_STILL_NON_UNIQUE
        ),
        "blockedMissingLocatorContextRows": sum(
            1
            for row in rows
            if row.get("disambiguation_status") == DISAMBIGUATION_STATUS_BLOCKED_MISSING_LOCATOR
        ),
        "blockedMissingCandidateMatchOffsetsRows": sum(
            1
            for row in rows
            if row.get("disambiguation_status") == DISAMBIGUATION_STATUS_BLOCKED_MISSING_MATCH_OFFSETS
        ),
        "blockedSourceTextUnavailableRows": sum(
            1
            for row in rows
            if row.get("disambiguation_status") == DISAMBIGUATION_STATUS_BLOCKED_SOURCE_TEXT_UNAVAILABLE
        ),
        "blockedRequiresManualOrLaterExtractorReviewRows": sum(
            1
            for row in rows
            if row.get("disambiguation_status") == DISAMBIGUATION_STATUS_BLOCKED_MANUAL_OR_LATER
        ),
        "blockedInputSchemaViolationRows": (
            len(rows) if input_schema_violations and rows else int(bool(input_schema_violations))
        ),
        "sourceSpanUpdatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "schemaViolationCount": len(input_schema_violations),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in rows)),
        "byDisambiguationStatus": dict(
            Counter(str(row.get("disambiguation_status") or "") for row in rows)
        ),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_parsed_artifact_source_span_text_match_disambiguation_design(
    *,
    design_review_report_path: str | Path = DEFAULT_DESIGN_REVIEW_REPORT_PATH,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    parsed_root: str | Path | None = None,
    paper_ids: list[str] | None = None,
    page_loader: Callable[[str | Path], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(design_review_report_path)).expanduser()
    papers_path = Path(str(papers_dir)).expanduser()
    parsed_path = Path(str(parsed_root or (papers_path / "parsed"))).expanduser()
    loader = page_loader or _extract_pdf_pages

    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    review_report = _read_json(report_path)
    if not review_report:
        warnings.append("design_review_report_missing_or_unreadable")

    validation = validate_payload(
        review_report,
        PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        input_schema_violations = [str(error) for error in validation.errors]
        if not review_report:
            input_schema_violations.append("design_review_report_missing_or_unreadable")

    ambiguous_rows = [
        row
        for row in review_report.get("ambiguousDisambiguationRows", [])
        if isinstance(row, dict)
    ] if isinstance(review_report, dict) else []

    if not ambiguous_rows:
        ambiguous_rows = [
            row
            for row in review_report.get("rows", [])
            if isinstance(row, dict)
            and _safe_text(row.get("review_status")) == REVIEW_STATUS_BLOCKED_NON_UNIQUE
        ] if isinstance(review_report, dict) else []

    input_rows = int((review_report.get("counts") or {}).get("inputRows") or 0) if review_report else 0
    target_rows = len(ambiguous_rows)

    if requested_papers:
        found = {_safe_text(row.get("paper_id")) for row in ambiguous_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found:
            warnings.append("requested_paper_ids_not_found")
        ambiguous_rows = [
            row for row in ambiguous_rows if _safe_text(row.get("paper_id")) in requested_papers
        ]

    if not ambiguous_rows:
        warnings.append("ambiguous_disambiguation_rows_missing")

    paper_contexts: dict[str, _PaperSourceContext] = {}
    rows = [
        _classify_ambiguous_row(
            review_row,
            paper_contexts=paper_contexts,
            parsed_root=parsed_path,
            papers_dir=papers_path,
            page_loader=loader,
        )
        for review_row in ambiguous_rows
    ]
    for index, row in enumerate(rows):
        row["disambiguation_row_id"] = (
            f"parsed-artifact-source-span-text-match-disambiguation-design:{index:04d}"
        )

    if input_schema_violations:
        for row in rows:
            row["disambiguation_status"] = DISAMBIGUATION_STATUS_BLOCKED_INPUT_SCHEMA
            row["disambiguation_blockers"] = _dedupe(
                [*row.get("disambiguation_blockers", []), *input_schema_violations]
            )
            row["proposed_chars"] = {}
            row["recommended_action"] = "repair_design_review_report_schema_before_disambiguation_design"

    disambiguation_candidates = [row for row in rows if row.get("disambiguation_status") == DISAMBIGUATION_STATUS_CANDIDATE]
    still_ambiguous = [
        row
        for row in rows
        if row.get("disambiguation_status") == DISAMBIGUATION_STATUS_BLOCKED_STILL_NON_UNIQUE
    ]

    counts = _count_rows(
        rows=rows,
        input_rows=input_rows if not requested_papers else len(rows),
        target_rows=target_rows if not requested_papers else len(rows),
        input_schema_violations=_dedupe(input_schema_violations),
    )
    candidate_count = int(counts.get("disambiguationDesignCandidateOnlyRows") or 0)
    still_blocked = int(counts.get("blockedStillNonUniqueAfterLocatorContextRows") or 0)

    status = "ok"
    if input_schema_violations or not rows:
        status = "blocked"
    elif candidate_count + still_blocked != len(rows):
        status = "blocked"

    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_TEXT_MATCH_DISAMBIGUATION_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "designReviewReportPath": str(report_path),
            "designReviewSchema": _safe_text(review_report.get("schema")) if review_report else "",
            "papersDir": str(papers_path),
            "parsedRoot": str(parsed_path),
            "requestedPaperIds": sorted(requested_papers),
        },
        "counts": counts,
        "gate": {
            "disambiguationDesignComplete": bool(disambiguation_candidates) and not input_schema_violations,
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "parsed_artifact_source_span_text_match_disambiguation_design_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": (
                "parsed_artifact_source_span_strict_evidence_design_review_reconciliation"
                if candidate_count > 0 and not input_schema_violations
                else "parsed_artifact_source_span_text_match_disambiguation_manual_review"
                if still_blocked > 0
                else "parsed_artifact_source_span_text_match_disambiguation_design_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "sourceSpanStoreWrite": False,
            "strictEvidenceCreated": False,
            "strictEligibleMutation": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "warnings": _dedupe(warnings),
        "disambiguationDesignRows": disambiguation_candidates,
        "stillAmbiguousRows": still_ambiguous,
        "rows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "input",
            "counts",
            "gate",
            "policy",
            "warnings",
            "disambiguationDesignRows",
            "stillAmbiguousRows",
            "rows",
        )
        if key in report
    }


def render_parsed_artifact_source_span_text_match_disambiguation_design_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byDisambiguationStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact SourceSpan Text Match Disambiguation Design",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- target ambiguous rows: {int(counts.get('targetRows') or 0)}",
            f"- disambiguation design candidates: {int(counts.get('disambiguationDesignCandidateOnlyRows') or 0)}",
            f"- still non-unique after locator context: {int(counts.get('blockedStillNonUniqueAfterLocatorContextRows') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            "",
            "## Disambiguation status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_source_span_text_match_disambiguation_design_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-span-text-match-disambiguation-design.json"
    summary_path = root / "parsed-artifact-source-span-text-match-disambiguation-design-summary.json"
    markdown_path = root / "parsed-artifact-source-span-text-match-disambiguation-design.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_text_match_disambiguation_design_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Design text-match disambiguation for ambiguous SourceSpan offset proposals "
            "without mutating SourceSpan records or creating StrictEvidence."
        )
    )
    parser.add_argument(
        "--design-review-report",
        default=str(DEFAULT_DESIGN_REVIEW_REPORT_PATH),
        help="Strict-evidence design review JSON report.",
    )
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--parsed-root", default="")
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_source_span_text_match_disambiguation_design(
        design_review_report_path=args.design_review_report,
        papers_dir=args.papers_dir,
        parsed_root=args.parsed_root or None,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_source_span_text_match_disambiguation_design_reports(
            report,
            args.output_dir,
        )
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")

    if args.json or not args.output_dir:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PARSED_ARTIFACT_SOURCE_SPAN_TEXT_MATCH_DISAMBIGUATION_DESIGN_SCHEMA_ID",
    "DISAMBIGUATION_STATUS_CANDIDATE",
    "build_parsed_artifact_source_span_text_match_disambiguation_design",
    "render_parsed_artifact_source_span_text_match_disambiguation_design_markdown",
    "write_parsed_artifact_source_span_text_match_disambiguation_design_reports",
]
