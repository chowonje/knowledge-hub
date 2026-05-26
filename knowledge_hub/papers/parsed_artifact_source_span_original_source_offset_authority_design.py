"""Report-only original-source char offset authority design for SourceSpan rows.

This helper consumes typed strict-evidence policy-gate rows blocked for missing
char-offset authority and attempts to design recoverable `chars` proposals against
canonical PDF-extracted text for the recorded `sourceContentHash`. It does not
mutate SourceSpan JSONL, create StrictEvidence, or change runtime/parser/answer/DB
state.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed import (
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_V2_TYPED_SCHEMA_ID,
    POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY,
)
from knowledge_hub.papers.sectionspan_pdf_offset_recovery_dry_run import (
    _exact_matches,
    _extract_pdf_pages,
    _normalized_matches,
    _source_pdf_from_manifest,
    _with_offsets,
)
from knowledge_hub.papers.source_text import source_hash_for_path


PARSED_ARTIFACT_SOURCE_SPAN_ORIGINAL_SOURCE_OFFSET_AUTHORITY_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-original-source-offset-authority-design.v1"
)

DESIGN_STATUS_OFFSET_AUTHORITY_CANDIDATE = "offset_authority_design_candidate_only"
DESIGN_STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
DESIGN_STATUS_BLOCKED_MISSING_SOURCE_FILE = "blocked_missing_source_file"
DESIGN_STATUS_BLOCKED_MISSING_LOCATOR_CONTEXT = "blocked_missing_locator_context"
DESIGN_STATUS_BLOCKED_MISSING_TEXT_SURFACE = "blocked_missing_text_surface"
DESIGN_STATUS_BLOCKED_SOURCE_TEXT_UNAVAILABLE = "blocked_source_text_unavailable"
DESIGN_STATUS_BLOCKED_NON_UNIQUE_TEXT_MATCH = "blocked_non_unique_text_match"
DESIGN_STATUS_BLOCKED_HASH_BASIS_UNAVAILABLE = "blocked_hash_basis_unavailable"
DESIGN_STATUS_BLOCKED_MANUAL_OR_LATER = "blocked_requires_manual_or_later_extractor_review"
DESIGN_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

TEXT_ARTIFACT_TYPES = {"section"}
CAPTION_ARTIFACT_TYPES = {"figure"}
TARGET_ARTIFACT_TYPES = TEXT_ARTIFACT_TYPES | CAPTION_ARTIFACT_TYPES

CHARS_NORMALIZATION = "nfkc_whitespace_casefold_v1"
CHARS_BASIS = "sourceContentHash"

DEFAULT_POLICY_GATE_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-strict-evidence-policy-gate-v2-typed"
    / "01-parsed-artifact-source-span-strict-evidence-policy-gate-v2-typed"
    / "parsed-artifact-source-span-strict-evidence-policy-gate-v2-typed.json"
)

DEFAULT_SECTIONSPAN_CANDIDATE_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-18"
    / "tex-sectionspan-candidate-audit"
    / "tex-sectionspan-candidates.json"
)

DEFAULT_FIGURE_CAPTION_CANDIDATE_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-18"
    / "tex-figure-caption-candidate-audit"
    / "tex-figure-caption-candidates.json"
)

DEFAULT_PAPERS_DIR = Path.home() / ".khub" / "papers"

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-original-source-offset-authority-design"
    / "01-parsed-artifact-source-span-original-source-offset-authority-design"
)


@dataclass(frozen=True)
class _PaperSourceContext:
    status: str
    paper_id: str
    manifest_path: str
    source_pdf_path: str
    source_content_hash: str
    pages: list[dict[str, Any]]
    canonical_text: str


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


def _substring_sha256(canonical_text: str, start: int, end: int) -> str:
    return hashlib.sha256(canonical_text[start:end].encode("utf-8")).hexdigest()


def _has_locator_context(row: dict[str, Any]) -> bool:
    locator = row.get("locator") if isinstance(row.get("locator"), dict) else {}
    page = _safe_int(locator.get("page"))
    bbox = _safe_list(locator.get("bbox"))
    block_indexes = _safe_list(locator.get("blockIndexes"))
    return page is not None or bool(bbox) or bool(block_indexes)


def _canonical_text_from_pages(pages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for page in pages:
        text = str(page.get("text") or "")
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _ledger_source_pdf(papers_dir: Path, paper_id: str) -> str:
    ledger_path = papers_dir.parent / "source_ledger" / "paper" / f"{paper_id.replace('.', '-')}.json"
    if not ledger_path.exists():
        ledger_path = papers_dir.parent / "source_ledger" / "paper" / f"{paper_id}.json"
    ledger = _read_json(ledger_path)
    artifacts = dict(ledger.get("artifacts") or {})
    return _safe_text(artifacts.get("raw_ref"))


def _resolve_paper_source_context(
    *,
    paper_id: str,
    parsed_root: Path,
    papers_dir: Path,
    page_loader: Callable[[str | Path], list[dict[str, Any]]],
    expected_hash: str,
) -> _PaperSourceContext:
    manifest_path = parsed_root / paper_id / "manifest.json"
    manifest = _read_json(manifest_path)
    source_pdf = _source_pdf_from_manifest(manifest) or _ledger_source_pdf(papers_dir, paper_id)
    if not manifest and not source_pdf:
        return _PaperSourceContext(
            status=DESIGN_STATUS_BLOCKED_MISSING_SOURCE_FILE,
            paper_id=paper_id,
            manifest_path=str(manifest_path),
            source_pdf_path="",
            source_content_hash="",
            pages=[],
            canonical_text="",
        )
    if not source_pdf:
        return _PaperSourceContext(
            status=DESIGN_STATUS_BLOCKED_MISSING_SOURCE_FILE,
            paper_id=paper_id,
            manifest_path=str(manifest_path),
            source_pdf_path="",
            source_content_hash="",
            pages=[],
            canonical_text="",
        )
    source_pdf_path = Path(source_pdf).expanduser()
    if not source_pdf_path.exists():
        return _PaperSourceContext(
            status=DESIGN_STATUS_BLOCKED_MISSING_SOURCE_FILE,
            paper_id=paper_id,
            manifest_path=str(manifest_path),
            source_pdf_path=str(source_pdf_path),
            source_content_hash="",
            pages=[],
            canonical_text="",
        )
    source_hash = source_hash_for_path(str(source_pdf_path))
    if not source_hash:
        return _PaperSourceContext(
            status=DESIGN_STATUS_BLOCKED_HASH_BASIS_UNAVAILABLE,
            paper_id=paper_id,
            manifest_path=str(manifest_path),
            source_pdf_path=str(source_pdf_path),
            source_content_hash="",
            pages=[],
            canonical_text="",
        )
    if expected_hash and source_hash != expected_hash:
        return _PaperSourceContext(
            status=DESIGN_STATUS_BLOCKED_HASH_BASIS_UNAVAILABLE,
            paper_id=paper_id,
            manifest_path=str(manifest_path),
            source_pdf_path=str(source_pdf_path),
            source_content_hash=source_hash,
            pages=[],
            canonical_text="",
        )
    pages = _with_offsets(page_loader(source_pdf_path))
    pages_with_text = sum(1 for item in pages if str(item.get("text") or "").strip())
    if not pages or pages_with_text <= 0:
        return _PaperSourceContext(
            status=DESIGN_STATUS_BLOCKED_SOURCE_TEXT_UNAVAILABLE,
            paper_id=paper_id,
            manifest_path=str(manifest_path),
            source_pdf_path=str(source_pdf_path),
            source_content_hash=source_hash,
            pages=pages,
            canonical_text="",
        )
    return _PaperSourceContext(
        status="ok",
        paper_id=paper_id,
        manifest_path=str(manifest_path),
        source_pdf_path=str(source_pdf_path),
        source_content_hash=source_hash,
        pages=pages,
        canonical_text=_canonical_text_from_pages(pages),
    )


def _index_candidate_text_surfaces(
    *,
    sectionspan_report_path: Path,
    figure_caption_report_path: Path,
) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    sectionspan = _read_json(sectionspan_report_path)
    for candidate in sectionspan.get("candidates", []) if isinstance(sectionspan, dict) else []:
        if not isinstance(candidate, dict):
            continue
        source_id = _safe_text(candidate.get("source_candidate_id"))
        if not source_id:
            continue
        text = _safe_text(candidate.get("section_title") or candidate.get("candidate_text"))
        if text:
            index.setdefault(source_id, {"text_surface": text, "source_file": _safe_text(candidate.get("source_file"))})
    figure_report = _read_json(figure_caption_report_path)
    for candidate in figure_report.get("candidates", []) if isinstance(figure_report, dict) else []:
        if not isinstance(candidate, dict):
            continue
        source_id = _safe_text(candidate.get("source_candidate_id"))
        if not source_id:
            continue
        text = _safe_text(candidate.get("caption_text") or candidate.get("candidate_text"))
        if text:
            index.setdefault(source_id, {"text_surface": text, "source_file": _safe_text(candidate.get("source_file"))})
    return index


def _figure_caption_from_manifest(parsed_root: Path, paper_id: str, page: int | None) -> str:
    if page is None:
        return ""
    manifest = _read_json(parsed_root / paper_id / "manifest.json")
    for item in manifest.get("figure_artifacts", []) if isinstance(manifest, dict) else []:
        if not isinstance(item, dict):
            continue
        if _safe_int(item.get("page")) == page:
            caption = _safe_text(item.get("caption"))
            if caption:
                return caption
    return ""


def _resolve_text_surface(
    *,
    policy_row: dict[str, Any],
    candidate_index: dict[str, dict[str, str]],
    parsed_root: Path,
) -> tuple[str, str]:
    source_candidate_id = _safe_text(policy_row.get("source_candidate_id"))
    indexed = candidate_index.get(source_candidate_id, {})
    text_surface = _safe_text(indexed.get("text_surface"))
    source_file = _safe_text(policy_row.get("source_file")) or _safe_text(indexed.get("source_file"))
    if text_surface:
        return text_surface, source_file
    artifact_type = _safe_text(policy_row.get("artifact_type"))
    if artifact_type == "figure":
        locator = policy_row.get("locator") if isinstance(policy_row.get("locator"), dict) else {}
        caption = _figure_caption_from_manifest(parsed_root, _safe_text(policy_row.get("paper_id")), _safe_int(locator.get("page")))
        if caption:
            return caption, source_file
    return "", source_file


def _proposed_chars(
    *,
    canonical_text: str,
    match: dict[str, Any],
    source_hash: str,
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
        "matchMethod": _safe_text(match.get("match_method")),
        "matchConfidence": match.get("match_confidence"),
        "page": _safe_int(match.get("page")),
    }


def _classify_design_row(
    *,
    policy_row: dict[str, Any],
    paper_contexts: dict[str, _PaperSourceContext],
    candidate_index: dict[str, dict[str, str]],
    parsed_root: Path,
    papers_dir: Path,
    page_loader: Callable[[str | Path], list[dict[str, Any]]],
) -> dict[str, Any]:
    paper_id = _safe_text(policy_row.get("paper_id"))
    artifact_type = _safe_text(policy_row.get("artifact_type"))
    source_hash = _safe_text(policy_row.get("sourceContentHash"))
    source_candidate_id = _safe_text(policy_row.get("source_candidate_id"))
    text_surface, source_file = _resolve_text_surface(
        policy_row=policy_row,
        candidate_index=candidate_index,
        parsed_root=parsed_root,
    )

    base = {
        "design_row_id": "",
        "policy_gate_row_id": _safe_text(policy_row.get("policy_gate_row_id")),
        "sourceSpanId": _safe_text(policy_row.get("sourceSpanId")),
        "candidateRecordId": _safe_text(policy_row.get("candidateRecordId")),
        "paper_id": paper_id,
        "artifact_type": artifact_type,
        "source_candidate_id": source_candidate_id,
        "sourceContentHash": source_hash,
        "source_file": source_file,
        "text_surface": text_surface,
        "locator": policy_row.get("locator") if isinstance(policy_row.get("locator"), dict) else {},
        "design_status": "",
        "design_blockers": [],
        "proposed_chars": {},
        "source_resolution": {},
        "recommended_action": "",
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "sourceSpanUpdatedRows": 0,
    }

    blockers: list[str] = ["original_source_offset_authority_design_only"]

    if not source_hash:
        base["design_status"] = DESIGN_STATUS_BLOCKED_MISSING_SOURCE_HASH
        base["design_blockers"] = _dedupe([*blockers, "sourceContentHash_missing"])
        base["recommended_action"] = "recover_source_content_hash_before_offset_authority_design"
        return base

    if not _has_locator_context(policy_row):
        base["design_status"] = DESIGN_STATUS_BLOCKED_MISSING_LOCATOR_CONTEXT
        base["design_blockers"] = _dedupe([*blockers, "locator_page_bbox_or_blockIndexes_missing"])
        base["recommended_action"] = "recover_locator_context_before_offset_authority_design"
        return base

    if not text_surface:
        base["design_status"] = DESIGN_STATUS_BLOCKED_MISSING_TEXT_SURFACE
        base["design_blockers"] = _dedupe([*blockers, "text_surface_unavailable_from_candidate_or_parsed_context"])
        base["recommended_action"] = "recover_text_surface_from_candidate_or_extractor_before_offset_design"
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

    base["source_resolution"] = {
        "manifestPath": context.manifest_path,
        "sourcePdfPath": context.source_pdf_path,
        "resolvedSourceContentHash": context.source_content_hash,
        "canonicalTextLength": len(context.canonical_text),
        "pageCount": len(context.pages),
    }

    if context.status != "ok":
        base["design_status"] = context.status
        base["design_blockers"] = _dedupe([*blockers, context.status])
        base["recommended_action"] = {
            DESIGN_STATUS_BLOCKED_MISSING_SOURCE_FILE: "register_or_restore_source_pdf_before_offset_design",
            DESIGN_STATUS_BLOCKED_HASH_BASIS_UNAVAILABLE: "reconcile_source_content_hash_with_registered_pdf_before_offset_design",
            DESIGN_STATUS_BLOCKED_SOURCE_TEXT_UNAVAILABLE: "restore_pdf_text_extraction_before_offset_design",
        }.get(context.status, "repair_source_context_before_offset_design")
        return base

    pages = list(context.pages)
    exact = _exact_matches(pages, text_surface)
    if len(exact) == 1:
        match = exact[0]
        method = "exact"
    elif len(exact) > 1:
        base["design_status"] = DESIGN_STATUS_BLOCKED_NON_UNIQUE_TEXT_MATCH
        base["design_blockers"] = _dedupe([*blockers, f"exact_match_count={len(exact)}"])
        base["recommended_action"] = "disambiguate_duplicate_exact_matches_before_chars_authority_design"
        return base
    else:
        normalized = _normalized_matches(pages, text_surface)
        if len(normalized) == 1:
            match = normalized[0]
            method = "normalized_whitespace_case"
        elif len(normalized) > 1:
            base["design_status"] = DESIGN_STATUS_BLOCKED_NON_UNIQUE_TEXT_MATCH
            base["design_blockers"] = _dedupe([*blockers, f"normalized_match_count={len(normalized)}"])
            base["recommended_action"] = "disambiguate_duplicate_normalized_matches_before_chars_authority_design"
            return base
        else:
            base["design_status"] = DESIGN_STATUS_BLOCKED_MANUAL_OR_LATER
            base["design_blockers"] = _dedupe([*blockers, "unique_original_source_text_match_not_found"])
            base["recommended_action"] = "manual_or_later_extractor_review_required_for_offset_authority"
            return base

    proposed = _proposed_chars(
        canonical_text=context.canonical_text,
        match=match,
        source_hash=context.source_content_hash,
    )
    if not proposed:
        base["design_status"] = DESIGN_STATUS_BLOCKED_MANUAL_OR_LATER
        base["design_blockers"] = _dedupe([*blockers, "proposed_chars_invalid"])
        base["recommended_action"] = "manual_or_later_extractor_review_required_for_offset_authority"
        return base

    base["design_status"] = DESIGN_STATUS_OFFSET_AUTHORITY_CANDIDATE
    base["design_blockers"] = _dedupe(
        [
            *blockers,
            "design_only_not_applied_to_source_span_store",
            "strict_evidence_creation_disabled_for_tranche",
        ]
    )
    base["proposed_chars"] = proposed
    base["recommended_action"] = (
        "review_proposed_chars_before_source_span_strict_evidence_design_review"
        if method == "exact"
        else "review_normalized_chars_proposal_before_source_span_strict_evidence_design_review"
    )
    return base


def _design_rows(
    policy_rows: list[dict[str, Any]],
    *,
    parsed_root: Path,
    papers_dir: Path,
    sectionspan_report_path: Path,
    figure_caption_report_path: Path,
    page_loader: Callable[[str | Path], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    candidate_index = _index_candidate_text_surfaces(
        sectionspan_report_path=sectionspan_report_path,
        figure_caption_report_path=figure_caption_report_path,
    )
    paper_contexts: dict[str, _PaperSourceContext] = {}
    rows: list[dict[str, Any]] = []
    for index, policy_row in enumerate(policy_rows):
        row = _classify_design_row(
            policy_row=policy_row,
            paper_contexts=paper_contexts,
            candidate_index=candidate_index,
            parsed_root=parsed_root,
            papers_dir=papers_dir,
            page_loader=page_loader,
        )
        row["design_row_id"] = (
            f"parsed-artifact-source-span-original-source-offset-authority-design:{index:04d}"
        )
        rows.append(row)
    return rows


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
        "offsetAuthorityDesignCandidateOnlyRows": sum(
            1 for row in rows if row.get("design_status") == DESIGN_STATUS_OFFSET_AUTHORITY_CANDIDATE
        ),
        "blockedMissingSourceHashRows": sum(
            1 for row in rows if row.get("design_status") == DESIGN_STATUS_BLOCKED_MISSING_SOURCE_HASH
        ),
        "blockedMissingSourceFileRows": sum(
            1 for row in rows if row.get("design_status") == DESIGN_STATUS_BLOCKED_MISSING_SOURCE_FILE
        ),
        "blockedMissingLocatorContextRows": sum(
            1 for row in rows if row.get("design_status") == DESIGN_STATUS_BLOCKED_MISSING_LOCATOR_CONTEXT
        ),
        "blockedMissingTextSurfaceRows": sum(
            1 for row in rows if row.get("design_status") == DESIGN_STATUS_BLOCKED_MISSING_TEXT_SURFACE
        ),
        "blockedSourceTextUnavailableRows": sum(
            1 for row in rows if row.get("design_status") == DESIGN_STATUS_BLOCKED_SOURCE_TEXT_UNAVAILABLE
        ),
        "blockedNonUniqueTextMatchRows": sum(
            1 for row in rows if row.get("design_status") == DESIGN_STATUS_BLOCKED_NON_UNIQUE_TEXT_MATCH
        ),
        "blockedRequiresManualOrLaterExtractorReviewRows": sum(
            1 for row in rows if row.get("design_status") == DESIGN_STATUS_BLOCKED_MANUAL_OR_LATER
        ),
        "blockedHashBasisUnavailableRows": sum(
            1 for row in rows if row.get("design_status") == DESIGN_STATUS_BLOCKED_HASH_BASIS_UNAVAILABLE
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
        "byDesignStatus": dict(Counter(str(row.get("design_status") or "") for row in rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_parsed_artifact_source_span_original_source_offset_authority_design(
    *,
    policy_gate_report_path: str | Path = DEFAULT_POLICY_GATE_REPORT_PATH,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    parsed_root: str | Path | None = None,
    sectionspan_candidate_report_path: str | Path = DEFAULT_SECTIONSPAN_CANDIDATE_REPORT_PATH,
    figure_caption_candidate_report_path: str | Path = DEFAULT_FIGURE_CAPTION_CANDIDATE_REPORT_PATH,
    paper_ids: list[str] | None = None,
    page_loader: Callable[[str | Path], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(policy_gate_report_path)).expanduser()
    papers_path = Path(str(papers_dir)).expanduser()
    parsed_path = Path(str(parsed_root or (papers_path / "parsed"))).expanduser()
    sectionspan_path = Path(str(sectionspan_candidate_report_path)).expanduser()
    figure_path = Path(str(figure_caption_candidate_report_path)).expanduser()
    loader = page_loader or _extract_pdf_pages

    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    policy_gate_report = _read_json(report_path)
    if not policy_gate_report:
        warnings.append("policy_gate_report_missing_or_unreadable")

    validation = validate_payload(
        policy_gate_report,
        PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_V2_TYPED_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        input_schema_violations = [str(error) for error in validation.errors]
        if not policy_gate_report:
            input_schema_violations.append("policy_gate_report_missing_or_unreadable")

    all_policy_rows = [
        row for row in policy_gate_report.get("rows", []) if isinstance(row, dict)
    ] if isinstance(policy_gate_report, dict) else []

    input_rows = len(all_policy_rows)
    target_policy_rows = [
        row
        for row in all_policy_rows
        if _safe_text(row.get("policy_gate_status")) == POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY
        and _safe_text(row.get("artifact_type")) in TARGET_ARTIFACT_TYPES
    ]

    if requested_papers:
        found = {_safe_text(row.get("paper_id")) for row in target_policy_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found:
            warnings.append("requested_paper_ids_not_found")
        target_policy_rows = [
            row for row in target_policy_rows if _safe_text(row.get("paper_id")) in requested_papers
        ]

    if not target_policy_rows:
        warnings.append("target_policy_gate_rows_missing")

    rows = _design_rows(
        target_policy_rows,
        parsed_root=parsed_path,
        papers_dir=papers_path,
        sectionspan_report_path=sectionspan_path,
        figure_caption_report_path=figure_path,
        page_loader=loader,
    )

    if input_schema_violations:
        for row in rows:
            row["design_status"] = DESIGN_STATUS_BLOCKED_INPUT_SCHEMA
            row["design_blockers"] = _dedupe(
                [*row.get("design_blockers", []), *input_schema_violations]
            )
            row["proposed_chars"] = {}
            row["recommended_action"] = "repair_policy_gate_report_schema_before_offset_authority_design"

    counts = _count_rows(
        rows=rows,
        input_rows=input_rows,
        target_rows=len(target_policy_rows),
        input_schema_violations=_dedupe(input_schema_violations),
    )
    candidate_rows = int(counts.get("offsetAuthorityDesignCandidateOnlyRows") or 0)
    status = "ok"
    if input_schema_violations or not rows or candidate_rows != len(rows):
        status = "blocked"

    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_ORIGINAL_SOURCE_OFFSET_AUTHORITY_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "policyGateReportPath": str(report_path),
            "policyGateSchema": _safe_text(policy_gate_report.get("schema")) if policy_gate_report else "",
            "papersDir": str(papers_path),
            "parsedRoot": str(parsed_path),
            "sectionspanCandidateReportPath": str(sectionspan_path),
            "figureCaptionCandidateReportPath": str(figure_path),
            "requestedPaperIds": sorted(requested_papers),
        },
        "counts": counts,
        "gate": {
            "offsetAuthorityDesignComplete": bool(candidate_rows) and candidate_rows == len(rows) and not input_schema_violations,
            "readyForStrictEvidenceDesignReview": False,
            "strictEvidenceCreated": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "parsed_artifact_source_span_original_source_offset_authority_design_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": (
                "parsed_artifact_source_span_strict_evidence_design_review"
                if status == "ok"
                else "parsed_artifact_source_span_original_source_offset_authority_design_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "sourceSpanStoreWrite": False,
            "strictEvidenceCreated": False,
            "strictEligibleMutation": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "warnings": _dedupe(warnings),
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
            "rows",
        )
        if key in report
    }


def render_parsed_artifact_source_span_original_source_offset_authority_design_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byDesignStatus") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact SourceSpan Original Source Offset Authority Design",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- target rows: {int(counts.get('targetRows') or 0)}",
            f"- offset authority design candidates: {int(counts.get('offsetAuthorityDesignCandidateOnlyRows') or 0)}",
            f"- blocked missing text surface: {int(counts.get('blockedMissingTextSurfaceRows') or 0)}",
            f"- blocked non-unique match: {int(counts.get('blockedNonUniqueTextMatchRows') or 0)}",
            f"- blocked manual/later review: {int(counts.get('blockedRequiresManualOrLaterExtractorReviewRows') or 0)}",
            f"- source span updated: {int(counts.get('sourceSpanUpdatedRows') or 0)}",
            "",
            "## Design status breakdown",
            *[f"- {item}" for item in by_status],
        ]
    )


def write_parsed_artifact_source_span_original_source_offset_authority_design_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = (
        root / "parsed-artifact-source-span-original-source-offset-authority-design.json"
    )
    summary_path = (
        root / "parsed-artifact-source-span-original-source-offset-authority-design-summary.json"
    )
    markdown_path = (
        root / "parsed-artifact-source-span-original-source-offset-authority-design.md"
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_original_source_offset_authority_design_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Design original-source char offset authority proposals for SourceSpan rows "
            "blocked on missing offset authority without mutating SourceSpan records."
        )
    )
    parser.add_argument(
        "--policy-gate-report",
        default=str(DEFAULT_POLICY_GATE_REPORT_PATH),
        help="Typed strict-evidence policy gate v2 JSON report.",
    )
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--parsed-root", default="")
    parser.add_argument(
        "--sectionspan-candidate-report",
        default=str(DEFAULT_SECTIONSPAN_CANDIDATE_REPORT_PATH),
    )
    parser.add_argument(
        "--figure-caption-candidate-report",
        default=str(DEFAULT_FIGURE_CAPTION_CANDIDATE_REPORT_PATH),
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_source_span_original_source_offset_authority_design(
        policy_gate_report_path=args.policy_gate_report,
        papers_dir=args.papers_dir,
        parsed_root=args.parsed_root or None,
        sectionspan_candidate_report_path=args.sectionspan_candidate_report,
        figure_caption_candidate_report_path=args.figure_caption_candidate_report,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_source_span_original_source_offset_authority_design_reports(
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
    "PARSED_ARTIFACT_SOURCE_SPAN_ORIGINAL_SOURCE_OFFSET_AUTHORITY_DESIGN_SCHEMA_ID",
    "DESIGN_STATUS_OFFSET_AUTHORITY_CANDIDATE",
    "build_parsed_artifact_source_span_original_source_offset_authority_design",
    "render_parsed_artifact_source_span_original_source_offset_authority_design_markdown",
    "write_parsed_artifact_source_span_original_source_offset_authority_design_reports",
]
