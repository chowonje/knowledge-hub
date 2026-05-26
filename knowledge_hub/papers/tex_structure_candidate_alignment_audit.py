"""Report-only TeX structure candidate alignment audit.

This helper consumes the arXiv source/TeX availability report and checks
whether TeX structure rows can be linked to canonical generated Markdown
spans, recovered pages, and source-content hashes.

It is an audit-only candidate layer.  It does not mutate SQLite, scan vault
content, reindex, reembed, route parsers, write canonical parsed artifacts, or
promote any row to strict/runtime evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.source_text import source_hash_for_path


TEX_STRUCTURE_CANDIDATE_ALIGNMENT_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.tex-structure-candidate-alignment-audit.v1"
)

DEFAULT_ARXIV_SOURCE_TEX_AVAILABILITY_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-18"
    / "arxiv-source-tex-availability-audit"
    / "arxiv-source-tex-availability-report.json"
)
DEFAULT_PARSED_ROOT = Path.home() / ".khub" / "papers" / "parsed"

_PAGE_HEADING_RE = re.compile(r"^##\s+Page\s+(\d+)\s*$", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[0-9a-z]+", re.IGNORECASE)
_LIGATURES = str.maketrans(
    {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "−": "-",
        "–": "-",
        "—": "-",
        "×": "x",
    }
)
_TEXT_SPAN_STRUCTURE_TYPES = {
    "section",
    "subsection",
    "subsubsection",
    "caption",
    "figure_caption",
    "table_caption",
}
_HEADING_STRUCTURE_TYPES = {"section", "subsection", "subsubsection"}
_MARKDOWN_HEADING_PREFIX_RE = re.compile(r"^\s*#{1,6}\s+")
_HEADING_CONTEXT_PREFIX_RE = re.compile(r"(?:^|[\n\r]|[\.\?!]\s+|\s)(?:\d+(?:\.\d+)*|[A-Z])\.?\s+$")


@dataclass(frozen=True)
class _NormalizedText:
    text: str
    original_index_by_char: list[int]


@dataclass(frozen=True)
class _Alignment:
    status: str
    method: str
    chars_start: int | None
    chars_end: int | None
    confidence: float
    reason: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_text(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "").strip())


def _fold_text(value: Any) -> str:
    return _clean_text(value).translate(_LIGATURES).casefold()


def _tokenize(value: Any) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(_fold_text(value)) if len(token) >= 2]


def _line_starts(text: str) -> list[int]:
    starts = [0]
    offset = 0
    for line in text.splitlines(keepends=True):
        offset += len(line)
        starts.append(offset)
    return starts


def _line_number(starts: list[int], offset: int) -> int:
    return max(1, bisect_right(starts, max(0, offset)))


def _normalized_text_with_map(text: str) -> _NormalizedText:
    parts: list[str] = []
    indexes: list[int] = []
    pending_space = False
    pending_index = 0
    for index, raw_char in enumerate(text):
        folded = raw_char.translate(_LIGATURES).casefold()
        if not folded:
            continue
        if folded.isspace():
            if parts:
                pending_space = True
                pending_index = index
            continue
        if pending_space and parts and parts[-1] != " ":
            parts.append(" ")
            indexes.append(pending_index)
        pending_space = False
        for char in folded:
            if char.isspace():
                if parts and parts[-1] != " ":
                    parts.append(" ")
                    indexes.append(index)
                continue
            parts.append(char)
            indexes.append(index)
    raw = "".join(parts)
    start = 0
    end = len(raw)
    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end - 1].isspace():
        end -= 1
    return _NormalizedText(raw[start:end], indexes[start:end])


def _find_all(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    starts: list[int] = []
    cursor = text.find(needle)
    while cursor >= 0:
        starts.append(cursor)
        cursor = text.find(needle, cursor + 1)
    return starts


def _exact_alignment(markdown_text: str, candidate_text: str) -> _Alignment | None:
    starts = _find_all(markdown_text, candidate_text)
    if len(starts) == 1:
        start = starts[0]
        return _Alignment("aligned", "exact", start, start + len(candidate_text), 0.99, "single_exact_text_match")
    if len(starts) > 1:
        return _Alignment("ambiguous", "exact", None, None, 0.2, "ambiguous_exact_text_match")
    return None


def _normalized_alignment(markdown_text: str, candidate_text: str) -> _Alignment | None:
    normalized_markdown = _normalized_text_with_map(markdown_text)
    normalized_candidate = _normalized_text_with_map(candidate_text).text
    starts = _find_all(normalized_markdown.text, normalized_candidate)
    if len(starts) == 1:
        start = starts[0]
        end = start + len(normalized_candidate)
        if end <= len(normalized_markdown.original_index_by_char):
            chars_start = normalized_markdown.original_index_by_char[start]
            chars_end = normalized_markdown.original_index_by_char[end - 1] + 1
            return _Alignment("aligned", "normalized", chars_start, chars_end, 0.82, "single_normalized_text_match")
    if len(starts) > 1:
        return _Alignment("ambiguous", "normalized", None, None, 0.18, "ambiguous_normalized_text_match")
    return None


def _ordered_token_alignment(markdown_text: str, candidate_text: str, *, max_window_chars: int = 4000) -> _Alignment | None:
    tokens = _tokenize(candidate_text)
    if len(tokens) < 3:
        return None
    normalized_markdown = _normalized_text_with_map(markdown_text)
    haystack = normalized_markdown.text
    first = tokens[0]
    candidates: list[tuple[int, int]] = []
    for start in _find_all(haystack, first):
        cursor = start + len(first)
        last_end = cursor
        matched = True
        for token in tokens[1:]:
            next_index = haystack.find(token, cursor)
            if next_index < 0 or next_index - start > max_window_chars:
                matched = False
                break
            cursor = next_index + len(token)
            last_end = cursor
        if matched and last_end <= len(normalized_markdown.original_index_by_char):
            original_start = normalized_markdown.original_index_by_char[start]
            original_end = normalized_markdown.original_index_by_char[last_end - 1] + 1
            candidates.append((original_start, original_end))
    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1:
        start, end = unique[0]
        return _Alignment("aligned", "ordered_token", start, end, 0.56, "single_ordered_token_match")
    if len(unique) > 1:
        return _Alignment("ambiguous", "ordered_token", None, None, 0.12, "ambiguous_ordered_token_match")
    return None


def _align_text(markdown_text: str, candidate_text: str) -> _Alignment:
    text = _clean_text(candidate_text)
    if not markdown_text:
        return _Alignment("blocked", "none", None, None, 0.0, "canonical_markdown_missing")
    if not text:
        return _Alignment("blocked", "none", None, None, 0.0, "candidate_text_empty")
    exact = _exact_alignment(markdown_text, text)
    if exact is not None:
        return exact
    normalized = _normalized_alignment(markdown_text, text)
    if normalized is not None:
        return normalized
    ordered = _ordered_token_alignment(markdown_text, text)
    if ordered is not None:
        return ordered
    return _Alignment("failed", "none", None, None, 0.0, "no_canonical_text_match")


def _heading_line_span(line: str) -> tuple[int, int, str]:
    content = line.rstrip("\r\n")
    prefix = _MARKDOWN_HEADING_PREFIX_RE.match(content)
    segment_start = prefix.end() if prefix else 0
    segment = content[segment_start:]
    left_trimmed = len(segment) - len(segment.lstrip())
    right_trimmed = len(segment.rstrip())
    start = segment_start + left_trimmed
    end = segment_start + right_trimmed
    return start, end, segment[left_trimmed:right_trimmed]


def _heading_match_allowed(markdown_text: str, start: int, end: int, candidate_text: str) -> bool:
    line_start = markdown_text.rfind("\n", 0, start) + 1
    line_end = markdown_text.find("\n", end)
    if line_end < 0:
        line_end = len(markdown_text)
    _segment_start, _segment_end, line_text = _heading_line_span(markdown_text[line_start:line_end])
    if line_text == candidate_text or _fold_text(line_text) == _fold_text(candidate_text):
        return True
    prefix = markdown_text[max(0, start - 80) : start]
    return bool(_HEADING_CONTEXT_PREFIX_RE.search(prefix))


def _heading_text_alignment(markdown_text: str, candidate_text: str) -> _Alignment:
    text = _clean_text(candidate_text)
    if not markdown_text:
        return _Alignment("blocked", "none", None, None, 0.0, "canonical_markdown_missing")
    if not text:
        return _Alignment("blocked", "none", None, None, 0.0, "candidate_text_empty")

    exact_matches: list[tuple[int, int]] = []
    normalized_matches: list[tuple[int, int]] = []
    for start in _find_all(markdown_text, text):
        end = start + len(text)
        if _heading_match_allowed(markdown_text, start, end, text):
            exact_matches.append((start, end))

    exact_unique = list(dict.fromkeys(exact_matches))
    if len(exact_unique) == 1:
        start, end = exact_unique[0]
        return _Alignment("aligned", "exact", start, end, 0.99, "single_heading_context_exact_match")
    if len(exact_unique) > 1:
        return _Alignment("ambiguous", "exact", None, None, 0.2, "ambiguous_heading_context_exact_match")

    normalized_markdown = _normalized_text_with_map(markdown_text)
    normalized_candidate = _normalized_text_with_map(text).text
    for start in _find_all(normalized_markdown.text, normalized_candidate):
        end = start + len(normalized_candidate)
        if end <= len(normalized_markdown.original_index_by_char):
            original_start = normalized_markdown.original_index_by_char[start]
            original_end = normalized_markdown.original_index_by_char[end - 1] + 1
            if _heading_match_allowed(markdown_text, original_start, original_end, text):
                normalized_matches.append((original_start, original_end))
    normalized_unique = list(dict.fromkeys(normalized_matches))
    if len(normalized_unique) == 1:
        start, end = normalized_unique[0]
        return _Alignment("aligned", "normalized", start, end, 0.82, "single_heading_context_normalized_match")
    if len(normalized_unique) > 1:
        return _Alignment("ambiguous", "normalized", None, None, 0.18, "ambiguous_heading_context_normalized_match")

    return _Alignment("failed", "none", None, None, 0.0, "no_heading_context_text_match")


def _source_hash_from_manifest(manifest: dict[str, Any]) -> tuple[str, str]:
    parser_meta = dict(manifest.get("parser_meta") or {})
    for key in (
        "sourceContentHash",
        "source_content_hash",
        "expectedSourceContentHash",
        "observedSourceContentHash",
    ):
        value = str(parser_meta.get(key) or manifest.get(key) or "").strip()
        if value:
            return value, key
    for key in ("source_pdf", "extracted_from", "pdf_path", "sourcePath"):
        path_value = str(parser_meta.get(key) or manifest.get(key) or "").strip()
        digest = source_hash_for_path(path_value)
        if digest:
            return digest, f"computed_from_manifest_{key}"
    return "", "unavailable"


def _page_ranges(markdown_text: str) -> list[dict[str, int]]:
    matches = list(_PAGE_HEADING_RE.finditer(markdown_text))
    ranges: list[dict[str, int]] = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(markdown_text)
        ranges.append(
            {
                "page": int(match.group(1)),
                "start": match.start(),
                "contentStart": match.end(),
                "end": next_start,
            }
        )
    return ranges


def _page_for_offset(offset: int | None, ranges: list[dict[str, int]]) -> int | None:
    if offset is None:
        return None
    for item in ranges:
        if int(item["start"]) <= offset < int(item["end"]):
            return int(item["page"])
    return None


def _markdown_span_locator(markdown_text: str, chars_start: int | None, chars_end: int | None) -> dict[str, Any]:
    if chars_start is None or chars_end is None:
        return {}
    starts = _line_starts(markdown_text)
    return {
        "path": "document.md",
        "locatorKind": "canonical_generated_markdown",
        "chars": {"start": chars_start, "end": chars_end},
        "lineStart": _line_number(starts, chars_start),
        "lineEnd": _line_number(starts, max(chars_start, chars_end - 1)),
    }


def _strict_blockers(
    *,
    structure_type: str,
    alignment: _Alignment,
    page: int | None,
    source_hash: str,
    mineru_link_status: str,
) -> list[str]:
    blockers = [
        "source_structure_candidate_only",
        "runtime_promotion_disabled_for_tranche",
        "strict_promotion_requires_later_explicit_tranche",
        "tex_offsets_are_not_canonical_source_spans",
        "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
    ]
    if alignment.status != "aligned":
        blockers.append("canonical_text_alignment_not_available")
    if alignment.status == "ambiguous":
        blockers.append("ambiguous_canonical_text_match")
    if alignment.method != "exact":
        blockers.append("non_exact_or_missing_canonical_alignment")
    if alignment.chars_start is None or alignment.chars_end is None:
        blockers.append("missing_chars_start_end")
    if page is None:
        blockers.append("missing_page")
    if not source_hash:
        blockers.append("missing_source_content_hash")
    if mineru_link_status != "linked":
        blockers.append("mineru_layout_or_bbox_link_not_unique")
    if structure_type.startswith("table"):
        blockers.append("table_cell_row_column_bbox_provenance_missing")
    if structure_type.startswith("figure"):
        blockers.append("figure_region_link_unverified")
    if structure_type.startswith("equation"):
        blockers.append("equation_text_or_semantics_not_citation_grade")
    return list(dict.fromkeys(blockers))


def _classification(
    *,
    structure_type: str,
    candidate_text: str,
    alignment: _Alignment,
    page: int | None,
    source_hash: str,
    mineru_link_status: str,
) -> str:
    if structure_type not in _TEXT_SPAN_STRUCTURE_TYPES or not candidate_text:
        return "raw_tex_environment_only"
    if alignment.status == "ambiguous":
        return "blocked_ambiguous_canonical_match"
    if alignment.status != "aligned":
        return "blocked_no_canonical_match"
    if not source_hash:
        return "canonical_span_without_source_hash_non_strict"
    if page is None:
        return "canonical_span_without_page_non_strict"
    if mineru_link_status == "linked":
        return "source_span_and_layout_candidate_only"
    return "source_span_candidate_only"


def _paper_inputs(parsed_root: Path, paper_id: str) -> tuple[dict[str, Any], str, list[dict[str, int]], str, str]:
    parsed_dir = parsed_root / paper_id
    manifest = _read_json(parsed_dir / "manifest.json")
    markdown_path = parsed_dir / "document.md"
    try:
        markdown_text = markdown_path.read_text(encoding="utf-8")
    except Exception:
        document = _read_json(parsed_dir / "document.json")
        markdown_text = str(document.get("markdown_text") or "")
    source_hash, source_hash_source = _source_hash_from_manifest(manifest)
    return manifest, markdown_text, _page_ranges(markdown_text), source_hash, source_hash_source


def _aligned_row(
    *,
    row: dict[str, Any],
    markdown_text: str,
    ranges: list[dict[str, int]],
    source_hash: str,
    source_hash_source: str,
) -> dict[str, Any]:
    structure_type = str(row.get("structure_type") or "")
    candidate_text = _clean_text(row.get("candidate_text"))
    if structure_type in _HEADING_STRUCTURE_TYPES:
        alignment = _heading_text_alignment(markdown_text, candidate_text)
    elif structure_type in _TEXT_SPAN_STRUCTURE_TYPES:
        alignment = _align_text(markdown_text, candidate_text)
    else:
        alignment = _Alignment("blocked", "none", None, None, 0.0, "structure_type_has_no_text_span")
    page = _page_for_offset(alignment.chars_start, ranges)
    mineru_link_status = str(row.get("mineru_layout_link_status") or "blocked")
    blockers = _strict_blockers(
        structure_type=structure_type,
        alignment=alignment,
        page=page,
        source_hash=source_hash,
        mineru_link_status=mineru_link_status,
    )
    classification = _classification(
        structure_type=structure_type,
        candidate_text=candidate_text,
        alignment=alignment,
        page=page,
        source_hash=source_hash,
        mineru_link_status=mineru_link_status,
    )
    source_span_ready = bool(
        alignment.status == "aligned"
        and alignment.chars_start is not None
        and alignment.chars_end is not None
        and page is not None
        and source_hash
    )
    return {
        "candidate_id": str(row.get("structure_row_id") or ""),
        "candidate_type": "tex_structure_candidate",
        "paper_id": str(row.get("paper_id") or ""),
        "source_parser": "arxiv_tex",
        "source_file": str(row.get("source_file") or ""),
        "structure_type": structure_type,
        "tex_command": str(row.get("tex_command") or ""),
        "tex_environment": str(row.get("tex_environment") or ""),
        "candidate_text": candidate_text,
        "tex_locator": {
            "source_file": str(row.get("source_file") or ""),
            "chars": {
                "start": int(row.get("tex_chars_start") or 0),
                "end": int(row.get("tex_chars_end") or 0),
            },
            "command": str(row.get("tex_command") or ""),
            "environment": str(row.get("tex_environment") or ""),
        },
        "alignment_status": alignment.status,
        "alignment_method": alignment.method,
        "alignment_reason": alignment.reason,
        "chars_start": alignment.chars_start,
        "chars_end": alignment.chars_end,
        "page": page,
        "sourceContentHash": source_hash or None,
        "sourceContentHashSource": source_hash_source,
        "confidence": round(alignment.confidence, 3),
        "source_span_locator": _markdown_span_locator(markdown_text, alignment.chars_start, alignment.chars_end),
        "source_span_candidate_ready": source_span_ready,
        "classification": classification,
        "mineru_layout_link_status": mineru_link_status,
        "mineru_layout_link_method": str(row.get("mineru_layout_link_method") or ""),
        "mineru_candidate_ids": list(row.get("mineru_candidate_ids") or []),
        "mineru_bbox_link_count": int(row.get("mineru_bbox_link_count") or 0),
        "evidence_tier": "source_structure_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": blockers,
        "non_strict_reason": blockers,
    }


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(row.get("structure_type") or "") for row in rows)
    by_status = Counter(str(row.get("alignment_status") or "") for row in rows)
    by_method = Counter(str(row.get("alignment_method") or "") for row in rows)
    by_class = Counter(str(row.get("classification") or "") for row in rows)
    text_rows = sum(1 for row in rows if row.get("candidate_text"))
    aligned = sum(1 for row in rows if row.get("alignment_status") == "aligned")
    page_recovered = sum(1 for row in rows if row.get("page") is not None)
    source_hash_linked = sum(1 for row in rows if row.get("sourceContentHash"))
    source_span_ready = sum(1 for row in rows if row.get("source_span_candidate_ready"))
    return {
        "totalRows": len(rows),
        "textRows": text_rows,
        "canonicalAlignedRows": aligned,
        "pageRecoveredRows": page_recovered,
        "sourceContentHashLinkedRows": source_hash_linked,
        "sourceSpanCandidateReadyRows": source_span_ready,
        "mineruLayoutLinkedRows": sum(1 for row in rows if row.get("mineru_layout_link_status") == "linked"),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "byStructureType": dict(by_type),
        "byAlignmentStatus": dict(by_status),
        "byAlignmentMethod": dict(by_method),
        "byClassification": dict(by_class),
        "alignmentSuccessRate": round(aligned / text_rows, 6) if text_rows else 0.0,
        "pageRecoveryRate": round(page_recovered / text_rows, 6) if text_rows else 0.0,
        "sourceSpanCandidateReadyRate": round(source_span_ready / text_rows, 6) if text_rows else 0.0,
    }


def _strict_blocker_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(str(value) for value in list(row.get("strict_blockers") or []))
    return dict(counter)


def build_tex_structure_candidate_alignment_audit(
    *,
    input_report: str | Path = DEFAULT_ARXIV_SOURCE_TEX_AVAILABILITY_REPORT,
    parsed_root: str | Path = DEFAULT_PARSED_ROOT,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build a report-only alignment audit for TeX structure candidates."""

    input_path = Path(str(input_report)).expanduser()
    parsed_root_path = Path(str(parsed_root)).expanduser()
    source_report = _read_json(input_path)
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    source_rows = [
        dict(row)
        for row in list(source_report.get("structureRows") or [])
        if isinstance(row, dict) and (not allowed or str(row.get("paper_id") or "") in allowed)
    ]
    paper_ids_seen = sorted({str(row.get("paper_id") or "") for row in source_rows if row.get("paper_id")})
    paper_context: dict[str, tuple[dict[str, Any], str, list[dict[str, int]], str, str]] = {
        paper_id: _paper_inputs(parsed_root_path, paper_id) for paper_id in paper_ids_seen
    }
    aligned_rows: list[dict[str, Any]] = []
    for row in source_rows:
        paper_id = str(row.get("paper_id") or "")
        _manifest, markdown_text, ranges, source_hash, source_hash_source = paper_context.get(
            paper_id,
            ({}, "", [], "", "unavailable"),
        )
        aligned_rows.append(
            _aligned_row(
                row=row,
                markdown_text=markdown_text,
                ranges=ranges,
                source_hash=source_hash,
                source_hash_source=source_hash_source,
            )
        )

    counts = _counts(aligned_rows)
    papers = []
    for paper_id in paper_ids_seen:
        _manifest, markdown_text, ranges, source_hash, source_hash_source = paper_context[paper_id]
        paper_rows = [row for row in aligned_rows if row.get("paper_id") == paper_id]
        papers.append(
            {
                "paper_id": paper_id,
                "status": "ok" if markdown_text and paper_rows else "degraded",
                "canonical_document_present": bool(markdown_text),
                "pageBoundaryCount": len(ranges),
                "sourceContentHashAvailable": bool(source_hash),
                "sourceContentHashSource": source_hash_source,
                "sourceContentHash": source_hash or None,
                "counts": _counts(paper_rows),
            }
        )

    status = "ok" if aligned_rows else "blocked"
    decision = "source_structure_candidates_ready_for_policy_review" if counts["sourceSpanCandidateReadyRows"] else "blocked_no_source_span_candidates"
    return {
        "schema": TEX_STRUCTURE_CANDIDATE_ALIGNMENT_AUDIT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "inputReport": str(input_path),
            "parsedRoot": str(parsed_root_path),
            "paperIds": requested,
        },
        "counts": counts,
        "gate": {
            "auditReady": bool(aligned_rows),
            "sourceSpanCandidatesReady": bool(counts["sourceSpanCandidateReadyRows"]),
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "recommendedNextTranche": "tex_sectionspan_candidate_report"
            if counts["sourceSpanCandidateReadyRows"]
            else "tex_source_or_parsed_artifact_repair",
        },
        "policy": {
            "reportOnly": True,
            "sourceStructureCandidateOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "strictBlockerSummary": _strict_blocker_summary(aligned_rows),
        "papers": papers,
        "candidates": aligned_rows,
        "warnings": [
            "all_outputs_are_source_structure_candidate_only",
            "sourceContentHash_plus_chars_page_does_not_create_strict_evidence",
            "tex_offsets_are_not_canonical_source_spans",
            "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
            "table_candidates_remain_non_strict_without_cell_row_column_bbox_provenance",
            "figure_candidates_remain_non_strict_without_verified_figure_region_links",
            "equation_candidates_remain_non_strict_without_equation_text_and_semantics_review",
        ],
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "strictBlockerSummary", "warnings")
        if key in report
    }


def render_tex_structure_candidate_alignment_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# TeX Structure Candidate Alignment Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Total rows: `{int(counts.get('totalRows') or 0)}`",
        f"- Text rows: `{int(counts.get('textRows') or 0)}`",
        f"- Canonical aligned rows: `{int(counts.get('canonicalAlignedRows') or 0)}`",
        f"- Page recovered rows: `{int(counts.get('pageRecoveredRows') or 0)}`",
        f"- Source-span candidate-ready rows: `{int(counts.get('sourceSpanCandidateReadyRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This audit is report-only. It does not create strict evidence, route parsers, write canonical parsed artifacts, mutate DB state, scan vault content, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By structure type: `{json.dumps(counts.get('byStructureType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By alignment status: `{json.dumps(counts.get('byAlignmentStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By alignment method: `{json.dumps(counts.get('byAlignmentMethod') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By classification: `{json.dumps(counts.get('byClassification') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Papers",
        "",
    ]
    for paper in list(report.get("papers") or []):
        paper_counts = dict(paper.get("counts") or {})
        lines.extend(
            [
                f"### {paper.get('paper_id', '')}",
                "",
                f"- Status: `{paper.get('status', '')}`",
                f"- Page boundaries: `{int(paper.get('pageBoundaryCount') or 0)}`",
                f"- Source hash available: `{bool(paper.get('sourceContentHashAvailable'))}`",
                f"- Total rows: `{int(paper_counts.get('totalRows') or 0)}`",
                f"- Source-span candidate-ready rows: `{int(paper_counts.get('sourceSpanCandidateReadyRows') or 0)}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_tex_structure_candidate_alignment_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "tex-structure-candidate-alignment-report.json"
    summary_path = root / "tex-structure-candidate-alignment-summary.json"
    markdown_path = root / "tex-structure-candidate-alignment-audit.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_tex_structure_candidate_alignment_audit_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only TeX structure candidate alignment audit.")
    parser.add_argument("--input-report", default=str(DEFAULT_ARXIV_SOURCE_TEX_AVAILABILITY_REPORT))
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT))
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to a paper id; can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_tex_structure_candidate_alignment_audit(
        input_report=args.input_report,
        parsed_root=args.parsed_root,
        paper_ids=args.paper_id or None,
    )
    paths = write_tex_structure_candidate_alignment_audit_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TEX_STRUCTURE_CANDIDATE_ALIGNMENT_AUDIT_SCHEMA_ID",
    "build_tex_structure_candidate_alignment_audit",
    "render_tex_structure_candidate_alignment_audit_markdown",
    "write_tex_structure_candidate_alignment_audit_reports",
]
