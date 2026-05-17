"""Report-only MinerU candidate source-alignment audit helpers.

This module inspects existing MinerU normalizer candidates and existing
PyMuPDF/canonical parsed artifacts.  It attempts to align candidate text back
to canonical generated markdown spans, recover a page from PyMuPDF page
markers, and attach a source-content hash when one is available from existing
manifest metadata or the manifest-declared source PDF.

It does not parse PDFs, mutate SQLite, write canonical paper artifacts,
reindex, reembed, change parser routing, or promote candidates into strict
evidence.
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


MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID = "knowledge-hub.paper.mineru-source-alignment-audit.v1"

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
_ALIGNABLE_TYPES = {
    "section_candidate",
    "table_candidate",
    "equation_candidate",
    "figure_caption_candidate",
}


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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


def _align_candidate(markdown_text: str, candidate_text: str) -> _Alignment:
    text = _clean_text(candidate_text)
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


def _strict_blockers(
    *,
    candidate_type: str,
    alignment: _Alignment,
    page: int | None,
    source_hash: str,
) -> list[str]:
    blockers = ["runtime_promotion_disabled_for_tranche"]
    if alignment.status != "aligned":
        blockers.append("text_alignment_not_available")
    if alignment.status == "ambiguous":
        blockers.append("ambiguous_match")
    if alignment.method not in {"exact"}:
        blockers.append("fuzzy_or_ambiguous_alignment")
    if alignment.chars_start is None or alignment.chars_end is None:
        blockers.append("missing_chars_start_end")
    if page is None:
        blockers.append("missing_page")
    if not source_hash:
        blockers.append("missing_source_content_hash")
    if candidate_type == "table_candidate":
        blockers.append("table_cell_provenance_missing")
    if candidate_type == "equation_candidate":
        blockers.append("equation_alignment_missing" if alignment.status != "aligned" else "equation_quote_candidate_only")
    if candidate_type == "figure_caption_candidate":
        blockers.append("figure_region_link_incomplete")
    blockers.append("markdown_offsets_are_generated_not_original_pdf_offsets")
    return list(dict.fromkeys(blockers))


def _classification(
    *,
    candidate_type: str,
    alignment: _Alignment,
    page: int | None,
    source_hash: str,
) -> str:
    if candidate_type not in _ALIGNABLE_TYPES:
        return "raw_candidate_only"
    if alignment.status != "aligned":
        return "blocked" if alignment.status in {"blocked", "failed", "ambiguous"} else "raw_candidate_only"
    if not source_hash or alignment.chars_start is None or alignment.chars_end is None:
        return "text_aligned_non_strict"
    if page is None:
        return "source_span_aligned_non_strict"
    if candidate_type == "section_candidate" and alignment.method == "exact":
        return "potential_strict_candidate"
    return "page_recovered_non_strict"


def _aligned_candidate(
    *,
    candidate: dict[str, Any],
    markdown_text: str,
    page_ranges: list[dict[str, int]],
    source_hash: str,
    source_hash_source: str,
) -> dict[str, Any]:
    candidate_type = str(candidate.get("candidate_type") or "")
    text = _clean_text(candidate.get("text") or candidate.get("caption"))
    alignment = _align_candidate(markdown_text, text) if candidate_type in _ALIGNABLE_TYPES else _Alignment(
        "blocked",
        "none",
        None,
        None,
        0.0,
        "candidate_type_not_text_span_aligned",
    )
    page = _page_for_offset(alignment.chars_start, page_ranges)
    blockers = _strict_blockers(
        candidate_type=candidate_type,
        alignment=alignment,
        page=page,
        source_hash=source_hash,
    )
    classification = _classification(
        candidate_type=candidate_type,
        alignment=alignment,
        page=page,
        source_hash=source_hash,
    )
    source_locator = _markdown_span_locator(markdown_text, alignment.chars_start, alignment.chars_end)
    payload = {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_type": candidate_type,
        "source_parser": "mineru",
        "paper_id": str(candidate.get("paper_id") or ""),
        "candidate_text": text,
        "alignment_status": alignment.status,
        "alignment_method": alignment.method,
        "alignment_reason": alignment.reason,
        "chars_start": alignment.chars_start,
        "chars_end": alignment.chars_end,
        "page": page,
        "sourceContentHash": source_hash or None,
        "sourceContentHashSource": source_hash_source,
        "confidence": round(alignment.confidence, 3),
        "classification": classification,
        "strict_eligible": False,
        "strict_requirements_met": classification == "potential_strict_candidate",
        "strict_blockers": blockers,
        "non_strict_reason": blockers,
        "evidence_tier": "source_alignment_candidate_only",
        "citation_grade": False,
        "source_span_locator": source_locator,
        "mineruCandidate": {
            "markdown_locator": candidate.get("markdown_locator") or {},
            "layout_element_ids": list(candidate.get("layout_element_ids") or []),
            "bbox": candidate.get("bbox"),
            "page": candidate.get("page"),
            "confidence": candidate.get("confidence"),
            "link_reason": candidate.get("link_reason"),
            "non_strict_reason": list(candidate.get("non_strict_reason") or []),
            "tableCellCitationGrade": bool(candidate.get("tableCellCitationGrade", False)),
        },
    }
    if candidate_type == "table_candidate":
        payload["tableCellCitationGrade"] = False
        payload["tableRowsPresent"] = bool(candidate.get("tableRows"))
    return payload


def _counts(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(item.get("candidate_type") or "") for item in candidates)
    by_status = Counter(str(item.get("alignment_status") or "") for item in candidates)
    by_method = Counter(str(item.get("alignment_method") or "") for item in candidates)
    by_class = Counter(str(item.get("classification") or "") for item in candidates)
    alignable = sum(1 for item in candidates if item.get("candidate_type") in _ALIGNABLE_TYPES)
    aligned = sum(1 for item in candidates if item.get("alignment_status") == "aligned")
    page_recovered = sum(1 for item in candidates if item.get("page") is not None)
    source_hash_linked = sum(
        1
        for item in candidates
        if item.get("candidate_type") in _ALIGNABLE_TYPES and item.get("sourceContentHash")
    )
    strict_requirement_complete = sum(1 for item in candidates if item.get("strict_requirements_met"))
    strict_eligible = sum(1 for item in candidates if item.get("strict_eligible"))
    return {
        "totalCandidates": len(candidates),
        "alignableCandidates": alignable,
        "alignedCandidates": aligned,
        "pageRecoveredCandidates": page_recovered,
        "sourceContentHashLinkedCandidates": source_hash_linked,
        "strictRequirementCompleteCandidates": strict_requirement_complete,
        "strictEligibleCandidates": strict_eligible,
        "citationGradeCandidates": sum(1 for item in candidates if item.get("citation_grade")),
        "byType": dict(by_type),
        "byAlignmentStatus": dict(by_status),
        "byAlignmentMethod": dict(by_method),
        "byClassification": dict(by_class),
        "alignmentSuccessRate": round(aligned / alignable, 6) if alignable else 0.0,
        "pageRecoveryRate": round(page_recovered / alignable, 6) if alignable else 0.0,
        "sourceContentHashLinkageRate": round(source_hash_linked / alignable, 6) if alignable else 0.0,
    }


def _strict_blocker_summary(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in candidates:
        counter.update(str(value) for value in list(item.get("strict_blockers") or []))
    return dict(counter)


def _paper_payload(
    *,
    paper_id: str,
    pymupdf_parsed_dir: Path,
    normalizer_report_path: Path,
) -> dict[str, Any]:
    manifest = _read_json(pymupdf_parsed_dir / "manifest.json")
    markdown_path = pymupdf_parsed_dir / "document.md"
    try:
        markdown_text = markdown_path.read_text(encoding="utf-8")
    except Exception:
        document = _read_json(pymupdf_parsed_dir / "document.json")
        markdown_text = str(document.get("markdown_text") or "")
    source_hash, source_hash_source = _source_hash_from_manifest(manifest)
    normalizer_report = _read_json(normalizer_report_path)
    raw_candidates = [dict(item) for item in list(normalizer_report.get("candidates") or []) if isinstance(item, dict)]
    ranges = _page_ranges(markdown_text)
    aligned = [
        _aligned_candidate(
            candidate=item,
            markdown_text=markdown_text,
            page_ranges=ranges,
            source_hash=source_hash,
            source_hash_source=source_hash_source,
        )
        for item in raw_candidates
    ]
    counts = _counts(aligned)
    return {
        "paperId": paper_id,
        "status": "ok" if markdown_text and raw_candidates else "degraded",
        "input": {
            "pymupdfParsedDir": str(pymupdf_parsed_dir),
            "pymupdfManifestPath": str(pymupdf_parsed_dir / "manifest.json"),
            "pymupdfDocumentMarkdownPath": str(markdown_path),
            "mineruNormalizerCandidatesPath": str(normalizer_report_path),
        },
        "source": {
            "sourceContentHashAvailable": bool(source_hash),
            "sourceContentHashSource": source_hash_source,
            "sourceContentHash": source_hash or None,
            "pageBoundaryCount": len(ranges),
        },
        "counts": counts,
        "strictBlockerSummary": _strict_blocker_summary(aligned),
        "candidates": aligned,
    }


def build_mineru_source_alignment_audit(
    *,
    input_root: str | Path,
    paper_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Build a report-only source-alignment audit for MinerU candidates."""

    root = Path(str(input_root)).expanduser()
    papers = []
    for paper_id in paper_ids:
        token = str(paper_id)
        papers.append(
            _paper_payload(
                paper_id=token,
                pymupdf_parsed_dir=root / "parser-runs" / "pymupdf" / "parsed" / token,
                normalizer_report_path=root / "normalizer" / token / "mineru-normalizer-candidates.json",
            )
        )
    all_candidates = [candidate for paper in papers for candidate in list(paper.get("candidates") or [])]
    counts = _counts(all_candidates)
    status = "ok" if papers and all(paper.get("status") == "ok" for paper in papers) else "degraded"
    return {
        "schema": MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "input": {
            "inputRoot": str(root),
            "paperIds": list(paper_ids),
        },
        "counts": {
            **counts,
            "paperCount": len(papers),
            "papersWithPageRecovery": sum(1 for paper in papers if int((paper.get("counts") or {}).get("pageRecoveredCandidates") or 0) > 0),
            "papersWithSourceContentHash": sum(1 for paper in papers if (paper.get("source") or {}).get("sourceContentHashAvailable")),
        },
        "policy": {
            "allOutputsNonStrict": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
        },
        "strictBlockerSummary": _strict_blocker_summary(all_candidates),
        "papers": [
            {
                key: paper[key]
                for key in ("paperId", "status", "input", "source", "counts", "strictBlockerSummary")
                if key in paper
            }
            for paper in papers
        ],
        "candidates": all_candidates,
        "warnings": [
            "all_outputs_are_source_alignment_candidate_only",
            "sourceContentHash_and_chars_start_end_do_not_promote_MinerU_candidates_to_runtime_strict_evidence",
            "document_md_offsets_are_canonical_generated_markdown_offsets_not_original_pdf_byte_offsets",
            "table_candidates_remain_non_strict_without_cell_row_column_bbox_provenance",
            "figure_caption_candidates_remain_non_strict_until_caption_to_figure_region_links_are_verified",
        ],
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
            "policy",
            "strictBlockerSummary",
            "papers",
            "warnings",
        )
        if key in report
    }


def render_mineru_source_alignment_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    blockers = dict(report.get("strictBlockerSummary") or {})
    lines = [
        "# MinerU Source Alignment Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Papers: `{int(counts.get('paperCount') or 0)}`",
        f"- Candidates: `{int(counts.get('totalCandidates') or 0)}` total, `{int(counts.get('alignableCandidates') or 0)}` alignable",
        f"- Alignment success rate: `{float(counts.get('alignmentSuccessRate') or 0.0):.2%}`",
        f"- Page recovery rate: `{float(counts.get('pageRecoveryRate') or 0.0):.2%}`",
        f"- Source-content-hash linkage rate: `{float(counts.get('sourceContentHashLinkageRate') or 0.0):.2%}`",
        f"- Potential strict candidates: `{int(counts.get('strictRequirementCompleteCandidates') or 0)}`",
        f"- Strict eligible candidates: `{int(counts.get('strictEligibleCandidates') or 0)}`",
        "",
        "## Evidence Tier",
        "",
        "All outputs remain `source_alignment_candidate_only`. This audit does not create runtime strict evidence.",
        "A recovered `sourceContentHash` plus canonical generated-markdown `chars:start-end` is a promotion input, not promotion itself.",
        "Tables remain non-strict because row/column/cell bbox provenance is not established; figure captions remain non-strict until caption-to-region links are verified.",
        "",
        "## Papers",
        "",
    ]
    for paper in list(report.get("papers") or []):
        paper_counts = dict(paper.get("counts") or {})
        source = dict(paper.get("source") or {})
        lines.extend(
            [
                f"### `{paper.get('paperId', '')}`",
                "",
                f"- Alignable: `{int(paper_counts.get('alignableCandidates') or 0)}`",
                f"- Aligned: `{int(paper_counts.get('alignedCandidates') or 0)}`",
                f"- Page recovered: `{int(paper_counts.get('pageRecoveredCandidates') or 0)}`",
                f"- Source hash available: `{bool(source.get('sourceContentHashAvailable'))}`",
                f"- Strict eligible: `{int(paper_counts.get('strictEligibleCandidates') or 0)}`",
                "",
            ]
        )
    lines.extend(["## Strict Blockers", ""])
    for reason, count in sorted(blockers.items(), key=lambda item: (-int(item[1]), item[0])):
        lines.append(f"- `{reason}`: `{count}`")
    lines.append("")
    return "\n".join(lines)


def write_mineru_source_alignment_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "mineru-source-alignment-report.json"
    summary_path = root / "mineru-source-alignment-summary.json"
    markdown_path = root / "mineru-source-alignment-audit.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_mineru_source_alignment_audit_markdown(report), encoding="utf-8")
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only MinerU source-alignment audit.")
    parser.add_argument("--input-root", required=True, help="Root of a layout-parser-pilot report with parser-runs/ and normalizer/.")
    parser.add_argument("--paper-id", action="append", default=[], help="Paper id to inspect. Repeatable.")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print the summary payload as JSON.")
    args = parser.parse_args(argv)

    paper_ids = [str(item) for item in args.paper_id if str(item).strip()]
    if not paper_ids:
        raise SystemExit("--paper-id is required at least once")
    report = build_mineru_source_alignment_audit(input_root=args.input_root, paper_ids=paper_ids)
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_mineru_source_alignment_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID",
    "build_mineru_source_alignment_audit",
    "render_mineru_source_alignment_audit_markdown",
    "write_mineru_source_alignment_reports",
]
