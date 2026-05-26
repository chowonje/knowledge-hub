"""Report-only MinerU output normalizer audit helpers.

This module inspects existing MinerU parsed artifacts and emits candidate-only
links between generated Markdown blocks and layout elements.  It does not parse
PDFs, write canonical paper artifacts, mutate SQLite, reindex, reembed, or
promote candidates into strict evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any


MINERU_NORMALIZER_AUDIT_SCHEMA_ID = "knowledge-hub.paper.mineru-normalizer-audit.v1"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_FIGURE_CAPTION_RE = re.compile(r"^Figure\s+\d+\s*:\s+.+", re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r"^Table\s+\d+\s*:\s+.+", re.IGNORECASE)

_BASE_NON_STRICT_REASONS = (
    "no_source_content_hash_strict_span",
    "no_original_chars_start_end",
    "markdown_offsets_not_original_source_chars",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _compact_text(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", _clean_text(value).casefold())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _line_starts(markdown_text: str) -> list[int]:
    starts = [0]
    offset = 0
    for line in markdown_text.splitlines(keepends=True):
        offset += len(line)
        starts.append(offset)
    return starts


def _line_number(starts: list[int], offset: int) -> int:
    return max(1, bisect_right(starts, max(0, offset)))


def _markdown_locator(
    *,
    starts: list[int],
    start: int,
    end: int,
    block_type: str,
) -> dict[str, Any]:
    return {
        "path": "document.md",
        "locatorKind": "generated_markdown",
        "blockType": block_type,
        "lineStart": _line_number(starts, start),
        "lineEnd": _line_number(starts, max(start, end - 1)),
        "markdownChars": {"start": max(0, int(start)), "end": max(0, int(end))},
    }


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    coords: list[float] = []
    for item in value[:4]:
        number = _safe_float(item)
        if number is None:
            return None
        coords.append(number)
    return coords


def _bbox_union(items: list[dict[str, Any]]) -> list[float] | None:
    boxes = [_bbox(item.get("bbox")) for item in items]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _page_from_items(items: list[dict[str, Any]]) -> int | None:
    pages = {_safe_int(item.get("page"), default=-1) for item in items if item.get("page") is not None}
    pages = {page for page in pages if page >= 0}
    if len(pages) == 1:
        page = next(iter(pages))
        return page + 1 if page == 0 else page
    return None


def _element_type(element: dict[str, Any]) -> str:
    return _clean_text(element.get("type") or element.get("kind") or element.get("content_type")).casefold()


def _element_text(element: dict[str, Any]) -> str:
    return _clean_text(element.get("text") or element.get("markdown") or element.get("content"))


def _layout_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    layout: list[dict[str, Any]] = []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue
        item = dict(element)
        item["layout_id"] = _clean_text(element.get("id")) or f"mineru:element:{index}"
        item["layout_index"] = index
        item["layout_type"] = _element_type(element) or "element"
        item["layout_text"] = _element_text(element)
        item["layout_compact_text"] = _compact_text(item["layout_text"])
        layout.append(item)
    return layout


def _find_text_element(layout: list[dict[str, Any]], text: str) -> tuple[list[dict[str, Any]], str, float]:
    compact = _compact_text(text)
    if not compact:
        return [], "no_text_to_match", 0.0
    text_items = [item for item in layout if item.get("layout_text")]
    exact = [item for item in text_items if item.get("layout_compact_text") == compact]
    if exact:
        return exact[:1], "exact_markdown_text_to_layout_text", 0.78
    contains = [
        item
        for item in text_items
        if compact in str(item.get("layout_compact_text") or "")
        or str(item.get("layout_compact_text") or "") in compact
    ]
    if contains:
        return contains[:1], "partial_markdown_text_to_layout_text", 0.52
    return [], "no_layout_text_match", 0.0


def _is_table_like(item: dict[str, Any]) -> bool:
    return "table" in str(item.get("layout_type") or "")


def _is_equation_like(item: dict[str, Any]) -> bool:
    element_type = str(item.get("layout_type") or "")
    return "equation" in element_type or "formula" in element_type


def _is_figure_like(item: dict[str, Any]) -> bool:
    element_type = str(item.get("layout_type") or "")
    return any(token in element_type for token in ("image", "figure", "chart"))


def _layout_groups(
    layout: list[dict[str, Any]],
    predicate: Any,
    *,
    max_index_gap: int = 25,
) -> list[list[dict[str, Any]]]:
    selected = [item for item in layout if predicate(item)]
    selected.sort(key=lambda item: int(item.get("layout_index") or 0))
    groups: list[list[dict[str, Any]]] = []
    for item in selected:
        if not groups:
            groups.append([item])
            continue
        previous = groups[-1][-1]
        gap = int(item.get("layout_index") or 0) - int(previous.get("layout_index") or 0)
        if gap <= max_index_gap:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def _figure_layout_groups(layout: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group primary image/chart blocks without merging later appendix figures.

    MinerU often emits duplicate `image` or `chart` nodes for the same visual,
    followed by body/caption nodes.  A broad index-gap group can merge several
    nearby appendix figures, so figure-caption linking uses primary visual nodes
    and merges only duplicate bboxes.
    """

    primaries = [
        item
        for item in layout
        if str(item.get("layout_type") or "") in {"image", "chart"}
    ]
    primaries.sort(key=lambda item: int(item.get("layout_index") or 0))
    groups: list[list[dict[str, Any]]] = []
    for item in primaries:
        box = _bbox(item.get("bbox"))
        if groups and box is not None:
            previous_boxes = [_bbox(existing.get("bbox")) for existing in groups[-1]]
            if any(previous == box for previous in previous_boxes):
                groups[-1].append(item)
                continue
        groups.append([item])
    return groups


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, Any]]] = []
        self._current_row: list[dict[str, Any]] | None = None
        self._current_cell: dict[str, Any] | None = None
        self._cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._current_row = []
            return
        if tag not in {"td", "th"}:
            return
        attr_map = {key.lower(): value for key, value in attrs}
        self._current_cell = {
            "text": "",
            "isHeader": tag == "th",
            "rowspan": max(1, _safe_int(attr_map.get("rowspan"), default=1)),
            "colspan": max(1, _safe_int(attr_map.get("colspan"), default=1)),
        }
        self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_cell is not None:
            self._current_cell["text"] = _clean_text(" ".join(self._cell_parts))
            if self._current_row is not None:
                self._current_row.append(dict(self._current_cell))
            self._current_cell = None
            self._cell_parts = []
            return
        if tag == "tr" and self._current_row is not None:
            self.rows.append(list(self._current_row))
            self._current_row = None


def _parse_table(html: str) -> list[list[dict[str, Any]]]:
    parser = _TableParser()
    try:
        parser.feed(html)
    except Exception:
        return []
    return parser.rows


def _markdown_blocks(markdown_text: str) -> dict[str, list[dict[str, Any]]]:
    starts = _line_starts(markdown_text)
    headings: list[dict[str, Any]] = []
    for match in _HEADING_RE.finditer(markdown_text):
        level = len(match.group(1))
        text = _clean_text(match.group(2))
        headings.append(
            {
                "type": "heading",
                "level": level,
                "text": text,
                "start": match.start(),
                "end": match.end(),
                "locator": _markdown_locator(starts=starts, start=match.start(), end=match.end(), block_type="heading"),
            }
        )

    lines = markdown_text.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    tables: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    equations: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        line_start = line_offsets[i]
        if _TABLE_CAPTION_RE.match(stripped):
            cursor = i + 1
            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1
            if cursor < len(lines) and lines[cursor].lstrip().startswith("<table"):
                table_start = line_offsets[cursor]
                table_end_index = cursor
                table_parts = []
                while table_end_index < len(lines):
                    table_parts.append(lines[table_end_index])
                    if "</table>" in lines[table_end_index].casefold():
                        break
                    table_end_index += 1
                table_html = "".join(table_parts)
                block_end = line_offsets[min(table_end_index, len(lines) - 1)] + len(lines[min(table_end_index, len(lines) - 1)])
                tables.append(
                    {
                        "type": "table",
                        "caption": stripped,
                        "html": table_html,
                        "rows": _parse_table(table_html),
                        "start": line_start,
                        "end": block_end,
                        "locator": _markdown_locator(starts=starts, start=line_start, end=block_end, block_type="table"),
                    }
                )
                i = table_end_index + 1
                continue
        if _FIGURE_CAPTION_RE.match(stripped):
            image_line = ""
            block_start = line_start
            if i > 0 and lines[i - 1].strip().startswith("![]("):
                block_start = line_offsets[i - 1]
                image_line = lines[i - 1].strip()
            block_end = line_start + len(raw)
            figures.append(
                {
                    "type": "figure_caption",
                    "caption": stripped,
                    "imageRef": image_line,
                    "start": block_start,
                    "end": block_end,
                    "locator": _markdown_locator(starts=starts, start=block_start, end=block_end, block_type="figure_caption"),
                }
            )
        if stripped == "$$":
            block_start = line_start
            cursor = i + 1
            content_parts: list[str] = []
            while cursor < len(lines):
                if lines[cursor].strip() == "$$":
                    block_end = line_offsets[cursor] + len(lines[cursor])
                    equations.append(
                        {
                            "type": "equation",
                            "text": _clean_text("".join(content_parts)),
                            "start": block_start,
                            "end": block_end,
                            "locator": _markdown_locator(starts=starts, start=block_start, end=block_end, block_type="equation"),
                        }
                    )
                    i = cursor + 1
                    break
                content_parts.append(lines[cursor])
                cursor += 1
            else:
                i += 1
            continue
        i += 1

    return {
        "headings": headings,
        "tables": tables,
        "figures": figures,
        "equations": equations,
    }


def _linked_fields(items: list[dict[str, Any]]) -> tuple[list[str], list[float] | None, int | None, int | None]:
    linked_ids = [str(item.get("layout_id")) for item in items if item.get("layout_id")]
    bbox = _bbox_union(items)
    page = _page_from_items(items)
    order_values = [
        _safe_int(item.get("reading_order"), default=-1)
        for item in items
        if item.get("reading_order") is not None
    ]
    order = min((value for value in order_values if value >= 0), default=None)
    return linked_ids, bbox, page, order


def _non_strict_reasons(*, page: int | None, bbox: list[float] | None, extra: list[str] | None = None) -> list[str]:
    reasons = list(_BASE_NON_STRICT_REASONS)
    if page is None:
        reasons.append("page_not_recovered")
    if bbox is not None:
        reasons.append("bbox_only_non_strict")
    if extra:
        reasons.extend(extra)
    return list(dict.fromkeys(reasons))


def _candidate(
    *,
    candidate_id: str,
    candidate_type: str,
    paper_id: str,
    text: str,
    markdown_locator: dict[str, Any] | None,
    linked_items: list[dict[str, Any]],
    confidence: float,
    link_reason: str,
    non_strict_extra: list[str] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    linked_ids, bbox, page, order = _linked_fields(linked_items)
    payload = {
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "source_parser": "mineru",
        "paper_id": paper_id,
        "text": _clean_text(text),
        "markdown_locator": markdown_locator or {},
        "layout_element_ids": linked_ids,
        "bbox": bbox,
        "page": page,
        "reading_order_index": order,
        "confidence": round(max(0.0, min(1.0, float(confidence))), 3),
        "link_reason": link_reason,
        "non_strict_reason": _non_strict_reasons(page=page, bbox=bbox, extra=non_strict_extra),
        "evidence_tier": "candidate_only",
        "strict": False,
        "citation_grade": False,
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


def _build_candidates(*, paper_id: str, markdown_text: str, elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    layout = _layout_elements(elements)
    blocks = _markdown_blocks(markdown_text)
    candidates: list[dict[str, Any]] = []

    for index, block in enumerate(blocks["headings"], start=1):
        linked, reason, confidence = _find_text_element(layout, str(block.get("text") or ""))
        candidates.append(
            _candidate(
                candidate_id=f"{paper_id}:section:{index:04d}",
                candidate_type="section_candidate",
                paper_id=paper_id,
                text=str(block.get("text") or ""),
                markdown_locator=dict(block.get("locator") or {}),
                linked_items=linked,
                confidence=confidence,
                link_reason=reason,
                non_strict_extra=["heading_candidate_from_generated_markdown"],
                extra_fields={"headingLevel": int(block.get("level") or 0)},
            )
        )

    table_groups = _layout_groups(layout, _is_table_like)
    for index, block in enumerate(blocks["tables"], start=1):
        linked = table_groups[index - 1] if index <= len(table_groups) else []
        reason = "ordinal_table_like_layout_match_without_page" if linked else "no_table_like_layout_match"
        candidates.append(
            _candidate(
                candidate_id=f"{paper_id}:table:{index:04d}",
                candidate_type="table_candidate",
                paper_id=paper_id,
                text=str(block.get("caption") or ""),
                markdown_locator=dict(block.get("locator") or {}),
                linked_items=linked,
                confidence=0.45 if linked else 0.0,
                link_reason=reason,
                non_strict_extra=[
                    "table_region_or_caption_only",
                    "no_cell_bbox_or_row_column_provenance",
                    "not_table_cell_citation_grade",
                ],
                extra_fields={
                    "caption": str(block.get("caption") or ""),
                    "tableRows": list(block.get("rows") or []),
                    "tableCellCitationGrade": False,
                    "tableHtmlPresent": bool(block.get("html")),
                },
            )
        )

    equation_groups = _layout_groups(layout, _is_equation_like)
    for index, block in enumerate(blocks["equations"], start=1):
        linked = equation_groups[index - 1] if index <= len(equation_groups) else []
        reason = "ordinal_equation_like_layout_match_without_page" if linked else "no_equation_like_layout_match"
        candidates.append(
            _candidate(
                candidate_id=f"{paper_id}:equation:{index:04d}",
                candidate_type="equation_candidate",
                paper_id=paper_id,
                text=str(block.get("text") or ""),
                markdown_locator=dict(block.get("locator") or {}),
                linked_items=linked,
                confidence=0.5 if linked else 0.0,
                link_reason=reason,
                non_strict_extra=["equation_quote_candidate_only"],
            )
        )

    figure_groups = _figure_layout_groups(layout)
    for index, block in enumerate(blocks["figures"], start=1):
        linked = figure_groups[index - 1] if index <= len(figure_groups) else []
        reason = "ordinal_figure_like_layout_match_without_page" if linked else "no_figure_like_layout_match"
        candidates.append(
            _candidate(
                candidate_id=f"{paper_id}:figure-caption:{index:04d}",
                candidate_type="figure_caption_candidate",
                paper_id=paper_id,
                text=str(block.get("caption") or ""),
                markdown_locator=dict(block.get("locator") or {}),
                linked_items=linked,
                confidence=0.45 if linked else 0.0,
                link_reason=reason,
                non_strict_extra=["caption_to_figure_link_incomplete"],
                extra_fields={"caption": str(block.get("caption") or ""), "imageRef": str(block.get("imageRef") or "")},
            )
        )

    reading_items = [item for item in layout if item.get("reading_order") is not None]
    for index, item in enumerate(reading_items, start=1):
        text = item.get("layout_text") or item.get("layout_type") or ""
        candidates.append(
            _candidate(
                candidate_id=f"{paper_id}:reading-order:{index:04d}",
                candidate_type="reading_order_candidate",
                paper_id=paper_id,
                text=str(text),
                markdown_locator={},
                linked_items=[item],
                confidence=0.3,
                link_reason="layout_element_has_parser_reading_order",
                non_strict_extra=["reading_order_layout_candidate_only"],
                extra_fields={"layoutType": str(item.get("layout_type") or "")},
            )
        )

    return candidates


def _candidate_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    by_type = Counter(str(item.get("candidate_type") or "") for item in candidates)
    return {
        "totalCandidates": len(candidates),
        "sectionCandidates": by_type.get("section_candidate", 0),
        "tableCandidates": by_type.get("table_candidate", 0),
        "equationCandidates": by_type.get("equation_candidate", 0),
        "figureCaptionCandidates": by_type.get("figure_caption_candidate", 0),
        "readingOrderCandidates": by_type.get("reading_order_candidate", 0),
        "linkedCandidates": sum(1 for item in candidates if item.get("layout_element_ids")),
        "citationGradeCandidates": sum(1 for item in candidates if item.get("citation_grade")),
    }


def _layout_counts(elements: list[dict[str, Any]]) -> dict[str, Any]:
    layout = _layout_elements(elements)
    pages = [item.get("page") for item in layout if item.get("page") is not None]
    return {
        "totalElements": len(layout),
        "typeDistribution": dict(Counter(str(item.get("layout_type") or "") for item in layout)),
        "bboxElementCount": sum(1 for item in layout if _bbox(item.get("bbox")) is not None),
        "pageElementCount": len(pages),
        "readingOrderElementCount": sum(1 for item in layout if item.get("reading_order") is not None),
        "nonEmptyTextElementCount": sum(1 for item in layout if item.get("layout_text")),
        "tableLikeElementCount": sum(1 for item in layout if _is_table_like(item)),
        "equationLikeElementCount": sum(1 for item in layout if _is_equation_like(item)),
        "figureLikeElementCount": sum(1 for item in layout if _is_figure_like(item)),
    }


def _markdown_counts(markdown_text: str) -> dict[str, int]:
    blocks = _markdown_blocks(markdown_text)
    return {
        "markdownChars": len(markdown_text),
        "markdownHeadings": len(blocks["headings"]),
        "markdownHtmlTables": len(blocks["tables"]),
        "markdownEquationBlocks": len(blocks["equations"]),
        "markdownFigureCaptions": len(blocks["figures"]),
    }


def build_mineru_normalizer_audit(
    parsed_dir: str | Path,
    *,
    paper_id: str | None = None,
) -> dict[str, Any]:
    """Build a candidate-only MinerU normalization audit from parsed artifacts."""

    root = Path(str(parsed_dir)).expanduser()
    manifest = _read_json(root / "manifest.json")
    document = _read_json(root / "document.json")
    markdown_path = Path(str(manifest.get("markdown_path") or root / "document.md"))
    if not markdown_path.is_absolute():
        markdown_path = root / markdown_path
    try:
        markdown_text = markdown_path.read_text(encoding="utf-8")
    except Exception:
        markdown_text = str(document.get("markdown_text") or "")
    token = _clean_text(paper_id or manifest.get("paper_id") or root.name)
    elements = [dict(item) for item in list(document.get("elements") or []) if isinstance(item, dict)]
    candidates = _build_candidates(paper_id=token, markdown_text=markdown_text, elements=elements)
    candidate_counts = _candidate_counts(candidates)
    layout_counts = _layout_counts(elements)
    markdown_counts = _markdown_counts(markdown_text)
    page_recovered = bool(layout_counts["pageElementCount"])
    summary = {
        "schema": MINERU_NORMALIZER_AUDIT_SCHEMA_ID,
        "status": "ok" if markdown_text and elements else "degraded",
        "generatedAt": _now(),
        "paperId": token,
        "sourceParser": "mineru",
        "input": {
            "parsedDir": str(root),
            "manifestPath": str(root / "manifest.json"),
            "documentJsonPath": str(root / "document.json"),
            "documentMarkdownPath": str(markdown_path),
        },
        "counts": {
            **candidate_counts,
            **markdown_counts,
            **layout_counts,
        },
        "provenance": {
            "allCandidatesNonStrict": True,
            "anyCitationGrade": False,
            "sourceContentHashBackedStrictSpan": False,
            "originalCharsStartEndAvailable": False,
            "pageRecovered": page_recovered,
            "pageRecoveryStatus": "layout_page_fields_present" if page_recovered else "unavailable_no_page_fields",
            "bboxOnlyCandidatesAreNonStrict": True,
            "tableCellCitationGradeAvailable": False,
        },
        "promotionAssessment": {
            "worthThreePaperPilot": True,
            "promotableCandidateTypes": [
                "section_candidate",
                "equation_candidate",
                "figure_caption_candidate",
                "reading_order_candidate",
            ],
            "notCitationGradeYet": [
                "table_candidate",
                "figure_caption_candidate",
                "equation_candidate",
                "section_candidate",
            ],
            "requiredBeforeParserRouting": [
                "recover_page_locators",
                "link_markdown_offsets_to_original_source_spans",
                "establish_table_cell_row_column_bbox_provenance",
                "verify_caption_to_figure_region_links",
            ],
        },
        "warnings": [
            "all_outputs_are_candidate_only",
            "markdown_offsets_are_generated_artifact_offsets_not_original_source_spans",
            "bbox_without_page_is_non_strict",
            "table_region_or_caption_candidates_are_not_table_cell_evidence",
        ],
    }
    return {
        **summary,
        "candidates": candidates,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "paperId",
            "sourceParser",
            "input",
            "counts",
            "provenance",
            "promotionAssessment",
            "warnings",
        )
        if key in report
    }


def render_mineru_normalizer_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    provenance = dict(report.get("provenance") or {})
    assessment = dict(report.get("promotionAssessment") or {})
    linked = int(counts.get("linkedCandidates") or 0)
    total = int(counts.get("totalCandidates") or 0)
    lines = [
        "# MinerU Normalizer Audit",
        "",
        f"- Paper: `{report.get('paperId', '')}`",
        f"- Source parser: `{report.get('sourceParser', '')}`",
        f"- Status: `{report.get('status', '')}`",
        f"- Candidates: `{total}` total, `{linked}` linked to layout elements",
        f"- Page recovered: `{bool(provenance.get('pageRecovered'))}`",
        f"- Citation-grade candidates: `{int(counts.get('citationGradeCandidates') or 0)}`",
        "",
        "## Candidate Counts",
        "",
        f"- Section candidates: `{int(counts.get('sectionCandidates') or 0)}`",
        f"- Table candidates: `{int(counts.get('tableCandidates') or 0)}`",
        f"- Equation candidates: `{int(counts.get('equationCandidates') or 0)}`",
        f"- Figure caption candidates: `{int(counts.get('figureCaptionCandidates') or 0)}`",
        f"- Reading order candidates: `{int(counts.get('readingOrderCandidates') or 0)}`",
        "",
        "## Evidence Tier",
        "",
        "All emitted records are `candidate_only`. They are not strict evidence.",
        "Generated Markdown offsets are not original source `chars:start-end` spans, and bbox-only links remain non-strict when page recovery is unavailable.",
        "Table candidates represent Markdown HTML grids plus table-region/caption links; they are not table-cell citation-grade evidence.",
        "",
        "## Promotion Assessment",
        "",
        f"- Worth a 3-paper pilot: `{bool(assessment.get('worthThreePaperPilot'))}`",
        "- Candidate types worth formalizing later: "
        + ", ".join(f"`{item}`" for item in list(assessment.get("promotableCandidateTypes") or [])),
        "- Required before parser routing: "
        + ", ".join(f"`{item}`" for item in list(assessment.get("requiredBeforeParserRouting") or [])),
        "",
    ]
    return "\n".join(lines)


def write_mineru_normalizer_audit_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    candidates_path = root / "mineru-normalizer-candidates.json"
    summary_path = root / "mineru-normalizer-summary.json"
    markdown_path = root / "mineru-normalizer-audit.md"
    candidates_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_mineru_normalizer_audit_markdown(report), encoding="utf-8")
    return {
        "candidates": str(candidates_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only MinerU normalizer audit from parsed artifacts.")
    parser.add_argument("--parsed-dir", required=True, help="Path to MinerU parsed/<paper_id> artifact directory.")
    parser.add_argument("--paper-id", default="", help="Optional paper id override.")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print the summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_mineru_normalizer_audit(args.parsed_dir, paper_id=args.paper_id or None)
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_mineru_normalizer_audit_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MINERU_NORMALIZER_AUDIT_SCHEMA_ID",
    "build_mineru_normalizer_audit",
    "render_mineru_normalizer_audit_markdown",
    "write_mineru_normalizer_audit_reports",
]
