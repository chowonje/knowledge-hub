"""Report-only TeX equation PDF-region anchor audit.

This helper checks whether TeX equation line-local anchor rows can be narrowed
to candidate PDF text regions using local PyMuPDF block text and bbox metadata.
It is intentionally non-authoritative: PDF regions are diagnostic layout
candidates only, not source spans, strict evidence, or runtime citations.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable
import unicodedata

from knowledge_hub.papers.source_text import source_hash_for_path
from knowledge_hub.papers.tex_equation_line_local_anchor_audit import (
    TEX_EQUATION_LINE_LOCAL_ANCHOR_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_structure_candidate_alignment_audit import DEFAULT_PARSED_ROOT


TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.tex-equation-pdf-region-anchor-audit.v1"
)

DEFAULT_TEX_EQUATION_LINE_LOCAL_ANCHOR_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-18"
    / "tex-equation-line-local-anchor-audit"
    / "tex-equation-line-local-anchor-report.json"
)

_TOKEN_RE = re.compile(r"[a-z]+[a-z0-9]*|[0-9]+(?:\.[0-9]+)?")
_EQUATION_NUMBER_RE = re.compile(r"\((\d{1,3})\)")
_FORMULA_SYMBOL_RE = re.compile(r"[=+\-*/∗×√()|_^{}<>≤≥]")
_SPACE_RE = re.compile(r"\s+")


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


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _clean_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip())


def _source_pdf_from_manifest(manifest: dict[str, Any]) -> str:
    parser_meta = dict(manifest.get("parser_meta") or {})
    for key in ("source_pdf", "extracted_from", "pdf_path", "sourcePath"):
        value = str(parser_meta.get(key) or manifest.get(key) or "").strip()
        if value:
            return value
    return ""


def _block_text(block: dict[str, Any]) -> str:
    if "text" in block:
        return str(block.get("text") or "")
    lines: list[str] = []
    for line in list(block.get("lines") or []):
        spans = list(line.get("spans") or []) if isinstance(line, dict) else []
        text = "".join(str(span.get("text") or "") for span in spans if isinstance(span, dict))
        if text:
            lines.append(text)
    return "\n".join(lines)


def _extract_pdf_blocks(source_pdf: str | Path) -> list[dict[str, Any]]:
    try:
        import fitz  # type: ignore
    except Exception:
        return []
    path = Path(str(source_pdf)).expanduser()
    try:
        document = fitz.open(str(path))
    except Exception:
        return []
    pages: list[dict[str, Any]] = []
    try:
        for page_index in range(int(getattr(document, "page_count", 0) or 0)):
            try:
                page = document.load_page(page_index)
                data = page.get_text("dict")
            except Exception:
                data = {}
            blocks: list[dict[str, Any]] = []
            for block_index, block in enumerate(list(data.get("blocks") or [])):
                if not isinstance(block, dict) or int(block.get("type") or 0) != 0:
                    continue
                text = _block_text(block)
                if not text.strip():
                    continue
                bbox = list(block.get("bbox") or [])
                blocks.append(
                    {
                        "block_index": block_index,
                        "bbox": [float(item) for item in bbox[:4]] if len(bbox) >= 4 else [],
                        "text": text,
                    }
                )
            pages.append({"page": page_index + 1, "blocks": blocks})
    finally:
        try:
            document.close()
        except Exception:
            pass
    return pages


def _normalize_text(value: str) -> str:
    chars: list[str] = []
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "×": " x ",
        "∗": " * ",
        "−": " - ",
        "–": " - ",
        "—": " - ",
        "√": " sqrt ",
        "|": " ",
    }
    for char in str(value or ""):
        expanded = replacements.get(char, char)
        folded = unicodedata.normalize("NFKC", expanded).casefold()
        for item in folded:
            if item.isalnum():
                chars.append(item)
            elif chars and chars[-1] != " ":
                chars.append(" ")
    return _clean_text("".join(chars))


def _tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(_normalize_text(value))


def _term_token(value: str) -> str:
    return _normalize_text(value).replace(" ", "")


def _token_matches_anchor(token: str, anchor: str) -> bool:
    if not token or not anchor:
        return False
    if token == anchor:
        return True
    if len(anchor) < 2:
        return False
    return token.startswith(anchor) or anchor in token


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    tokens = _tokens(text)
    matched: list[str] = []
    for term in terms:
        anchor = _term_token(term)
        if anchor and any(_token_matches_anchor(token, anchor) for token in tokens):
            matched.append(term)
    return list(dict.fromkeys(matched))


def _equation_numbers(text: str) -> list[str]:
    return list(dict.fromkeys(_EQUATION_NUMBER_RE.findall(text)))


def _bbox_union(blocks: list[dict[str, Any]]) -> list[float]:
    bboxes = [list(block.get("bbox") or []) for block in blocks if len(list(block.get("bbox") or [])) >= 4]
    if not bboxes:
        return []
    return [
        round(min(float(bbox[0]) for bbox in bboxes), 3),
        round(min(float(bbox[1]) for bbox in bboxes), 3),
        round(max(float(bbox[2]) for bbox in bboxes), 3),
        round(max(float(bbox[3]) for bbox in bboxes), 3),
    ]


def _formula_score(*, text: str, coverage: float, matched_count: int) -> dict[str, Any]:
    clean = _clean_text(text)
    text_length = len(clean)
    line_count = max(1, str(text or "").count("\n") + 1)
    has_equal = "=" in text
    equation_numbers = _equation_numbers(text)
    symbol_count = len(_FORMULA_SYMBOL_RE.findall(text))
    symbol_ratio = round(symbol_count / max(text_length, 1), 6)
    score = coverage
    if has_equal:
        score += 0.25
    if equation_numbers:
        score += 0.2
    if symbol_ratio >= 0.08:
        score += 0.15
    elif symbol_ratio >= 0.04:
        score += 0.05
    if line_count <= 4:
        score += 0.1
    if text_length > 700:
        score -= 0.4
    elif text_length > 350:
        score -= 0.2
    if matched_count <= 1:
        score -= 0.2
    return {
        "formula_score": round(max(0.0, score), 6),
        "has_equal_sign": has_equal,
        "equation_numbers": equation_numbers,
        "symbol_ratio": symbol_ratio,
        "line_count": line_count,
        "text_length": text_length,
    }


def _block_windows(blocks: list[dict[str, Any]], max_window_size: int = 2) -> list[list[dict[str, Any]]]:
    windows: list[list[dict[str, Any]]] = []
    for start in range(len(blocks)):
        for size in range(1, max_window_size + 1):
            end = start + size
            if end <= len(blocks):
                window = blocks[start:end]
                indexes = [_safe_int(block.get("block_index")) for block in window]
                if indexes and max(indexes) - min(indexes) != len(indexes) - 1:
                    continue
                windows.append(window)
    return windows


def _page_filter(row: dict[str, Any]) -> list[int]:
    pages = [
        _safe_int(detail.get("page_marker"))
        for detail in list(row.get("window_details") or [])
        if isinstance(detail, dict) and _safe_int(detail.get("page_marker")) > 0
    ]
    return sorted(set(pages))


def _source_context(
    *,
    paper_id: str,
    parsed_root: Path,
    pdf_block_loader: Callable[[str | Path], list[dict[str, Any]]],
) -> dict[str, Any]:
    manifest_path = parsed_root / paper_id / "manifest.json"
    manifest = _read_json(manifest_path)
    if not manifest:
        return {"status": "blocked_manifest_missing", "manifestPath": str(manifest_path), "sourcePdfPath": ""}
    source_pdf = _source_pdf_from_manifest(manifest)
    if not source_pdf:
        return {"status": "blocked_source_pdf_unregistered", "manifestPath": str(manifest_path), "sourcePdfPath": ""}
    source_pdf_path = Path(source_pdf).expanduser()
    if not source_pdf_path.exists():
        return {"status": "blocked_source_pdf_missing", "manifestPath": str(manifest_path), "sourcePdfPath": str(source_pdf_path)}
    source_hash = source_hash_for_path(str(source_pdf_path))
    if not source_hash:
        return {"status": "blocked_source_hash_unavailable", "manifestPath": str(manifest_path), "sourcePdfPath": str(source_pdf_path)}
    pages = pdf_block_loader(source_pdf_path)
    page_count = len(pages)
    block_count = sum(len(list(page.get("blocks") or [])) for page in pages if isinstance(page, dict))
    if page_count <= 0 or block_count <= 0:
        return {
            "status": "blocked_pdf_block_extraction_unavailable",
            "manifestPath": str(manifest_path),
            "sourcePdfPath": str(source_pdf_path),
            "sourceContentHash": source_hash,
            "pages": [],
        }
    return {
        "status": "ok",
        "manifestPath": str(manifest_path),
        "sourcePdfPath": str(source_pdf_path),
        "sourceContentHash": source_hash,
        "pages": pages,
        "pageCount": page_count,
        "blockCount": block_count,
    }


def _pdf_region_candidates(row: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
    terms = [str(item) for item in list(row.get("normalized_terms") or []) if str(item).strip()]
    pages = set(_page_filter(row))
    candidates: list[dict[str, Any]] = []
    for page in list(context.get("pages") or []):
        page_number = _safe_int(page.get("page"))
        if pages and page_number not in pages:
            continue
        blocks = [dict(block) for block in list(page.get("blocks") or []) if isinstance(block, dict)]
        for window in _block_windows(blocks):
            text = "\n".join(_block_text(block) for block in window)
            matched = _matched_terms(text, terms)
            coverage = round(len(matched) / max(len(terms), 1), 6)
            features = _formula_score(text=text, coverage=coverage, matched_count=len(matched))
            formula_like = bool(features["has_equal_sign"] or features["equation_numbers"])
            if coverage < 0.75 or not formula_like or _safe_float(features["formula_score"]) < 1.0:
                continue
            candidates.append(
                {
                    "candidate_index": len(candidates) + 1,
                    "page": page_number,
                    "block_indexes": [_safe_int(block.get("block_index")) for block in window],
                    "bbox": _bbox_union(window),
                    "window_size": len(window),
                    "matched_terms": matched,
                    "coverage": coverage,
                    "formula_score": features["formula_score"],
                    "has_equal_sign": features["has_equal_sign"],
                    "equation_numbers": features["equation_numbers"],
                    "symbol_ratio": features["symbol_ratio"],
                    "line_count": features["line_count"],
                    "text_length": features["text_length"],
                    "text_preview": _clean_text(text[:500]),
                }
            )
    candidates.sort(
        key=lambda item: (
            -_safe_float(item.get("formula_score")),
            -_safe_float(item.get("coverage")),
            _safe_int(item.get("page")),
            list(item.get("bbox") or [0, 0, 0, 0])[1] if item.get("bbox") else 0,
            list(item.get("bbox") or [0, 0, 0, 0])[0] if item.get("bbox") else 0,
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["rank"] = index
    return candidates[:8]


def _select_region(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, bool]:
    if not candidates:
        return None, False
    best = candidates[0]
    ties = [
        candidate
        for candidate in candidates
        if abs(_safe_float(candidate.get("formula_score")) - _safe_float(best.get("formula_score"))) <= 0.05
    ]
    if len(ties) == 1:
        return best, True
    return best, False


def _row(index: int, source_row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    candidate_text = _clean_text(source_row.get("candidate_text"))
    source_status = str(context.get("status") or "blocked")
    normalized_window_count = _safe_int(source_row.get("normalized_window_count"))
    terms = [str(item) for item in list(source_row.get("normalized_terms") or []) if str(item).strip()]
    line_status = str(source_row.get("line_local_anchor_status") or "")
    page_markers = _page_filter(source_row)
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    unique = False
    status = "blocked"
    method = "none"
    confidence = 0.0
    failure_reason = ""

    if source_status != "ok":
        status = source_status
        failure_reason = source_status
    elif not candidate_text:
        status = "empty_equation_text"
        failure_reason = "empty_equation_text"
    elif normalized_window_count <= 0:
        status = "blocked_no_line_local_normalized_window"
        failure_reason = "line_local_normalized_window_missing"
    elif len(terms) < 2:
        status = "insufficient_normalized_terms"
        failure_reason = "insufficient_normalized_terms"
    else:
        candidates = _pdf_region_candidates(source_row, context)
        selected, unique = _select_region(candidates)
        if selected and unique:
            if line_status.startswith("ambiguous"):
                status = "pdf_region_resolves_line_local_ambiguity_candidate_only"
            else:
                status = "unique_pdf_region_anchor_candidate_only"
            method = "formula_like_pdf_block_window"
            confidence = min(0.78, 0.5 + _safe_float(selected.get("coverage")) * 0.2 + _safe_float(selected.get("formula_score")) * 0.05)
        elif candidates:
            status = "ambiguous_pdf_region_anchor_candidate_only"
            method = "formula_like_pdf_block_window"
            confidence = 0.35
            failure_reason = "pdf_region_anchor_ambiguous"
        else:
            status = "no_pdf_region_anchor_candidate"
            failure_reason = "no_formula_like_pdf_block_window"

    selected_region = selected or {}
    source_hash = str(context.get("sourceContentHash") or "")
    input_hash = str(source_row.get("sourceContentHash") or "")
    blockers = list(
        dict.fromkeys(
            [
                "tex_equation_pdf_region_anchor_audit_only",
                "pdf_region_bbox_is_not_source_span",
                "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
                "equation_semantics_not_interpreted",
                "equation_region_link_unverified_for_runtime",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_later_explicit_tranche",
                *[str(item) for item in list(source_row.get("strict_blockers") or [])],
            ]
        )
    )
    return {
        "pdf_region_anchor_id": f"tex-equation-pdf-region-anchor:{index:04d}",
        "source_line_local_anchor_id": str(source_row.get("anchor_id") or ""),
        "source_design_id": str(source_row.get("source_design_id") or ""),
        "source_candidate_id": str(source_row.get("source_candidate_id") or ""),
        "paper_id": str(source_row.get("paper_id") or ""),
        "candidate_type": "tex_equation_pdf_region_anchor_candidate",
        "source_parser": "arxiv_tex+pymupdf_pdf_blocks",
        "source_file": str(source_row.get("source_file") or ""),
        "equation_environment": str(source_row.get("equation_environment") or ""),
        "candidate_text": candidate_text,
        "normalized_terms": terms,
        "normalized_window_count": normalized_window_count,
        "line_local_anchor_status": line_status,
        "line_local_anchor_method": str(source_row.get("line_local_anchor_method") or ""),
        "canonical_page_markers": page_markers,
        "source_pdf_path": str(context.get("sourcePdfPath") or ""),
        "source_manifest_path": str(context.get("manifestPath") or ""),
        "sourceContentHash": source_hash,
        "input_sourceContentHash": input_hash,
        "source_hash_agrees_with_input": bool(source_hash and input_hash and source_hash == input_hash),
        "pdf_region_anchor_status": status,
        "pdf_region_anchor_method": method,
        "pdf_region_candidate_count": len(candidates),
        "pdf_region_candidates": candidates,
        "selected_pdf_region": {
            "page": selected_region.get("page"),
            "bbox": selected_region.get("bbox") or [],
            "block_indexes": selected_region.get("block_indexes") or [],
            "matched_terms": selected_region.get("matched_terms") or [],
            "coverage": selected_region.get("coverage", 0.0),
            "formula_score": selected_region.get("formula_score", 0.0),
            "equation_numbers": selected_region.get("equation_numbers") or [],
            "text_preview": selected_region.get("text_preview", ""),
        },
        "pdf_region_anchor_unique": bool(selected and unique),
        "line_local_ambiguity_resolved_by_pdf_region": bool(selected and unique and line_status.startswith("ambiguous")),
        "feasibility_failure_reason": failure_reason,
        "chars_start": None,
        "chars_end": None,
        "page": None,
        "bbox": selected_region.get("bbox") or [],
        "source_span_created": False,
        "equation_semantics_interpreted": False,
        "equation_region_verified": False,
        "evidence_tier": "tex_equation_pdf_region_anchor_candidate_only",
        "confidence": round(confidence, 6),
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": blockers,
        "non_strict_reason": [
            "pdf_region_anchor_rows_are_not_evidence",
            "pdf_region_bbox_is_diagnostic_not_provenance",
            "later_explicit_promotion_tranche_required",
        ],
    }


def _counts(rows: list[dict[str, Any]], schema_violations: list[str]) -> dict[str, Any]:
    by_status = Counter(str(row.get("pdf_region_anchor_status") or "") for row in rows)
    by_paper = Counter(str(row.get("paper_id") or "") for row in rows)
    return {
        "lineLocalAnchorRows": len(rows),
        "normalizedWindowRows": sum(1 for row in rows if _safe_int(row.get("normalized_window_count")) > 0),
        "pdfRegionCandidateRows": sum(1 for row in rows if _safe_int(row.get("pdf_region_candidate_count")) > 0),
        "uniquePdfRegionAnchorRows": sum(1 for row in rows if bool(row.get("pdf_region_anchor_unique"))),
        "pdfRegionResolvedAmbiguousRows": sum(
            1 for row in rows if bool(row.get("line_local_ambiguity_resolved_by_pdf_region"))
        ),
        "ambiguousPdfRegionAnchorRows": int(by_status.get("ambiguous_pdf_region_anchor_candidate_only", 0)),
        "failedPdfRegionAnchorRows": sum(
            count
            for status, count in by_status.items()
            if status
            not in {
                "unique_pdf_region_anchor_candidate_only",
                "pdf_region_resolves_line_local_ambiguity_candidate_only",
                "ambiguous_pdf_region_anchor_candidate_only",
            }
        ),
        "sourceSpanCreatedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(by_paper),
        "byPdfRegionAnchorStatus": dict(by_status),
    }


def build_tex_equation_pdf_region_anchor_audit(
    *,
    line_local_anchor_report: str | Path = DEFAULT_TEX_EQUATION_LINE_LOCAL_ANCHOR_REPORT,
    parsed_root: str | Path = DEFAULT_PARSED_ROOT,
    paper_ids: list[str] | None = None,
    pdf_block_loader: Callable[[str | Path], list[dict[str, Any]]] = _extract_pdf_blocks,
) -> dict[str, Any]:
    input_path = Path(str(line_local_anchor_report)).expanduser()
    parsed_root_path = Path(str(parsed_root)).expanduser()
    payload = _read_json(input_path)
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    schema = str(payload.get("schema") or "")
    schema_violations = [] if schema == TEX_EQUATION_LINE_LOCAL_ANCHOR_AUDIT_SCHEMA_ID else [
        "tex_equation_line_local_anchor_audit_schema_mismatch"
    ]
    source_rows = [
        dict(row)
        for row in list(payload.get("rows") or [])
        if isinstance(row, dict) and (not allowed or str(row.get("paper_id") or "") in allowed)
    ]
    contexts: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    if not schema_violations:
        for index, source_row in enumerate(source_rows, start=1):
            paper_id = str(source_row.get("paper_id") or "")
            if paper_id not in contexts:
                contexts[paper_id] = _source_context(
                    paper_id=paper_id,
                    parsed_root=parsed_root_path,
                    pdf_block_loader=pdf_block_loader,
                )
            rows.append(_row(index, source_row, contexts[paper_id]))
    counts = _counts(rows, schema_violations)
    return {
        "schema": TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
        "status": "ok" if rows and not schema_violations else "blocked",
        "generatedAt": _now(),
        "input": {
            "lineLocalAnchorReportPath": str(input_path),
            "lineLocalAnchorReportSchema": schema,
            "parsedRoot": str(parsed_root_path),
            "paperIds": requested,
        },
        "counts": counts,
        "gate": {
            "pdfRegionAnchorAuditReady": bool(rows) and not schema_violations,
            "sourceSpanCreationReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "equation_pdf_region_anchor_audited" if rows and not schema_violations else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "tex_equation_quote_candidate_v2_design" if counts["pdfRegionResolvedAmbiguousRows"] else "tex_pdf_sync_or_equation_rendering_anchor_audit",
        },
        "policy": {
            "reportOnly": True,
            "pdfRegionAnchorAuditOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
            "equationInterpretationAllowed": False,
        },
        "warnings": [
            "pdf_region_bboxes_are_diagnostic_not_source_spans",
            "equation_semantics_are_not_interpreted",
            "pdf_region_candidates_do_not_create_runtime_evidence",
            "equation_candidates_remain_non_strict_until_an_explicit_promotion_tranche",
        ],
        "rows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "input", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_tex_equation_pdf_region_anchor_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# TeX Equation PDF Region Anchor Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Line-local anchor rows: `{int(counts.get('lineLocalAnchorRows') or 0)}`",
        f"- Normalized-window rows: `{int(counts.get('normalizedWindowRows') or 0)}`",
        f"- PDF-region candidate rows: `{int(counts.get('pdfRegionCandidateRows') or 0)}`",
        f"- Unique PDF-region anchor rows: `{int(counts.get('uniquePdfRegionAnchorRows') or 0)}`",
        f"- Ambiguous rows resolved by PDF region: `{int(counts.get('pdfRegionResolvedAmbiguousRows') or 0)}`",
        f"- Ambiguous PDF-region anchor rows: `{int(counts.get('ambiguousPdfRegionAnchorRows') or 0)}`",
        f"- Failed PDF-region anchor rows: `{int(counts.get('failedPdfRegionAnchorRows') or 0)}`",
        f"- Source spans created: `{int(counts.get('sourceSpanCreatedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Policy",
        "",
        "This report is a PDF-region anchor audit only. Bboxes and block windows are diagnostic candidates, not source spans or evidence.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By PDF-region anchor status: `{json.dumps(counts.get('byPdfRegionAnchorStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_tex_equation_pdf_region_anchor_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "tex-equation-pdf-region-anchor-report.json"
    summary_path = root / "tex-equation-pdf-region-anchor-summary.json"
    markdown_path = root / "tex-equation-pdf-region-anchor-audit.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_tex_equation_pdf_region_anchor_audit_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only TeX equation PDF-region anchor audit.")
    parser.add_argument("--line-local-anchor-report", default=str(DEFAULT_TEX_EQUATION_LINE_LOCAL_ANCHOR_REPORT))
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT))
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to a paper id; can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_tex_equation_pdf_region_anchor_audit(
        line_local_anchor_report=args.line_local_anchor_report,
        parsed_root=args.parsed_root,
        paper_ids=args.paper_id or None,
    )
    paths = write_tex_equation_pdf_region_anchor_audit_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID",
    "build_tex_equation_pdf_region_anchor_audit",
    "render_tex_equation_pdf_region_anchor_audit_markdown",
    "write_tex_equation_pdf_region_anchor_audit_reports",
]
