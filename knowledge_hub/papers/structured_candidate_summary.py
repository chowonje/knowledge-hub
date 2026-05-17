"""Report-only structured paper candidate summary helpers.

This module summarizes report-only SectionSpan/FigureCaption/EquationQuote/
TableRegion candidate layers.  It does not read PDFs, mutate SQLite, change
parser routing, write canonical parsed artifacts, or promote candidates into
strict evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID = "knowledge-hub.paper.structured-candidate-summary.v1"
SECTIONSPAN_PDF_OFFSET_RECOVERY_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-recovery-review-pack.v1"
)
FIGURE_CAPTION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID = (
    "knowledge-hub.paper.figure-caption-pdf-offset-feasibility.v1"
)

_LAYER_CONFIG = {
    "sectionspan": {
        "countKey": "sectionSpanCandidates",
        "alignedKey": "sectionSpanCandidates",
        "blockedKey": "heldOutCandidates",
        "tier": "sectionspan_candidate_only",
    },
    "figure_caption": {
        "countKey": "figureCaptionCandidates",
        "alignedKey": "alignedCaptionSpanCandidates",
        "blockedReadinessPrefixes": ("blocked_",),
        "tier": "figure_caption_candidate_only",
    },
    "equation_quote": {
        "countKey": "equationQuoteCandidates",
        "alignedKey": "alignedEquationQuoteCandidates",
        "blockedReadinessPrefixes": ("blocked_",),
        "tier": "equation_quote_candidate_only",
    },
    "table_region": {
        "countKey": "tableRegionCandidates",
        "alignedKey": "alignedTableCaptionCandidates",
        "blockedReadinessPrefixes": ("blocked_",),
        "tier": "table_region_candidate_only",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
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


def _blocked_from_readiness(counts: dict[str, Any], prefixes: tuple[str, ...]) -> int:
    by_readiness = dict(counts.get("byReadiness") or {})
    total = 0
    for key, value in by_readiness.items():
        if any(str(key).startswith(prefix) for prefix in prefixes):
            total += _safe_int(value)
    return total


def _layer_summary(layer: str, path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = _LAYER_CONFIG[layer]
    counts = dict(payload.get("counts") or {})
    count_key = str(config["countKey"])
    aligned_key = str(config["alignedKey"])
    blocked = _safe_int(counts.get(str(config.get("blockedKey") or "")))
    prefixes = config.get("blockedReadinessPrefixes")
    if isinstance(prefixes, tuple):
        blocked = _blocked_from_readiness(counts, prefixes)
    blocker_summary = dict(counts.get("strictBlockerSummary") or {})
    held_out = dict(counts.get("heldOutByReason") or {})
    primary_blockers = dict(Counter({**blocker_summary, **held_out}).most_common(8))
    return {
        "layer": layer,
        "path": str(Path(str(path)).expanduser()),
        "schema": str(payload.get("schema") or ""),
        "status": str(payload.get("status") or ""),
        "evidenceTier": str(config["tier"]),
        "candidateCount": _safe_int(counts.get(count_key)),
        "alignedCount": _safe_int(counts.get(aligned_key)),
        "blockedOrHeldOutCount": blocked,
        "strictEligibleCandidates": _safe_int(counts.get("strictEligibleCandidates")),
        "citationGradeCandidates": _safe_int(counts.get("citationGradeCandidates")),
        "byPaper": dict(counts.get("byPaper") or {}),
        "byReadiness": dict(counts.get("byReadiness") or {}),
        "primaryBlockers": primary_blockers,
    }


def _sectionspan_pdf_offset_supplement(path: str | Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    counts = dict(payload.get("counts") or {})
    schema = str(payload.get("schema") or "")
    provided = bool(path)
    ready = (
        provided
        and schema == SECTIONSPAN_PDF_OFFSET_RECOVERY_REVIEW_PACK_SCHEMA_ID
        and str(payload.get("status") or "") == "review_pack_ready"
        and _safe_int(counts.get("unsafeUpstreamFlagCount")) == 0
    )
    return {
        "supplement": "sectionspan_pdf_offset_recovery_review_pack",
        "path": str(Path(str(path)).expanduser()) if path else "",
        "schema": schema,
        "status": str(payload.get("status") or ("not_provided" if not provided else "")),
        "evidenceTier": "sectionspan_pdf_offset_recovery_review_card_only",
        "reviewCardRows": _safe_int(counts.get("reviewCardRows")),
        "readyForHumanReviewRows": _safe_int(counts.get("readyForHumanReviewRows")),
        "readyForRegionReviewRows": 0,
        "heldOutRows": _safe_int(counts.get("heldOutRows")),
        "pageAgreementRows": _safe_int(counts.get("pageAgreementRows")),
        "sourceHashAgreementRows": _safe_int(counts.get("sourceHashAgreementRows")),
        "strictEligibleRows": _safe_int(counts.get("strictEligibleRows")),
        "citationGradeRows": _safe_int(counts.get("citationGradeRows")),
        "runtimeEvidenceRows": _safe_int(counts.get("runtimeEvidenceRows")),
        "readyForReview": ready,
        "byMatchMethod": dict(counts.get("byMatchMethod") or {}),
        "byReviewStatus": dict(counts.get("byReviewStatus") or {}),
    }


def _figure_caption_pdf_offset_supplement(path: str | Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    counts = dict(payload.get("counts") or {})
    schema = str(payload.get("schema") or "")
    provided = bool(path)
    ready = (
        provided
        and schema == FIGURE_CAPTION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID
        and str(payload.get("status") or "") == "feasibility_complete"
        and _safe_int(counts.get("schemaViolationCount")) == 0
    )
    return {
        "supplement": "figure_caption_pdf_offset_feasibility",
        "path": str(Path(str(path)).expanduser()) if path else "",
        "schema": schema,
        "status": str(payload.get("status") or ("not_provided" if not provided else "")),
        "evidenceTier": "figure_caption_pdf_offset_feasibility_only",
        "reviewCardRows": _safe_int(counts.get("feasibilityRows")),
        "readyForHumanReviewRows": 0,
        "readyForRegionReviewRows": _safe_int(counts.get("originalPdfOffsetRecoveredRows")),
        "heldOutRows": _safe_int(counts.get("blockedRows")),
        "pageAgreementRows": _safe_int(counts.get("pageAgreementRows")),
        "sourceHashAgreementRows": _safe_int(counts.get("sourceHashAgreementRows")),
        "strictEligibleRows": _safe_int(counts.get("strictEligibleRows")),
        "citationGradeRows": _safe_int(counts.get("citationGradeRows")),
        "runtimeEvidenceRows": _safe_int(counts.get("runtimeEvidenceRows")),
        "readyForReview": ready,
        "byMatchMethod": {
            "exact": _safe_int(counts.get("exactRecoveredRows")),
            "normalized_whitespace_case": _safe_int(counts.get("normalizedRecoveredRows")),
        },
        "byReviewStatus": dict(counts.get("byFeasibilityStatus") or {}),
    }


def _main_blockers(*, supplements: list[dict[str, Any]]) -> list[str]:
    sectionspan = next(
        (
            item
            for item in supplements
            if item.get("supplement") == "sectionspan_pdf_offset_recovery_review_pack"
        ),
        {},
    )
    figure_caption = next(
        (
            item
            for item in supplements
            if item.get("supplement") == "figure_caption_pdf_offset_feasibility"
        ),
        {},
    )
    blockers = [
        "equation_quote_alignment_missing",
        "table_cell_row_column_bbox_provenance_missing",
        "figure_region_link_unverified",
    ]
    if sectionspan.get("readyForReview"):
        blockers.append("sectionspan_pdf_offsets_require_human_review_before_strict_promotion")
        blockers.append("non_sectionspan_layers_lack_original_pdf_offsets")
    else:
        blockers.append("generated_markdown_offsets_are_not_original_pdf_offsets")
    if _safe_int(figure_caption.get("readyForRegionReviewRows")) > 0:
        blockers.append("figure_caption_pdf_offsets_require_region_link_review")
    return blockers


def build_structured_candidate_summary(
    *,
    sectionspan_report: str | Path,
    figure_caption_report: str | Path,
    equation_quote_report: str | Path,
    table_region_report: str | Path,
    sectionspan_pdf_offset_review_pack_report: str | Path | None = None,
    figure_caption_pdf_offset_feasibility_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build a consolidated report over all current structured candidate layers."""

    inputs = {
        "sectionspan": str(Path(str(sectionspan_report)).expanduser()),
        "figure_caption": str(Path(str(figure_caption_report)).expanduser()),
        "equation_quote": str(Path(str(equation_quote_report)).expanduser()),
        "table_region": str(Path(str(table_region_report)).expanduser()),
    }
    if sectionspan_pdf_offset_review_pack_report:
        inputs["sectionspan_pdf_offset_review_pack"] = str(
            Path(str(sectionspan_pdf_offset_review_pack_report)).expanduser()
        )
    if figure_caption_pdf_offset_feasibility_report:
        inputs["figure_caption_pdf_offset_feasibility"] = str(
            Path(str(figure_caption_pdf_offset_feasibility_report)).expanduser()
        )
    payloads = {
        "sectionspan": _read_json(sectionspan_report),
        "figure_caption": _read_json(figure_caption_report),
        "equation_quote": _read_json(equation_quote_report),
        "table_region": _read_json(table_region_report),
    }
    layers = [
        _layer_summary(layer, inputs[layer], payload)
        for layer, payload in payloads.items()
    ]
    by_layer = {item["layer"]: item["candidateCount"] for item in layers}
    aligned_by_layer = {item["layer"]: item["alignedCount"] for item in layers}
    blocked_by_layer = {item["layer"]: item["blockedOrHeldOutCount"] for item in layers}
    strict_total = sum(_safe_int(item.get("strictEligibleCandidates")) for item in layers)
    citation_total = sum(_safe_int(item.get("citationGradeCandidates")) for item in layers)
    offset_payload = _read_json(sectionspan_pdf_offset_review_pack_report) if sectionspan_pdf_offset_review_pack_report else {}
    figure_offset_payload = (
        _read_json(figure_caption_pdf_offset_feasibility_report)
        if figure_caption_pdf_offset_feasibility_report
        else {}
    )
    supplements = [
        _sectionspan_pdf_offset_supplement(sectionspan_pdf_offset_review_pack_report, offset_payload),
        _figure_caption_pdf_offset_supplement(
            figure_caption_pdf_offset_feasibility_report, figure_offset_payload
        ),
    ]
    return {
        "schema": STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID,
        "status": "ok" if any(item["candidateCount"] for item in layers) else "empty",
        "generatedAt": _now(),
        "inputs": inputs,
        "counts": {
            "layerCount": len(layers),
            "totalCandidates": sum(by_layer.values()),
            "byLayer": by_layer,
            "alignedByLayer": aligned_by_layer,
            "blockedOrHeldOutByLayer": blocked_by_layer,
            "strictEligibleCandidates": strict_total,
            "citationGradeCandidates": citation_total,
            "runtimeEvidenceCandidates": 0,
            "sourceAlignmentSupplementCount": len(supplements),
            "sectionspanOriginalPdfOffsetReviewCards": _safe_int(supplements[0].get("reviewCardRows")),
            "sectionspanOriginalPdfOffsetReadyForReviewRows": _safe_int(
                supplements[0].get("readyForHumanReviewRows")
            ),
            "sectionspanOriginalPdfOffsetHeldOutRows": _safe_int(supplements[0].get("heldOutRows")),
            "sectionspanOriginalPdfOffsetPageAgreementRows": _safe_int(supplements[0].get("pageAgreementRows")),
            "sectionspanOriginalPdfOffsetSourceHashAgreementRows": _safe_int(
                supplements[0].get("sourceHashAgreementRows")
            ),
            "figureCaptionOriginalPdfOffsetFeasibilityRows": _safe_int(supplements[1].get("reviewCardRows")),
            "figureCaptionOriginalPdfOffsetRecoveredRows": _safe_int(
                supplements[1].get("readyForRegionReviewRows")
            ),
            "figureCaptionOriginalPdfOffsetBlockedRows": _safe_int(supplements[1].get("heldOutRows")),
            "figureCaptionOriginalPdfOffsetPageAgreementRows": _safe_int(supplements[1].get("pageAgreementRows")),
            "figureCaptionOriginalPdfOffsetSourceHashAgreementRows": _safe_int(
                supplements[1].get("sourceHashAgreementRows")
            ),
        },
        "policy": {
            "allCandidatesNonStrict": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "releaseCandidateAssessment": {
            "candidateLayerReviewReady": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "recommendedNextTranche": "candidate_layer_review_gate_refresh",
            "mainBlockers": _main_blockers(supplements=supplements),
        },
        "warnings": [
            "summary_rows_are_not_runtime_evidence",
            "candidate_layers_do_not_create_strict_citations",
            "parser_routing_and_answer_integration_remain_out_of_scope",
            "source_alignment_supplements_do_not_promote_candidates_to_runtime_evidence",
        ],
        "layers": layers,
        "sourceAlignmentSupplements": supplements,
    }


def render_structured_candidate_summary_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    assessment = dict(report.get("releaseCandidateAssessment") or {})
    lines = [
        "# Structured Candidate Layer Summary",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Total candidates: `{int(counts.get('totalCandidates') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleCandidates') or 0)}`",
        f"- Citation-grade: `{int(counts.get('citationGradeCandidates') or 0)}`",
        f"- Runtime evidence candidates: `{int(counts.get('runtimeEvidenceCandidates') or 0)}`",
        f"- SectionSpan original-PDF-offset review cards: `{int(counts.get('sectionspanOriginalPdfOffsetReviewCards') or 0)}`",
        f"- SectionSpan original-PDF-offset ready for review: `{int(counts.get('sectionspanOriginalPdfOffsetReadyForReviewRows') or 0)}`",
        f"- FigureCaption original-PDF-offset recovered: `{int(counts.get('figureCaptionOriginalPdfOffsetRecoveredRows') or 0)}`",
        f"- Candidate-layer review ready: `{bool(assessment.get('candidateLayerReviewReady'))}`",
        f"- Strict evidence ready: `{bool(assessment.get('strictEvidenceReady'))}`",
        f"- Parser routing ready: `{bool(assessment.get('parserRoutingReady'))}`",
        "",
        "## Evidence Tier",
        "",
        "All layer outputs remain non-strict candidates. This summary does not create runtime citations.",
        "",
        "## Layer Counts",
        "",
    ]
    for layer in list(report.get("layers") or []):
        lines.extend(
            [
                f"### `{layer.get('layer', '')}`",
                "",
                f"- Candidates: `{layer.get('candidateCount')}`",
                f"- Aligned: `{layer.get('alignedCount')}`",
                f"- Blocked or held out: `{layer.get('blockedOrHeldOutCount')}`",
                f"- Evidence tier: `{layer.get('evidenceTier')}`",
                f"- Primary blockers: `{json.dumps(layer.get('primaryBlockers') or {}, ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(["## Source Alignment Supplements", ""])
    for supplement in list(report.get("sourceAlignmentSupplements") or []):
        lines.extend(
            [
                f"### `{supplement.get('supplement', '')}`",
                "",
                f"- Status: `{supplement.get('status', '')}`",
                f"- Ready for review: `{bool(supplement.get('readyForReview'))}`",
                f"- Review cards: `{supplement.get('reviewCardRows')}`",
                f"- Ready rows: `{supplement.get('readyForHumanReviewRows')}`",
                f"- Held out: `{supplement.get('heldOutRows')}`",
                f"- Page agreement: `{supplement.get('pageAgreementRows')}`",
                f"- Source-hash agreement: `{supplement.get('sourceHashAgreementRows')}`",
                f"- Match methods: `{json.dumps(supplement.get('byMatchMethod') or {}, ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Next Tranche",
            "",
            f"- Recommended: `{assessment.get('recommendedNextTranche', '')}`",
            f"- Main blockers: `{json.dumps(assessment.get('mainBlockers') or [], ensure_ascii=False)}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_structured_candidate_summary_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "structured-candidate-summary.json"
    markdown_path = root / "structured-candidate-summary.md"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_structured_candidate_summary_markdown(report), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only structured candidate summary.")
    parser.add_argument("--sectionspan-report", required=True, help="Path to sectionspan-candidates.json.")
    parser.add_argument("--figure-caption-report", required=True, help="Path to figure-caption-candidates.json.")
    parser.add_argument("--equation-quote-report", required=True, help="Path to equation-quote-candidates.json.")
    parser.add_argument("--table-region-report", required=True, help="Path to table-region-candidates.json.")
    parser.add_argument(
        "--sectionspan-pdf-offset-review-pack-report",
        default="",
        help="Optional path to SectionSpan PDF offset recovery review cards.",
    )
    parser.add_argument(
        "--figure-caption-pdf-offset-feasibility-report",
        default="",
        help="Optional path to FigureCaption original PDF offset feasibility report.",
    )
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_structured_candidate_summary(
        sectionspan_report=args.sectionspan_report,
        figure_caption_report=args.figure_caption_report,
        equation_quote_report=args.equation_quote_report,
        table_region_report=args.table_region_report,
        sectionspan_pdf_offset_review_pack_report=args.sectionspan_pdf_offset_review_pack_report or None,
        figure_caption_pdf_offset_feasibility_report=args.figure_caption_pdf_offset_feasibility_report or None,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_structured_candidate_summary_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID",
    "build_structured_candidate_summary",
    "render_structured_candidate_summary_markdown",
    "write_structured_candidate_summary_reports",
]
