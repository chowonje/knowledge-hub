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


def build_structured_candidate_summary(
    *,
    sectionspan_report: str | Path,
    figure_caption_report: str | Path,
    equation_quote_report: str | Path,
    table_region_report: str | Path,
) -> dict[str, Any]:
    """Build a consolidated report over all current structured candidate layers."""

    inputs = {
        "sectionspan": str(Path(str(sectionspan_report)).expanduser()),
        "figure_caption": str(Path(str(figure_caption_report)).expanduser()),
        "equation_quote": str(Path(str(equation_quote_report)).expanduser()),
        "table_region": str(Path(str(table_region_report)).expanduser()),
    }
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
            "recommendedNextTranche": "complex_paper_qa_eval_design",
            "mainBlockers": [
                "equation_quote_alignment_missing",
                "table_cell_row_column_bbox_provenance_missing",
                "figure_region_link_unverified",
                "generated_markdown_offsets_are_not_original_pdf_offsets",
            ],
        },
        "warnings": [
            "summary_rows_are_not_runtime_evidence",
            "candidate_layers_do_not_create_strict_citations",
            "parser_routing_and_answer_integration_remain_out_of_scope",
        ],
        "layers": layers,
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
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_structured_candidate_summary(
        sectionspan_report=args.sectionspan_report,
        figure_caption_report=args.figure_caption_report,
        equation_quote_report=args.equation_quote_report,
        table_region_report=args.table_region_report,
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
