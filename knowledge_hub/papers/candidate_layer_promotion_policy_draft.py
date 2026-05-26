"""Report-only candidate-layer promotion policy draft.

This helper turns the current structured candidate reports and feasibility
audits into an explicit promotion-policy draft.  It does not change runtime
policy: every track remains non-strict, parser routing and answer integration
stay blocked, and any real promotion requires a later explicit tranche.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_PROMOTION_POLICY_DRAFT_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-promotion-policy-draft.v1"
)
STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID = "knowledge-hub.paper.structured-candidate-summary.v1"
CANDIDATE_LAYER_REVIEW_GATE_SCHEMA_ID = "knowledge-hub.paper.candidate-layer-review-gate.v1"
CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID = "knowledge-hub.paper.candidate-layer-blocker-backlog.v1"
SOURCE_SPAN_OFFSET_AUTHORITY_AUDIT_SCHEMA_ID = "knowledge-hub.paper.source-span-offset-authority-audit.v1"
EQUATION_ALIGNMENT_FEASIBILITY_AUDIT_SCHEMA_ID = "knowledge-hub.paper.equation-alignment-feasibility-audit.v1"
TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-provenance-feasibility-audit.v1"
)
FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID = "knowledge-hub.paper.figure-region-link-feasibility-audit.v1"


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


def _schema_violations(payloads: dict[str, dict[str, Any]]) -> list[str]:
    expected = {
        "structuredSummary": STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID,
        "candidateLayerReviewGate": CANDIDATE_LAYER_REVIEW_GATE_SCHEMA_ID,
        "candidateLayerBlockerBacklog": CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID,
        "sourceSpanOffsetAuthorityAudit": SOURCE_SPAN_OFFSET_AUTHORITY_AUDIT_SCHEMA_ID,
        "equationAlignmentFeasibilityAudit": EQUATION_ALIGNMENT_FEASIBILITY_AUDIT_SCHEMA_ID,
        "tableCellProvenanceFeasibilityAudit": TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID,
        "figureRegionLinkFeasibilityAudit": FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID,
    }
    violations: list[str] = []
    for key, schema_id in expected.items():
        if payloads.get(key, {}).get("schema") != schema_id:
            violations.append(f"{key}_schema_mismatch")
    return violations


def _unsafe_upstream_flags(payloads: dict[str, dict[str, Any]]) -> list[str]:
    unsafe: list[str] = []
    for name, payload in payloads.items():
        counts = dict(payload.get("counts") or {})
        policy = dict(payload.get("policy") or {})
        gate = dict(payload.get("gate") or {})
        for count_key in (
            "strictEligibleCandidates",
            "citationGradeCandidates",
            "runtimeEvidenceCandidates",
            "strictEvidenceCreated",
            "currentRuntimeAnswerableQuestions",
            "tableCellCitationGradeCandidates",
            "figureRegionLinkVerifiedCandidates",
            "canonicalSourceSpanCreatedCandidates",
        ):
            if _safe_int(counts.get(count_key)) > 0:
                unsafe.append(f"{name}_{count_key}_nonzero")
        for policy_key in (
            "strictEvidenceCreated",
            "runtimePromotionAllowed",
            "parserRoutingChanged",
            "canonicalParsedArtifactsWritten",
            "databaseMutation",
            "reindexOrReembed",
            "answerIntegrationChanged",
        ):
            if bool(policy.get(policy_key)) is True:
                unsafe.append(f"{name}_{policy_key}_true")
        for gate_key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady"):
            if bool(gate.get(gate_key)) is True:
                unsafe.append(f"{name}_{gate_key}_true")
    return list(dict.fromkeys(unsafe))


def _by_layer(summary: dict[str, Any]) -> dict[str, int]:
    return {
        str(key): _safe_int(value)
        for key, value in dict((summary.get("counts") or {}).get("byLayer") or {}).items()
    }


def _blocker_map(backlog: dict[str, Any]) -> dict[str, list[str]]:
    by_layer: dict[str, list[str]] = {
        "sectionspan": [],
        "figure_caption": [],
        "equation_quote": [],
        "table_region": [],
    }
    for item in list(backlog.get("backlog") or []):
        if not isinstance(item, dict):
            continue
        blocker = str(item.get("blocker") or "")
        for layer in list(item.get("affected_layers") or []):
            if layer in by_layer and blocker:
                by_layer[layer].append(blocker)
    return {key: list(dict.fromkeys(value)) for key, value in by_layer.items()}


def _track(
    *,
    layer: str,
    candidate_count: int,
    source_span_count: int,
    blockers: list[str],
    readiness: str,
    next_tranche: str,
    evidence_needed: list[str],
) -> dict[str, Any]:
    strict_blockers = list(
        dict.fromkeys(
            [
                *blockers,
                "candidate_layer_promotion_policy_draft_only",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_explicit_later_tranche",
                "parser_routing_requires_explicit_later_tranche",
                "answer_integration_requires_explicit_later_tranche",
            ]
        )
    )
    return {
        "track_id": f"candidate-layer-promotion:{layer}",
        "layer": layer,
        "candidate_count": candidate_count,
        "canonical_source_span_candidate_count": source_span_count,
        "promotion_readiness": readiness,
        "candidate_layer_formalization_ready": readiness == "candidate_formalization_ready_non_strict",
        "strict_promotion_ready": False,
        "parser_routing_ready": False,
        "answer_integration_ready": False,
        "allowed_next_actions": [
            "human_operator_review",
            "schema_backed_report_only_refinement",
            "candidate_contract_review",
            next_tranche,
        ],
        "disallowed_actions": [
            "strict_evidence_promotion",
            "parser_routing",
            "answer_integration",
            "canonical_parsed_artifact_write",
            "database_mutation",
            "reindex_or_reembed",
        ],
        "evidence_needed_before_strict_promotion": evidence_needed,
        "recommended_next_tranche": next_tranche,
        "evidence_tier": "promotion_policy_draft_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": [
            "promotion_policy_draft_is_not_runtime_policy",
            "candidate_layers_remain_report_only",
            "later_explicit_promotion_tranche_required",
        ],
    }


def _tracks(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    summary = payloads["structuredSummary"]
    source_offsets = payloads["sourceSpanOffsetAuthorityAudit"]
    equation = payloads["equationAlignmentFeasibilityAudit"]
    table = payloads["tableCellProvenanceFeasibilityAudit"]
    figure = payloads["figureRegionLinkFeasibilityAudit"]
    by_layer = _by_layer(summary)
    source_by_layer = {
        str(key): _safe_int(value)
        for key, value in dict((source_offsets.get("counts") or {}).get("byLayer") or {}).items()
    }
    blockers = _blocker_map(payloads["candidateLayerBlockerBacklog"])
    return [
        _track(
            layer="sectionspan",
            candidate_count=_safe_int(by_layer.get("sectionspan")),
            source_span_count=_safe_int(source_by_layer.get("sectionspan")),
            blockers=[
                *blockers.get("sectionspan", []),
                "original_pdf_offset_not_available",
                "canonical_text_span_is_candidate_only",
            ],
            readiness="candidate_formalization_ready_non_strict",
            next_tranche="sectionspan_candidate_contract_review",
            evidence_needed=[
                "explicit candidate contract approval",
                "proof candidate rows are not runtime citations",
                "later strict promotion rule if SectionSpan is used by answers",
            ],
        ),
        _track(
            layer="figure_caption",
            candidate_count=_safe_int(by_layer.get("figure_caption")),
            source_span_count=_safe_int((figure.get("counts") or {}).get("captionSourceSpanCandidates")),
            blockers=[
                *blockers.get("figure_caption", []),
                "figure_region_page_missing",
                "figure_region_type_unverified",
            ],
            readiness="blocked_figure_region_link_unverified",
            next_tranche="figure_region_link_authority_design",
            evidence_needed=[
                "caption source span",
                "linked figure/image-like region with page",
                "verified caption-to-region relation",
                "sourceContentHash linkage",
            ],
        ),
        _track(
            layer="equation_quote",
            candidate_count=_safe_int(by_layer.get("equation_quote")),
            source_span_count=_safe_int((equation.get("counts") or {}).get("canonicalSourceSpanCreatedCandidates")),
            blockers=[
                *blockers.get("equation_quote", []),
                "canonical_equation_source_span_missing",
                "equation_semantics_not_interpreted",
            ],
            readiness="blocked_equation_source_span_missing",
            next_tranche="equation_quote_alignment_design",
            evidence_needed=[
                "exact equation text alignment to canonical parsed text",
                "quote-only policy with no equation interpretation",
                "page and sourceContentHash linkage",
            ],
        ),
        _track(
            layer="table_region",
            candidate_count=_safe_int(by_layer.get("table_region")),
            source_span_count=_safe_int((table.get("counts") or {}).get("rowColumnTextCandidates")),
            blockers=[
                *blockers.get("table_region", []),
                "table_cell_bbox_missing",
                "table_cell_chars_start_end_missing",
                "table_cell_source_content_hash_missing",
            ],
            readiness="blocked_table_cell_provenance_missing",
            next_tranche="table_cell_provenance_authority_design",
            evidence_needed=[
                "row and column labels",
                "cell values",
                "cell-level bbox or source span provenance",
                "cell-level sourceContentHash linkage",
            ],
        ),
    ]


def _counts(tracks: list[dict[str, Any]], payloads: dict[str, dict[str, Any]], violations: list[str]) -> dict[str, Any]:
    summary_counts = dict(payloads["structuredSummary"].get("counts") or {})
    review_counts = dict(payloads["candidateLayerReviewGate"].get("counts") or {})
    source_counts = dict(payloads["sourceSpanOffsetAuthorityAudit"].get("counts") or {})
    by_readiness = Counter(str(item.get("promotion_readiness") or "") for item in tracks)
    return {
        "totalCandidates": _safe_int(summary_counts.get("totalCandidates")),
        "promotionTrackCount": len(tracks),
        "candidateOnlyPromotionTracks": len(tracks),
        "candidateFormalizationReadyTracks": sum(1 for item in tracks if item.get("candidate_layer_formalization_ready")),
        "strictPromotionReadyTracks": 0,
        "parserRoutingReadyTracks": 0,
        "answerIntegrationReadyTracks": 0,
        "runtimePromotionAllowedTracks": 0,
        "strictEligibleCandidates": _safe_int(summary_counts.get("strictEligibleCandidates")),
        "citationGradeCandidates": _safe_int(summary_counts.get("citationGradeCandidates")),
        "runtimeEvidenceCandidates": _safe_int(summary_counts.get("runtimeEvidenceCandidates")),
        "currentRuntimeAnswerableQuestions": _safe_int(review_counts.get("currentRuntimeAnswerableQuestions")),
        "canonicalParsedTextSpanCandidates": _safe_int(source_counts.get("canonicalParsedTextSpanCandidates")),
        "originalPdfOffsetCandidates": _safe_int(source_counts.get("originalPdfOffsetCandidates")),
        "schemaViolationCount": len([item for item in violations if item.endswith("_schema_mismatch")]),
        "unsafeUpstreamFlagCount": len([item for item in violations if not item.endswith("_schema_mismatch")]),
        "byReadiness": dict(by_readiness),
        "byLayer": {str(item.get("layer")): _safe_int(item.get("candidate_count")) for item in tracks},
    }


def build_candidate_layer_promotion_policy_draft(
    *,
    structured_summary_report: str | Path,
    candidate_layer_review_gate_report: str | Path,
    candidate_layer_blocker_backlog_report: str | Path,
    source_span_offset_authority_audit_report: str | Path,
    equation_alignment_feasibility_audit_report: str | Path,
    table_cell_provenance_feasibility_audit_report: str | Path,
    figure_region_link_feasibility_audit_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only candidate-layer promotion policy draft."""

    paths = {
        "structuredSummary": Path(str(structured_summary_report)).expanduser(),
        "candidateLayerReviewGate": Path(str(candidate_layer_review_gate_report)).expanduser(),
        "candidateLayerBlockerBacklog": Path(str(candidate_layer_blocker_backlog_report)).expanduser(),
        "sourceSpanOffsetAuthorityAudit": Path(str(source_span_offset_authority_audit_report)).expanduser(),
        "equationAlignmentFeasibilityAudit": Path(str(equation_alignment_feasibility_audit_report)).expanduser(),
        "tableCellProvenanceFeasibilityAudit": Path(str(table_cell_provenance_feasibility_audit_report)).expanduser(),
        "figureRegionLinkFeasibilityAudit": Path(str(figure_region_link_feasibility_audit_report)).expanduser(),
    }
    payloads = {key: _read_json(path) for key, path in paths.items()}
    violations = [*_schema_violations(payloads), *_unsafe_upstream_flags(payloads)]
    tracks = _tracks(payloads)
    status = "draft_ready" if not violations else "blocked"
    return {
        "schema": CANDIDATE_LAYER_PROMOTION_POLICY_DRAFT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            key: str(path)
            for key, path in paths.items()
        }
        | {
            f"{key}Schema": str(payloads[key].get("schema") or "")
            for key in paths
        },
        "counts": _counts(tracks, payloads, violations),
        "gate": {
            "promotionPolicyDraftReady": not violations,
            "candidateLayerReviewReady": bool((payloads["candidateLayerReviewGate"].get("gate") or {}).get("candidateLayerReviewReady")),
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "decision": "candidate_layer_promotion_policy_draft_ready" if not violations else "blocked",
            "schemaViolations": [item for item in violations if item.endswith("_schema_mismatch")],
            "unsafeUpstreamFlags": [item for item in violations if not item.endswith("_schema_mismatch")],
            "recommendedNextTranche": "human_review_or_sectionspan_candidate_contract_review",
        },
        "policy": {
            "draftOnly": True,
            "runtimePolicyChanged": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "promotionPrinciples": [
            "candidate_only_rows_are_never_runtime_citations",
            "sectionspan_can_be_reviewed_first_as_a_candidate_contract_not_strict_evidence",
            "figure_caption_requires_verified_figure_region_authority_before_runtime_use",
            "equation_quote_requires_exact_source_span_alignment_and_no_semantic_interpretation",
            "table_region_requires_cell_level_bbox_or_source_span_and_source_hash_before_numeric_qa_use",
            "parser_routing_and_answer_integration_require_later_explicit_approval",
        ],
        "warnings": [
            "this_draft_does_not_modify_runtime_evidence_policy",
            "no_candidate_type_is_strict_or_citation_grade_in_this_report",
            "promotion_tracks_are_review_guidance_only",
        ],
        "promotionTracks": tracks,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "inputs",
            "counts",
            "gate",
            "policy",
            "promotionPrinciples",
            "warnings",
            "promotionTracks",
        )
        if key in report
    }


def render_candidate_layer_promotion_policy_draft_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Candidate Layer Promotion Policy Draft",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Total candidates: `{int(counts.get('totalCandidates') or 0)}`",
        f"- Promotion tracks: `{int(counts.get('promotionTrackCount') or 0)}`",
        f"- Candidate formalization ready tracks: `{int(counts.get('candidateFormalizationReadyTracks') or 0)}`",
        f"- Strict promotion ready tracks: `{int(counts.get('strictPromotionReadyTracks') or 0)}`",
        f"- Runtime promotion allowed tracks: `{int(counts.get('runtimePromotionAllowedTracks') or 0)}`",
        f"- Original PDF offset candidates: `{int(counts.get('originalPdfOffsetCandidates') or 0)}`",
        "",
        "## Policy Boundary",
        "",
        "This is a draft-only review artifact. It does not modify runtime evidence policy, parser routing, answer integration, canonical parsed artifacts, SQLite, indexes, or embeddings.",
        "",
        "## Promotion Tracks",
        "",
    ]
    for item in list(report.get("promotionTracks") or []):
        lines.extend(
            [
                f"### `{item.get('layer', '')}`",
                "",
                f"- Candidates: `{int(item.get('candidate_count') or 0)}`",
                f"- Readiness: `{item.get('promotion_readiness', '')}`",
                f"- Strict promotion ready: `{bool(item.get('strict_promotion_ready'))}`",
                f"- Recommended next tranche: `{item.get('recommended_next_tranche', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_candidate_layer_promotion_policy_draft_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    draft_path = root / "candidate-layer-promotion-policy-draft.json"
    summary_path = root / "candidate-layer-promotion-policy-summary.json"
    markdown_path = root / "candidate-layer-promotion-policy-draft.md"
    draft_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_candidate_layer_promotion_policy_draft_markdown(report), encoding="utf-8")
    return {"draft": str(draft_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only candidate-layer promotion policy draft.")
    parser.add_argument("--structured-summary-report", required=True)
    parser.add_argument("--candidate-layer-review-gate-report", required=True)
    parser.add_argument("--candidate-layer-blocker-backlog-report", required=True)
    parser.add_argument("--source-span-offset-authority-audit-report", required=True)
    parser.add_argument("--equation-alignment-feasibility-audit-report", required=True)
    parser.add_argument("--table-cell-provenance-feasibility-audit-report", required=True)
    parser.add_argument("--figure-region-link-feasibility-audit-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_candidate_layer_promotion_policy_draft(
        structured_summary_report=args.structured_summary_report,
        candidate_layer_review_gate_report=args.candidate_layer_review_gate_report,
        candidate_layer_blocker_backlog_report=args.candidate_layer_blocker_backlog_report,
        source_span_offset_authority_audit_report=args.source_span_offset_authority_audit_report,
        equation_alignment_feasibility_audit_report=args.equation_alignment_feasibility_audit_report,
        table_cell_provenance_feasibility_audit_report=args.table_cell_provenance_feasibility_audit_report,
        figure_region_link_feasibility_audit_report=args.figure_region_link_feasibility_audit_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_promotion_policy_draft_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_PROMOTION_POLICY_DRAFT_SCHEMA_ID",
    "build_candidate_layer_promotion_policy_draft",
    "render_candidate_layer_promotion_policy_draft_markdown",
    "write_candidate_layer_promotion_policy_draft_reports",
]
