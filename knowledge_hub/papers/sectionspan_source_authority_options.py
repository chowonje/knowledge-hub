"""Report-only SectionSpan source-authority decision options.

This helper turns the SectionSpan strict-promotion design into explicit options
for the next authority decision.  It does not select an authority, implement
strict evidence, change parser routing, wire answers to candidate layers, write
canonical parsed artifacts, mutate SQLite, reindex, or reembed.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_SOURCE_AUTHORITY_OPTIONS_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-source-authority-options.v1"
)
SECTIONSPAN_STRICT_PROMOTION_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-strict-promotion-design.v1"
)


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


def _schema_violations(design: dict[str, Any]) -> list[str]:
    if design.get("schema") != SECTIONSPAN_STRICT_PROMOTION_DESIGN_SCHEMA_ID:
        return ["sectionspan_strict_promotion_design_schema_mismatch"]
    return []


def _unsafe_flags(design: dict[str, Any]) -> list[str]:
    unsafe: list[str] = []
    counts = dict(design.get("counts") or {})
    gate = dict(design.get("gate") or {})
    policy = dict(design.get("policy") or {})
    if design.get("status") != "design_ready":
        unsafe.append("sectionspan_strict_promotion_design_not_ready")
    for key in (
        "strictPromotionReadyRows",
        "runtimePromotionAllowedRows",
        "strictEligibleRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            unsafe.append(f"{key}_nonzero")
    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(gate.get(key)):
            unsafe.append(f"{key}_true")
    for key in (
        "strictPromotionImplemented",
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(policy.get(key)):
            unsafe.append(f"{key}_true")
    return list(dict.fromkeys(unsafe))


def _option(
    *,
    option_id: str,
    title: str,
    authority_posture: str,
    recommendation: str,
    affected_rows: int,
    pros: list[str],
    cons: list[str],
    required_before_use: list[str],
    blocked_actions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "option_id": option_id,
        "title": title,
        "authority_posture": authority_posture,
        "recommendation": recommendation,
        "affected_row_count": affected_rows,
        "pros": pros,
        "cons": cons,
        "required_before_use": required_before_use,
        "blocked_actions": blocked_actions
        or [
            "strict_evidence_promotion",
            "runtime_answer_citation",
            "parser_routing",
            "canonical_parsed_artifact_write",
            "database_mutation",
            "reindex_or_reembed",
        ],
        "authority_decision_made": False,
        "strict_promotion_ready": False,
        "runtime_promotion_allowed": False,
        "evidence_tier": "sectionspan_source_authority_option_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
    }


def _options(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    affected = len(rows)
    return [
        _option(
            option_id="keep_candidate_layer_only",
            title="Keep SectionSpan as report-only candidate structure",
            authority_posture="no_strict_authority",
            recommendation="safe_default_until_authority_decision",
            affected_rows=affected,
            pros=[
                "preserves current evidence policy",
                "requires no parser routing or runtime answer changes",
                "keeps SectionSpan useful for operator review and future design",
            ],
            cons=[
                "does not improve runtime citation quality",
                "does not make complex-paper QA answerable",
            ],
            required_before_use=[
                "none_for_report_only_review",
            ],
        ),
        _option(
            option_id="recover_original_pdf_offsets_first",
            title="Recover original PDF offset or source-span authority before strict use",
            authority_posture="strict_authority_candidate_after_recovery",
            recommendation="recommended_before_any_runtime_strict_promotion",
            affected_rows=affected,
            pros=[
                "keeps strict evidence tied to original source authority",
                "avoids treating generated Markdown offsets as citation authority",
                "matches fail-closed provenance posture",
            ],
            cons=[
                "requires additional offset recovery or PDF text span mapping work",
                "may be expensive or parser-specific",
            ],
            required_before_use=[
                "original_pdf_offset_recovery_design",
                "sourceContentHash_and_original_source_span_mapping",
                "protecting_tests_for_no_summary_or_generated_offset_promotion",
                "later_explicit_strict_promotion_tranche",
            ],
        ),
        _option(
            option_id="explicitly_authorize_canonical_generated_markdown_offsets",
            title="Explicitly authorize canonical generated Markdown offsets as SectionSpan authority",
            authority_posture="policy_decision_required",
            recommendation="requires_user_and_project_authority_approval_before_use",
            affected_rows=affected,
            pros=[
                "could make current SectionSpan candidates usable sooner",
                "leverages already aligned canonical parsed text spans",
            ],
            cons=[
                "changes evidence authority semantics",
                "may blur original-source vs generated-artifact provenance",
                "requires ADR or equivalent durable policy decision before runtime use",
            ],
            required_before_use=[
                "explicit_authority_decision",
                "ADR_or_project_state_decision_record",
                "runtime_evidence_policy_update",
                "answer_contract_tests_preventing_summary_or_paraphrase_promotion",
                "later_explicit_strict_promotion_tranche",
            ],
        ),
    ]


def _counts(rows: list[dict[str, Any]], options: list[dict[str, Any]], violations: list[str]) -> dict[str, Any]:
    by_status = Counter(str(item.get("promotion_design_status") or "") for item in rows)
    return {
        "inputDesignRows": len(rows),
        "optionCount": len(options),
        "authorityDecisionMadeOptions": 0,
        "strictPromotionReadyOptions": 0,
        "runtimePromotionAllowedOptions": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len([item for item in violations if item.endswith("_mismatch")]),
        "unsafeUpstreamFlagCount": len([item for item in violations if not item.endswith("_mismatch")]),
        "byPromotionDesignStatus": dict(by_status),
        "byRecommendedOption": {
            "safe_default_until_authority_decision": 1,
            "recommended_before_any_runtime_strict_promotion": 1,
            "requires_user_and_project_authority_approval_before_use": 1,
        },
    }


def build_sectionspan_source_authority_options(
    *,
    sectionspan_strict_promotion_design_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only source-authority options payload."""

    path = Path(str(sectionspan_strict_promotion_design_report)).expanduser()
    design = _read_json(path)
    violations = [*_schema_violations(design), *_unsafe_flags(design)]
    rows = [dict(item) for item in list(design.get("designRows") or []) if isinstance(item, dict)]
    options = _options(rows)
    counts = _counts(rows, options, violations)
    ready = not violations and bool(rows)
    return {
        "schema": SECTIONSPAN_SOURCE_AUTHORITY_OPTIONS_SCHEMA_ID,
        "status": "options_ready" if ready else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "sectionspanStrictPromotionDesignReport": str(path),
            "sectionspanStrictPromotionDesignSchema": str(design.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "authorityOptionsReady": ready,
            "authorityDecisionMade": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "source_authority_options_ready_no_decision_made" if ready else "blocked",
            "schemaViolations": [item for item in violations if item.endswith("_mismatch")],
            "unsafeUpstreamFlags": [item for item in violations if not item.endswith("_mismatch")],
            "recommendedNextTranche": "choose_original_pdf_offset_recovery_or_canonical_markdown_authority_policy",
        },
        "policy": {
            "reportOnly": True,
            "authorityDecisionMade": False,
            "strictPromotionImplemented": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "this_report_does_not_choose_an_authority_option",
            "canonical_generated_markdown_offsets_are_not_currently_strict_evidence",
            "runtime_use_requires_later_explicit_policy_and_implementation_tranche",
        ],
        "options": options,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings", "options")
        if key in report
    }


def render_sectionspan_source_authority_options_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan Source-Authority Options",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Input design rows: `{int(counts.get('inputDesignRows') or 0)}`",
        f"- Authority decision made: `{bool(gate.get('authorityDecisionMade'))}`",
        f"- Strict promotion ready options: `{int(counts.get('strictPromotionReadyOptions') or 0)}`",
        f"- Runtime promotion allowed options: `{int(counts.get('runtimePromotionAllowedOptions') or 0)}`",
        "",
        "## Boundary",
        "",
        "This report lists options only. It does not choose source authority, create strict evidence, allow runtime citations, change parser routing, write canonical parsed artifacts, mutate SQLite, reindex, or reembed.",
        "",
        "## Options",
        "",
    ]
    for option in list(report.get("options") or []):
        lines.extend(
            [
                f"### `{option.get('option_id', '')}`",
                "",
                f"- Title: {option.get('title', '')}",
                f"- Recommendation: `{option.get('recommendation', '')}`",
                f"- Authority decision made: `{bool(option.get('authority_decision_made'))}`",
                f"- Runtime promotion allowed: `{bool(option.get('runtime_promotion_allowed'))}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_sectionspan_source_authority_options_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    options_path = root / "sectionspan-source-authority-options.json"
    summary_path = root / "sectionspan-source-authority-options-summary.json"
    markdown_path = root / "sectionspan-source-authority-options.md"
    options_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_source_authority_options_markdown(report), encoding="utf-8")
    return {"options": str(options_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only SectionSpan source-authority options.")
    parser.add_argument("--sectionspan-strict-promotion-design-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_source_authority_options(
        sectionspan_strict_promotion_design_report=args.sectionspan_strict_promotion_design_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_source_authority_options_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_SOURCE_AUTHORITY_OPTIONS_SCHEMA_ID",
    "build_sectionspan_source_authority_options",
    "render_sectionspan_source_authority_options_markdown",
    "write_sectionspan_source_authority_options_reports",
]
