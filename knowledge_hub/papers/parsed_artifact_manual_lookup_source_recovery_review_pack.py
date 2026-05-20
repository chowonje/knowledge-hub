"""Report-only review pack for manual parsed-artifact source recovery.

This helper consumes the manual lookup source recovery plan and turns its
candidate rows into operator review cards plus an inert allowlist template. It
does not perform external lookup, download PDFs, mutate source registrations,
materialize parsed artifacts, touch evidence stores, scan the vault, or change
answer behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_coverage_batch_report import _counter_items
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_plan import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_PLAN_SCHEMA_ID,
)


PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-manual-lookup-source-recovery-review-pack.v1"
)

DEFAULT_MANUAL_LOOKUP_PLAN_REPORT_PATH = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-21/"
    "parsed-artifact-manual-lookup-source-recovery-plan/"
    "01-parsed-artifact-manual-lookup-source-recovery-plan/"
    "parsed-artifact-manual-lookup-source-recovery-plan.json"
).expanduser()
DEFAULT_OUTPUT_DIR = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-21/"
    "parsed-artifact-manual-lookup-source-recovery-review-pack/"
    "01-parsed-artifact-manual-lookup-source-recovery-review-pack"
).expanduser()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))


def _mutation_policy() -> dict[str, Any]:
    return {
        "externalLookup": False,
        "sourceDownload": False,
        "sourcePathRewrite": False,
        "sourceRegistrationMutation": False,
        "parsedArtifactWrite": False,
        "parserRouting": False,
        "sourceSpanCreated": False,
        "strictEvidence": False,
        "citationEvidence": False,
        "runtimeEvidence": False,
        "databaseMutation": False,
        "indexMutation": False,
        "reindexOrReembed": False,
        "vaultScan": False,
        "vaultWrite": False,
        "answerIntegration": False,
        "manualBlockerResolution": False,
        "runManifestWrite": False,
    }


def _mutation_counters() -> dict[str, int]:
    return {
        "externalLookupRows": 0,
        "sourceDownloadRows": 0,
        "sourcePathRewriteRows": 0,
        "sourceRegistrationMutationRows": 0,
        "parsedArtifactWriteRows": 0,
        "parserRoutingRows": 0,
        "sourceSpanCreatedRows": 0,
        "strictEvidenceRows": 0,
        "citationEvidenceRows": 0,
        "runtimeEvidenceRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultReadRows": 0,
        "vaultWriteRows": 0,
        "answerIntegrationRows": 0,
        "manualBlockerResolutionRows": 0,
        "runManifestWriteRows": 0,
    }


def _priority_rank(row: dict[str, Any]) -> tuple[int, int, str]:
    priority = _clean_text(row.get("lookupPriority"))
    mode = _clean_text(row.get("manualLookupMode"))
    priority_rank = 0 if priority == "high" else 1
    mode_rank = 0 if mode == "registered_pdf_path_missing_manual_lookup" else 1
    return (priority_rank, mode_rank, _clean_text(row.get("paperId")))


def _approval_template(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paperId": _clean_text(row.get("paperId")),
        "paperTitle": _clean_text(row.get("paperTitle")),
        "decision": "needs_review",
        "approvedSourceType": "",
        "approvedSourceUrl": "",
        "approvedLocalPdfPath": "",
        "approvedSourceContentHash": "",
        "approvedBy": "",
        "approvedAt": "",
        "notes": "",
        "templateOnly": True,
        "acceptedByThisTranche": False,
    }


def _review_card(index: int, row: dict[str, Any]) -> dict[str, Any]:
    paper_id = _clean_text(row.get("paperId"))
    return {
        "reviewCardId": f"manual-lookup-source-recovery-review-card:{index:04d}:{paper_id}",
        "paperId": paper_id,
        "paperTitle": _clean_text(row.get("paperTitle")),
        "manualLookupMode": _clean_text(row.get("manualLookupMode")),
        "lookupPriority": _clean_text(row.get("lookupPriority")),
        "sourceArtifact": dict(row.get("sourceArtifact") or {}),
        "lookupQueries": dict(row.get("lookupQueries") or {}),
        "preferredReviewOrder": list(row.get("preferredReviewOrder") or []),
        "requiredFutureApproval": _clean_text(row.get("requiredFutureApproval")),
        "futureApplyTarget": _clean_text(row.get("futureApplyTarget")),
        "operatorDecisionDefault": "needs_review",
        "reviewStatus": "manual_lookup_source_recovery_review_card_candidate_only",
        "approvalTemplate": _approval_template(row),
        "externalLookupAttempted": False,
        "sourceDownloadAttempted": False,
        "sourceRegistrationMutationAttempted": False,
        "parsedArtifactWriteAttempted": False,
        "applyReady": False,
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerIntegration": False,
        "nonScopeReasons": [
            "review_pack_report_only",
            "approval_template_is_inert",
            "explicit_source_url_or_local_pdf_path_allowlist_required_before_apply",
            "parsed_artifact_materialization_requires_later_apply_tranche",
        ],
    }


def _text_holdout(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paperId": _clean_text(row.get("paperId")),
        "paperTitle": _clean_text(row.get("paperTitle")),
        "sourceArtifact": dict(row.get("sourceArtifact") or {}),
        "textSourceHoldoutStatus": "text_source_contract_holdout",
        "futurePolicyOption": _clean_text(row.get("futurePolicyOption")),
        "decisionRecommendation": _clean_text(row.get("decisionRecommendation")),
        "requiredFutureApproval": _clean_text(row.get("requiredFutureApproval")),
        "recommendedAction": "parsed_artifact_text_source_materialization_contract",
        "externalLookupAttempted": False,
        "sourceDownloadAttempted": False,
        "sourceRegistrationMutationAttempted": False,
        "parsedArtifactWriteAttempted": False,
        "applyReady": False,
    }


def build_parsed_artifact_manual_lookup_source_recovery_review_pack(
    *,
    manual_lookup_plan_report: dict[str, Any],
    manual_lookup_plan_report_path: str | Path | None = None,
    report_name: str = "parsed-artifact-manual-lookup-source-recovery-review-pack",
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a non-mutating review pack from a manual lookup recovery plan."""

    validation = validate_payload(
        manual_lookup_plan_report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_PLAN_SCHEMA_ID,
        strict=True,
    )
    input_validation = {"ok": bool(validation.ok), "errors": list(validation.errors)}
    if not validation.ok:
        manual_rows: list[dict[str, Any]] = []
        text_rows: list[dict[str, Any]] = []
        coverage_target: dict[str, Any] = {
            "missingParsedArtifactsBefore": 0,
            "targetMissingParsedArtifacts": 0,
            "manualRecoveriesNeededForTarget": 0,
            "manualLookupCandidateRows": 0,
            "targetReachableIfEnoughManualSourcesResolved": False,
            "potentialMissingParsedArtifactsAfterAllManualLookupRecovered": 0,
            "textSourceRowsRemainAfterManualLookup": 0,
        }
        warnings = ["blocked_input_schema_violation"]
        status = "blocked_input_schema_violation"
    else:
        manual_rows = [
            dict(row)
            for row in list(manual_lookup_plan_report.get("manualLookupCandidates") or [])
            if isinstance(row, dict)
        ]
        text_rows = [
            dict(row)
            for row in list(manual_lookup_plan_report.get("textSourceHoldouts") or [])
            if isinstance(row, dict)
        ]
        coverage_target = dict(manual_lookup_plan_report.get("coverageTarget") or {})
        warnings = list(manual_lookup_plan_report.get("warnings") or [])
        status = (
            "manual_lookup_source_recovery_review_pack_candidate_only"
            if manual_rows
            else "no_manual_lookup_source_recovery_review_candidates"
        )

    ordered_manual_rows = sorted(manual_rows, key=_priority_rank)
    review_cards = [_review_card(index, row) for index, row in enumerate(ordered_manual_rows, start=1)]
    holdouts = [_text_holdout(row) for row in text_rows]
    approval_templates = [dict(card.get("approvalTemplate") or {}) for card in review_cards]
    mode_counter = Counter(_clean_text(card.get("manualLookupMode")) for card in review_cards)
    priority_counter = Counter(_clean_text(card.get("lookupPriority")) for card in review_cards)
    holdout_counter = Counter(_clean_text(row.get("textSourceHoldoutStatus")) for row in holdouts)

    counts = {
        "inputRows": len(manual_rows) + len(text_rows),
        "manualLookupReviewCardRows": len(review_cards),
        "textSourceHoldoutRows": len(holdouts),
        "approvalTemplateRows": len(approval_templates),
        "approvedSourceRows": 0,
        "applyReadyRows": 0,
        "manualLookupHighPriorityRows": sum(1 for card in review_cards if card.get("lookupPriority") == "high"),
        "manualLookupNormalPriorityRows": sum(1 for card in review_cards if card.get("lookupPriority") == "normal"),
        "manualLookupModeTaxonomy": _counter_items(mode_counter),
        "lookupPriorityTaxonomy": _counter_items(priority_counter),
        "textSourceHoldoutTaxonomy": _counter_items(holdout_counter),
    }
    counts.update(_mutation_counters())
    report = {
        "schema": PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or _utc_now(),
        "report": {
            "name": report_name,
            "inputManualLookupPlanReportPath": str(Path(str(manual_lookup_plan_report_path)).expanduser())
            if manual_lookup_plan_report_path
            else "",
            "selectionRule": (
                "manualLookupCandidates become inert review cards; textSourceHoldouts stay held out for "
                "a separate text-source contract"
            ),
            "nonScope": [
                "external_lookup",
                "source_download",
                "source_path_rewrite",
                "source_registration_mutation",
                "parsed_artifact_write",
                "parser_routing",
                "source_span_creation",
                "strict_or_citation_or_runtime_evidence",
                "database_or_index_or_reembed",
                "vault_scan_or_write",
                "answer_integration",
                "manual_blocker_resolution",
                "approval_acceptance",
                "run_manifest_write",
            ],
        },
        "inputPlan": {
            "schemaValidation": input_validation,
            "schema": _clean_text(manual_lookup_plan_report.get("schema")),
            "status": _clean_text(manual_lookup_plan_report.get("status")),
            "manualLookupCandidateOnlyRows": int(
                dict(manual_lookup_plan_report.get("candidatePool") or {}).get("manualLookupCandidateOnlyRows") or 0
            )
            if validation.ok
            else 0,
            "textSourceContractHoldRows": int(
                dict(manual_lookup_plan_report.get("candidatePool") or {}).get("textSourceContractHoldRows") or 0
            )
            if validation.ok
            else 0,
        },
        "counts": counts,
        "coverageTarget": coverage_target,
        "manualLookupReviewCardPaperIds": [_clean_text(card.get("paperId")) for card in review_cards],
        "reviewCards": review_cards,
        "allowlistTemplateRows": approval_templates,
        "textSourceHoldoutPaperIds": [_clean_text(row.get("paperId")) for row in holdouts],
        "textSourceHoldouts": holdouts,
        "gate": {
            "reviewPackReady": bool(review_cards) and validation.ok,
            "manualApprovalsAccepted": False,
            "applyReady": False,
            "sourceRegistrationMutationReady": False,
            "parsedArtifactMaterializationReady": False,
            "parserRoutingReady": False,
            "strictEvidenceReady": False,
            "answerIntegrationReady": False,
            "decision": status,
            "recommendedNextTranche": "parsed_artifact_manual_lookup_source_recovery_decision_file_draft"
            if review_cards
            else "parsed_artifact_text_source_materialization_contract",
        },
        "mutationPolicy": _mutation_policy(),
        "mutationCounters": _mutation_counters(),
        "warnings": warnings,
    }
    output_validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID,
        strict=True,
    )
    if not output_validation.ok:
        raise ValueError(
            "parsed artifact manual lookup source recovery review pack schema validation failed: "
            + "; ".join(output_validation.errors[:5])
        )
    return report


def write_parsed_artifact_manual_lookup_source_recovery_review_pack(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON, summary JSON, Markdown, cards, allowlist, and holdout files."""

    validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        raise ValueError(
            "parsed artifact manual lookup source recovery review pack schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-manual-lookup-source-recovery-review-pack.json"
    summary_json_path = root / "parsed-artifact-manual-lookup-source-recovery-review-pack-summary.json"
    markdown_path = root / "parsed-artifact-manual-lookup-source-recovery-review-pack.md"
    cards_path = root / "manual-lookup-source-recovery-review-cards.json"
    allowlist_path = root / "manual-lookup-source-recovery-allowlist-template.json"
    text_ids_path = root / "text-source-contract-holdout-paper-ids.txt"
    summary = {
        "reportSchema": report.get("schema"),
        "status": report.get("status"),
        "counts": report.get("counts"),
        "coverageTarget": report.get("coverageTarget"),
        "gate": report.get("gate"),
        "mutationCounters": report.get("mutationCounters"),
        "reportFiles": {
            "reportJsonPath": str(report_path),
            "reviewCardsPath": str(cards_path),
            "allowlistTemplatePath": str(allowlist_path),
            "textSourceHoldoutIdsPath": str(text_ids_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cards_path.write_text(
        json.dumps(report.get("reviewCards") or [], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    allowlist_path.write_text(
        json.dumps(report.get("allowlistTemplateRows") or [], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    text_ids_path.write_text(
        "\n".join(report.get("textSourceHoldoutPaperIds") or []) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown_summary(report), encoding="utf-8")
    return {
        "reportJsonPath": str(report_path),
        "summaryJsonPath": str(summary_json_path),
        "reportMarkdownPath": str(markdown_path),
        "reviewCardsPath": str(cards_path),
        "allowlistTemplatePath": str(allowlist_path),
        "textSourceHoldoutIdsPath": str(text_ids_path),
    }


def _render_markdown_summary(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    target = dict(report.get("coverageTarget") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Parsed Artifact Manual Lookup Source Recovery Review Pack",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- input rows: {counts.get('inputRows', 0)}",
        f"- review card rows: {counts.get('manualLookupReviewCardRows', 0)}",
        f"- text-source holdout rows: {counts.get('textSourceHoldoutRows', 0)}",
        f"- approval template rows: {counts.get('approvalTemplateRows', 0)}",
        f"- approved source rows: {counts.get('approvedSourceRows', 0)}",
        f"- apply-ready rows: {counts.get('applyReadyRows', 0)}",
        f"- missing parsed artifacts before: {target.get('missingParsedArtifactsBefore', 0)}",
        f"- manual recoveries needed for target: {target.get('manualRecoveriesNeededForTarget', 0)}",
        f"- target missing parsed artifacts: {target.get('targetMissingParsedArtifacts', 0)}",
        f"- recommended next tranche: `{gate.get('recommendedNextTranche', '')}`",
        "",
        "## Boundary",
        "",
        "This review pack is report-only. It does not perform lookup, download files, accept approvals, mutate source registrations, materialize parsed artifacts, create evidence, scan the vault, or change answer behavior.",
        "",
        "## Manual Lookup Mode Taxonomy",
        "",
    ]
    for item in list(counts.get("manualLookupModeTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Review Cards", ""])
    for row in list(report.get("reviewCards") or []):
        lines.append(
            f"- `{row.get('paperId')}`: {row.get('paperTitle')} "
            f"(`{row.get('manualLookupMode')}`, priority `{row.get('lookupPriority')}`)"
        )
    lines.extend(["", "## Text Source Holdouts", ""])
    for row in list(report.get("textSourceHoldouts") or []):
        lines.append(f"- `{row.get('paperId')}`: {row.get('paperTitle')}")
    lines.extend(["", "## Mutation Counters", ""])
    for key, value in sorted(dict(report.get("mutationCounters") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-lookup-plan-report", default=str(DEFAULT_MANUAL_LOOKUP_PLAN_REPORT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-name", default="parsed-artifact-manual-lookup-source-recovery-review-pack")
    args = parser.parse_args(argv)

    plan_report_path = Path(args.manual_lookup_plan_report).expanduser()
    plan_report = _load_json(plan_report_path)
    report = build_parsed_artifact_manual_lookup_source_recovery_review_pack(
        manual_lookup_plan_report=plan_report,
        manual_lookup_plan_report_path=plan_report_path,
        report_name=args.report_name,
    )
    paths = write_parsed_artifact_manual_lookup_source_recovery_review_pack(report, args.output_dir)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "paths": paths,
                "counts": report["counts"],
                "coverageTarget": report["coverageTarget"],
                "gate": report["gate"],
                "mutationCounters": report["mutationCounters"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID",
    "build_parsed_artifact_manual_lookup_source_recovery_review_pack",
    "write_parsed_artifact_manual_lookup_source_recovery_review_pack",
]
