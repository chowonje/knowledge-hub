"""Report-only draft decision file for manual parsed-artifact source recovery.

The draft file is an editable starting point for a human/operator. Every manual
lookup row remains ``needs_review`` and no approved source URL, local PDF path,
source hash, source registration, parsed artifact, or downstream evidence is
created by this helper.
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
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_review_pack import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID,
)


PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_DRAFT_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-manual-lookup-source-recovery-decision-file-draft.v1"
)

DEFAULT_REVIEW_PACK_REPORT_PATH = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-21/"
    "parsed-artifact-manual-lookup-source-recovery-review-pack/"
    "01-parsed-artifact-manual-lookup-source-recovery-review-pack/"
    "parsed-artifact-manual-lookup-source-recovery-review-pack.json"
).expanduser()
DEFAULT_OUTPUT_DIR = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-21/"
    "parsed-artifact-manual-lookup-source-recovery-decision-file-draft/"
    "01-parsed-artifact-manual-lookup-source-recovery-decision-file-draft"
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


def _unsafe_flags(review_pack: dict[str, Any], validation_ok: bool) -> list[str]:
    flags: list[str] = []
    counts = dict(review_pack.get("counts") or {})
    gate = dict(review_pack.get("gate") or {})
    mutation_counters = dict(review_pack.get("mutationCounters") or {})
    mutation_policy = dict(review_pack.get("mutationPolicy") or {})
    if not validation_ok:
        flags.append("manual_lookup_source_recovery_review_pack_schema_violation")
    if review_pack.get("schema") != PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID:
        flags.append("manual_lookup_source_recovery_review_pack_schema_mismatch")
    if str(review_pack.get("status") or "") == "blocked_input_schema_violation":
        flags.append("manual_lookup_source_recovery_review_pack_blocked")
    for key in ("approvedSourceRows", "applyReadyRows"):
        if int(counts.get(key) or 0) > 0:
            flags.append(f"reviewPack_{key}_nonzero")
    for key in (
        "manualApprovalsAccepted",
        "applyReady",
        "sourceRegistrationMutationReady",
        "parsedArtifactMaterializationReady",
        "parserRoutingReady",
        "strictEvidenceReady",
        "answerIntegrationReady",
    ):
        if bool(gate.get(key)):
            flags.append(f"reviewPack_{key}_true")
    for key, value in mutation_policy.items():
        if bool(value):
            flags.append(f"reviewPack_policy_{key}_true")
    for key, value in mutation_counters.items():
        if int(value or 0) > 0:
            flags.append(f"reviewPack_counter_{key}_nonzero")
    return list(dict.fromkeys(flags))


def _allowed_decisions(card: dict[str, Any]) -> list[str]:
    mode = _clean_text(card.get("manualLookupMode"))
    decisions = [
        "needs_review",
        "approve_source_url_for_later_apply",
        "approve_local_pdf_path_for_later_apply",
        "reject_candidate_keep_missing",
        "hold_for_manual_lookup",
    ]
    if mode == "registered_pdf_path_missing_manual_lookup":
        return decisions
    return decisions


def _draft_row(index: int, card: dict[str, Any]) -> dict[str, Any]:
    template = dict(card.get("approvalTemplate") or {})
    paper_id = _clean_text(card.get("paperId"))
    return {
        "draftRowId": f"manual-lookup-source-recovery-decision-file-draft:{index:04d}:{paper_id}",
        "sourceReviewCardId": _clean_text(card.get("reviewCardId")),
        "paperId": paper_id,
        "paperTitle": _clean_text(card.get("paperTitle")),
        "manualLookupMode": _clean_text(card.get("manualLookupMode")),
        "lookupPriority": _clean_text(card.get("lookupPriority")),
        "sourceArtifact": dict(card.get("sourceArtifact") or {}),
        "lookupQueries": dict(card.get("lookupQueries") or {}),
        "preferredReviewOrder": list(card.get("preferredReviewOrder") or []),
        "allowedDecisions": _allowed_decisions(card),
        "decision": "needs_review",
        "approvedSourceType": "",
        "approvedSourceUrl": "",
        "approvedLocalPdfPath": "",
        "approvedSourceContentHash": "",
        "approvedBy": "",
        "approvedAt": "",
        "reviewer": "",
        "notes": "",
        "templateSource": {
            "decision": _clean_text(template.get("decision")),
            "templateOnly": bool(template.get("templateOnly")),
            "acceptedByThisTranche": bool(template.get("acceptedByThisTranche")),
        },
        "draftOnly": True,
        "decisionScope": "manual_lookup_source_recovery_decision_file_draft_only_no_source_or_parsed_mutation",
        "evidenceTier": "manual_lookup_source_recovery_decision_file_draft_only",
        "reportOnly": True,
        "applyReady": False,
        "externalLookupAttempted": False,
        "sourceDownloadAttempted": False,
        "sourceRegistrationMutationAttempted": False,
        "parsedArtifactWriteAttempted": False,
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerIntegration": False,
        "blockers": [
            "decision_file_draft_only",
            "decision_not_validated",
            "approved_source_missing",
            "source_recovery_requires_later_allowlist_apply_tranche",
            "parsed_artifact_materialization_requires_later_apply_tranche",
        ],
    }


def _decision_file_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "draftOnly": True,
        "instructions": [
            "Edit a copy of this file before using it as a decision file.",
            "Keep decision=needs_review unless a human/operator has made an explicit source decision.",
            "Source approvals require exactly one approvedSourceUrl or approvedLocalPdfPath, approvedSourceContentHash, plus reviewer notes.",
            "This draft does not perform lookup, download sources, mutate registrations, or materialize parsed artifacts.",
        ],
        "decisions": [
            {
                "sourceReviewCardId": _clean_text(row.get("sourceReviewCardId")),
                "paperId": _clean_text(row.get("paperId")),
                "paperTitle": _clean_text(row.get("paperTitle")),
                "manualLookupMode": _clean_text(row.get("manualLookupMode")),
                "lookupPriority": _clean_text(row.get("lookupPriority")),
                "lookupQueries": dict(row.get("lookupQueries") or {}),
                "allowedDecisions": list(row.get("allowedDecisions") or []),
                "decision": "needs_review",
                "approvedSourceType": "",
                "approvedSourceUrl": "",
                "approvedLocalPdfPath": "",
                "approvedSourceContentHash": "",
                "approvedBy": "",
                "approvedAt": "",
                "reviewer": "",
                "notes": "",
            }
            for row in rows
        ],
    }


def build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft(
    *,
    review_pack_report: dict[str, Any],
    review_pack_report_path: str | Path | None = None,
    report_name: str = "parsed-artifact-manual-lookup-source-recovery-decision-file-draft",
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a needs-review-only manual lookup source recovery decision draft."""

    validation = validate_payload(
        review_pack_report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID,
        strict=True,
    )
    unsafe_flags = _unsafe_flags(review_pack_report, bool(validation.ok))
    if validation.ok:
        cards = [
            dict(card)
            for card in list(review_pack_report.get("reviewCards") or [])
            if isinstance(card, dict)
        ]
        holdouts = [
            dict(row)
            for row in list(review_pack_report.get("textSourceHoldouts") or [])
            if isinstance(row, dict)
        ]
        coverage_target = dict(review_pack_report.get("coverageTarget") or {})
    else:
        cards = []
        holdouts = []
        coverage_target = {
            "missingParsedArtifactsBefore": 0,
            "targetMissingParsedArtifacts": 0,
            "manualRecoveriesNeededForTarget": 0,
            "manualLookupCandidateRows": 0,
            "targetReachableIfEnoughManualSourcesResolved": False,
            "potentialMissingParsedArtifactsAfterAllManualLookupRecovered": 0,
            "textSourceRowsRemainAfterManualLookup": 0,
        }
    rows = [_draft_row(index, card) for index, card in enumerate(cards, start=1)]
    priority_counter = Counter(_clean_text(row.get("lookupPriority")) for row in rows)
    mode_counter = Counter(_clean_text(row.get("manualLookupMode")) for row in rows)
    counts = {
        "inputRows": len(cards) + len(holdouts),
        "sourceReviewCardRows": len(cards),
        "draftDecisionRows": len(rows),
        "needsReviewRows": len(rows),
        "approvedDecisionRows": 0,
        "rejectedDecisionRows": 0,
        "applyReadyRows": 0,
        "textSourceHoldoutRows": len(holdouts),
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "manualLookupModeTaxonomy": _counter_items(mode_counter),
        "lookupPriorityTaxonomy": _counter_items(priority_counter),
    }
    counts.update(_mutation_counters())
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif rows:
        status = "decision_file_draft_ready"
        decision = "needs_review_draft_ready_for_manual_edit"
    else:
        status = "no_draft_rows"
        decision = "no_manual_lookup_review_cards"
    report = {
        "schema": PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_DRAFT_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or _utc_now(),
        "report": {
            "name": report_name,
            "inputReviewPackReportPath": str(Path(str(review_pack_report_path)).expanduser())
            if review_pack_report_path
            else "",
            "selectionRule": "reviewCards become needs_review draft decision rows; textSourceHoldouts remain excluded",
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
        "inputReviewPack": {
            "schemaValidation": {"ok": bool(validation.ok), "errors": list(validation.errors)},
            "schema": _clean_text(review_pack_report.get("schema")),
            "status": _clean_text(review_pack_report.get("status")),
            "reviewCardRows": int(dict(review_pack_report.get("counts") or {}).get("manualLookupReviewCardRows") or 0)
            if validation.ok
            else 0,
            "textSourceHoldoutRows": int(dict(review_pack_report.get("counts") or {}).get("textSourceHoldoutRows") or 0)
            if validation.ok
            else 0,
        },
        "counts": counts,
        "coverageTarget": coverage_target,
        "manualLookupDecisionDraftPaperIds": [_clean_text(row.get("paperId")) for row in rows],
        "decisionFileDraft": _decision_file_from_rows(rows),
        "draftRows": rows,
        "textSourceHoldoutPaperIds": [_clean_text(row.get("paperId")) for row in holdouts],
        "textSourceHoldouts": holdouts,
        "gate": {
            "decisionFileDraftReady": bool(rows) and not unsafe_flags,
            "containsOnlyNeedsReviewDefaults": True,
            "containsAcceptedSourceApprovals": False,
            "containsRejectedDecisions": False,
            "applyReady": False,
            "sourceRegistrationMutationReady": False,
            "parsedArtifactMaterializationReady": False,
            "parserRoutingReady": False,
            "strictEvidenceReady": False,
            "answerIntegrationReady": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "parsed_artifact_manual_lookup_source_recovery_decision_file_validation"
            if rows
            else "parsed_artifact_manual_lookup_source_recovery_review_pack_refresh",
        },
        "mutationPolicy": _mutation_policy(),
        "mutationCounters": _mutation_counters(),
        "warnings": [
            "draft_rows_are_not_recorded_decisions",
            "draft_decision_file_defaults_every_row_to_needs_review",
            "source_approval_requires_human_edit_and_later_validation",
            "source_recovery_and_parsed_artifact_materialization_require_later_explicit_apply_tranches",
        ],
    }
    output_validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_DRAFT_SCHEMA_ID,
        strict=True,
    )
    if not output_validation.ok:
        raise ValueError(
            "parsed artifact manual lookup source recovery decision file draft schema validation failed: "
            + "; ".join(output_validation.errors[:5])
        )
    return report


def write_parsed_artifact_manual_lookup_source_recovery_decision_file_draft(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write report JSON, editable draft JSON, summary JSON, and Markdown."""

    validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_DRAFT_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        raise ValueError(
            "parsed artifact manual lookup source recovery decision file draft schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-manual-lookup-source-recovery-decision-file-draft.json"
    decision_file_path = root / "manual-lookup-source-recovery-decisions.draft.json"
    summary_json_path = root / "parsed-artifact-manual-lookup-source-recovery-decision-file-draft-summary.json"
    markdown_path = root / "parsed-artifact-manual-lookup-source-recovery-decision-file-draft.md"
    summary = {
        "reportSchema": report.get("schema"),
        "status": report.get("status"),
        "counts": report.get("counts"),
        "coverageTarget": report.get("coverageTarget"),
        "gate": report.get("gate"),
        "mutationCounters": report.get("mutationCounters"),
        "reportFiles": {
            "reportJsonPath": str(report_path),
            "decisionFileDraftPath": str(decision_file_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    decision_file_path.write_text(
        json.dumps(report.get("decisionFileDraft") or {}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown_summary(report), encoding="utf-8")
    return {
        "reportJsonPath": str(report_path),
        "decisionFileDraftPath": str(decision_file_path),
        "summaryJsonPath": str(summary_json_path),
        "reportMarkdownPath": str(markdown_path),
    }


def _render_markdown_summary(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    target = dict(report.get("coverageTarget") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Parsed Artifact Manual Lookup Source Recovery Decision File Draft",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- draft decision rows: {counts.get('draftDecisionRows', 0)}",
        f"- `needs_review` rows: {counts.get('needsReviewRows', 0)}",
        f"- approved decision rows: {counts.get('approvedDecisionRows', 0)}",
        f"- apply-ready rows: {counts.get('applyReadyRows', 0)}",
        f"- text-source holdout rows: {counts.get('textSourceHoldoutRows', 0)}",
        f"- missing parsed artifacts before: {target.get('missingParsedArtifactsBefore', 0)}",
        f"- manual recoveries needed for target: {target.get('manualRecoveriesNeededForTarget', 0)}",
        f"- recommended next tranche: `{gate.get('recommendedNextTranche', '')}`",
        "",
        "## Boundary",
        "",
        "This draft is an editable starting point only. It does not record approvals, perform lookup, download files, mutate source registrations, materialize parsed artifacts, create evidence, scan the vault, or change answer behavior.",
        "",
        "## Manual Lookup Mode Taxonomy",
        "",
    ]
    for item in list(counts.get("manualLookupModeTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Draft Rows", ""])
    for row in list(report.get("draftRows") or []):
        lines.append(
            f"- `{row.get('paperId')}`: {row.get('paperTitle')} "
            f"(`{row.get('manualLookupMode')}`, priority `{row.get('lookupPriority')}`)"
        )
    lines.extend(["", "## Mutation Counters", ""])
    for key, value in sorted(dict(report.get("mutationCounters") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack-report", default=str(DEFAULT_REVIEW_PACK_REPORT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-name", default="parsed-artifact-manual-lookup-source-recovery-decision-file-draft")
    args = parser.parse_args(argv)

    review_pack_report_path = Path(args.review_pack_report).expanduser()
    review_pack_report = _load_json(review_pack_report_path)
    report = build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft(
        review_pack_report=review_pack_report,
        review_pack_report_path=review_pack_report_path,
        report_name=args.report_name,
    )
    paths = write_parsed_artifact_manual_lookup_source_recovery_decision_file_draft(report, args.output_dir)
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
    "PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_DRAFT_SCHEMA_ID",
    "build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft",
    "write_parsed_artifact_manual_lookup_source_recovery_decision_file_draft",
]
