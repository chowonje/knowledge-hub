"""Report-only operator decision packet for manual lookup source recovery.

Builds a concise operator review sheet from the human review report and the
editable decision draft. It does not download sources, mutate registrations,
materialize parsed artifacts, create evidence, scan the vault, or change answer
behavior.
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
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_human_review import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_validation import (
    DECISION_APPROVE_LOCAL_PDF,
    DECISION_APPROVE_SOURCE_URL,
    DECISION_HOLD,
    DECISION_NEEDS_REVIEW,
    DECISION_REJECT,
)


PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_OPERATOR_DECISION_PACKET_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-manual-lookup-source-recovery-operator-decision-packet.v1"
)

APPROVED_DECISIONS = frozenset({DECISION_APPROVE_SOURCE_URL, DECISION_APPROVE_LOCAL_PDF})

MISSING_REASON_BY_MODE = {
    "registered_pdf_path_missing_manual_lookup": "registered_pdf_path_missing",
    "no_registered_source_artifact_manual_lookup": "no_registered_source_artifact",
}

RISK_NOTE_BY_MODE = {
    "registered_pdf_path_missing_manual_lookup": (
        "Registered PDF path is absent on disk. Do not invent URLs; approve only an "
        "explicit publisher/open-access PDF URL or a verified local PDF path."
    ),
    "no_registered_source_artifact_manual_lookup": (
        "No registered PDF source artifact exists. Operator must supply an explicit "
        "URL or local path after manual lookup; never guess or fabricate source URLs."
    ),
}

REQUIRED_FIELDS_BY_DECISION = {
    DECISION_NEEDS_REVIEW: [],
    DECISION_APPROVE_SOURCE_URL: [
        "approvedSourceType",
        "approvedSourceUrl",
        "approvedSourceContentHash",
        "approvedBy",
        "approvedAt",
        "notes",
    ],
    DECISION_APPROVE_LOCAL_PDF: [
        "approvedSourceType",
        "approvedLocalPdfPath",
        "approvedSourceContentHash",
        "approvedBy",
        "approvedAt",
        "notes",
    ],
    DECISION_REJECT: ["reviewer", "notes"],
    DECISION_HOLD: ["reviewer", "notes"],
}


def _default_report_root() -> Path:
    return Path.home() / ("." + "khub") / "reports" / "parsed-artifact-coverage" / "2026-05-21"


DEFAULT_HUMAN_REVIEW_REPORT_PATH = (
    _default_report_root()
    / "parsed-artifact-manual-lookup-source-recovery-decision-file-human-review"
    / "01-parsed-artifact-manual-lookup-source-recovery-decision-file-human-review"
    / "parsed-artifact-manual-lookup-source-recovery-decision-file-human-review.json"
)
DEFAULT_DECISION_FILE_PATH = (
    _default_report_root()
    / "parsed-artifact-manual-lookup-source-recovery-decision-file-draft"
    / "01-parsed-artifact-manual-lookup-source-recovery-decision-file-draft"
    / "manual-lookup-source-recovery-decisions.draft.json"
)
DEFAULT_OUTPUT_DIR = (
    _default_report_root()
    / "parsed-artifact-manual-lookup-source-recovery-operator-decision-packet"
    / "01-parsed-artifact-manual-lookup-source-recovery-operator-decision-packet"
)

EXPECTED_MANUAL_LOOKUP_ROWS = 15


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def _mutation_policy() -> dict[str, Any]:
    return {
        "externalLookup": False,
        "sourceDownload": False,
        "sourceRegistrationMutation": False,
        "parsedArtifactWrite": False,
        "strictEvidence": False,
        "citationEvidence": False,
        "runtimeEvidence": False,
        "databaseMutation": False,
        "indexMutation": False,
        "reindexOrReembed": False,
        "vaultScan": False,
        "vaultWrite": False,
        "answerIntegration": False,
        "decisionFileMutation": False,
        "humanDecisionRecording": False,
    }


def _mutation_counters() -> dict[str, int]:
    return {
        "externalLookupRows": 0,
        "sourceDownloadRows": 0,
        "sourceRegistrationMutationRows": 0,
        "parsedArtifactWriteRows": 0,
        "strictEvidenceRows": 0,
        "citationEvidenceRows": 0,
        "runtimeEvidenceRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultReadRows": 0,
        "vaultWriteRows": 0,
        "answerIntegrationRows": 0,
        "decisionFileMutationRows": 0,
        "humanDecisionRecordingRows": 0,
    }


def _row_key(paper_id: str, source_review_card_id: str) -> str:
    return f"{paper_id}::{source_review_card_id}"


def _lookup_query_text(queries: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("titlePdf", "exactTitle", "paperId", "registeredFileName"):
        value = _clean_text(queries.get(key))
        if value:
            parts.append(f"{key}={value}")
    return "; ".join(parts)


def _missing_reason(mode: str) -> str:
    return MISSING_REASON_BY_MODE.get(_clean_text(mode), "manual_lookup_required")


def _risk_note(mode: str) -> str:
    return RISK_NOTE_BY_MODE.get(
        _clean_text(mode),
        "Manual lookup required; do not invent source URLs or approve without explicit evidence.",
    )


def _approval_fields_present(row: dict[str, Any], decision: str) -> bool:
    if not _clean_text(row.get("approvedSourceContentHash")):
        return False
    if decision == DECISION_APPROVE_SOURCE_URL:
        return bool(_clean_text(row.get("approvedSourceType"))) and bool(_clean_text(row.get("approvedSourceUrl")))
    if decision == DECISION_APPROVE_LOCAL_PDF:
        return bool(_clean_text(row.get("approvedSourceType"))) and bool(_clean_text(row.get("approvedLocalPdfPath")))
    return False


def _reviewer_fields_present(row: dict[str, Any]) -> bool:
    return bool(_clean_text(row.get("reviewer"))) and bool(_clean_text(row.get("notes")))


def _effective_decision(decision_row: dict[str, Any]) -> str:
    decision = _clean_text(decision_row.get("decision")) or DECISION_NEEDS_REVIEW
    if decision in APPROVED_DECISIONS:
        if not _approval_fields_present(decision_row, decision):
            return DECISION_NEEDS_REVIEW
        if not _clean_text(decision_row.get("approvedBy")) or not _clean_text(decision_row.get("approvedAt")):
            return DECISION_NEEDS_REVIEW
        if not _clean_text(decision_row.get("notes")):
            return DECISION_NEEDS_REVIEW
        return decision
    if decision in {DECISION_REJECT, DECISION_HOLD}:
        return decision if _reviewer_fields_present(decision_row) else DECISION_NEEDS_REVIEW
    if decision == DECISION_NEEDS_REVIEW:
        return DECISION_NEEDS_REVIEW
    return DECISION_NEEDS_REVIEW


def _required_fields_for_decision(decision: str) -> list[str]:
    return list(REQUIRED_FIELDS_BY_DECISION.get(decision, REQUIRED_FIELDS_BY_DECISION[DECISION_NEEDS_REVIEW]))


def _operator_review_row(
    index: int,
    *,
    human_row: dict[str, Any],
    decision_row: dict[str, Any] | None,
) -> dict[str, Any]:
    mode = _clean_text(human_row.get("manualLookupMode"))
    queries = dict(human_row.get("lookupQueries") or {})
    decision = _effective_decision(decision_row or human_row)
    allowed = list(human_row.get("allowedDecisions") or decision_row.get("allowedDecisions") or [])
    return {
        "operatorReviewRowId": f"manual-lookup-source-recovery-operator-review:{index:04d}",
        "paperId": _clean_text(human_row.get("paperId")),
        "paperTitle": _clean_text(human_row.get("paperTitle")),
        "sourceReviewCardId": _clean_text(human_row.get("sourceReviewCardId")),
        "currentMissingReason": _missing_reason(mode),
        "manualLookupMode": mode,
        "lookupPriority": _clean_text(human_row.get("lookupPriority")),
        "lookupQuery": _lookup_query_text(queries),
        "lookupQueries": queries,
        "currentDecision": decision,
        "allowedDecisions": allowed,
        "requiredApprovalFields": _required_fields_for_decision(decision),
        "requiredFieldsByDecision": {
            key: _required_fields_for_decision(key) for key in allowed if key in REQUIRED_FIELDS_BY_DECISION
        },
        "riskNote": _risk_note(mode),
        "decisionEditTarget": "manual-lookup-source-recovery-decisions.draft.json",
        "reportOnly": True,
    }


def _unsafe_flags(human_review_report: dict[str, Any], human_review_ok: bool) -> list[str]:
    flags: list[str] = []
    gate = dict(human_review_report.get("gate") or {})
    if not human_review_ok:
        flags.append("human_review_report_schema_violation")
    if human_review_report.get("schema") != PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID:
        flags.append("human_review_report_schema_mismatch")
    if _clean_text(human_review_report.get("status")) != "decision_file_human_review_ready":
        flags.append(f"human_review_status={_clean_text(human_review_report.get('status')) or 'unknown'}")
    if not bool(gate.get("humanReviewSheetReady")):
        flags.append("human_review_sheet_not_ready")
    if list(gate.get("schemaViolations") or []):
        flags.extend(str(item) for item in gate.get("schemaViolations") or [])
    return list(dict.fromkeys(flags))


def build_parsed_artifact_manual_lookup_source_recovery_operator_decision_packet(
    *,
    human_review_report: dict[str, Any],
    decision_file: dict[str, Any],
    human_review_report_path: str | Path | None = None,
    decision_file_path: str | Path | None = None,
    report_name: str = "parsed-artifact-manual-lookup-source-recovery-operator-decision-packet",
    generated_at: str | None = None,
    expected_manual_lookup_rows: int = EXPECTED_MANUAL_LOOKUP_ROWS,
) -> dict[str, Any]:
    """Build a concise operator decision packet from human review inputs."""

    human_review_validation = validate_payload(
        human_review_report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID,
        strict=True,
    )
    unsafe_flags = _unsafe_flags(human_review_report, bool(human_review_validation.ok))

    human_rows = [
        dict(row)
        for row in list(human_review_report.get("humanReviewRows") or [])
        if isinstance(row, dict)
    ] if human_review_validation.ok else []

    decision_index = {
        _row_key(_clean_text(row.get("paperId")), _clean_text(row.get("sourceReviewCardId"))): dict(row)
        for row in list(decision_file.get("decisions") or [])
        if isinstance(row, dict)
    }

    operator_rows: list[dict[str, Any]] = []
    for index, human_row in enumerate(human_rows, start=1):
        key = _row_key(
            _clean_text(human_row.get("paperId")),
            _clean_text(human_row.get("sourceReviewCardId")),
        )
        operator_rows.append(
            _operator_review_row(index, human_row=human_row, decision_row=decision_index.get(key))
        )

    decision_counter = Counter(_clean_text(row.get("currentDecision")) for row in operator_rows)
    counts = {
        "operatorReviewRows": len(operator_rows),
        "needsReviewRows": decision_counter.get(DECISION_NEEDS_REVIEW, 0),
        "approvedDecisionRows": sum(decision_counter.get(item, 0) for item in APPROVED_DECISIONS),
        "rejectedDecisionRows": decision_counter.get(DECISION_REJECT, 0),
        "holdForManualLookupRows": decision_counter.get(DECISION_HOLD, 0),
        "textSourceHoldoutRows": int(dict(human_review_report.get("counts") or {}).get("textSourceHoldoutRows") or 0),
        "unexpectedBlockerCount": int(
            dict(human_review_report.get("remainingBlockerClosure") or {}).get("unexpectedBlockerCount") or 0
        ),
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "decisionTaxonomy": _counter_items(decision_counter),
    }
    counts.update(_mutation_counters())

    ready = (
        not unsafe_flags
        and len(operator_rows) == expected_manual_lookup_rows
        and counts["needsReviewRows"] == expected_manual_lookup_rows
        and counts["approvedDecisionRows"] == 0
    )
    status = "operator_decision_packet_ready" if ready else "blocked"
    closure = dict(human_review_report.get("remainingBlockerClosure") or {})

    report = {
        "schema": PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_OPERATOR_DECISION_PACKET_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or _utc_now(),
        "report": {
            "name": report_name,
            "inputHumanReviewReportPath": str(Path(str(human_review_report_path)).expanduser())
            if human_review_report_path
            else "",
            "inputDecisionFilePath": str(Path(str(decision_file_path)).expanduser()) if decision_file_path else "",
            "decisionDraftFileName": "manual-lookup-source-recovery-decisions.draft.json",
            "selectionRule": (
                "emit concise operator review rows from validated human review context; "
                "preserve explicit approvals only when already present in the decision draft"
            ),
            "nonScope": [
                "external_lookup",
                "source_download",
                "source_registration_mutation",
                "parsed_artifact_write",
                "strict_or_citation_or_runtime_evidence",
                "database_or_index_or_reembed",
                "vault_scan_or_write",
                "answer_integration",
                "decision_file_mutation",
                "invented_source_urls",
            ],
        },
        "inputHumanReview": {
            "schemaValidation": {
                "ok": bool(human_review_validation.ok),
                "errors": list(human_review_validation.errors),
            },
            "schema": _clean_text(human_review_report.get("schema")),
            "status": _clean_text(human_review_report.get("status")),
            "manualReviewRows": len(human_rows),
        },
        "inputDecisionFile": {
            "draftOnly": bool(decision_file.get("draftOnly")),
            "decisionRows": len(list(decision_file.get("decisions") or [])),
        },
        "counts": counts,
        "coverageTarget": dict(human_review_report.get("coverageTarget") or {}),
        "remainingBlockerClosure": closure,
        "operatorReviewRows": operator_rows,
        "humanRequiredFields": {
            "sharedEditTarget": "manual-lookup-source-recovery-decisions.draft.json",
            "approveSourceUrl": REQUIRED_FIELDS_BY_DECISION[DECISION_APPROVE_SOURCE_URL],
            "approveLocalPdfPath": REQUIRED_FIELDS_BY_DECISION[DECISION_APPROVE_LOCAL_PDF],
            "rejectOrHold": REQUIRED_FIELDS_BY_DECISION[DECISION_REJECT],
            "needsReview": [],
            "notes": [
                "Do not invent source URLs.",
                "Leave decision=needs_review when unsure.",
                "Re-run decision-file validation after edits.",
            ],
        },
        "gate": {
            "operatorDecisionPacketReady": ready,
            "containsOnlyNeedsReviewRows": counts["needsReviewRows"] == len(operator_rows) and bool(operator_rows),
            "applyReady": False,
            "sourceRegistrationMutationReady": False,
            "parsedArtifactMaterializationReady": False,
            "decision": (
                "manual_lookup_operator_decision_packet_ready_pending_human_edit"
                if ready
                else "manual_lookup_operator_decision_packet_blocked"
            ),
            "schemaViolations": unsafe_flags,
            "recommendedNextTranche": (
                "parsed_artifact_manual_lookup_source_recovery_decision_file_validation_after_human_edit"
                if ready
                else "parsed_artifact_manual_lookup_source_recovery_decision_file_human_review_repair"
            ),
        },
        "mutationPolicy": _mutation_policy(),
        "mutationCounters": _mutation_counters(),
        "warnings": [
            "operator_packet_does_not_edit_decision_file",
            "operator_packet_does_not_invent_source_urls",
            "needs_review_is_default_until_explicit_approval_fields_are_present",
        ],
    }
    output_validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_OPERATOR_DECISION_PACKET_SCHEMA_ID,
        strict=True,
    )
    if not output_validation.ok:
        raise ValueError(
            "parsed artifact manual lookup source recovery operator decision packet schema failed: "
            + "; ".join(output_validation.errors[:5])
        )
    return report


def render_operator_review_sheet_markdown(report: dict[str, Any]) -> str:
    closure = dict(report.get("remainingBlockerClosure") or {})
    required = dict(report.get("humanRequiredFields") or {})
    lines = [
        "# Manual Lookup Source Recovery Operator Review Sheet",
        "",
        f"- packet status: `{report.get('status')}`",
        f"- decision draft: `{dict(report.get('report') or {}).get('decisionDraftFileName', '')}`",
        f"- operator rows: `{int(dict(report.get('counts') or {}).get('operatorReviewRows') or 0)}`",
        f"- needs_review rows: `{int(dict(report.get('counts') or {}).get('needsReviewRows') or 0)}`",
        "",
        "## Remaining Blockers",
        "",
        f"- missing parsed artifacts remaining: `{closure.get('missingParsedArtifactsRemaining', 0)}`",
    ]
    for item in list(closure.get("blockerTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(
        [
            "",
            "## Fields The Human Must Fill",
            "",
            f"- edit target: `{required.get('sharedEditTarget', '')}`",
            f"- approve URL: `{', '.join(required.get('approveSourceUrl') or [])}`",
            f"- approve local PDF: `{', '.join(required.get('approveLocalPdfPath') or [])}`",
            f"- reject/hold: `{', '.join(required.get('rejectOrHold') or [])}`",
            "",
            "## Rows",
            "",
        ]
    )
    for row in list(report.get("operatorReviewRows") or []):
        lines.extend(
            [
                f"### {row.get('paperId', '')}",
                "",
                f"- paper_id: `{row.get('paperId', '')}`",
                f"- current missing reason: `{row.get('currentMissingReason', '')}`",
                f"- lookup query: `{row.get('lookupQuery', '')}`",
                f"- current decision: `{row.get('currentDecision', '')}`",
                f"- allowed decisions: `{', '.join(row.get('allowedDecisions') or [])}`",
                f"- required approval fields (current decision): `{', '.join(row.get('requiredApprovalFields') or []) or 'none'}`",
                f"- risk note: {row.get('riskNote', '')}",
                "",
            ]
        )
    return "\n".join(lines)


def write_parsed_artifact_manual_lookup_source_recovery_operator_decision_packet(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_OPERATOR_DECISION_PACKET_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        raise ValueError(
            "parsed artifact manual lookup source recovery operator decision packet schema failed: "
            + "; ".join(validation.errors[:5])
        )
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    packet_path = root / "manual-lookup-source-recovery-operator-decision-packet.json"
    sheet_json_path = root / "manual-lookup-source-recovery-operator-review-sheet.json"
    sheet_md_path = root / "manual-lookup-source-recovery-operator-review-sheet.md"
    summary_path = root / "manual-lookup-source-recovery-operator-decision-packet-summary.json"

    sheet_payload = {
        "schema": "knowledge-hub.paper.parsed-artifact-manual-lookup-source-recovery-operator-review-sheet.v1",
        "generatedAt": report.get("generatedAt"),
        "decisionDraftPath": dict(report.get("report") or {}).get("inputDecisionFilePath", ""),
        "humanRequiredFields": report.get("humanRequiredFields"),
        "remainingBlockerClosure": report.get("remainingBlockerClosure"),
        "operatorReviewRows": report.get("operatorReviewRows"),
    }
    summary = {
        "reportSchema": report.get("schema"),
        "status": report.get("status"),
        "counts": report.get("counts"),
        "remainingBlockerClosure": report.get("remainingBlockerClosure"),
        "gate": report.get("gate"),
        "reportFiles": {
            "packetJsonPath": str(packet_path),
            "reviewSheetJsonPath": str(sheet_json_path),
            "reviewSheetMarkdownPath": str(sheet_md_path),
        },
    }

    packet_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sheet_json_path.write_text(
        json.dumps(sheet_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sheet_md_path.write_text(render_operator_review_sheet_markdown(report), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "packetJsonPath": str(packet_path),
        "reviewSheetJsonPath": str(sheet_json_path),
        "reviewSheetMarkdownPath": str(sheet_md_path),
        "summaryJsonPath": str(summary_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-review-report", default=str(DEFAULT_HUMAN_REVIEW_REPORT_PATH))
    parser.add_argument("--decision-file", default=str(DEFAULT_DECISION_FILE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write local operator packet and review sheet files. Default is dry-run summary only.",
    )
    args = parser.parse_args(argv)

    human_review_path = Path(args.human_review_report).expanduser()
    decision_file_path = Path(args.decision_file).expanduser()
    report = build_parsed_artifact_manual_lookup_source_recovery_operator_decision_packet(
        human_review_report=_load_json(human_review_path),
        decision_file=_load_json(decision_file_path),
        human_review_report_path=human_review_path,
        decision_file_path=decision_file_path,
    )
    payload: dict[str, Any] = {
        "schema": report["schema"],
        "status": report["status"],
        "counts": report["counts"],
        "remainingBlockerClosure": report["remainingBlockerClosure"],
        "humanRequiredFields": report["humanRequiredFields"],
        "gate": report["gate"],
        "decisionDraftPath": str(decision_file_path),
    }
    if args.apply:
        payload["paths"] = write_parsed_artifact_manual_lookup_source_recovery_operator_decision_packet(
            report,
            args.output_dir,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_OPERATOR_DECISION_PACKET_SCHEMA_ID",
    "build_parsed_artifact_manual_lookup_source_recovery_operator_decision_packet",
    "render_operator_review_sheet_markdown",
    "write_parsed_artifact_manual_lookup_source_recovery_operator_decision_packet",
]
