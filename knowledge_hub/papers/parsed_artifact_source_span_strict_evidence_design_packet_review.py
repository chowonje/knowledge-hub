"""Report-only design packet review for reconciled strict-evidence design rows.

Reviews the 99 reconciled design-ready rows for internal consistency before any
StrictEvidence executor or SourceSpan apply tranche. Does not create StrictEvidence,
mutate SourceSpan JSONL, or change runtime/parser/answer/DB state.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_design_review_reconciliation import (
    FINAL_STATUS_READY,
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_RECONCILIATION_SCHEMA_ID,
    RECOMMENDED_ACTION_READY as RECONCILIATION_PACKET_QUEUE_ACTION,
    SOURCE_DISAMBIGUATION_DESIGN,
    SOURCE_ORIGINAL_DESIGN_REVIEW,
)


PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_PACKET_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-strict-evidence-design-packet-review.v1"
)

PACKET_REVIEW_STATUS_READY = "design_packet_review_ready"
PACKET_REVIEW_STATUS_BLOCKED_MISSING_IDENTITY = "blocked_missing_record_identity"
PACKET_REVIEW_STATUS_BLOCKED_MISSING_CHARS = "blocked_missing_proposed_chars"
PACKET_REVIEW_STATUS_BLOCKED_INVALID_BASIS = "blocked_invalid_chars_basis"
PACKET_REVIEW_STATUS_BLOCKED_MISSING_HASH = "blocked_missing_expected_substring_hash"
PACKET_REVIEW_STATUS_BLOCKED_RUNTIME_FLAGS = "blocked_runtime_or_strict_flag_violation"
PACKET_REVIEW_STATUS_BLOCKED_UNEXPECTED_MANUAL = "blocked_unexpected_manual_row_in_packet"
PACKET_REVIEW_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

PACKET_REVIEW_RECOMMENDED_ACTION_READY = "queue_for_strict_evidence_record_contract"
PACKET_REVIEW_RECOMMENDED_ACTION_BLOCKED = "hold_packet_row_until_blockers_resolved"
PACKET_REVIEW_RECOMMENDED_ACTION_REPAIR = "repair_reconciliation_report_before_packet_review"

CHARS_BASIS = "sourceContentHash"
ALLOWED_PACKET_SOURCES = {SOURCE_ORIGINAL_DESIGN_REVIEW, SOURCE_DISAMBIGUATION_DESIGN}

DEFAULT_RECONCILIATION_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-strict-evidence-design-review-reconciliation"
    / "01-parsed-artifact-source-span-strict-evidence-design-review-reconciliation"
    / "parsed-artifact-source-span-strict-evidence-design-review-reconciliation.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-strict-evidence-design-packet-review"
    / "01-parsed-artifact-source-span-strict-evidence-design-packet-review"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = _safe_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _proposed_chars_dict(row: dict[str, Any]) -> dict[str, Any]:
    proposed = row.get("proposed_chars")
    return proposed if isinstance(proposed, dict) else {}


def _paper_id(row: dict[str, Any]) -> str:
    return _safe_text(row.get("paper_id")) or _safe_text(row.get("paperId"))


def _artifact_type(row: dict[str, Any]) -> str:
    return _safe_text(row.get("artifact_type")) or _safe_text(row.get("artifactType"))


def _source_file(row: dict[str, Any]) -> str:
    if "source_file" in row:
        return _safe_text(row.get("source_file"))
    if "sourceFile" in row:
        return _safe_text(row.get("sourceFile"))
    return ""


def _has_source_file_field(row: dict[str, Any]) -> bool:
    return "source_file" in row or "sourceFile" in row


def _runtime_flag_violations(row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    forbidden_true_fields = (
        "strictEligible",
        "citationGrade",
        "runtimeEvidence",
        "strictEvidenceCreated",
        "parserRoutingChanged",
        "answerIntegrationChanged",
        "databaseMutation",
        "canonicalParsedArtifactsWritten",
    )
    for field in forbidden_true_fields:
        if _safe_bool(row.get(field)) is True:
            violations.append(f"{field}=true")
    return _dedupe(violations)


def _is_packet_candidate(row: dict[str, Any]) -> bool:
    return (
        _safe_text(row.get("final_status")) == FINAL_STATUS_READY
        and _safe_text(row.get("recommended_action")) == RECONCILIATION_PACKET_QUEUE_ACTION
        and bool(_safe_bool(row.get("readyForStrictEvidenceDesignReview")))
    )


def _classify_packet_row(row: dict[str, Any]) -> tuple[str, list[str], str]:
    blockers: list[str] = ["strict_evidence_design_packet_review_only"]

    if not _is_packet_candidate(row):
        return (
            PACKET_REVIEW_STATUS_BLOCKED_UNEXPECTED_MANUAL,
            _dedupe([*blockers, "row_not_in_final_ready_packet_queue"]),
            PACKET_REVIEW_RECOMMENDED_ACTION_BLOCKED,
        )

    source = _safe_text(row.get("source"))
    if source not in ALLOWED_PACKET_SOURCES:
        blockers.append(f"packet_source_invalid:{source or 'missing'}")
        return (
            PACKET_REVIEW_STATUS_BLOCKED_MISSING_IDENTITY,
            blockers,
            PACKET_REVIEW_RECOMMENDED_ACTION_BLOCKED,
        )

    source_span_id = _safe_text(row.get("sourceSpanId"))
    candidate_record_id = _safe_text(row.get("candidateRecordId"))
    paper_id = _paper_id(row)
    artifact_type = _artifact_type(row)
    source_content_hash = _safe_text(row.get("sourceContentHash"))

    if not source_span_id or not candidate_record_id or not paper_id or not artifact_type:
        missing = []
        if not source_span_id:
            missing.append("sourceSpanId")
        if not candidate_record_id:
            missing.append("candidateRecordId")
        if not paper_id:
            missing.append("paper_id")
        if not artifact_type:
            missing.append("artifact_type")
        return (
            PACKET_REVIEW_STATUS_BLOCKED_MISSING_IDENTITY,
            _dedupe([*blockers, *[f"missing_{field}" for field in missing]]),
            PACKET_REVIEW_RECOMMENDED_ACTION_BLOCKED,
        )

    if not source_content_hash:
        return (
            PACKET_REVIEW_STATUS_BLOCKED_MISSING_IDENTITY,
            _dedupe([*blockers, "missing_sourceContentHash"]),
            PACKET_REVIEW_RECOMMENDED_ACTION_BLOCKED,
        )

    if not _has_source_file_field(row):
        return (
            PACKET_REVIEW_STATUS_BLOCKED_MISSING_IDENTITY,
            _dedupe([*blockers, "missing_source_file_field"]),
            PACKET_REVIEW_RECOMMENDED_ACTION_BLOCKED,
        )

    runtime_violations = _runtime_flag_violations(row)
    if runtime_violations:
        return (
            PACKET_REVIEW_STATUS_BLOCKED_RUNTIME_FLAGS,
            _dedupe([*blockers, *runtime_violations]),
            PACKET_REVIEW_RECOMMENDED_ACTION_BLOCKED,
        )

    proposed = _proposed_chars_dict(row)
    start = _safe_int(proposed.get("start"))
    end = _safe_int(proposed.get("end"))
    if start is None or end is None or end <= start:
        return (
            PACKET_REVIEW_STATUS_BLOCKED_MISSING_CHARS,
            _dedupe([*blockers, "chars_start_or_end_missing_or_invalid"]),
            PACKET_REVIEW_RECOMMENDED_ACTION_BLOCKED,
        )

    basis = _safe_text(proposed.get("basis"))
    if basis != CHARS_BASIS:
        return (
            PACKET_REVIEW_STATUS_BLOCKED_INVALID_BASIS,
            _dedupe([*blockers, f"chars_basis_invalid:{basis or 'missing'}"]),
            PACKET_REVIEW_RECOMMENDED_ACTION_BLOCKED,
        )

    if not _safe_text(proposed.get("normalization")):
        return (
            PACKET_REVIEW_STATUS_BLOCKED_INVALID_BASIS,
            _dedupe([*blockers, "chars_normalization_missing"]),
            PACKET_REVIEW_RECOMMENDED_ACTION_BLOCKED,
        )

    if not _safe_text(proposed.get("expectedSubstringSha256")):
        return (
            PACKET_REVIEW_STATUS_BLOCKED_MISSING_HASH,
            _dedupe([*blockers, "chars_expectedSubstringSha256_missing"]),
            PACKET_REVIEW_RECOMMENDED_ACTION_BLOCKED,
        )

    proposed_hash = _safe_text(proposed.get("sourceContentHash"))
    if proposed_hash and proposed_hash != source_content_hash:
        blockers.append("proposed_chars_sourceContentHash_mismatch")

    return (
        PACKET_REVIEW_STATUS_READY,
        _dedupe(
            [
                *blockers,
                "source_span_remains_non_strict",
                "strict_evidence_creation_disabled_for_tranche",
            ]
        ),
        PACKET_REVIEW_RECOMMENDED_ACTION_READY,
    )


def _packet_review_row(reconciliation_row: dict[str, Any], *, index: int) -> dict[str, Any]:
    review_status, review_blockers, recommended_action = _classify_packet_row(reconciliation_row)
    proposed = _proposed_chars_dict(reconciliation_row)
    return {
        "packet_review_row_id": (
            f"parsed-artifact-source-span-strict-evidence-design-packet-review:{index:04d}"
        ),
        "reconciliation_row_id": _safe_text(reconciliation_row.get("reconciliation_row_id")),
        "source": _safe_text(reconciliation_row.get("source")),
        "review_row_id": _safe_text(reconciliation_row.get("review_row_id")),
        "design_row_id": _safe_text(reconciliation_row.get("design_row_id")),
        "disambiguation_row_id": _safe_text(reconciliation_row.get("disambiguation_row_id")),
        "sourceSpanId": _safe_text(reconciliation_row.get("sourceSpanId")),
        "candidateRecordId": _safe_text(reconciliation_row.get("candidateRecordId")),
        "paper_id": _paper_id(reconciliation_row),
        "artifact_type": _artifact_type(reconciliation_row),
        "sourceContentHash": _safe_text(reconciliation_row.get("sourceContentHash")),
        "source_file": _source_file(reconciliation_row),
        "text_surface": _safe_text(reconciliation_row.get("text_surface")),
        "proposed_chars": proposed,
        "packet_review_status": review_status,
        "packet_review_blockers": review_blockers,
        "recommended_action": recommended_action,
        "designPacketReviewReady": review_status == PACKET_REVIEW_STATUS_READY,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "sourceSpanUpdatedRows": 0,
    }


def _count_rows(
    *,
    packet_rows: list[dict[str, Any]],
    excluded_manual_rows: list[dict[str, Any]],
    input_rows: int,
    input_schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": input_rows,
        "packetCandidateRows": len(packet_rows),
        "designPacketReviewReadyRows": sum(
            1 for row in packet_rows if row.get("packet_review_status") == PACKET_REVIEW_STATUS_READY
        ),
        "excludedManualOrExtractorRows": len(excluded_manual_rows),
        "blockedMissingRecordIdentityRows": sum(
            1
            for row in packet_rows
            if row.get("packet_review_status") == PACKET_REVIEW_STATUS_BLOCKED_MISSING_IDENTITY
        ),
        "blockedMissingProposedCharsRows": sum(
            1
            for row in packet_rows
            if row.get("packet_review_status") == PACKET_REVIEW_STATUS_BLOCKED_MISSING_CHARS
        ),
        "blockedInvalidCharsBasisRows": sum(
            1
            for row in packet_rows
            if row.get("packet_review_status") == PACKET_REVIEW_STATUS_BLOCKED_INVALID_BASIS
        ),
        "blockedMissingExpectedSubstringHashRows": sum(
            1
            for row in packet_rows
            if row.get("packet_review_status") == PACKET_REVIEW_STATUS_BLOCKED_MISSING_HASH
        ),
        "blockedRuntimeOrStrictFlagViolationRows": sum(
            1
            for row in packet_rows
            if row.get("packet_review_status") == PACKET_REVIEW_STATUS_BLOCKED_RUNTIME_FLAGS
        ),
        "blockedUnexpectedManualRowInPacketRows": sum(
            1
            for row in packet_rows
            if row.get("packet_review_status") == PACKET_REVIEW_STATUS_BLOCKED_UNEXPECTED_MANUAL
        ),
        "blockedInputSchemaViolationRows": int(bool(input_schema_violations)),
        "sourceSpanUpdatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "schemaViolationCount": len(input_schema_violations),
        "byPaperId": dict(Counter(str(row.get("paper_id") or "") for row in packet_rows)),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in packet_rows)),
        "byPacketSource": dict(Counter(str(row.get("source") or "") for row in packet_rows)),
        "byReviewStatus": dict(
            Counter(str(row.get("packet_review_status") or "") for row in packet_rows)
        ),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in packet_rows)),
    }


def build_parsed_artifact_source_span_strict_evidence_design_packet_review(
    *,
    reconciliation_report_path: str | Path = DEFAULT_RECONCILIATION_REPORT_PATH,
) -> dict[str, Any]:
    report_path = Path(str(reconciliation_report_path)).expanduser()
    warnings: list[str] = []
    input_schema_violations: list[str] = []

    reconciliation_report = _read_json(report_path)
    if not reconciliation_report:
        warnings.append("reconciliation_report_missing_or_unreadable")

    validation = validate_payload(
        reconciliation_report,
        PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_RECONCILIATION_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        input_schema_violations = [str(error) for error in validation.errors]
        if not reconciliation_report:
            input_schema_violations.append("reconciliation_report_missing_or_unreadable")

    input_rows = int((reconciliation_report.get("counts") or {}).get("inputDesignReviewRows") or 0)
    if not input_rows and reconciliation_report:
        input_rows = len(reconciliation_report.get("rows", []) or [])

    final_ready_rows = [
        row
        for row in reconciliation_report.get("finalReadyRows", [])
        if isinstance(row, dict)
    ] if isinstance(reconciliation_report, dict) else []

    if not final_ready_rows and isinstance(reconciliation_report, dict):
        final_ready_rows = [
            row
            for row in reconciliation_report.get("rows", [])
            if isinstance(row, dict) and _is_packet_candidate(row)
        ]

    excluded_manual_rows = [
        row
        for row in reconciliation_report.get("manualOrExtractorFollowUpRows", [])
        if isinstance(row, dict)
    ] if isinstance(reconciliation_report, dict) else []

    if not excluded_manual_rows and isinstance(reconciliation_report, dict):
        excluded_manual_rows = [
            row
            for row in reconciliation_report.get("rows", [])
            if isinstance(row, dict) and not _is_packet_candidate(row)
        ]

    packet_candidates = [row for row in final_ready_rows if _is_packet_candidate(row)]
    if len(packet_candidates) != len(final_ready_rows):
        warnings.append("final_ready_rows_contains_non_packet_candidates")

    unexpected_in_packet = [
        row for row in packet_candidates if not _is_packet_candidate(row)
    ]
    if unexpected_in_packet:
        warnings.append("unexpected_rows_in_packet_candidate_set")

    packet_rows = [_packet_review_row(row, index=index) for index, row in enumerate(packet_candidates)]

    if input_schema_violations:
        for row in packet_rows:
            row["packet_review_status"] = PACKET_REVIEW_STATUS_BLOCKED_INPUT_SCHEMA
            row["recommended_action"] = PACKET_REVIEW_RECOMMENDED_ACTION_REPAIR
            row["designPacketReviewReady"] = False
            row["packet_review_blockers"] = _dedupe(
                [
                    *(row.get("packet_review_blockers") or []),
                    *input_schema_violations,
                ]
            )

    counts = _count_rows(
        packet_rows=packet_rows,
        excluded_manual_rows=excluded_manual_rows,
        input_rows=input_rows,
        input_schema_violations=_dedupe(input_schema_violations),
    )

    ready_count = int(counts.get("designPacketReviewReadyRows") or 0)
    packet_candidate_count = int(counts.get("packetCandidateRows") or 0)
    excluded_count = int(counts.get("excludedManualOrExtractorRows") or 0)
    blocker_total = packet_candidate_count - ready_count

    status = "ok"
    if input_schema_violations:
        status = "blocked"
    elif blocker_total > 0:
        status = "blocked"
    elif ready_count == 0:
        status = "blocked"

    packet_ready = (
        status == "ok"
        and ready_count == packet_candidate_count
        and packet_candidate_count > 0
        and not input_schema_violations
    )

    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_PACKET_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "reconciliationReportPath": str(report_path),
            "reconciliationSchema": _safe_text(reconciliation_report.get("schema"))
            if reconciliation_report
            else "",
        },
        "counts": counts,
        "gate": {
            "designPacketReviewReady": packet_ready,
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "parsed_artifact_source_span_strict_evidence_design_packet_review_ready"
                if packet_ready
                else "blocked"
            ),
            "recommendedNextTranche": (
                "parsed_artifact_strict_evidence_record_contract"
                if packet_ready
                else "parsed_artifact_source_span_strict_evidence_design_packet_review_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "sourceSpanStoreWrite": False,
            "strictEvidenceCreated": False,
            "strictEligibleMutation": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "warnings": _dedupe(warnings),
        "packetRows": packet_rows,
        "excludedManualOrExtractorRows": excluded_manual_rows,
        "rows": [*packet_rows, *excluded_manual_rows],
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
            "gate",
            "policy",
            "warnings",
            "packetRows",
            "excludedManualOrExtractorRows",
            "rows",
        )
        if key in report
    }


def render_parsed_artifact_source_span_strict_evidence_design_packet_review_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byReviewStatus") or {})).items())
    ]
    by_source = [
        f"{source}: {count}"
        for source, count in sorted((dict(counts.get("byPacketSource") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact SourceSpan Strict Evidence Design Packet Review",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- packet candidate rows: {int(counts.get('packetCandidateRows') or 0)}",
            f"- design packet review ready: {int(counts.get('designPacketReviewReadyRows') or 0)}",
            f"- excluded manual/extractor rows: {int(counts.get('excludedManualOrExtractorRows') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            "",
            "## Review status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            "## Packet source breakdown",
            *[f"- {item}" for item in by_source],
        ]
    )


def write_parsed_artifact_source_span_strict_evidence_design_packet_review_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-span-strict-evidence-design-packet-review.json"
    summary_path = root / "parsed-artifact-source-span-strict-evidence-design-packet-review-summary.json"
    markdown_path = root / "parsed-artifact-source-span-strict-evidence-design-packet-review.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_strict_evidence_design_packet_review_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Review reconciled strict-evidence design packet rows for internal consistency "
            "without mutating SourceSpan records or creating StrictEvidence."
        )
    )
    parser.add_argument(
        "--reconciliation-report",
        default=str(DEFAULT_RECONCILIATION_REPORT_PATH),
        help="Strict-evidence design review reconciliation JSON report.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_source_span_strict_evidence_design_packet_review(
        reconciliation_report_path=args.reconciliation_report,
    )

    if args.output_dir:
        paths = write_parsed_artifact_source_span_strict_evidence_design_packet_review_reports(
            report,
            args.output_dir,
        )
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")

    if args.json or not args.output_dir:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_PACKET_REVIEW_SCHEMA_ID",
    "PACKET_REVIEW_STATUS_READY",
    "build_parsed_artifact_source_span_strict_evidence_design_packet_review",
    "write_parsed_artifact_source_span_strict_evidence_design_packet_review_reports",
]
