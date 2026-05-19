"""Report-only strict-evidence design review for SourceSpan offset proposals.

This helper consumes the original-source offset authority design report and
classifies rows into a strict-evidence design-review queue or a non-unique text
match disambiguation queue. It does not create StrictEvidence records, mutate
SourceSpan JSONL, or change runtime/parser/answer/DB state.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_original_source_offset_authority_design import (
    DESIGN_STATUS_BLOCKED_MANUAL_OR_LATER,
    DESIGN_STATUS_BLOCKED_NON_UNIQUE_TEXT_MATCH,
    DESIGN_STATUS_OFFSET_AUTHORITY_CANDIDATE,
    PARSED_ARTIFACT_SOURCE_SPAN_ORIGINAL_SOURCE_OFFSET_AUTHORITY_DESIGN_SCHEMA_ID,
)


PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-strict-evidence-design-review.v1"
)

REVIEW_STATUS_READY = "ready_for_strict_evidence_design_review"
REVIEW_STATUS_BLOCKED_NON_UNIQUE = "blocked_non_unique_text_match_needs_disambiguation"
REVIEW_STATUS_BLOCKED_MISSING_SUBSTRING_HASH = "blocked_missing_expected_substring_hash"
REVIEW_STATUS_BLOCKED_MISSING_CHARS_BASIS = "blocked_missing_chars_basis"
REVIEW_STATUS_BLOCKED_MISSING_CHARS_OFFSETS = "blocked_missing_chars_offsets"
REVIEW_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"
REVIEW_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE = "blocked_unsupported_artifact_type"
REVIEW_STATUS_BLOCKED_MANUAL_OR_LATER = "blocked_requires_manual_or_later_extractor_review"

TEXT_ARTIFACT_TYPES = {"section"}
CAPTION_ARTIFACT_TYPES = {"figure"}
SUPPORTED_ARTIFACT_TYPES = TEXT_ARTIFACT_TYPES | CAPTION_ARTIFACT_TYPES

CHARS_BASIS = "sourceContentHash"

DEFAULT_OFFSET_AUTHORITY_DESIGN_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-original-source-offset-authority-design"
    / "01-parsed-artifact-source-span-original-source-offset-authority-design"
    / "parsed-artifact-source-span-original-source-offset-authority-design.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-strict-evidence-design-review"
    / "01-parsed-artifact-source-span-strict-evidence-design-review"
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


def _validate_proposed_chars(proposed: dict[str, Any]) -> tuple[str, list[str]]:
    blockers: list[str] = []
    start = _safe_int(proposed.get("start"))
    end = _safe_int(proposed.get("end"))
    if start is None or end is None or end <= start:
        return REVIEW_STATUS_BLOCKED_MISSING_CHARS_OFFSETS, ["chars_start_or_end_missing_or_invalid"]

    basis = _safe_text(proposed.get("basis"))
    if basis != CHARS_BASIS:
        blockers.append(f"chars_basis_invalid:{basis or 'missing'}")
        return REVIEW_STATUS_BLOCKED_MISSING_CHARS_BASIS, blockers

    if not _safe_text(proposed.get("normalization")):
        blockers.append("chars_normalization_missing")
        return REVIEW_STATUS_BLOCKED_MISSING_CHARS_BASIS, blockers

    if not _safe_text(proposed.get("expectedSubstringSha256")):
        blockers.append("chars_expectedSubstringSha256_missing")
        return REVIEW_STATUS_BLOCKED_MISSING_SUBSTRING_HASH, blockers

    return "", []


def _classify_design_row(design_row: dict[str, Any]) -> tuple[str, list[str], str]:
    blockers = ["strict_evidence_design_review_only"]
    artifact_type = _safe_text(design_row.get("artifact_type"))
    design_status = _safe_text(design_row.get("design_status"))

    if artifact_type and artifact_type not in SUPPORTED_ARTIFACT_TYPES:
        return (
            REVIEW_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE,
            _dedupe([*blockers, f"artifact_type={artifact_type}"]),
            "hold_unsupported_artifact_type_before_strict_evidence_design_review",
        )

    if design_status == DESIGN_STATUS_BLOCKED_NON_UNIQUE_TEXT_MATCH:
        return (
            REVIEW_STATUS_BLOCKED_NON_UNIQUE,
            _dedupe([*blockers, "non_unique_text_match_in_original_source_text"]),
            "run_text_match_disambiguation_before_strict_evidence_design_review",
        )

    if design_status == DESIGN_STATUS_BLOCKED_MANUAL_OR_LATER:
        return (
            REVIEW_STATUS_BLOCKED_MANUAL_OR_LATER,
            _dedupe([*blockers, *list(design_row.get("design_blockers") or [])]),
            "manual_or_later_extractor_review_required_before_strict_evidence_design_review",
        )

    if design_status != DESIGN_STATUS_OFFSET_AUTHORITY_CANDIDATE:
        return (
            REVIEW_STATUS_BLOCKED_MANUAL_OR_LATER,
            _dedupe([*blockers, f"unexpected_design_status={design_status or 'unknown'}"]),
            "repair_offset_authority_design_row_before_strict_evidence_design_review",
        )

    proposed = _proposed_chars_dict(design_row)
    chars_status, chars_blockers = _validate_proposed_chars(proposed)
    if chars_status:
        return (
            chars_status,
            _dedupe([*blockers, *chars_blockers]),
            "repair_proposed_chars_before_strict_evidence_design_review",
        )

    return (
        REVIEW_STATUS_READY,
        _dedupe(
            [
                *blockers,
                "source_span_remains_non_strict",
                "strict_evidence_creation_disabled_for_tranche",
            ]
        ),
        "queue_for_strict_evidence_design_review_packet_only",
    )


def _review_row(index: int, design_row: dict[str, Any]) -> dict[str, Any]:
    review_status, review_blockers, recommended_action = _classify_design_row(design_row)
    proposed = _proposed_chars_dict(design_row)
    return {
        "review_row_id": f"parsed-artifact-source-span-strict-evidence-design-review:{index:04d}",
        "design_row_id": _safe_text(design_row.get("design_row_id")),
        "policy_gate_row_id": _safe_text(design_row.get("policy_gate_row_id")),
        "sourceSpanId": _safe_text(design_row.get("sourceSpanId")),
        "candidateRecordId": _safe_text(design_row.get("candidateRecordId")),
        "paper_id": _safe_text(design_row.get("paper_id")),
        "artifact_type": _safe_text(design_row.get("artifact_type")),
        "source_candidate_id": _safe_text(design_row.get("source_candidate_id")),
        "sourceContentHash": _safe_text(design_row.get("sourceContentHash")),
        "source_file": _safe_text(design_row.get("source_file")),
        "text_surface": _safe_text(design_row.get("text_surface")),
        "design_status": _safe_text(design_row.get("design_status")),
        "review_status": review_status,
        "review_blockers": review_blockers,
        "proposed_chars": proposed,
        "locator": design_row.get("locator") if isinstance(design_row.get("locator"), dict) else {},
        "source_resolution": (
            design_row.get("source_resolution")
            if isinstance(design_row.get("source_resolution"), dict)
            else {}
        ),
        "recommended_action": recommended_action,
        "readyForStrictEvidenceDesignReview": review_status == REVIEW_STATUS_READY,
        "disambiguationRequired": review_status == REVIEW_STATUS_BLOCKED_NON_UNIQUE,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "sourceSpanUpdatedRows": 0,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "canonicalParsedArtifactsWritten": False,
    }


def _sample_blockers(rows: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        for blocker in row.get("review_blockers") or []:
            text = _safe_text(blocker)
            if text:
                counter[text] += 1
    return [
        {"blocker": blocker, "count": count}
        for blocker, count in counter.most_common(limit)
    ]


def _count_rows(
    *,
    rows: list[dict[str, Any]],
    input_rows: int,
    input_schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": input_rows,
        "readyForStrictEvidenceDesignReviewRows": sum(
            1 for row in rows if row.get("review_status") == REVIEW_STATUS_READY
        ),
        "blockedNonUniqueTextMatchRows": sum(
            1 for row in rows if row.get("review_status") == REVIEW_STATUS_BLOCKED_NON_UNIQUE
        ),
        "blockedMissingExpectedSubstringHashRows": sum(
            1 for row in rows if row.get("review_status") == REVIEW_STATUS_BLOCKED_MISSING_SUBSTRING_HASH
        ),
        "blockedMissingCharsBasisRows": sum(
            1 for row in rows if row.get("review_status") == REVIEW_STATUS_BLOCKED_MISSING_CHARS_BASIS
        ),
        "blockedMissingCharsOffsetsRows": sum(
            1 for row in rows if row.get("review_status") == REVIEW_STATUS_BLOCKED_MISSING_CHARS_OFFSETS
        ),
        "blockedRequiresManualOrLaterExtractorReviewRows": sum(
            1 for row in rows if row.get("review_status") == REVIEW_STATUS_BLOCKED_MANUAL_OR_LATER
        ),
        "blockedUnsupportedArtifactTypeRows": sum(
            1 for row in rows if row.get("review_status") == REVIEW_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT_TYPE
        ),
        "blockedInputSchemaViolationRows": (
            len(rows) if input_schema_violations and rows else int(bool(input_schema_violations))
        ),
        "sourceSpanUpdatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "schemaViolationCount": len(input_schema_violations),
        "byPaperId": dict(Counter(str(row.get("paper_id") or "") for row in rows)),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in rows)),
        "byReviewStatus": dict(Counter(str(row.get("review_status") or "") for row in rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_parsed_artifact_source_span_strict_evidence_design_review(
    *,
    offset_authority_design_report_path: str | Path = DEFAULT_OFFSET_AUTHORITY_DESIGN_REPORT_PATH,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(offset_authority_design_report_path)).expanduser()
    warnings: list[str] = []
    input_schema_violations: list[str] = []
    requested_papers = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    design_report = _read_json(report_path)
    if not design_report:
        warnings.append("offset_authority_design_report_missing_or_unreadable")

    validation = validate_payload(
        design_report,
        PARSED_ARTIFACT_SOURCE_SPAN_ORIGINAL_SOURCE_OFFSET_AUTHORITY_DESIGN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        input_schema_violations = [str(error) for error in validation.errors]
        if not design_report:
            input_schema_violations.append("offset_authority_design_report_missing_or_unreadable")

    design_rows = [
        row for row in design_report.get("rows", []) if isinstance(row, dict)
    ] if isinstance(design_report, dict) else []

    input_rows = int((design_report.get("counts") or {}).get("targetRows") or len(design_rows)) if design_report else 0

    if requested_papers:
        found = {_safe_text(row.get("paper_id")) for row in design_rows if _safe_text(row.get("paper_id"))}
        if requested_papers - found:
            warnings.append("requested_paper_ids_not_found")
        design_rows = [
            row for row in design_rows if _safe_text(row.get("paper_id")) in requested_papers
        ]

    if not design_rows:
        warnings.append("offset_authority_design_rows_missing")

    rows = [_review_row(index, design_row) for index, design_row in enumerate(design_rows)]

    if input_schema_violations:
        for row in rows:
            row["review_status"] = REVIEW_STATUS_BLOCKED_INPUT_SCHEMA
            row["review_blockers"] = _dedupe(
                [*row.get("review_blockers", []), *input_schema_violations]
            )
            row["readyForStrictEvidenceDesignReview"] = False
            row["disambiguationRequired"] = False
            row["recommended_action"] = "repair_offset_authority_design_report_schema_before_design_review"

    ready_rows = [row for row in rows if row.get("review_status") == REVIEW_STATUS_READY]
    ambiguous_rows = [row for row in rows if row.get("review_status") == REVIEW_STATUS_BLOCKED_NON_UNIQUE]

    counts = _count_rows(
        rows=rows,
        input_rows=input_rows if not requested_papers else len(rows),
        input_schema_violations=_dedupe(input_schema_violations),
    )
    ready_count = int(counts.get("readyForStrictEvidenceDesignReviewRows") or 0)
    ambiguous_count = int(counts.get("blockedNonUniqueTextMatchRows") or 0)

    status = "ok"
    if input_schema_violations:
        status = "blocked"
    elif not rows:
        status = "blocked"
    elif ready_count + ambiguous_count != len(rows):
        status = "blocked"

    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "offsetAuthorityDesignReportPath": str(report_path),
            "offsetAuthorityDesignSchema": _safe_text(design_report.get("schema")) if design_report else "",
            "requestedPaperIds": sorted(requested_papers),
        },
        "counts": counts,
        "gate": {
            "designReviewPacketReady": bool(ready_rows) and not input_schema_violations,
            "disambiguationQueuePresent": bool(ambiguous_rows),
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "parsed_artifact_source_span_strict_evidence_design_review_ready"
                if status == "ok"
                else "blocked"
            ),
            "recommendedNextTranche": (
                "parsed_artifact_source_span_strict_evidence_design_packet_review"
                if ready_count > 0 and not input_schema_violations
                else "parsed_artifact_source_span_text_match_disambiguation_design"
                if ambiguous_count > 0
                else "parsed_artifact_source_span_strict_evidence_design_review_repair"
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
        "sampleBlockers": _sample_blockers(rows),
        "readyDesignRows": ready_rows,
        "ambiguousDisambiguationRows": ambiguous_rows,
        "rows": rows,
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
            "sampleBlockers",
            "readyDesignRows",
            "ambiguousDisambiguationRows",
            "rows",
        )
        if key in report
    }


def render_parsed_artifact_source_span_strict_evidence_design_review_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byReviewStatus") or {})).items())
    ]
    sample_blockers = report.get("sampleBlockers") or []
    return "\n".join(
        [
            "# Parsed Artifact SourceSpan Strict Evidence Design Review",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- ready for strict evidence design review: {int(counts.get('readyForStrictEvidenceDesignReviewRows') or 0)}",
            f"- blocked non-unique text match: {int(counts.get('blockedNonUniqueTextMatchRows') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            f"- source span updated: {int(counts.get('sourceSpanUpdatedRows') or 0)}",
            "",
            "## Review status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            "## Sample blockers",
            *[
                f"- {item.get('blocker', '')}: {int(item.get('count') or 0)}"
                for item in sample_blockers
            ],
        ]
    )


def write_parsed_artifact_source_span_strict_evidence_design_review_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-span-strict-evidence-design-review.json"
    summary_path = root / "parsed-artifact-source-span-strict-evidence-design-review-summary.json"
    markdown_path = root / "parsed-artifact-source-span-strict-evidence-design-review.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_strict_evidence_design_review_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Review offset-authority design rows for strict-evidence design readiness "
            "without creating StrictEvidence or mutating SourceSpan records."
        )
    )
    parser.add_argument(
        "--offset-authority-design-report",
        default=str(DEFAULT_OFFSET_AUTHORITY_DESIGN_REPORT_PATH),
        help="Original-source offset authority design JSON report.",
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_source_span_strict_evidence_design_review(
        offset_authority_design_report_path=args.offset_authority_design_report,
        paper_ids=args.paper_id or None,
    )

    if args.output_dir:
        paths = write_parsed_artifact_source_span_strict_evidence_design_review_reports(
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
    "PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID",
    "REVIEW_STATUS_READY",
    "REVIEW_STATUS_BLOCKED_NON_UNIQUE",
    "build_parsed_artifact_source_span_strict_evidence_design_review",
    "render_parsed_artifact_source_span_strict_evidence_design_review_markdown",
    "write_parsed_artifact_source_span_strict_evidence_design_review_reports",
]
