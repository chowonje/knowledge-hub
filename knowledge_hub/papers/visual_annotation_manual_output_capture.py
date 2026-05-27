"""Validate manually supplied web/VLM visual annotation output.

This helper captures operator-supplied web/VLM output as retrieval hints only.
It validates the output against the small visual annotation web/attachment
packs and never indexes, promotes, or exposes the derived text as evidence.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_attachment_pack import (
    VISUAL_ANNOTATION_ATTACHMENT_PACK_SCHEMA_ID,
)
from knowledge_hub.papers.visual_annotation_web_pack import VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID


VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID = "knowledge-hub.paper.visual-annotation-web-output.v1"
VISUAL_ANNOTATION_WEB_OUTPUT_VALIDATION_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-web-output-validation.v1"
)
VISUAL_ANNOTATION_CAPTURED_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-captured-row.v1"
)

READY_DECISION = "ready_for_retrieval_hint_candidate_store_design"
NEXT_RECOMMENDED_TRANCHE = "visual_retrieval_hint_candidate_store_design"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)

OBSERVATION_STATUSES = {"image_attached", "image_not_attached", "unclear"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitized_report_ref(path: Path, *, project_root: Path | None = None) -> str:
    resolved = path.expanduser()
    if project_root is not None:
        try:
            rel = resolved.resolve().relative_to(project_root.resolve())
            return rel.as_posix()
        except Exception:
            pass
    return f"input_reports/{resolved.name}"


def _scope(output_rows: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "manualWebModelOutputRows": int(output_rows),
        "vectorIndexing": False,
        "strictEvidencePromotionRows": 0,
        "runtimeAnswerVisibleExposureRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "answerabilityGateBypassRows": 0,
        "cropWriteRows": 0,
        "pageImageWriteRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _source_rows_by_id(web_pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [row for row in list(web_pack.get("packRowsDetail") or []) if isinstance(row, dict)]
    return {normalize_text(row.get("sourceCandidateId")): row for row in rows}


def _attachment_rows_by_id(attachment_pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [
        row
        for row in list(attachment_pack.get("attachmentRowsDetail") or [])
        if isinstance(row, dict)
    ]
    return {normalize_text(row.get("sourceCandidateId")): row for row in rows}


def _duplicate_ids(rows: list[dict[str, Any]]) -> list[str]:
    counts = Counter(normalize_text(row.get("sourceCandidateId")) for row in rows)
    return sorted(candidate_id for candidate_id, count in counts.items() if candidate_id and count > 1)


def _output_rows(output: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in list(output.get("rows") or []) if isinstance(row, dict)]


def _schema_errors(payload: dict[str, Any], schema_id: str) -> list[str]:
    result = validate_payload(payload, schema_id, strict=True)
    return [str(error) for error in result.errors]


def _policy_violations(output_row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if normalize_text(output_row.get("visualObservationStatus")) not in OBSERVATION_STATUSES:
        violations.append("invalid_visual_observation_status")
    if not normalize_text(output_row.get("derivedTextForRetrieval")):
        violations.append("empty_derived_text")
    if not normalize_text(output_row.get("visibleText")):
        violations.append("empty_visible_text")
    keywords = output_row.get("retrievalKeywords")
    if not isinstance(keywords, list) or not any(normalize_text(item) for item in keywords):
        violations.append("empty_retrieval_keywords")
    if not normalize_text(output_row.get("uncertainty")):
        violations.append("empty_uncertainty")
    if not normalize_text(output_row.get("limitations")):
        violations.append("empty_limitations")
    if output_row.get("strictEvidence") is not False:
        violations.append("strict_evidence_not_false")
    if output_row.get("citationGrade") is not False:
        violations.append("citation_grade_not_false")
    if output_row.get("answerableWithoutTextEvidence") is not False:
        violations.append("answerable_without_text_evidence_not_false")
    if _contains_private_path(output_row):
        violations.append("private_path_leak")
    return violations


def _retrieval_hint_plan() -> dict[str, Any]:
    return {
        "targetDerivedTextField": "derivedTextForRetrieval",
        "allowedUse": "retrieval_hint_only",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
    }


def _captured_row(
    *,
    source_row: dict[str, Any],
    attachment_row: dict[str, Any] | None,
    output_row: dict[str, Any],
    violations: list[str],
) -> dict[str, Any]:
    return {
        "schema": VISUAL_ANNOTATION_CAPTURED_ROW_SCHEMA_ID,
        "sourceCandidateId": normalize_text(output_row.get("sourceCandidateId")),
        "paperId": normalize_text(source_row.get("paperId")),
        "paperRef": normalize_text(source_row.get("paperRef")),
        "sourceContentHash": normalize_text(source_row.get("sourceContentHash")),
        "page": int(source_row.get("page") or 0),
        "bbox": list(source_row.get("bbox") or []),
        "candidateType": normalize_text(source_row.get("candidateType")),
        "attachmentRef": normalize_text((attachment_row or {}).get("attachmentRef")),
        "visualObservationStatus": normalize_text(output_row.get("visualObservationStatus")),
        "derivedTextForRetrieval": normalize_text(output_row.get("derivedTextForRetrieval")),
        "visibleText": normalize_text(output_row.get("visibleText")),
        "retrievalKeywords": [
            normalize_text(item)
            for item in list(output_row.get("retrievalKeywords") or [])
            if normalize_text(item)
        ],
        "uncertainty": normalize_text(output_row.get("uncertainty")),
        "limitations": normalize_text(output_row.get("limitations")),
        "retrievalHintPlan": _retrieval_hint_plan(),
        "provenance": {
            "sourceWebPackSchema": VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID,
            "sourceAttachmentPackSchema": VISUAL_ANNOTATION_ATTACHMENT_PACK_SCHEMA_ID,
            "sourceCandidateId": normalize_text(output_row.get("sourceCandidateId")),
            "sourceContentHash": normalize_text(source_row.get("sourceContentHash")),
            "page": int(source_row.get("page") or 0),
            "bbox": list(source_row.get("bbox") or []),
            "extractionMethod": "manual_web_vlm_output_capture_v1",
        },
        "validation": {
            "matchedSourcePackRow": True,
            "matchedAttachmentRow": bool(attachment_row),
            "policyCompliant": not violations,
            "violationReasons": violations,
        },
    }


def build_visual_annotation_web_output_validation(
    output: dict[str, Any],
    web_pack: dict[str, Any],
    attachment_pack: dict[str, Any],
    *,
    output_ref: str = "eval/knowledgeos/reports/visual_annotation_web_output_001.manual.json",
    source_web_pack_ref: str = "eval/knowledgeos/reports/visual_annotation_web_pack_001.v1.json",
    source_attachment_pack_ref: str = "eval/knowledgeos/reports/visual_annotation_attachment_pack_001.v1.json",
    generated_at: str | None = None,
) -> dict[str, Any]:
    output_rows = _output_rows(output)
    web_rows = _source_rows_by_id(web_pack)
    attachment_rows = _attachment_rows_by_id(attachment_pack)
    output_schema_errors = _schema_errors(output, VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID)
    duplicate_ids = _duplicate_ids(output_rows)
    output_ids = [normalize_text(row.get("sourceCandidateId")) for row in output_rows]
    output_id_set = {candidate_id for candidate_id in output_ids if candidate_id}
    source_id_set = set(web_rows)
    missing_ids = sorted(source_id_set - output_id_set)
    extra_ids = sorted(output_id_set - source_id_set)

    violations: list[dict[str, Any]] = []
    captured_rows: list[dict[str, Any]] = []
    policy_violation_ids: set[str] = set()
    empty_derived_text_ids: set[str] = set()
    keyword_empty_ids: set[str] = set()
    private_path_leak_ids: set[str] = set()

    for error in output_schema_errors:
        violations.append(
            {
                "sourceCandidateId": "",
                "kind": "output_schema_violation",
                "message": error,
            }
        )
    for candidate_id in duplicate_ids:
        violations.append(
            {
                "sourceCandidateId": candidate_id,
                "kind": "duplicate_source_candidate_id",
                "message": "Output contains more than one row for sourceCandidateId.",
            }
        )
    for candidate_id in missing_ids:
        violations.append(
            {
                "sourceCandidateId": candidate_id,
                "kind": "missing_source_candidate_id",
                "message": "Output is missing a required web-pack source candidate.",
            }
        )
    for candidate_id in extra_ids:
        violations.append(
            {
                "sourceCandidateId": candidate_id,
                "kind": "extra_source_candidate_id",
                "message": "Output contains a candidate absent from the source web pack.",
            }
        )

    for output_row in output_rows:
        candidate_id = normalize_text(output_row.get("sourceCandidateId"))
        row_violations = _policy_violations(output_row)
        if row_violations:
            policy_violation_ids.add(candidate_id)
            for reason in row_violations:
                violations.append(
                    {
                        "sourceCandidateId": candidate_id,
                        "kind": reason,
                        "message": "Output row violates retrieval-hint-only policy or row contract.",
                    }
                )
            if "empty_derived_text" in row_violations:
                empty_derived_text_ids.add(candidate_id)
            if "empty_retrieval_keywords" in row_violations:
                keyword_empty_ids.add(candidate_id)
            if "private_path_leak" in row_violations:
                private_path_leak_ids.add(candidate_id)
        source_row = web_rows.get(candidate_id)
        if source_row and candidate_id not in duplicate_ids:
            captured_rows.append(
                _captured_row(
                    source_row=source_row,
                    attachment_row=attachment_rows.get(candidate_id),
                    output_row=output_row,
                    violations=row_violations,
                )
            )

    matched_rows = [
        row
        for row in captured_rows
        if row.get("validation", {}).get("matchedSourcePackRow")
        and row.get("validation", {}).get("policyCompliant")
    ]
    private_path_leak_rows = len(private_path_leak_ids)
    if _contains_private_path({k: v for k, v in output.items() if k != "rows"}):
        private_path_leak_rows += 1
        violations.append(
            {
                "sourceCandidateId": "",
                "kind": "private_path_leak",
                "message": "Top-level output payload contains a private local path token.",
            }
        )

    counts = {
        "sourcePackRows": len(web_rows),
        "attachmentRows": len(attachment_rows),
        "outputRows": len(output_rows),
        "matchedRows": len(matched_rows),
        "missingRows": len(missing_ids),
        "extraRows": len(extra_ids),
        "duplicateRows": len(duplicate_ids),
        "policyViolationRows": len(policy_violation_ids),
        "emptyDerivedTextRows": len(empty_derived_text_ids),
        "keywordEmptyRows": len(keyword_empty_ids),
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": len(output_schema_errors),
        "blockedRows": 0,
    }
    blocked_rows = (
        counts["missingRows"]
        + counts["extraRows"]
        + counts["duplicateRows"]
        + counts["policyViolationRows"]
        + counts["privatePathLeakRows"]
        + counts["schemaViolationCount"]
    )
    counts["blockedRows"] = int(blocked_rows)

    report: dict[str, Any] = {
        "schema": VISUAL_ANNOTATION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceOutput": {
            "schema": normalize_text(output.get("schema")),
            "reportRef": normalize_text(output_ref),
            "rows": len(output_rows),
        },
        "sourceWebPack": {
            "schema": normalize_text(web_pack.get("schema")),
            "status": normalize_text(web_pack.get("status")),
            "reportRef": normalize_text(source_web_pack_ref),
            "packRows": len(web_rows),
        },
        "sourceAttachmentPack": {
            "schema": normalize_text(attachment_pack.get("schema")),
            "status": normalize_text(attachment_pack.get("status")),
            "reportRef": normalize_text(source_attachment_pack_ref),
            "attachmentRows": len(attachment_rows),
        },
        "scope": _scope(len(output_rows)),
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "targetDerivedTextField": "derivedTextForRetrieval",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "storagePlan": "candidate_report_only_no_index_write",
        },
        "counts": counts,
        "capturedRowsDetail": captured_rows,
        "violations": violations,
        "warnings": [
            "derivedTextForRetrieval is accepted only as a retrieval hint candidate.",
            "No visual annotation row is promoted to strict evidence or citation-grade evidence.",
            "The next tranche must design a candidate store before any vectorization decision.",
        ],
    }
    if (
        output.get("schema") != VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID
        or web_pack.get("schema") != VISUAL_ANNOTATION_WEB_PACK_SCHEMA_ID
        or web_pack.get("status") != "ready"
        or attachment_pack.get("schema") != VISUAL_ANNOTATION_ATTACHMENT_PACK_SCHEMA_ID
        or attachment_pack.get("status") != "ready"
        or counts["blockedRows"]
        or counts["matchedRows"] != counts["sourcePackRows"]
        or counts["outputRows"] != counts["sourcePackRows"]
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def render_markdown_validation(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    output = dict(report.get("sourceOutput") or {})
    lines = [
        "# Visual Annotation Web Output 001 Validation",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- sourceOutput: `{output.get('reportRef')}`",
        f"- sourcePackRows: `{counts.get('sourcePackRows')}`",
        f"- outputRows: `{counts.get('outputRows')}`",
        f"- matchedRows: `{counts.get('matchedRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- apiCalls: `{scope.get('apiCalls')}`",
        f"- modelCalls: `{scope.get('modelCalls')}`",
        f"- webModelCalls: `{scope.get('webModelCalls')}`",
        f"- manualWebModelOutputRows: `{scope.get('manualWebModelOutputRows')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        f"- databaseMutationRows: `{scope.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{scope.get('indexMutationRows')}`",
        f"- reindexOrReembedRows: `{scope.get('reindexOrReembedRows')}`",
        f"- vaultScanRows: `{scope.get('vaultScanRows')}`",
        f"- externalDownloadRows: `{scope.get('externalDownloadRows')}`",
        f"- answerabilityGateBypassRows: `{scope.get('answerabilityGateBypassRows')}`",
        f"- cropWriteRows: `{scope.get('cropWriteRows')}`",
        "",
        "## Captured Rows",
        "",
        "| # | paperId | type | page | sourceCandidateId | status | keywords |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for index, row in enumerate(report.get("capturedRowsDetail", []), start=1):
        keywords = ", ".join(list(row.get("retrievalKeywords") or [])[:6])
        lines.append(
            "| {index} | {paperId} | {candidateType} | {page} | `{candidateId}` | `{status}` | {keywords} |".format(
                index=index,
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                candidateId=row.get("sourceCandidateId"),
                status=row.get("visualObservationStatus"),
                keywords=keywords.replace("|", "\\|"),
            )
        )
    if report.get("violations"):
        lines.extend(["", "## Violations", ""])
        for violation in report.get("violations", []):
            lines.append(
                "- `{kind}` `{candidate}`: {message}".format(
                    kind=violation.get("kind"),
                    candidate=violation.get("sourceCandidateId"),
                    message=violation.get("message"),
                )
            )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_visual_annotation_web_output_validation(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_validation(report), encoding="utf-8")
    return {
        "json": str(report_json),
        "markdown": str(report_md),
    }


__all__ = [
    "NEXT_RECOMMENDED_TRANCHE",
    "READY_DECISION",
    "VISUAL_ANNOTATION_CAPTURED_ROW_SCHEMA_ID",
    "VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID",
    "VISUAL_ANNOTATION_WEB_OUTPUT_VALIDATION_SCHEMA_ID",
    "build_visual_annotation_web_output_validation",
    "load_json",
    "render_markdown_validation",
    "sanitized_report_ref",
    "write_visual_annotation_web_output_validation",
]
