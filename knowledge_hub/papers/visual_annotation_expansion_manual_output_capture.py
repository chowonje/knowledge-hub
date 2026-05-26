"""Prepare and validate manual web/VLM output for the expansion attachment pack.

This helper creates the operator-facing manual run packet for the 24 expansion
context crops, then validates manually supplied web/VLM output as retrieval
hints only. It does not call models, write indexes, mutate stores, or promote
derived visual text to evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_expansion_attachment_pack import (
    VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID,
)
from knowledge_hub.papers.visual_annotation_expansion_pack_design import (
    VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.visual_annotation_manual_output_capture import (
    OBSERVATION_STATUSES,
    VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
)


VISUAL_ANNOTATION_EXPANSION_MANUAL_RUN_PACKET_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-expansion-manual-run-packet.v1"
)
VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-expansion-web-output-validation.v1"
)
VISUAL_ANNOTATION_EXPANSION_OPERATOR_HANDOFF_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-expansion-operator-handoff.v1"
)
VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_TEMPLATE_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-expansion-web-output-template.v1"
)
VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-expansion-web-run-bundle.v1"
)
VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BATCH_TEMPLATE_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-expansion-web-run-batch-template.v1"
)
VISUAL_ANNOTATION_EXPANSION_CAPTURED_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-expansion-captured-row.v1"
)

DEFAULT_MANUAL_RUN_PACKET_ID = "visual_annotation_expansion_manual_run_packet_002"
DEFAULT_OPERATOR_HANDOFF_ID = "visual_annotation_expansion_operator_handoff_002"
DEFAULT_WEB_OUTPUT_TEMPLATE_ID = "visual_annotation_expansion_web_output_template_002"
DEFAULT_WEB_RUN_BUNDLE_ID = "visual_annotation_expansion_web_run_bundle_002"
READY_RUN_DECISION = "ready_for_manual_web_vlm_expansion_run"
READY_HANDOFF_DECISION = "ready_for_operator_web_vlm_run"
READY_TEMPLATE_DECISION = "ready_for_manual_web_output_fill"
READY_BUNDLE_DECISION = "ready_for_operator_web_batch_run"
READY_VALIDATION_DECISION = "ready_for_visual_retrieval_hint_candidate_store_expansion_design"
NEXT_AFTER_RUN_TRANCHE = "visual_annotation_expansion_manual_output_capture"
NEXT_AFTER_VALIDATION_TRANCHE = "visual_retrieval_hint_candidate_store_expansion_design"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


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


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _source_rows_by_id(expansion_pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [row for row in list(expansion_pack.get("packRowsDetail") or []) if isinstance(row, dict)]
    return {normalize_text(row.get("sourceCandidateId")): row for row in rows}


def _attachment_rows_by_id(attachment_pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [
        row
        for row in list(attachment_pack.get("attachmentRowsDetail") or [])
        if isinstance(row, dict)
    ]
    return {normalize_text(row.get("sourceCandidateId")): row for row in rows}


def _output_rows(output: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in list(output.get("rows") or []) if isinstance(row, dict)]


def _duplicate_ids(rows: Sequence[dict[str, Any]]) -> list[str]:
    counts = Counter(normalize_text(row.get("sourceCandidateId")) for row in rows)
    return sorted(candidate_id for candidate_id, count in counts.items() if candidate_id and count > 1)


def _schema_errors(payload: dict[str, Any], schema_id: str) -> list[str]:
    result = validate_payload(payload, schema_id, strict=True)
    return [str(error) for error in result.errors]


def _retrieval_hint_plan() -> dict[str, Any]:
    return {
        "targetDerivedTextField": "derivedTextForRetrieval",
        "allowedUse": "retrieval_hint_only",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
    }


def _scope(*, manual_rows: int = 0) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "manualWebModelOutputRows": int(manual_rows),
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
        "wholeImageWriteRows": 0,
        "wholeImageGptRows": 0,
        "candidateStoreMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _handoff_scope() -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "manualOperatorWebModelRunRequired": True,
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
        "wholeImageWriteRows": 0,
        "wholeImageGptRows": 0,
        "candidateStoreMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _template_scope() -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "manualOperatorWebModelRunRequired": True,
        "templateOnly": True,
        "completedWebOutputRows": 0,
        "manualWebModelOutputRows": 0,
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
        "wholeImageWriteRows": 0,
        "wholeImageGptRows": 0,
        "candidateStoreMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _bundle_scope(*, batch_rows: int = 0, bundle_artifact_rows: int = 0) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "manualOperatorWebModelRunRequired": True,
        "operatorBatchPromptRows": int(batch_rows),
        "bundleArtifactRows": int(bundle_artifact_rows),
        "completedWebOutputRows": 0,
        "manualWebModelOutputRows": 0,
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
        "wholeImageWriteRows": 0,
        "wholeImageGptRows": 0,
        "candidateStoreMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


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


def _manual_prompt(batch_number: int, total_batches: int) -> str:
    return (
        f"You are annotating visual retrieval hints for batch {batch_number} of {total_batches}. "
        "Use only the attached context-crop images and the provided row metadata. Return one JSON object "
        "with schema knowledge-hub.paper.visual-annotation-web-output.v1 and a rows array. For every row, "
        "fill sourceCandidateId, visualObservationStatus, derivedTextForRetrieval, visibleText, "
        "retrievalKeywords, uncertainty, limitations, strictEvidence=false, citationGrade=false, and "
        "answerableWithoutTextEvidence=false. Do not create citation-grade evidence, do not answer paper "
        "claims from the image, do not infer hidden details, and do not include local file paths."
    )


def _run_packet_rows(
    expansion_rows: Sequence[dict[str, Any]],
    attachment_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in expansion_rows:
        source_candidate_id = normalize_text(row.get("sourceCandidateId"))
        attachment = attachment_rows.get(source_candidate_id) or {}
        web_input = row.get("webInput") if isinstance(row.get("webInput"), dict) else {}
        rows.append(
            {
                "sourceCandidateId": source_candidate_id,
                "sourcePackCandidateId": normalize_text(row.get("packCandidateId")),
                "paperId": normalize_text(row.get("paperId")),
                "paperRef": normalize_text(row.get("paperRef")),
                "sourceContentHash": normalize_text(row.get("sourceContentHash")),
                "page": int(row.get("page") or 0),
                "bbox": list(row.get("bbox") or []),
                "candidateType": normalize_text(row.get("candidateType")),
                "attachmentRef": normalize_text(attachment.get("attachmentRef")),
                "copyPasteContext": normalize_text(web_input.get("copyPasteContext")),
                "expectedOutputSkeleton": {
                    "sourceCandidateId": source_candidate_id,
                    "visualObservationStatus": "image_attached",
                    "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
                    "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
                    "retrievalKeywords": ["FILL_IN_KEYWORD"],
                    "uncertainty": "FILL_IN_UNCERTAINTY.",
                    "limitations": "Retrieval hint only, not evidence.",
                    "strictEvidence": False,
                    "citationGrade": False,
                    "answerableWithoutTextEvidence": False,
                },
            }
        )
    return rows


def build_visual_annotation_expansion_manual_run_packet(
    expansion_pack: dict[str, Any],
    attachment_pack: dict[str, Any],
    *,
    packet_id: str = DEFAULT_MANUAL_RUN_PACKET_ID,
    source_expansion_pack_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_pack_design.v1.json",
    source_attachment_pack_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002.v1.json",
    batch_size: int = 8,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_rows = [row for row in list(expansion_pack.get("packRowsDetail") or []) if isinstance(row, dict)]
    attachment_rows = _attachment_rows_by_id(attachment_pack)
    packet_rows = _run_packet_rows(source_rows, attachment_rows)
    batches = []
    total_batches = max(1, (len(packet_rows) + max(1, batch_size) - 1) // max(1, batch_size))
    for offset in range(0, len(packet_rows), max(1, batch_size)):
        batch_rows = packet_rows[offset : offset + max(1, batch_size)]
        batch_number = len(batches) + 1
        batches.append(
            {
                "batchId": f"{packet_id}_batch_{batch_number:02d}",
                "batchNumber": batch_number,
                "totalBatches": total_batches,
                "prompt": _manual_prompt(batch_number, total_batches),
                "attachmentRefs": [row["attachmentRef"] for row in batch_rows],
                "rows": batch_rows,
            }
        )
    missing_attachment_rows = sum(1 for row in packet_rows if not normalize_text(row.get("attachmentRef")))
    private_path_leak_rows = 1 if _contains_private_path({"batches": batches}) else 0
    counts = {
        "sourcePackRows": len(source_rows),
        "attachmentRows": len(attachment_rows),
        "packetRows": len(packet_rows),
        "batchRows": len(batches),
        "missingAttachmentRows": int(missing_attachment_rows),
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }
    report: dict[str, Any] = {
        "schema": VISUAL_ANNOTATION_EXPANSION_MANUAL_RUN_PACKET_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_RUN_DECISION,
        "nextRecommendedTranche": NEXT_AFTER_RUN_TRANCHE,
        "packetId": packet_id,
        "sourceExpansionPack": {
            "schema": normalize_text(expansion_pack.get("schema")),
            "status": normalize_text(expansion_pack.get("status")),
            "reportRef": normalize_text(source_expansion_pack_ref),
            "packRows": len(source_rows),
        },
        "sourceAttachmentPack": {
            "schema": normalize_text(attachment_pack.get("schema")),
            "status": normalize_text(attachment_pack.get("status")),
            "reportRef": normalize_text(source_attachment_pack_ref),
            "attachmentRows": len(attachment_rows),
        },
        "scope": _scope(),
        "outputContract": {
            "schema": VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
        },
        "counts": counts,
        "batches": batches,
        "warnings": [
            "This packet is for manual web/VLM use only; this script makes no model calls.",
            "Return derivedTextForRetrieval as retrieval hints only, not evidence.",
            "Do not upload whole pages or whole images for this gate; use the listed context crops only.",
        ],
    }
    if (
        expansion_pack.get("schema") != VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID
        or expansion_pack.get("status") != "ready"
        or attachment_pack.get("schema") != VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID
        or attachment_pack.get("status") != "ready"
        or not packet_rows
        or missing_attachment_rows
        or private_path_leak_rows
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def _flatten_packet_rows(manual_run_packet: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in list(manual_run_packet.get("batches") or []):
        if not isinstance(batch, dict):
            continue
        for row in list(batch.get("rows") or []):
            if isinstance(row, dict):
                merged = dict(row)
                merged["batchId"] = normalize_text(batch.get("batchId"))
                merged["batchNumber"] = int(batch.get("batchNumber") or 0)
                rows.append(merged)
    return rows


def build_visual_annotation_expansion_operator_handoff(
    manual_run_packet: dict[str, Any],
    *,
    handoff_id: str = DEFAULT_OPERATOR_HANDOFF_ID,
    source_manual_run_packet_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_manual_run_packet_002.v1.json",
    expected_output_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json",
    validation_command: str = (
        "PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py"
    ),
    generated_at: str | None = None,
) -> dict[str, Any]:
    packet_rows = _flatten_packet_rows(manual_run_packet)
    template_rows = [
        {
            "batchId": normalize_text(row.get("batchId")),
            "batchNumber": int(row.get("batchNumber") or 0),
            "sourceCandidateId": normalize_text(row.get("sourceCandidateId")),
            "paperId": normalize_text(row.get("paperId")),
            "candidateType": normalize_text(row.get("candidateType")),
            "page": int(row.get("page") or 0),
            "attachmentRef": normalize_text(row.get("attachmentRef")),
            "fillRequired": [
                "visualObservationStatus",
                "derivedTextForRetrieval",
                "visibleText",
                "retrievalKeywords",
                "uncertainty",
                "limitations",
            ],
            "outputRowSkeleton": dict(row.get("expectedOutputSkeleton") or {}),
        }
        for row in packet_rows
    ]
    private_path_leak_rows = 1 if _contains_private_path(template_rows) else 0
    counts = {
        "packetRows": len(packet_rows),
        "templateRows": len(template_rows),
        "batchRows": len({int(row.get("batchNumber") or 0) for row in template_rows}),
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }
    report: dict[str, Any] = {
        "schema": VISUAL_ANNOTATION_EXPANSION_OPERATOR_HANDOFF_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_HANDOFF_DECISION,
        "nextRecommendedTranche": NEXT_AFTER_RUN_TRANCHE,
        "handoffId": handoff_id,
        "sourceManualRunPacket": {
            "schema": normalize_text(manual_run_packet.get("schema")),
            "status": normalize_text(manual_run_packet.get("status")),
            "reportRef": normalize_text(source_manual_run_packet_ref),
            "packetRows": len(packet_rows),
        },
        "scope": _handoff_scope(),
        "operatorSteps": [
            "Open each batch in the manual run packet.",
            "Upload only the listed context-crop PNG attachments for that batch.",
            "Paste the batch prompt and row metadata into web GPT/Pro.",
            "Ask for JSON only with schema knowledge-hub.paper.visual-annotation-web-output.v1.",
            f"Combine the returned rows into {expected_output_ref}.",
            f"Run validation with: {validation_command}",
        ],
        "expectedOutputRef": expected_output_ref,
        "validationCommand": validation_command,
        "outputContract": {
            "schema": VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
        },
        "counts": counts,
        "templateRows": template_rows,
        "warnings": [
            "This handoff is not web/VLM output and must not be validated as completed output.",
            "The operator must replace every FILL_IN placeholder with observations from attached context crops.",
            "Do not upload whole pages or whole images for this gate.",
            "Do not treat derivedTextForRetrieval as strict, citation-grade, or answer-visible evidence.",
        ],
    }
    if (
        manual_run_packet.get("schema") != VISUAL_ANNOTATION_EXPANSION_MANUAL_RUN_PACKET_SCHEMA_ID
        or manual_run_packet.get("status") != "ready"
        or not packet_rows
        or private_path_leak_rows
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def build_visual_annotation_expansion_web_output_template(
    operator_handoff: dict[str, Any],
    *,
    template_id: str = DEFAULT_WEB_OUTPUT_TEMPLATE_ID,
    source_operator_handoff_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_operator_handoff_002.v1.json",
    target_output_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json",
    validation_command: str = (
        "PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py"
    ),
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_rows = [
        row for row in list(operator_handoff.get("templateRows") or []) if isinstance(row, dict)
    ]
    template_rows = [
        {
            "batchId": normalize_text(row.get("batchId")),
            "batchNumber": int(row.get("batchNumber") or 0),
            "sourceCandidateId": normalize_text(row.get("sourceCandidateId")),
            "paperId": normalize_text(row.get("paperId")),
            "candidateType": normalize_text(row.get("candidateType")),
            "page": int(row.get("page") or 0),
            "attachmentRef": normalize_text(row.get("attachmentRef")),
            "fillRequired": list(row.get("fillRequired") or []),
            "targetRowTemplate": dict(row.get("outputRowSkeleton") or {}),
        }
        for row in source_rows
    ]
    placeholder_rows = sum(
        1 for row in template_rows if "FILL_IN" in json.dumps(row, ensure_ascii=False)
    )
    private_path_leak_rows = 1 if _contains_private_path(template_rows) else 0
    counts = {
        "sourceTemplateRows": len(source_rows),
        "outputTemplateRows": len(template_rows),
        "batchRows": len({int(row.get("batchNumber") or 0) for row in template_rows}),
        "placeholderRows": int(placeholder_rows),
        "completedWebOutputRows": 0,
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }
    report: dict[str, Any] = {
        "schema": VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_TEMPLATE_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_TEMPLATE_DECISION,
        "nextRecommendedTranche": NEXT_AFTER_RUN_TRANCHE,
        "templateId": template_id,
        "sourceOperatorHandoff": {
            "schema": normalize_text(operator_handoff.get("schema")),
            "status": normalize_text(operator_handoff.get("status")),
            "reportRef": normalize_text(source_operator_handoff_ref),
            "templateRows": len(source_rows),
        },
        "targetOutput": {
            "schema": VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
            "reportRef": normalize_text(target_output_ref),
            "validationCommand": validation_command,
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
        },
        "scope": _template_scope(),
        "counts": counts,
        "templateRows": template_rows,
        "instructions": [
            "Use this file as a fill template, not as completed web/VLM output.",
            "For the final output, return only schema knowledge-hub.paper.visual-annotation-web-output.v1 with a rows array.",
            "Replace every FILL_IN placeholder using only the attached context-crop image for that row.",
            "Keep strictEvidence=false, citationGrade=false, and answerableWithoutTextEvidence=false for every row.",
            f"Save the completed output at {target_output_ref} and run: {validation_command}",
        ],
        "warnings": [
            "This template intentionally uses a different schema so it cannot be accepted as completed visual annotation output.",
            "A placeholder row is not a manual web/VLM observation.",
            "Do not upload whole pages or whole images for this gate.",
            "Do not treat derivedTextForRetrieval as strict, citation-grade, or answer-visible evidence.",
        ],
    }
    if (
        operator_handoff.get("schema") != VISUAL_ANNOTATION_EXPANSION_OPERATOR_HANDOFF_SCHEMA_ID
        or operator_handoff.get("status") != "ready"
        or not source_rows
        or private_path_leak_rows
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def _batch_file_stem(batch_number: int) -> str:
    return f"batch_{int(batch_number):02d}"


def _web_run_bundle_rows(web_output_template: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in list(web_output_template.get("templateRows") or []) if isinstance(row, dict)
    ]


def _final_output_skeleton(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
        "rows": [dict(row.get("targetRowTemplate") or {}) for row in rows],
    }


def build_visual_annotation_expansion_web_run_batch_template(
    batch_bundle: dict[str, Any],
) -> dict[str, Any]:
    rows = [row for row in list(batch_bundle.get("rows") or []) if isinstance(row, dict)]
    return {
        "schema": VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BATCH_TEMPLATE_SCHEMA_ID,
        "batchId": normalize_text(batch_bundle.get("batchId")),
        "batchNumber": int(batch_bundle.get("batchNumber") or 0),
        "targetOutputSchema": VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
        "attachmentRefs": list(batch_bundle.get("attachmentRefs") or []),
        "rows": [dict(row.get("targetRowTemplate") or {}) for row in rows],
        "warnings": [
            "This is a fill template, not completed web/VLM output.",
            "Replace every FILL_IN placeholder before creating the final manual output file.",
            "Keep strictEvidence=false, citationGrade=false, and answerableWithoutTextEvidence=false.",
        ],
    }


def render_markdown_web_run_batch_prompt(batch_bundle: dict[str, Any]) -> str:
    rows = [row for row in list(batch_bundle.get("rows") or []) if isinstance(row, dict)]
    skeleton = _final_output_skeleton(rows)
    lines = [
        f"# Visual Annotation Expansion Batch {int(batch_bundle.get('batchNumber') or 0):02d}",
        "",
        "Use only the attached context-crop PNG files listed below. Do not use whole pages, whole images, or outside sources.",
        "Return JSON only with top-level schema `knowledge-hub.paper.visual-annotation-web-output.v1` and a `rows` array.",
        "Every row must keep `strictEvidence=false`, `citationGrade=false`, and `answerableWithoutTextEvidence=false`.",
        "`derivedTextForRetrieval` is a retrieval hint only; it is not evidence and not answer-visible text.",
        "",
        "## Attachments To Upload",
        "",
    ]
    for attachment_ref in list(batch_bundle.get("attachmentRefs") or []):
        lines.append(f"- `{attachment_ref}`")
    lines.extend(
        [
            "",
            "## Rows To Fill",
            "",
            "| # | paperId | type | page | sourceCandidateId | attachmentRef |",
            "|---:|---|---|---:|---|---|",
        ]
    )
    for index, row in enumerate(rows, start=1):
        lines.append(
            "| {index} | {paperId} | {candidateType} | {page} | `{candidateId}` | `{attachment}` |".format(
                index=index,
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                candidateId=row.get("sourceCandidateId"),
                attachment=row.get("attachmentRef"),
            )
        )
    lines.extend(
        [
            "",
            "## JSON Shape To Return",
            "",
            "```json",
            json.dumps(skeleton, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_visual_annotation_expansion_web_run_bundle(
    web_output_template: dict[str, Any],
    *,
    bundle_id: str = DEFAULT_WEB_RUN_BUNDLE_ID,
    source_web_output_template_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_web_output_template_002.v1.json",
    bundle_dir_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_web_run_bundle_002",
    target_output_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json",
    validation_command: str = (
        "PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py"
    ),
    generated_at: str | None = None,
) -> dict[str, Any]:
    template_rows = _web_run_bundle_rows(web_output_template)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in template_rows:
        grouped.setdefault(int(row.get("batchNumber") or 0), []).append(row)

    batch_bundles: list[dict[str, Any]] = []
    for batch_number in sorted(grouped):
        rows = grouped[batch_number]
        stem = _batch_file_stem(batch_number)
        batch_id = normalize_text(rows[0].get("batchId")) if rows else f"{bundle_id}_{stem}"
        attachment_refs = [normalize_text(row.get("attachmentRef")) for row in rows]
        batch_bundles.append(
            {
                "batchId": batch_id,
                "batchNumber": batch_number,
                "rowCount": len(rows),
                "promptRef": f"{bundle_dir_ref}/{stem}_prompt.md",
                "fillTemplateRef": f"{bundle_dir_ref}/{stem}_fill_template.v1.json",
                "attachmentRefs": attachment_refs,
                "rows": rows,
            }
        )

    placeholder_rows = sum(
        1 for row in template_rows if "FILL_IN" in json.dumps(row, ensure_ascii=False)
    )
    private_path_leak_rows = 1 if _contains_private_path(batch_bundles) else 0
    bundle_artifact_rows = len(batch_bundles) * 2
    counts = {
        "sourceTemplateRows": len(template_rows),
        "batchRows": len(batch_bundles),
        "operatorPromptRows": len(batch_bundles),
        "fillTemplateRows": len(batch_bundles),
        "bundleArtifactRows": bundle_artifact_rows,
        "attachmentRefRows": sum(len(batch.get("attachmentRefs") or []) for batch in batch_bundles),
        "placeholderRows": int(placeholder_rows),
        "completedWebOutputRows": 0,
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }
    report: dict[str, Any] = {
        "schema": VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_BUNDLE_DECISION,
        "nextRecommendedTranche": NEXT_AFTER_RUN_TRANCHE,
        "bundleId": bundle_id,
        "sourceWebOutputTemplate": {
            "schema": normalize_text(web_output_template.get("schema")),
            "status": normalize_text(web_output_template.get("status")),
            "reportRef": normalize_text(source_web_output_template_ref),
            "outputTemplateRows": len(template_rows),
        },
        "targetOutput": {
            "schema": VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
            "reportRef": normalize_text(target_output_ref),
            "validationCommand": validation_command,
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
        },
        "scope": _bundle_scope(
            batch_rows=len(batch_bundles),
            bundle_artifact_rows=bundle_artifact_rows,
        ),
        "counts": counts,
        "batchBundles": batch_bundles,
        "operatorInstructions": [
            "Run one batch at a time in web GPT/Pro.",
            "For each batch, upload only the attachment refs listed in that batch prompt.",
            "Paste the batch prompt and use the batch fill template only as a shape guide.",
            "After all batches return JSON rows, combine the rows into the target output file.",
            f"Validate the combined file with: {validation_command}",
        ],
        "warnings": [
            "This bundle is not web/VLM output and contains placeholder text.",
            "Batch fill templates use a batch-template schema, not the completed web-output schema.",
            "Do not upload whole pages or whole images for this gate.",
            "Do not treat derivedTextForRetrieval as strict, citation-grade, or answer-visible evidence.",
        ],
    }
    if (
        web_output_template.get("schema") != VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_TEMPLATE_SCHEMA_ID
        or web_output_template.get("status") != "ready"
        or not template_rows
        or private_path_leak_rows
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def _captured_row(
    *,
    source_row: dict[str, Any],
    attachment_row: dict[str, Any] | None,
    output_row: dict[str, Any],
    violations: list[str],
) -> dict[str, Any]:
    return {
        "schema": VISUAL_ANNOTATION_EXPANSION_CAPTURED_ROW_SCHEMA_ID,
        "sourceCandidateId": normalize_text(output_row.get("sourceCandidateId")),
        "sourcePackCandidateId": normalize_text(source_row.get("packCandidateId")),
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
            "sourceExpansionPackSchema": VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
            "sourceAttachmentPackSchema": VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID,
            "sourceCandidateId": normalize_text(output_row.get("sourceCandidateId")),
            "sourceContentHash": normalize_text(source_row.get("sourceContentHash")),
            "page": int(source_row.get("page") or 0),
            "bbox": list(source_row.get("bbox") or []),
            "extractionMethod": "manual_web_vlm_expansion_output_capture_v1",
        },
        "validation": {
            "matchedSourcePackRow": True,
            "matchedAttachmentRow": bool(attachment_row),
            "policyCompliant": not violations,
            "violationReasons": violations,
        },
    }


def build_visual_annotation_expansion_web_output_validation(
    output: dict[str, Any],
    expansion_pack: dict[str, Any],
    attachment_pack: dict[str, Any],
    *,
    output_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json",
    source_expansion_pack_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_pack_design.v1.json",
    source_attachment_pack_ref: str = "eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002.v1.json",
    generated_at: str | None = None,
) -> dict[str, Any]:
    output_rows = _output_rows(output)
    source_rows = _source_rows_by_id(expansion_pack)
    attachment_rows = _attachment_rows_by_id(attachment_pack)
    output_schema_errors = _schema_errors(output, VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID)
    duplicate_ids = _duplicate_ids(output_rows)
    output_ids = [normalize_text(row.get("sourceCandidateId")) for row in output_rows]
    output_id_set = {candidate_id for candidate_id in output_ids if candidate_id}
    source_id_set = set(source_rows)
    missing_ids = sorted(source_id_set - output_id_set)
    extra_ids = sorted(output_id_set - source_id_set)

    violations: list[dict[str, Any]] = []
    captured_rows: list[dict[str, Any]] = []
    policy_violation_ids: set[str] = set()
    empty_derived_text_ids: set[str] = set()
    keyword_empty_ids: set[str] = set()
    private_path_leak_ids: set[str] = set()

    for error in output_schema_errors:
        violations.append({"sourceCandidateId": "", "kind": "output_schema_violation", "message": error})
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
                "message": "Output is missing a required expansion source candidate.",
            }
        )
    for candidate_id in extra_ids:
        violations.append(
            {
                "sourceCandidateId": candidate_id,
                "kind": "extra_source_candidate_id",
                "message": "Output contains a candidate absent from the source expansion pack.",
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
        source_row = source_rows.get(candidate_id)
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
        "sourcePackRows": len(source_rows),
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
    counts["blockedRows"] = int(
        counts["missingRows"]
        + counts["extraRows"]
        + counts["duplicateRows"]
        + counts["policyViolationRows"]
        + counts["privatePathLeakRows"]
        + counts["schemaViolationCount"]
    )

    report: dict[str, Any] = {
        "schema": VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_VALIDATION_DECISION,
        "nextRecommendedTranche": NEXT_AFTER_VALIDATION_TRANCHE,
        "sourceOutput": {
            "schema": normalize_text(output.get("schema")),
            "reportRef": normalize_text(output_ref),
            "rows": len(output_rows),
        },
        "sourceExpansionPack": {
            "schema": normalize_text(expansion_pack.get("schema")),
            "status": normalize_text(expansion_pack.get("status")),
            "reportRef": normalize_text(source_expansion_pack_ref),
            "packRows": len(source_rows),
        },
        "sourceAttachmentPack": {
            "schema": normalize_text(attachment_pack.get("schema")),
            "status": normalize_text(attachment_pack.get("status")),
            "reportRef": normalize_text(source_attachment_pack_ref),
            "attachmentRows": len(attachment_rows),
        },
        "scope": _scope(manual_rows=len(output_rows)),
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
            "The next tranche must design an expansion candidate store before any vectorization decision.",
        ],
    }
    if (
        output.get("schema") != VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID
        or expansion_pack.get("schema") != VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID
        or expansion_pack.get("status") != "ready"
        or attachment_pack.get("schema") != VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID
        or attachment_pack.get("status") != "ready"
        or counts["blockedRows"]
        or counts["matchedRows"] != counts["sourcePackRows"]
        or counts["outputRows"] != counts["sourcePackRows"]
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def render_markdown_manual_run_packet(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Visual Annotation Expansion Manual Run Packet 002",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- packetId: `{report.get('packetId')}`",
        f"- packetRows: `{counts.get('packetRows')}`",
        f"- batchRows: `{counts.get('batchRows')}`",
        f"- missingAttachmentRows: `{counts.get('missingAttachmentRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        "",
        "## Batches",
        "",
    ]
    for batch in report.get("batches", []):
        lines.extend(
            [
                f"### Batch {batch.get('batchNumber')} of {batch.get('totalBatches')}",
                "",
                "Prompt:",
                "",
                "```text",
                str(batch.get("prompt") or ""),
                "```",
                "",
                "| # | paperId | type | page | sourceCandidateId | attachmentRef |",
                "|---:|---|---|---:|---|---|",
            ]
        )
        for index, row in enumerate(batch.get("rows", []), start=1):
            lines.append(
                "| {index} | {paperId} | {candidateType} | {page} | `{candidateId}` | `{attachment}` |".format(
                    index=index,
                    paperId=row.get("paperId"),
                    candidateType=row.get("candidateType"),
                    page=row.get("page"),
                    candidateId=row.get("sourceCandidateId"),
                    attachment=row.get("attachmentRef"),
                )
            )
        lines.append("")
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def render_markdown_operator_handoff(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    source = dict(report.get("sourceManualRunPacket") or {})
    lines = [
        "# Visual Annotation Expansion Operator Handoff 002",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- handoffId: `{report.get('handoffId')}`",
        f"- sourceManualRunPacket: `{source.get('reportRef')}`",
        f"- expectedOutputRef: `{report.get('expectedOutputRef')}`",
        f"- validationCommand: `{report.get('validationCommand')}`",
        f"- templateRows: `{counts.get('templateRows')}`",
        f"- batchRows: `{counts.get('batchRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- apiCalls: `{scope.get('apiCalls')}`",
        f"- modelCalls: `{scope.get('modelCalls')}`",
        f"- webModelCalls: `{scope.get('webModelCalls')}`",
        f"- manualOperatorWebModelRunRequired: `{scope.get('manualOperatorWebModelRunRequired')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        f"- candidateStoreMutationRows: `{scope.get('candidateStoreMutationRows')}`",
        f"- wholeImageGptRows: `{scope.get('wholeImageGptRows')}`",
        "",
        "## Operator Steps",
        "",
    ]
    for index, step in enumerate(report.get("operatorSteps") or [], start=1):
        lines.append(f"{index}. {step}")
    lines.extend(
        [
            "",
            "## Template Rows",
            "",
            "| # | batch | paperId | type | page | sourceCandidateId | attachmentRef |",
            "|---:|---:|---|---|---:|---|---|",
        ]
    )
    for index, row in enumerate(report.get("templateRows", []), start=1):
        lines.append(
            "| {index} | {batch} | {paperId} | {candidateType} | {page} | `{candidateId}` | `{attachment}` |".format(
                index=index,
                batch=row.get("batchNumber"),
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                candidateId=row.get("sourceCandidateId"),
                attachment=row.get("attachmentRef"),
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def render_markdown_web_output_template(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    target = dict(report.get("targetOutput") or {})
    source = dict(report.get("sourceOperatorHandoff") or {})
    lines = [
        "# Visual Annotation Expansion Web Output Template 002",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- templateId: `{report.get('templateId')}`",
        f"- sourceOperatorHandoff: `{source.get('reportRef')}`",
        f"- targetOutputRef: `{target.get('reportRef')}`",
        f"- validationCommand: `{target.get('validationCommand')}`",
        f"- outputTemplateRows: `{counts.get('outputTemplateRows')}`",
        f"- placeholderRows: `{counts.get('placeholderRows')}`",
        f"- completedWebOutputRows: `{counts.get('completedWebOutputRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- apiCalls: `{scope.get('apiCalls')}`",
        f"- modelCalls: `{scope.get('modelCalls')}`",
        f"- webModelCalls: `{scope.get('webModelCalls')}`",
        f"- templateOnly: `{scope.get('templateOnly')}`",
        f"- completedWebOutputRows: `{scope.get('completedWebOutputRows')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        f"- candidateStoreMutationRows: `{scope.get('candidateStoreMutationRows')}`",
        f"- wholeImageGptRows: `{scope.get('wholeImageGptRows')}`",
        "",
        "## Instructions",
        "",
    ]
    for index, instruction in enumerate(report.get("instructions") or [], start=1):
        lines.append(f"{index}. {instruction}")
    lines.extend(
        [
            "",
            "## Template Rows",
            "",
            "| # | batch | paperId | type | page | sourceCandidateId | attachmentRef |",
            "|---:|---:|---|---|---:|---|---|",
        ]
    )
    for index, row in enumerate(report.get("templateRows", []), start=1):
        lines.append(
            "| {index} | {batch} | {paperId} | {candidateType} | {page} | `{candidateId}` | `{attachment}` |".format(
                index=index,
                batch=row.get("batchNumber"),
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                candidateId=row.get("sourceCandidateId"),
                attachment=row.get("attachmentRef"),
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def render_markdown_web_run_bundle(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    target = dict(report.get("targetOutput") or {})
    source = dict(report.get("sourceWebOutputTemplate") or {})
    lines = [
        "# Visual Annotation Expansion Web Run Bundle 002",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- bundleId: `{report.get('bundleId')}`",
        f"- sourceWebOutputTemplate: `{source.get('reportRef')}`",
        f"- targetOutputRef: `{target.get('reportRef')}`",
        f"- validationCommand: `{target.get('validationCommand')}`",
        f"- sourceTemplateRows: `{counts.get('sourceTemplateRows')}`",
        f"- batchRows: `{counts.get('batchRows')}`",
        f"- bundleArtifactRows: `{counts.get('bundleArtifactRows')}`",
        f"- completedWebOutputRows: `{counts.get('completedWebOutputRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- apiCalls: `{scope.get('apiCalls')}`",
        f"- modelCalls: `{scope.get('modelCalls')}`",
        f"- webModelCalls: `{scope.get('webModelCalls')}`",
        f"- operatorBatchPromptRows: `{scope.get('operatorBatchPromptRows')}`",
        f"- completedWebOutputRows: `{scope.get('completedWebOutputRows')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        f"- candidateStoreMutationRows: `{scope.get('candidateStoreMutationRows')}`",
        f"- wholeImageGptRows: `{scope.get('wholeImageGptRows')}`",
        "",
        "## Batch Bundles",
        "",
        "| batch | rows | promptRef | fillTemplateRef |",
        "|---:|---:|---|---|",
    ]
    for batch in report.get("batchBundles", []):
        lines.append(
            "| {batch} | {rows} | `{prompt}` | `{template}` |".format(
                batch=batch.get("batchNumber"),
                rows=batch.get("rowCount"),
                prompt=batch.get("promptRef"),
                template=batch.get("fillTemplateRef"),
            )
        )
    lines.extend(["", "## Operator Instructions", ""])
    for index, instruction in enumerate(report.get("operatorInstructions") or [], start=1):
        lines.append(f"{index}. {instruction}")
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def render_markdown_validation(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    output = dict(report.get("sourceOutput") or {})
    lines = [
        "# Visual Annotation Expansion Web Output 002 Validation",
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


def write_visual_annotation_expansion_manual_run_packet(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_manual_run_packet(report), encoding="utf-8")
    return {"json": str(report_json), "markdown": str(report_md)}


def write_visual_annotation_expansion_operator_handoff(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_operator_handoff(report), encoding="utf-8")
    return {"json": str(report_json), "markdown": str(report_md)}


def write_visual_annotation_expansion_web_output_template(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_web_output_template(report), encoding="utf-8")
    return {"json": str(report_json), "markdown": str(report_md)}


def write_visual_annotation_expansion_web_run_bundle(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
    bundle_dir: Path,
) -> dict[str, Any]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    batch_paths = []
    for batch in report.get("batchBundles", []):
        stem = _batch_file_stem(int(batch.get("batchNumber") or 0))
        prompt_path = bundle_dir / f"{stem}_prompt.md"
        template_path = bundle_dir / f"{stem}_fill_template.v1.json"
        prompt_path.write_text(render_markdown_web_run_batch_prompt(batch), encoding="utf-8")
        batch_template = build_visual_annotation_expansion_web_run_batch_template(batch)
        template_path.write_text(
            json.dumps(batch_template, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        batch_paths.append({"prompt": str(prompt_path), "fillTemplate": str(template_path)})
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_web_run_bundle(report), encoding="utf-8")
    return {"json": str(report_json), "markdown": str(report_md), "batchFiles": batch_paths}


def write_visual_annotation_expansion_web_output_validation(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_validation(report), encoding="utf-8")
    return {"json": str(report_json), "markdown": str(report_md)}


__all__ = [
    "DEFAULT_MANUAL_RUN_PACKET_ID",
    "NEXT_AFTER_RUN_TRANCHE",
    "NEXT_AFTER_VALIDATION_TRANCHE",
    "READY_RUN_DECISION",
    "READY_HANDOFF_DECISION",
    "READY_TEMPLATE_DECISION",
    "READY_BUNDLE_DECISION",
    "READY_VALIDATION_DECISION",
    "VISUAL_ANNOTATION_EXPANSION_CAPTURED_ROW_SCHEMA_ID",
    "VISUAL_ANNOTATION_EXPANSION_MANUAL_RUN_PACKET_SCHEMA_ID",
    "VISUAL_ANNOTATION_EXPANSION_OPERATOR_HANDOFF_SCHEMA_ID",
    "VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BATCH_TEMPLATE_SCHEMA_ID",
    "VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID",
    "VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_TEMPLATE_SCHEMA_ID",
    "VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID",
    "build_visual_annotation_expansion_manual_run_packet",
    "build_visual_annotation_expansion_operator_handoff",
    "build_visual_annotation_expansion_web_run_batch_template",
    "build_visual_annotation_expansion_web_run_bundle",
    "build_visual_annotation_expansion_web_output_template",
    "build_visual_annotation_expansion_web_output_validation",
    "load_json",
    "render_markdown_manual_run_packet",
    "render_markdown_operator_handoff",
    "render_markdown_web_run_batch_prompt",
    "render_markdown_web_run_bundle",
    "render_markdown_web_output_template",
    "render_markdown_validation",
    "sanitized_report_ref",
    "write_visual_annotation_expansion_manual_run_packet",
    "write_visual_annotation_expansion_operator_handoff",
    "write_visual_annotation_expansion_web_run_bundle",
    "write_visual_annotation_expansion_web_output_template",
    "write_visual_annotation_expansion_web_output_validation",
]
