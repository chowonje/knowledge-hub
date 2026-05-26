from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_expansion_attachment_pack import (
    VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID,
)
from knowledge_hub.papers.visual_annotation_expansion_manual_output_capture import (
    VISUAL_ANNOTATION_EXPANSION_MANUAL_RUN_PACKET_SCHEMA_ID,
    VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
    VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
    build_visual_annotation_expansion_manual_run_packet,
    build_visual_annotation_expansion_web_output_validation,
    write_visual_annotation_expansion_manual_run_packet,
    write_visual_annotation_expansion_web_output_validation,
)
from knowledge_hub.papers.visual_annotation_expansion_pack_design import (
    VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
)


def _hash() -> str:
    return "sha256:" + "1" * 64


def _source_ids() -> list[str]:
    return [
        "visual-layout:sample-paper:image_region:1:1111111111111111",
        "visual-layout:sample-paper:figure_caption_region:2:2222222222222222",
        "visual-layout:sample-paper:table_region:3:3333333333333333",
    ]


def _expansion_pack() -> dict[str, object]:
    rows = []
    for index, candidate_id in enumerate(_source_ids(), start=1):
        candidate_type = candidate_id.split(":")[2]
        rows.append(
            {
                "schema": "knowledge-hub.paper.visual-annotation-expansion-pack-row.v1",
                "packCandidateId": f"visual-annotation-expansion-pack:test:{index}",
                "sourceCandidateId": candidate_id,
                "paperId": "sample-paper",
                "paperRef": "papers_dir/sample.pdf",
                "sourceContentHash": _hash(),
                "page": index,
                "bbox": [10.0, 20.0, 200.0, 260.0],
                "candidateType": candidate_type,
                "priority": index,
                "webInput": {
                    "copyPasteContext": f"paperId=sample-paper page={index} candidateType={candidate_type}"
                },
            }
        )
    return {
        "schema": VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
        "status": "ready",
        "packRowsDetail": rows,
    }


def _attachment_pack() -> dict[str, object]:
    return {
        "schema": VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID,
        "status": "ready",
        "attachmentRowsDetail": [
            {
                "sourceCandidateId": candidate_id,
                "attachmentRef": (
                    "eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/"
                    f"assets/{index}.png"
                ),
            }
            for index, candidate_id in enumerate(_source_ids(), start=1)
        ],
    }


def _output_row(candidate_id: str) -> dict[str, object]:
    return {
        "sourceCandidateId": candidate_id,
        "visualObservationStatus": "image_attached",
        "derivedTextForRetrieval": "Retrieval hint only: sample expansion visual region.",
        "visibleText": "Visible fragments include: sample figure and labels.",
        "retrievalKeywords": ["sample", "expansion", "visual"],
        "uncertainty": "Low. The sample crop is readable.",
        "limitations": "Retrieval hint only, not evidence.",
        "strictEvidence": False,
        "citationGrade": False,
        "answerableWithoutTextEvidence": False,
    }


def _output() -> dict[str, object]:
    return {
        "schema": VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
        "rows": [_output_row(candidate_id) for candidate_id in _source_ids()],
    }


def test_manual_run_packet_batches_attachment_refs_and_preserves_no_model_contract() -> None:
    report = build_visual_annotation_expansion_manual_run_packet(
        _expansion_pack(),
        _attachment_pack(),
        batch_size=2,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["schema"] == VISUAL_ANNOTATION_EXPANSION_MANUAL_RUN_PACKET_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_manual_web_vlm_expansion_run"
    assert report["nextRecommendedTranche"] == "visual_annotation_expansion_manual_output_capture"
    assert report["counts"]["packetRows"] == 3
    assert report["counts"]["batchRows"] == 2
    assert report["counts"]["missingAttachmentRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["scope"]["writes"] == "report_only"
    assert report["scope"]["apiCalls"] is False
    assert report["scope"]["modelCalls"] is False
    assert report["scope"]["webModelCalls"] is False
    assert report["scope"]["manualWebModelOutputRows"] == 0
    assert report["scope"]["vectorIndexing"] is False
    assert report["scope"]["strictEvidencePromotionRows"] == 0
    assert report["scope"]["runtimeAnswerVisibleExposureRows"] == 0
    assert report["scope"]["candidateStoreMutationRows"] == 0
    assert report["scope"]["wholeImageGptRows"] == 0
    assert report["outputContract"]["allowedUse"] == "retrieval_hint_only"
    assert report["batches"][0]["attachmentRefs"]
    assert "Do not create citation-grade evidence" in report["batches"][0]["prompt"]

    validation = validate_payload(
        report,
        VISUAL_ANNOTATION_EXPANSION_MANUAL_RUN_PACKET_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_expansion_web_output_validation_ready_and_no_mutation_contract() -> None:
    output = _output()
    report = build_visual_annotation_expansion_web_output_validation(
        output,
        _expansion_pack(),
        _attachment_pack(),
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["schema"] == VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_visual_retrieval_hint_candidate_store_expansion_design"
    assert report["nextRecommendedTranche"] == "visual_retrieval_hint_candidate_store_expansion_design"
    assert report["counts"]["sourcePackRows"] == 3
    assert report["counts"]["outputRows"] == 3
    assert report["counts"]["matchedRows"] == 3
    assert report["counts"]["blockedRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["counts"]["schemaViolationCount"] == 0
    assert report["scope"]["writes"] == "report_only"
    assert report["scope"]["apiCalls"] is False
    assert report["scope"]["modelCalls"] is False
    assert report["scope"]["webModelCalls"] is False
    assert report["scope"]["manualWebModelOutputRows"] == 3
    assert report["scope"]["vectorIndexing"] is False
    assert report["scope"]["strictEvidencePromotionRows"] == 0
    assert report["scope"]["runtimeAnswerVisibleExposureRows"] == 0
    assert report["scope"]["databaseMutationRows"] == 0
    assert report["scope"]["indexMutationRows"] == 0
    assert report["scope"]["vaultScanRows"] == 0
    assert report["scope"]["externalDownloadRows"] == 0
    assert report["scope"]["answerabilityGateBypassRows"] == 0
    assert report["scope"]["candidateStoreMutationRows"] == 0
    assert report["scope"]["wholeImageGptRows"] == 0

    output_validation = validate_payload(output, VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID, strict=True)
    assert output_validation.ok, output_validation.errors
    report_validation = validate_payload(
        report,
        VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
        strict=True,
    )
    assert report_validation.ok, report_validation.errors

    row = report["capturedRowsDetail"][0]
    assert row["retrievalHintPlan"]["allowedUse"] == "retrieval_hint_only"
    assert row["validation"]["matchedAttachmentRow"] is True
    assert row["validation"]["policyCompliant"] is True


def test_expansion_web_output_validation_blocks_duplicate_extra_and_policy_violations() -> None:
    output = _output()
    duplicate = copy.deepcopy(output["rows"][0])
    duplicate["strictEvidence"] = True
    output["rows"] = [
        output["rows"][0],
        duplicate,
        _output_row("visual-layout:extra:image_region:1:eeeeeeeeeeeeeeee"),
    ]

    report = build_visual_annotation_expansion_web_output_validation(
        output,
        _expansion_pack(),
        _attachment_pack(),
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["decision"] == "blocked"
    assert report["counts"]["missingRows"] == 2
    assert report["counts"]["extraRows"] == 1
    assert report["counts"]["duplicateRows"] == 1
    assert report["counts"]["policyViolationRows"] == 1
    assert report["counts"]["schemaViolationCount"] >= 1
    assert any(v["kind"] == "strict_evidence_not_false" for v in report["violations"])


def test_expansion_web_output_validation_detects_private_path_leak() -> None:
    output = _output()
    output["rows"][0]["derivedTextForRetrieval"] = (
        "Retrieval hint only: /" + "Users" + "/won/private.pdf"
    )

    report = build_visual_annotation_expansion_web_output_validation(
        output,
        _expansion_pack(),
        _attachment_pack(),
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert any(v["kind"] == "private_path_leak" for v in report["violations"])


def test_writers_output_sanitized_refs(tmp_path: Path) -> None:
    packet = build_visual_annotation_expansion_manual_run_packet(
        _expansion_pack(),
        _attachment_pack(),
        source_expansion_pack_ref="eval/knowledgeos/reports/visual_annotation_expansion_pack_design.v1.json",
        source_attachment_pack_ref=(
            "eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002.v1.json"
        ),
        generated_at="2026-05-26T00:00:00Z",
    )
    validation_report = build_visual_annotation_expansion_web_output_validation(
        _output(),
        _expansion_pack(),
        _attachment_pack(),
        output_ref="eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json",
        source_expansion_pack_ref="eval/knowledgeos/reports/visual_annotation_expansion_pack_design.v1.json",
        source_attachment_pack_ref=(
            "eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002.v1.json"
        ),
        generated_at="2026-05-26T00:00:00Z",
    )
    packet_json = tmp_path / "packet.json"
    packet_md = tmp_path / "packet.md"
    validation_json = tmp_path / "validation.json"
    validation_md = tmp_path / "validation.md"

    write_visual_annotation_expansion_manual_run_packet(
        packet,
        report_json=packet_json,
        report_md=packet_md,
    )
    write_visual_annotation_expansion_web_output_validation(
        validation_report,
        report_json=validation_json,
        report_md=validation_md,
    )

    combined = (
        packet_json.read_text(encoding="utf-8")
        + packet_md.read_text(encoding="utf-8")
        + validation_json.read_text(encoding="utf-8")
        + validation_md.read_text(encoding="utf-8")
    )
    parsed = json.loads(validation_json.read_text(encoding="utf-8"))
    assert parsed["sourceOutput"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert parsed["capturedRowsDetail"][0]["paperRef"] == "papers_dir/sample.pdf"
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
