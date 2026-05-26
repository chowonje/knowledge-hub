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
    VISUAL_ANNOTATION_EXPANSION_OPERATOR_HANDOFF_SCHEMA_ID,
    VISUAL_ANNOTATION_EXPANSION_WEB_BATCH_OUTPUT_COLLECTOR_SCHEMA_ID,
    VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BATCH_TEMPLATE_SCHEMA_ID,
    VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID,
    VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_TEMPLATE_SCHEMA_ID,
    VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_VALIDATION_SCHEMA_ID,
    VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
    build_visual_annotation_expansion_manual_run_packet,
    build_visual_annotation_expansion_operator_handoff,
    build_visual_annotation_expansion_web_batch_output_collector,
    build_visual_annotation_expansion_web_run_batch_template,
    build_visual_annotation_expansion_web_run_bundle,
    build_visual_annotation_expansion_web_output_template,
    build_visual_annotation_expansion_web_output_validation,
    combine_visual_annotation_expansion_web_batch_outputs,
    render_markdown_web_run_batch_prompt,
    write_visual_annotation_expansion_web_batch_output_collector,
    write_visual_annotation_expansion_manual_run_packet,
    write_visual_annotation_expansion_operator_handoff,
    write_visual_annotation_expansion_web_run_bundle,
    write_visual_annotation_expansion_web_output_template,
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


def _bundle() -> dict[str, object]:
    packet = build_visual_annotation_expansion_manual_run_packet(
        _expansion_pack(),
        _attachment_pack(),
        batch_size=2,
        generated_at="2026-05-26T00:00:00Z",
    )
    handoff = build_visual_annotation_expansion_operator_handoff(
        packet,
        generated_at="2026-05-26T00:00:00Z",
    )
    template = build_visual_annotation_expansion_web_output_template(
        handoff,
        generated_at="2026-05-26T00:00:00Z",
    )
    return build_visual_annotation_expansion_web_run_bundle(
        template,
        generated_at="2026-05-26T00:00:00Z",
    )


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


def test_operator_handoff_preserves_template_only_policy_and_validation_command() -> None:
    packet = build_visual_annotation_expansion_manual_run_packet(
        _expansion_pack(),
        _attachment_pack(),
        batch_size=2,
        generated_at="2026-05-26T00:00:00Z",
    )

    handoff = build_visual_annotation_expansion_operator_handoff(
        packet,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert handoff["schema"] == VISUAL_ANNOTATION_EXPANSION_OPERATOR_HANDOFF_SCHEMA_ID
    assert handoff["status"] == "ready"
    assert handoff["decision"] == "ready_for_operator_web_vlm_run"
    assert handoff["nextRecommendedTranche"] == "visual_annotation_expansion_manual_output_capture"
    assert handoff["expectedOutputRef"] == (
        "eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json"
    )
    assert handoff["counts"]["packetRows"] == 3
    assert handoff["counts"]["templateRows"] == 3
    assert handoff["counts"]["batchRows"] == 2
    assert handoff["counts"]["privatePathLeakRows"] == 0
    assert handoff["scope"]["writes"] == "report_only"
    assert handoff["scope"]["apiCalls"] is False
    assert handoff["scope"]["modelCalls"] is False
    assert handoff["scope"]["webModelCalls"] is False
    assert handoff["scope"]["manualOperatorWebModelRunRequired"] is True
    assert handoff["scope"]["vectorIndexing"] is False
    assert handoff["scope"]["strictEvidencePromotionRows"] == 0
    assert handoff["scope"]["runtimeAnswerVisibleExposureRows"] == 0
    assert handoff["scope"]["candidateStoreMutationRows"] == 0
    assert handoff["scope"]["wholeImageGptRows"] == 0
    assert "validate_visual_annotation_expansion_web_output.py" in handoff["validationCommand"]
    assert "FILL_IN" in handoff["templateRows"][0]["outputRowSkeleton"]["derivedTextForRetrieval"]
    assert handoff["outputContract"]["allowedUse"] == "retrieval_hint_only"

    validation = validate_payload(
        handoff,
        VISUAL_ANNOTATION_EXPANSION_OPERATOR_HANDOFF_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_web_output_template_is_fill_only_and_not_completed_output() -> None:
    packet = build_visual_annotation_expansion_manual_run_packet(
        _expansion_pack(),
        _attachment_pack(),
        batch_size=2,
        generated_at="2026-05-26T00:00:00Z",
    )
    handoff = build_visual_annotation_expansion_operator_handoff(
        packet,
        generated_at="2026-05-26T00:00:00Z",
    )

    template = build_visual_annotation_expansion_web_output_template(
        handoff,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert template["schema"] == VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_TEMPLATE_SCHEMA_ID
    assert template["schema"] != VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID
    assert template["status"] == "ready"
    assert template["decision"] == "ready_for_manual_web_output_fill"
    assert template["nextRecommendedTranche"] == "visual_annotation_expansion_manual_output_capture"
    assert template["targetOutput"]["schema"] == VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID
    assert template["targetOutput"]["reportRef"] == (
        "eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json"
    )
    assert template["counts"]["sourceTemplateRows"] == 3
    assert template["counts"]["outputTemplateRows"] == 3
    assert template["counts"]["batchRows"] == 2
    assert template["counts"]["placeholderRows"] == 3
    assert template["counts"]["completedWebOutputRows"] == 0
    assert template["counts"]["privatePathLeakRows"] == 0
    assert template["scope"]["templateOnly"] is True
    assert template["scope"]["completedWebOutputRows"] == 0
    assert template["scope"]["manualWebModelOutputRows"] == 0
    assert template["scope"]["modelCalls"] is False
    assert template["scope"]["webModelCalls"] is False
    assert template["scope"]["vectorIndexing"] is False
    assert template["scope"]["strictEvidencePromotionRows"] == 0
    assert template["scope"]["candidateStoreMutationRows"] == 0
    assert "FILL_IN" in template["templateRows"][0]["targetRowTemplate"]["derivedTextForRetrieval"]

    validation = validate_payload(
        template,
        VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_TEMPLATE_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors

    target_output_validation = validate_payload(
        template,
        VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
        strict=True,
    )
    assert not target_output_validation.ok


def test_web_run_bundle_splits_batches_without_completed_output() -> None:
    packet = build_visual_annotation_expansion_manual_run_packet(
        _expansion_pack(),
        _attachment_pack(),
        batch_size=2,
        generated_at="2026-05-26T00:00:00Z",
    )
    handoff = build_visual_annotation_expansion_operator_handoff(
        packet,
        generated_at="2026-05-26T00:00:00Z",
    )
    template = build_visual_annotation_expansion_web_output_template(
        handoff,
        generated_at="2026-05-26T00:00:00Z",
    )

    bundle = build_visual_annotation_expansion_web_run_bundle(
        template,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert bundle["schema"] == VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID
    assert bundle["status"] == "ready"
    assert bundle["decision"] == "ready_for_operator_web_batch_run"
    assert bundle["nextRecommendedTranche"] == "visual_annotation_expansion_manual_output_capture"
    assert bundle["targetOutput"]["reportRef"] == (
        "eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json"
    )
    assert bundle["counts"]["sourceTemplateRows"] == 3
    assert bundle["counts"]["batchRows"] == 2
    assert bundle["counts"]["operatorPromptRows"] == 2
    assert bundle["counts"]["fillTemplateRows"] == 2
    assert bundle["counts"]["bundleArtifactRows"] == 4
    assert bundle["counts"]["attachmentRefRows"] == 3
    assert bundle["counts"]["placeholderRows"] == 3
    assert bundle["counts"]["completedWebOutputRows"] == 0
    assert bundle["counts"]["privatePathLeakRows"] == 0
    assert bundle["scope"]["modelCalls"] is False
    assert bundle["scope"]["webModelCalls"] is False
    assert bundle["scope"]["manualWebModelOutputRows"] == 0
    assert bundle["scope"]["vectorIndexing"] is False
    assert bundle["scope"]["strictEvidencePromotionRows"] == 0
    assert bundle["scope"]["candidateStoreMutationRows"] == 0
    assert bundle["batchBundles"][0]["promptRef"].endswith("batch_01_prompt.md")
    assert bundle["batchBundles"][0]["fillTemplateRef"].endswith(
        "batch_01_fill_template.v1.json"
    )

    bundle_validation = validate_payload(
        bundle,
        VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID,
        strict=True,
    )
    assert bundle_validation.ok, bundle_validation.errors

    batch_template = build_visual_annotation_expansion_web_run_batch_template(
        bundle["batchBundles"][0]
    )
    assert batch_template["schema"] == VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BATCH_TEMPLATE_SCHEMA_ID
    assert batch_template["targetOutputSchema"] == VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID
    assert "FILL_IN" in batch_template["rows"][0]["derivedTextForRetrieval"]
    batch_validation = validate_payload(
        batch_template,
        VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BATCH_TEMPLATE_SCHEMA_ID,
        strict=True,
    )
    assert batch_validation.ok, batch_validation.errors
    assert not validate_payload(
        batch_template,
        VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
        strict=True,
    ).ok

    prompt = render_markdown_web_run_batch_prompt(bundle["batchBundles"][0])
    assert "Attachments To Upload" in prompt
    assert "knowledge-hub.paper.visual-annotation-web-output.v1" in prompt
    assert "strictEvidence=false" in prompt


def test_web_batch_output_collector_blocks_missing_batch_outputs() -> None:
    bundle = _bundle()

    report = build_visual_annotation_expansion_web_batch_output_collector(
        bundle,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["schema"] == VISUAL_ANNOTATION_EXPANSION_WEB_BATCH_OUTPUT_COLLECTOR_SCHEMA_ID
    assert report["status"] == "blocked"
    assert report["decision"] == "blocked_missing_or_invalid_batch_outputs"
    assert report["counts"]["expectedBatchRows"] == 2
    assert report["counts"]["presentBatchRows"] == 0
    assert report["counts"]["missingBatchRows"] == 2
    assert report["counts"]["expectedOutputRows"] == 3
    assert report["counts"]["collectedOutputRows"] == 0
    assert report["counts"]["blockedRows"] >= 2
    assert report["scope"]["writes"] == "report_only"
    assert report["scope"]["modelCalls"] is False
    assert report["scope"]["webModelCalls"] is False
    assert report["scope"]["combinedOutputWriteRows"] == 0
    assert report["scope"]["vectorIndexing"] is False
    assert report["batchOutputRowsDetail"][0]["outputRef"].endswith(
        "batch_01_web_output.manual.json"
    )
    assert report["batchOutputRowsDetail"][0]["blockerReasons"] == ["missing_batch_output_file"]

    validation = validate_payload(
        report,
        VISUAL_ANNOTATION_EXPANSION_WEB_BATCH_OUTPUT_COLLECTOR_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_web_batch_output_collector_accepts_valid_batches_and_combines_rows() -> None:
    bundle = _bundle()
    first_batch_ids = [
        row["sourceCandidateId"] for row in bundle["batchBundles"][0]["rows"]
    ]
    second_batch_ids = [
        row["sourceCandidateId"] for row in bundle["batchBundles"][1]["rows"]
    ]
    outputs = {
        (
            "eval/knowledgeos/reports/visual_annotation_expansion_web_batch_outputs_002/"
            "batch_01_web_output.manual.json"
        ): {
            "schema": VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
            "rows": [_output_row(candidate_id) for candidate_id in first_batch_ids],
        },
        (
            "eval/knowledgeos/reports/visual_annotation_expansion_web_batch_outputs_002/"
            "batch_02_web_output.manual.json"
        ): {
            "schema": VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
            "rows": [_output_row(candidate_id) for candidate_id in second_batch_ids],
        },
    }

    report = build_visual_annotation_expansion_web_batch_output_collector(
        bundle,
        outputs,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_combined_manual_output_validation"
    assert report["counts"]["presentBatchRows"] == 2
    assert report["counts"]["missingBatchRows"] == 0
    assert report["counts"]["validBatchRows"] == 2
    assert report["counts"]["expectedOutputRows"] == 3
    assert report["counts"]["matchedOutputRows"] == 3
    assert report["counts"]["placeholderRows"] == 0
    assert report["counts"]["policyViolationRows"] == 0
    assert report["counts"]["blockedRows"] == 0

    validation = validate_payload(
        report,
        VISUAL_ANNOTATION_EXPANSION_WEB_BATCH_OUTPUT_COLLECTOR_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors

    combined = combine_visual_annotation_expansion_web_batch_outputs(bundle, outputs)
    assert combined["schema"] == VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID
    assert [row["sourceCandidateId"] for row in combined["rows"]] == first_batch_ids + second_batch_ids
    combined_validation = validate_payload(
        combined,
        VISUAL_ANNOTATION_WEB_OUTPUT_SCHEMA_ID,
        strict=True,
    )
    assert combined_validation.ok, combined_validation.errors


def test_web_batch_output_collector_rejects_template_placeholders_as_output() -> None:
    bundle = _bundle()
    template_output = build_visual_annotation_expansion_web_run_batch_template(
        bundle["batchBundles"][0]
    )
    outputs = {
        (
            "eval/knowledgeos/reports/visual_annotation_expansion_web_batch_outputs_002/"
            "batch_01_web_output.manual.json"
        ): template_output
    }

    report = build_visual_annotation_expansion_web_batch_output_collector(
        bundle,
        outputs,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["presentBatchRows"] == 1
    assert report["counts"]["schemaViolationCount"] >= 1
    assert report["counts"]["placeholderRows"] == 2
    assert "schema_violation" in report["batchOutputRowsDetail"][0]["blockerReasons"]
    assert "placeholder_text_present" in report["batchOutputRowsDetail"][0]["blockerReasons"]


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
    handoff_json = tmp_path / "handoff.json"
    handoff_md = tmp_path / "handoff.md"
    template_json = tmp_path / "template.json"
    template_md = tmp_path / "template.md"
    bundle_json = tmp_path / "bundle.json"
    bundle_md = tmp_path / "bundle.md"
    bundle_dir = tmp_path / "bundle"
    collector_json = tmp_path / "collector.json"
    collector_md = tmp_path / "collector.md"
    validation_json = tmp_path / "validation.json"
    validation_md = tmp_path / "validation.md"

    write_visual_annotation_expansion_manual_run_packet(
        packet,
        report_json=packet_json,
        report_md=packet_md,
    )
    handoff = build_visual_annotation_expansion_operator_handoff(
        packet,
        source_manual_run_packet_ref=(
            "eval/knowledgeos/reports/visual_annotation_expansion_manual_run_packet_002.v1.json"
        ),
        generated_at="2026-05-26T00:00:00Z",
    )
    write_visual_annotation_expansion_operator_handoff(
        handoff,
        report_json=handoff_json,
        report_md=handoff_md,
    )
    template = build_visual_annotation_expansion_web_output_template(
        handoff,
        source_operator_handoff_ref=(
            "eval/knowledgeos/reports/visual_annotation_expansion_operator_handoff_002.v1.json"
        ),
        generated_at="2026-05-26T00:00:00Z",
    )
    write_visual_annotation_expansion_web_output_template(
        template,
        report_json=template_json,
        report_md=template_md,
    )
    bundle = build_visual_annotation_expansion_web_run_bundle(
        template,
        source_web_output_template_ref=(
            "eval/knowledgeos/reports/visual_annotation_expansion_web_output_template_002.v1.json"
        ),
        generated_at="2026-05-26T00:00:00Z",
    )
    bundle_paths = write_visual_annotation_expansion_web_run_bundle(
        bundle,
        report_json=bundle_json,
        report_md=bundle_md,
        bundle_dir=bundle_dir,
    )
    collector = build_visual_annotation_expansion_web_batch_output_collector(
        bundle,
        source_web_run_bundle_ref=(
            "eval/knowledgeos/reports/visual_annotation_expansion_web_run_bundle_002.v1.json"
        ),
        generated_at="2026-05-26T00:00:00Z",
    )
    write_visual_annotation_expansion_web_batch_output_collector(
        collector,
        report_json=collector_json,
        report_md=collector_md,
    )
    write_visual_annotation_expansion_web_output_validation(
        validation_report,
        report_json=validation_json,
        report_md=validation_md,
    )

    combined = (
        packet_json.read_text(encoding="utf-8")
        + packet_md.read_text(encoding="utf-8")
        + handoff_json.read_text(encoding="utf-8")
        + handoff_md.read_text(encoding="utf-8")
        + template_json.read_text(encoding="utf-8")
        + template_md.read_text(encoding="utf-8")
        + bundle_json.read_text(encoding="utf-8")
        + bundle_md.read_text(encoding="utf-8")
        + collector_json.read_text(encoding="utf-8")
        + collector_md.read_text(encoding="utf-8")
        + validation_json.read_text(encoding="utf-8")
        + validation_md.read_text(encoding="utf-8")
    )
    for batch_path in bundle_paths["batchFiles"]:
        combined += Path(batch_path["prompt"]).read_text(encoding="utf-8")
        combined += Path(batch_path["fillTemplate"]).read_text(encoding="utf-8")
    parsed = json.loads(validation_json.read_text(encoding="utf-8"))
    assert parsed["sourceOutput"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert parsed["capturedRowsDetail"][0]["paperRef"] == "papers_dir/sample.pdf"
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
