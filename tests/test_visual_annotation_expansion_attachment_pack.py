from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_expansion_attachment_pack import (
    VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID,
    build_visual_annotation_expansion_attachment_pack,
    sha256_file,
    write_visual_annotation_expansion_attachment_pack,
)
from knowledge_hub.papers.visual_annotation_expansion_pack_design import (
    VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
)


def _make_pdf(path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page(width=360, height=360)
    page.insert_text((72, 80), "Figure 1: expansion visual context")
    page.draw_rect(fitz.Rect(70, 105, 230, 210), color=(0, 0, 0), fill=(0.85, 0.9, 1.0))
    document.save(str(path))
    document.close()


def _expansion_pack(pdf_hash: str, *, paper_ref: str = "papers_dir/sample.pdf") -> dict[str, object]:
    return {
        "schema": VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
        "status": "ready",
        "packRowsDetail": [
            {
                "schema": "knowledge-hub.paper.visual-annotation-expansion-pack-row.v1",
                "packCandidateId": "visual-annotation-expansion-pack:test:1111111111111111",
                "sourceCandidateId": "visual-layout:sample-paper:image_region:1:1111111111111111",
                "paperId": "sample-paper",
                "paperRef": paper_ref,
                "sourceContentHash": pdf_hash,
                "page": 1,
                "bbox": [70.0, 100.0, 230.0, 210.0],
                "candidateType": "image_region",
                "priority": 1,
            }
        ],
    }


def test_expansion_attachment_pack_renders_context_crop_and_validates_schema(tmp_path: Path) -> None:
    papers_root = tmp_path / "papers"
    papers_root.mkdir()
    pdf_path = papers_root / "sample.pdf"
    _make_pdf(pdf_path)

    report = build_visual_annotation_expansion_attachment_pack(
        _expansion_pack(sha256_file(pdf_path)),
        papers_root=papers_root,
        output_dir=tmp_path / "out",
        output_dir_ref="eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002",
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_manual_web_vlm_expansion_run"
    assert report["nextRecommendedTranche"] == "visual_annotation_expansion_manual_output_capture"
    assert report["scope"]["writes"] == "report_and_attachment_files_only"
    assert report["scope"]["modelCalls"] is False
    assert report["scope"]["webModelCalls"] is False
    assert report["scope"]["vectorIndexing"] is False
    assert report["scope"]["databaseMutationRows"] == 0
    assert report["scope"]["indexMutationRows"] == 0
    assert report["scope"]["vaultScanRows"] == 0
    assert report["scope"]["externalDownloadRows"] == 0
    assert report["scope"]["strictEvidencePromotionRows"] == 0
    assert report["scope"]["runtimeAnswerVisibleExposureRows"] == 0
    assert report["scope"]["answerabilityGateBypassRows"] == 0
    assert report["scope"]["candidateStoreMutationRows"] == 0
    assert report["scope"]["cropWriteRows"] == 1
    assert report["scope"]["pageImageWriteRows"] == 0
    assert report["scope"]["wholeImageWriteRows"] == 0
    assert report["scope"]["wholeImageGptRows"] == 0
    assert report["counts"]["cropAttachmentRows"] == 1
    assert report["counts"]["imageCandidateRows"] == 1
    assert report["counts"]["wholeImageRows"] == 0
    assert report["counts"]["blockedRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0

    row = report["attachmentRowsDetail"][0]
    assert row["schema"] == "knowledge-hub.paper.visual-annotation-expansion-attachment-row.v1"
    assert row["sourcePackCandidateId"].startswith("visual-annotation-expansion-pack:")
    assert row["attachmentRef"].startswith(
        "eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/"
    )
    assert row["attachmentSha256"].startswith("sha256:")
    assert row["attachmentBytes"] > 0
    assert row["pixelWidth"] > 0
    assert row["pixelHeight"] > 0
    assert row["wholeImageAttachment"] is False
    assert row["pageImageAttachment"] is False
    assert row["fullImageEscalationStatus"] == "deferred_to_full_image_gate"
    assert row["retrievalHintUseOnly"] is True
    assert row["strictEvidence"] is False
    assert row["citationGrade"] is False
    assert row["answerableWithoutTextEvidence"] is False

    rendered = tmp_path / "out" / "assets" / Path(row["attachmentRef"]).name
    assert rendered.is_file()
    assert sha256_file(rendered) == row["attachmentSha256"]

    validation = validate_payload(
        report,
        VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_expansion_attachment_pack_blocks_missing_pdf_without_external_download(tmp_path: Path) -> None:
    report = build_visual_annotation_expansion_attachment_pack(
        _expansion_pack("sha256:" + "1" * 64),
        papers_root=tmp_path / "missing",
        output_dir=tmp_path / "out",
        output_dir_ref="eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002",
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["decision"] == "blocked"
    assert report["counts"]["blockedRows"] == 1
    assert report["scope"]["cropWriteRows"] == 0
    assert report["scope"]["externalDownloadRows"] == 0
    assert report["attachmentRowsDetail"][0]["blockerReason"] == "source_pdf_missing"


def test_expansion_attachment_pack_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    papers_root = tmp_path / "papers"
    papers_root.mkdir()
    pdf_path = papers_root / "sample.pdf"
    _make_pdf(pdf_path)
    report = build_visual_annotation_expansion_attachment_pack(
        _expansion_pack(sha256_file(pdf_path)),
        papers_root=papers_root,
        output_dir=tmp_path / "out",
        output_dir_ref="eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002",
        generated_at="2026-05-26T00:00:00Z",
    )
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    write_visual_annotation_expansion_attachment_pack(report, report_json=report_json, report_md=report_md)

    combined = report_json.read_text(encoding="utf-8") + report_md.read_text(encoding="utf-8")
    parsed = json.loads(report_json.read_text(encoding="utf-8"))
    assert parsed["attachmentRowsDetail"][0]["paperRef"] == "papers_dir/sample.pdf"
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
