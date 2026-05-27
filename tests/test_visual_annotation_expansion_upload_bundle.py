from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.papers.visual_annotation_expansion_attachment_pack import (
    VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID,
)
from knowledge_hub.papers.visual_annotation_expansion_manual_output_capture import (
    VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID,
)
from knowledge_hub.papers.visual_annotation_expansion_upload_bundle import (
    build_visual_annotation_expansion_web_upload_bundle,
    write_visual_annotation_expansion_web_upload_bundle,
)


def test_upload_bundle_copies_batch_prompt_template_and_png_with_sanitized_refs(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    prompt = project_root / "eval/knowledgeos/reports/run/batch_01_prompt.md"
    template = project_root / "eval/knowledgeos/reports/run/batch_01_fill_template.v1.json"
    asset = project_root / "eval/knowledgeos/reports/attachments/assets/crop.png"
    prompt.parent.mkdir(parents=True, exist_ok=True)
    template.parent.mkdir(parents=True, exist_ok=True)
    asset.parent.mkdir(parents=True)
    prompt.write_text("prompt", encoding="utf-8")
    template.write_text("{}", encoding="utf-8")
    asset.write_bytes(b"png")

    source_candidate_id = "visual-layout:sample:figure_caption_region:1:aaaaaaaaaaaaaaaa"
    web_run_bundle = {
        "schema": VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID,
        "status": "ready",
        "batchBundles": [
            {
                "batchId": "batch-1",
                "batchNumber": 1,
                "rowCount": 1,
                "promptRef": "eval/knowledgeos/reports/run/batch_01_prompt.md",
                "fillTemplateRef": "eval/knowledgeos/reports/run/batch_01_fill_template.v1.json",
                "rows": [
                    {
                        "sourceCandidateId": source_candidate_id,
                        "paperId": "sample-paper",
                        "candidateType": "figure_caption_region",
                        "page": 1,
                    }
                ],
            }
        ],
    }
    attachment_pack = {
        "schema": VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID,
        "status": "ready",
        "attachmentRowsDetail": [
            {
                "sourceCandidateId": source_candidate_id,
                "attachmentRef": "eval/knowledgeos/reports/attachments/assets/crop.png",
            }
        ],
    }

    report = build_visual_annotation_expansion_web_upload_bundle(
        web_run_bundle,
        attachment_pack,
        project_root=project_root,
        output_dir=project_root / "eval/knowledgeos/reports/upload",
        output_dir_ref="eval/knowledgeos/reports/upload",
        upload_bundle_id="upload-test",
        source_web_run_bundle_ref="eval/knowledgeos/reports/run.v1.json",
        source_attachment_pack_ref="eval/knowledgeos/reports/attachments.v1.json",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["counts"]["batchRows"] == 1
    assert report["counts"]["operatorPromptRows"] == 1
    assert report["counts"]["fillTemplateRows"] == 1
    assert report["counts"]["copiedAttachmentRows"] == 1
    assert report["counts"]["missingArtifactRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["scope"]["modelCalls"] is False
    assert report["scope"]["vectorIndexing"] is False
    assert report["scope"]["databaseMutationRows"] == 0
    assert report["scope"]["strictEvidencePromotionRows"] == 0

    paths = write_visual_annotation_expansion_web_upload_bundle(
        report,
        report_json=project_root / "eval/knowledgeos/reports/upload.v1.json",
        report_md=project_root / "eval/knowledgeos/reports/upload.v1.md",
        output_dir=project_root / "eval/knowledgeos/reports/upload",
    )

    assert (project_root / "eval/knowledgeos/reports/upload/batch_01/batch_01_prompt.md").is_file()
    assert (project_root / "eval/knowledgeos/reports/upload/batch_01/batch_01_fill_template.v1.json").is_file()
    copied_pngs = list((project_root / "eval/knowledgeos/reports/upload/batch_01").glob("*.png"))
    assert len(copied_pngs) == 1
    combined = (
        Path(paths["json"]).read_text(encoding="utf-8")
        + Path(paths["markdown"]).read_text(encoding="utf-8")
        + (project_root / "eval/knowledgeos/reports/upload/README.md").read_text(encoding="utf-8")
    )
    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["batchRowsDetail"][0]["folderRef"] == "eval/knowledgeos/reports/upload/batch_01"
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
