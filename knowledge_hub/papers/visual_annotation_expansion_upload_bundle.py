"""Create Finder-friendly upload folders for visual annotation batches.

This helper copies already-rendered context crops plus per-batch prompt/template
files into one folder per manual web GPT/Pro chat. It is an operator packaging
step only: no model calls, no vector indexing, no store mutation, and no
evidence promotion.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import json
import re
import shutil
from typing import Any

from knowledge_hub.papers.visual_annotation_expansion_manual_output_capture import (
    VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID,
    normalize_text,
    utc_now_iso,
)
from knowledge_hub.papers.visual_annotation_expansion_attachment_pack import (
    VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID,
    PRIVATE_PATH_RE,
)


VISUAL_ANNOTATION_EXPANSION_WEB_UPLOAD_BUNDLE_SCHEMA_ID = (
    "knowledge-hub.paper.visual-annotation-expansion-web-upload-bundle.v1"
)


def _slug(value: str) -> str:
    token = re.sub(r"[^a-z0-9_.-]+", "-", str(value or "").lower()).strip("-")
    return token or "unknown"


def _short_candidate_digest(candidate_id: str) -> str:
    return _slug(str(candidate_id).split(":")[-1])[:10] or "candidate"


def _project_path(project_root: Path, ref: str) -> Path:
    token = normalize_text(ref)
    if not token or token.startswith("/"):
        return Path("__invalid_absolute_or_empty_ref__")
    return project_root / token


def _attachment_rows_by_id(attachment_pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [
        row
        for row in list(attachment_pack.get("attachmentRowsDetail") or [])
        if isinstance(row, dict)
    ]
    return {normalize_text(row.get("sourceCandidateId")): row for row in rows}


def _scope(*, prompt_rows: int, template_rows: int, copied_attachment_rows: int) -> dict[str, Any]:
    return {
        "writes": "upload_bundle_files_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "manualOperatorWebModelRunRequired": True,
        "operatorPromptRows": int(prompt_rows),
        "fillTemplateRows": int(template_rows),
        "copiedAttachmentRows": int(copied_attachment_rows),
        "vectorIndexing": False,
        "strictEvidencePromotionRows": 0,
        "runtimeAnswerVisibleExposureRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "answerabilityGateBypassRows": 0,
        "pageImageWriteRows": 0,
        "wholeImageWriteRows": 0,
        "wholeImageGptRows": 0,
        "candidateStoreMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_visual_annotation_expansion_web_upload_bundle(
    web_run_bundle: dict[str, Any],
    attachment_pack: dict[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    output_dir_ref: str,
    upload_bundle_id: str,
    source_web_run_bundle_ref: str,
    source_attachment_pack_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    attachment_rows = _attachment_rows_by_id(attachment_pack)
    batch_rows: list[dict[str, Any]] = []
    missing_artifacts: list[str] = []
    copied_attachment_rows = 0
    prompt_rows = 0
    template_rows = 0

    for batch in list(web_run_bundle.get("batchBundles") or []):
        if not isinstance(batch, dict):
            continue
        missing_before = len([item for item in missing_artifacts if item])
        batch_number = int(batch.get("batchNumber") or 0)
        batch_dir_name = f"batch_{batch_number:02d}"
        batch_dir = output_dir / batch_dir_name
        batch_ref = f"{output_dir_ref.rstrip('/')}/{batch_dir_name}"
        prompt_ref = normalize_text(batch.get("promptRef"))
        template_ref = normalize_text(batch.get("fillTemplateRef"))
        copied_prompt_ref = f"{batch_ref}/batch_{batch_number:02d}_prompt.md"
        copied_template_ref = f"{batch_ref}/batch_{batch_number:02d}_fill_template.v1.json"

        if _copy_file(_project_path(project_root, prompt_ref), batch_dir / f"batch_{batch_number:02d}_prompt.md"):
            prompt_rows += 1
        else:
            missing_artifacts.append(prompt_ref)
        if _copy_file(
            _project_path(project_root, template_ref),
            batch_dir / f"batch_{batch_number:02d}_fill_template.v1.json",
        ):
            template_rows += 1
        else:
            missing_artifacts.append(template_ref)

        copied_refs: list[str] = []
        for row in [row for row in list(batch.get("rows") or []) if isinstance(row, dict)]:
            source_candidate_id = normalize_text(row.get("sourceCandidateId"))
            attachment = attachment_rows.get(source_candidate_id) or {}
            attachment_ref = normalize_text(attachment.get("attachmentRef"))
            source_path = _project_path(project_root, attachment_ref)
            filename = "{priority:02d}-{paper}-p{page}-{kind}-{digest}.png".format(
                priority=int(row.get("priority") or len(copied_refs) + 1),
                paper=_slug(normalize_text(row.get("paperId")))[:42],
                page=int(row.get("page") or 0),
                kind=_slug(normalize_text(row.get("candidateType")))[:28],
                digest=_short_candidate_digest(source_candidate_id),
            )
            target_path = batch_dir / filename
            target_ref = f"{batch_ref}/{filename}"
            if _copy_file(source_path, target_path):
                copied_attachment_rows += 1
                copied_refs.append(target_ref)
            else:
                missing_artifacts.append(attachment_ref)

        batch_rows.append(
            {
                "batchId": normalize_text(batch.get("batchId")),
                "batchNumber": batch_number,
                "rowCount": int(batch.get("rowCount") or 0),
                "folderRef": batch_ref,
                "promptRef": copied_prompt_ref,
                "fillTemplateRef": copied_template_ref,
                "attachmentRefs": copied_refs,
                "missingArtifactRows": len([item for item in missing_artifacts if item]) - missing_before,
            }
        )

    report: dict[str, Any] = {
        "schema": VISUAL_ANNOTATION_EXPANSION_WEB_UPLOAD_BUNDLE_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": "ready_for_operator_web_gpt_pro_upload",
        "nextRecommendedTranche": "visual_annotation_expansion_manual_output_capture",
        "uploadBundleId": upload_bundle_id,
        "sourceWebRunBundle": {
            "schema": normalize_text(web_run_bundle.get("schema")),
            "status": normalize_text(web_run_bundle.get("status")),
            "reportRef": normalize_text(source_web_run_bundle_ref),
            "batchRows": len(list(web_run_bundle.get("batchBundles") or [])),
        },
        "sourceAttachmentPack": {
            "schema": normalize_text(attachment_pack.get("schema")),
            "status": normalize_text(attachment_pack.get("status")),
            "reportRef": normalize_text(source_attachment_pack_ref),
            "attachmentRows": len(attachment_rows),
        },
        "uploadRootRef": normalize_text(output_dir_ref),
        "scope": _scope(
            prompt_rows=prompt_rows,
            template_rows=template_rows,
            copied_attachment_rows=copied_attachment_rows,
        ),
        "counts": {
            "batchRows": len(batch_rows),
            "operatorPromptRows": prompt_rows,
            "fillTemplateRows": template_rows,
            "copiedAttachmentRows": copied_attachment_rows,
            "missingArtifactRows": len([item for item in missing_artifacts if item]),
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "batchRowsDetail": batch_rows,
        "warnings": [
            "Use one batch folder per new web GPT/Pro chat.",
            "Upload only the PNG files inside that batch folder with the matching prompt.",
            "Generated visual text remains retrieval-hint-only and non-evidence.",
        ],
    }
    report["counts"]["privatePathLeakRows"] = 1 if _contains_private_path(report) else 0
    if (
        web_run_bundle.get("schema") != VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID
        or web_run_bundle.get("status") != "ready"
        or attachment_pack.get("schema") != VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID
        or attachment_pack.get("status") != "ready"
        or report["counts"]["missingArtifactRows"]
        or report["counts"]["privatePathLeakRows"]
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def render_markdown_upload_bundle(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    lines = [
        f"# {report.get('uploadBundleId')}",
        "",
        "Use one batch folder per new GPT/Pro chat. Each folder contains one prompt, one fill template, and PNG context crops.",
        "",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- uploadRootRef: `{report.get('uploadRootRef')}`",
        f"- batchRows: `{counts.get('batchRows')}`",
        f"- copiedAttachmentRows: `{counts.get('copiedAttachmentRows')}`",
        f"- missingArtifactRows: `{counts.get('missingArtifactRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        "",
        "## Mutation Guarantees",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- modelCalls: `{scope.get('modelCalls')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        f"- databaseMutationRows: `{scope.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{scope.get('indexMutationRows')}`",
        f"- vaultScanRows: `{scope.get('vaultScanRows')}`",
        f"- externalDownloadRows: `{scope.get('externalDownloadRows')}`",
        "",
        "## Batches",
        "",
    ]
    for batch in list(report.get("batchRowsDetail") or []):
        lines.append(
            "- `{folder}`: prompt + fill template + `{rows}` PNG files".format(
                folder=batch.get("folderRef"),
                rows=len(list(batch.get("attachmentRefs") or [])),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def write_visual_annotation_expansion_web_upload_bundle(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
    output_dir: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown_upload_bundle(report)
    report_md.write_text(markdown, encoding="utf-8")
    (output_dir / "README.md").write_text(markdown, encoding="utf-8")
    return {
        "json": str(report_json),
        "markdown": str(report_md),
        "uploadDir": str(output_dir),
    }


__all__ = [
    "VISUAL_ANNOTATION_EXPANSION_WEB_UPLOAD_BUNDLE_SCHEMA_ID",
    "build_visual_annotation_expansion_web_upload_bundle",
    "render_markdown_upload_bundle",
    "write_visual_annotation_expansion_web_upload_bundle",
]
