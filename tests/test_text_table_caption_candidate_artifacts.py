from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_layout_candidate_list_report import PaperSpec
from knowledge_hub.papers.text_table_caption_candidate_artifacts import (
    TABLE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID,
    TABLE_TEXT_ARTIFACT_CANDIDATE_SCHEMA_ID,
    build_text_table_caption_candidate_report,
    extract_table_text_candidates_from_blocks,
    write_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _blocks() -> list[tuple[int, list[tuple[float, float, float, float, str, int, int]]]]:
    return [
        (
            1,
            [
                (10.0, 10.0, 200.0, 30.0, "Introduction text", 0, 0),
                (10.0, 40.0, 200.0, 60.0, "Table 1: Accuracy on ImageNet validation.", 1, 0),
                (
                    10.0,
                    70.0,
                    220.0,
                    130.0,
                    "Model    Top-1    Top-5\nA        75.1     92.3\nB        76.5     93.1",
                    2,
                    0,
                ),
            ],
        )
    ]


def test_extracts_table_caption_and_table_like_text_candidate() -> None:
    candidates, diagnostics = extract_table_text_candidates_from_blocks(
        paper_id="sample-paper",
        paper_ref="papers_dir/sample.pdf",
        source_content_hash="sha256:" + "1" * 64,
        blocks_by_page=_blocks(),
    )

    assert diagnostics["captionRows"] == 1
    assert diagnostics["tableLikeTextRows"] == 1
    assert len(candidates) == 1
    row = candidates[0]
    assert row["tableLabel"] == "Table 1"
    assert row["captionText"] == "Accuracy on ImageNet validation."
    assert row["structureGrade"] == "table_like_text_candidate"
    assert row["numericCandidate"] is True
    assert row["sourceContentHash"] == "sha256:" + "1" * 64
    assert row["page"] == 1
    assert row["captionBbox"] == [10.0, 40.0, 200.0, 60.0]
    assert row["tableTextBbox"] == [10.0, 70.0, 220.0, 130.0]
    assert validate_payload(row, TABLE_TEXT_ARTIFACT_CANDIDATE_SCHEMA_ID, strict=True).ok


def test_caption_only_candidate_is_allowed_but_not_numeric() -> None:
    candidates, diagnostics = extract_table_text_candidates_from_blocks(
        paper_id="sample-paper",
        paper_ref="papers_dir/sample.pdf",
        source_content_hash="sha256:" + "1" * 64,
        blocks_by_page=[(1, [(10.0, 40.0, 200.0, 60.0, "Table 2: Ablation summary.", 0, 0)])],
    )

    assert diagnostics["captionRows"] == 1
    row = candidates[0]
    assert row["structureGrade"] == "caption_only"
    assert row["numericCandidate"] is False
    assert row["tableText"] == ""
    assert row["tableTextBbox"] == []
    assert validate_payload(row, TABLE_TEXT_ARTIFACT_CANDIDATE_SCHEMA_ID, strict=True).ok


def test_build_report_records_blocker_for_missing_source(tmp_path: Path) -> None:
    report = build_text_table_caption_candidate_report(
        papers_root=tmp_path,
        paper_specs=[PaperSpec(paper_id="missing-paper", filename="missing.pdf")],
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["candidateRows"] == 0
    assert report["blockerRows"] == 1
    assert report["blockers"][0]["blockerReason"] == "source_pdf_missing"
    assert validate_payload(report, TABLE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok


def test_report_schema_accepts_fake_table_candidates() -> None:
    candidates, diagnostics = extract_table_text_candidates_from_blocks(
        paper_id="sample-paper",
        paper_ref="papers_dir/sample.pdf",
        source_content_hash="sha256:" + "1" * 64,
        blocks_by_page=_blocks(),
    )
    diagnostics["pageCount"] = 1
    report = {
        "schema": TABLE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID,
        "status": "ready",
        "generatedAt": "2026-05-26T00:00:00Z",
        "scope": {
            "paperRows": 1,
            "paperRefs": ["papers_dir/sample.pdf"],
            "writes": "report_only",
            "canonicalParsedArtifactWriteRows": 0,
            "charBasis": "pymupdf_block_reading_order_normalized_text_v1",
            "tableGridGuarantee": "none_v0_1_candidate_only",
        },
        "paperDiagnostics": [diagnostics],
        "candidateRows": len(candidates),
        "captionOnlyRows": 0,
        "tableLikeTextRows": 1,
        "numericCandidateRows": 1,
        "candidates": candidates,
        "blockerRows": 0,
        "blockers": [],
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "strictEvidencePromotionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "warnings": [],
        "schemaErrors": [],
    }

    assert validate_payload(report, TABLE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok


def test_report_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_text_table_caption_candidate_report(
        papers_root=tmp_path,
        paper_specs=[PaperSpec(paper_id="missing-paper", filename="missing.pdf")],
        generated_at="2026-05-26T00:00:00Z",
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_report(report, json_path=json_path, markdown_path=md_path)

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "papers_dir/missing.pdf" in combined
    assert json.loads(json_path.read_text(encoding="utf-8"))["privatePathLeakRows"] == 0


def test_build_script_runs_from_repo_root_without_installed_package(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "eval/knowledgeos/scripts/build_text_table_caption_candidate_artifacts.py",
            "--papers-root",
            str(tmp_path),
            "--no-write",
            "--json",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "ModuleNotFoundError" not in result.stderr
    assert json.loads(result.stdout)["status"] == "blocked"
