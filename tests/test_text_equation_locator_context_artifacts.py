from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.figure_caption_artifact_vertical_slice import PaperSpec
from knowledge_hub.papers.text_equation_locator_context_artifacts import (
    EQUATION_CONTEXT_ARTIFACT_CANDIDATE_SCHEMA_ID,
    EQUATION_LOCATOR_CONTEXT_REPORT_SCHEMA_ID,
    build_text_equation_locator_context_report,
    extract_equation_context_candidates_from_blocks,
    write_report,
)


def _blocks() -> list[tuple[int, list[tuple[float, float, float, float, str, int, int]]]]:
    return [
        (
            1,
            [
                (
                    10.0,
                    10.0,
                    300.0,
                    38.0,
                    "The model computes a residual mapping from the input feature tensor and keeps the optimization target stable.",
                    0,
                    0,
                ),
                (20.0, 50.0, 260.0, 70.0, "y = F(x) + x (1)", 1, 0),
                (
                    10.0,
                    86.0,
                    310.0,
                    122.0,
                    "This residual form lets the network learn an additive correction while the skip path preserves the original signal.",
                    2,
                    0,
                ),
            ],
        )
    ]


def test_extracts_labeled_equation_context_candidate() -> None:
    candidates, diagnostics = extract_equation_context_candidates_from_blocks(
        paper_id="sample-paper",
        paper_ref="papers_dir/sample.pdf",
        source_content_hash="sha256:" + "1" * 64,
        blocks_by_page=_blocks(),
    )

    assert diagnostics["candidateRows"] == 1
    assert diagnostics["labeledEquationRows"] == 1
    assert diagnostics["equationLikeRows"] == 0
    assert diagnostics["contextRows"] == 1
    row = candidates[0]
    assert row["equationLabel"] == "Equation 1"
    assert row["equationText"] == "y = F(x) + x (1)"
    assert "residual mapping" in row["contextText"]
    assert row["locatorGrade"] == "labeled_equation_context"
    assert row["latexGuarantee"] == "none_v0_1_locator_only"
    assert row["sourceContentHash"] == "sha256:" + "1" * 64
    assert row["page"] == 1
    assert row["equationBbox"] == [20.0, 50.0, 260.0, 70.0]
    assert row["contextBbox"] == [10.0, 10.0, 310.0, 122.0]
    assert row["equationCharStart"] > 0
    assert row["equationCharEnd"] > row["equationCharStart"]
    assert validate_payload(row, EQUATION_CONTEXT_ARTIFACT_CANDIDATE_SCHEMA_ID, strict=True).ok


def test_unlabeled_equation_like_block_remains_candidate_grade() -> None:
    candidates, diagnostics = extract_equation_context_candidates_from_blocks(
        paper_id="sample-paper",
        paper_ref="papers_dir/sample.pdf",
        source_content_hash="sha256:" + "2" * 64,
        blocks_by_page=[
            (
                1,
                [
                    (
                        10.0,
                        10.0,
                        300.0,
                        38.0,
                        "The scoring rule combines normalized logits before the final classifier is applied.",
                        0,
                        0,
                    ),
                    (20.0, 50.0, 260.0, 70.0, "score = softmax(Wx + b)", 1, 0),
                ],
            )
        ],
    )

    assert diagnostics["candidateRows"] == 1
    assert diagnostics["labeledEquationRows"] == 0
    assert diagnostics["equationLikeRows"] == 1
    row = candidates[0]
    assert row["equationLabel"] == ""
    assert row["locatorGrade"] == "equation_like_context"
    assert row["latexGuarantee"] == "none_v0_1_locator_only"
    assert validate_payload(row, EQUATION_CONTEXT_ARTIFACT_CANDIDATE_SCHEMA_ID, strict=True).ok


def test_build_report_records_blocker_for_missing_source(tmp_path: Path) -> None:
    report = build_text_equation_locator_context_report(
        papers_root=tmp_path,
        paper_specs=[PaperSpec(paper_id="missing-paper", filename="missing.pdf")],
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["candidateRows"] == 0
    assert report["blockerRows"] == 1
    assert report["blockers"][0]["blockerReason"] == "source_pdf_missing"
    assert validate_payload(report, EQUATION_LOCATOR_CONTEXT_REPORT_SCHEMA_ID, strict=True).ok


def test_report_schema_accepts_fake_equation_candidates() -> None:
    candidates, diagnostics = extract_equation_context_candidates_from_blocks(
        paper_id="sample-paper",
        paper_ref="papers_dir/sample.pdf",
        source_content_hash="sha256:" + "1" * 64,
        blocks_by_page=_blocks(),
    )
    diagnostics["pageCount"] = 1
    report = {
        "schema": EQUATION_LOCATOR_CONTEXT_REPORT_SCHEMA_ID,
        "status": "ready",
        "generatedAt": "2026-05-26T00:00:00Z",
        "scope": {
            "paperRows": 1,
            "paperRefs": ["papers_dir/sample.pdf"],
            "writes": "report_only",
            "canonicalParsedArtifactWriteRows": 0,
            "charBasis": "pymupdf_block_reading_order_normalized_text_v1",
            "latexGuarantee": "none_v0_1_locator_only",
        },
        "paperDiagnostics": [diagnostics],
        "candidateRows": 1,
        "labeledEquationRows": 1,
        "equationLikeRows": 0,
        "contextRows": 1,
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

    assert validate_payload(report, EQUATION_LOCATOR_CONTEXT_REPORT_SCHEMA_ID, strict=True).ok


def test_report_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_text_equation_locator_context_report(
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
