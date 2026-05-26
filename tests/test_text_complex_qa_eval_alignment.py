from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.text_complex_qa_eval_alignment import (
    TEXT_COMPLEX_QA_ALIGNMENT_CASE_SCHEMA_ID,
    TEXT_COMPLEX_QA_ALIGNMENT_REPORT_SCHEMA_ID,
    align_complex_qa_cases,
    default_complex_qa_cases,
    write_report,
)


def _reports() -> dict[str, dict]:
    source_hash = "sha256:" + "1" * 64
    return {
        "figure_caption_text_qa": {
            "status": "ready",
            "rows": [
                {
                    "paperId": "alexnet-2012",
                    "requestedFigureLabel": "Figure 1",
                    "answerabilityStatus": "answerable",
                    "visualReasoningRequested": False,
                    "candidateAnswerPacket": {
                        "evidence": {
                            "artifactId": "figure-caption:alexnet-2012:figure-1:test",
                            "paperId": "alexnet-2012",
                            "sourceContentHash": source_hash,
                            "page": 3,
                            "bbox": [1.0, 2.0, 3.0, 4.0],
                            "figureLabel": "Figure 1",
                            "captionTextHash": "sha256:" + "2" * 64,
                        }
                    },
                }
            ],
        },
        "section_paragraph_spans": {
            "status": "ready",
            "candidates": [
                {
                    "paperId": "resnet-2015",
                    "artifactId": "paragraph-span:resnet:test",
                    "spanType": "paragraph",
                    "sourceContentHash": source_hash,
                    "page": 1,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "textHash": "sha256:" + "3" * 64,
                    "text": "The paper introduces residual learning for very deep neural networks.",
                },
                {
                    "paperId": "mae-2021",
                    "artifactId": "paragraph-span:mae:test",
                    "spanType": "paragraph",
                    "sourceContentHash": source_hash,
                    "page": 1,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "textHash": "sha256:" + "4" * 64,
                    "text": "This paper shows that masked autoencoders are scalable self-supervised learners.",
                },
            ],
        },
        "table_caption_candidates": {
            "status": "ready",
            "candidates": [
                {
                    "paperId": "alexnet-2012",
                    "artifactId": "table-text:alexnet:table-1:test",
                    "sourceContentHash": source_hash,
                    "page": 5,
                    "captionBbox": [1.0, 2.0, 3.0, 4.0],
                    "captionTextHash": "sha256:" + "5" * 64,
                    "tableLabel": "Table 1",
                    "structureGrade": "table_like_text_candidate",
                    "numericCandidate": True,
                }
            ],
        },
        "equation_locator_context": {
            "status": "ready",
            "candidates": [
                {
                    "paperId": "resnet-2015",
                    "artifactId": "equation-context:resnet:equation-1:test",
                    "sourceContentHash": source_hash,
                    "page": 3,
                    "equationBbox": [1.0, 2.0, 3.0, 4.0],
                    "equationTextHash": "sha256:" + "6" * 64,
                    "equationLabel": "Equation 1",
                    "locatorGrade": "labeled_equation_context",
                }
            ],
        },
    }


def test_aligns_default_cases_into_four_dispositions() -> None:
    report = align_complex_qa_cases(
        reports=_reports(),
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["caseRows"] == 10
    assert report["textAnswerableRows"] == 3
    assert report["candidateOnlyRows"] == 2
    assert report["visualUnsupportedRows"] == 2
    assert report["noAnswerRows"] == 3
    assert report["categoryCounts"]["table_numeric_qa"] == 2
    assert report["categoryCounts"]["equation_citation_qa"] == 2
    assert validate_payload(report, TEXT_COMPLEX_QA_ALIGNMENT_REPORT_SCHEMA_ID, strict=True).ok
    for row in report["rows"]:
        assert validate_payload(row, TEXT_COMPLEX_QA_ALIGNMENT_CASE_SCHEMA_ID, strict=True).ok


def test_table_numeric_cases_remain_candidate_only_without_cell_identity() -> None:
    report = align_complex_qa_cases(reports=_reports(), generated_at="2026-05-26T00:00:00Z")
    row = next(item for item in report["rows"] if item["caseId"] == "table-numeric-alexnet-table-1-result")

    assert row["disposition"] == "candidate_only"
    assert row["blockerReason"] == "table_cell_identity_not_available"
    assert "table_cell_identity" in row["missingContracts"]
    assert row["matchedEvidenceRows"] == 1
    assert row["matchedEvidence"][0]["strictEvidence"] is False


def test_report_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = align_complex_qa_cases(reports=_reports(), generated_at="2026-05-26T00:00:00Z")
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_report(report, json_path=json_path, markdown_path=md_path)

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert json.loads(json_path.read_text(encoding="utf-8"))["privatePathLeakRows"] == 0
