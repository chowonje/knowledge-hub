from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.text_evidence_pr149_disposition import (
    TEXT_EVIDENCE_PR149_DISPOSITION_SCHEMA_ID,
    build_text_evidence_pr149_disposition_report,
    write_report,
)


def _write_alignment_report(root: Path) -> None:
    report_path = root / "eval" / "knowledgeos" / "reports" / "text_complex_qa_eval_alignment.v1.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema": "knowledge-hub.paper.text-complex-qa-eval-alignment-report.v1",
                "status": "ready",
                "caseRows": 10,
                "textAnswerableRows": 3,
                "visualUnsupportedRows": 2,
            }
        ),
        encoding="utf-8",
    )


def test_pr149_disposition_abandons_current_pr_for_text_rc(tmp_path: Path) -> None:
    _write_alignment_report(tmp_path)

    report = build_text_evidence_pr149_disposition_report(
        project_root=tmp_path,
        include_gh=False,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"]["decision"] == "abandon_current_pr_before_public_rc"
    assert report["decision"]["mergeRecommended"] is False
    assert report["decision"]["recutRecommendedForV01"] is False
    assert report["decision"]["laterSideTrackAllowed"] is True
    assert report["missingDependencyRows"] == 2
    assert report["textAlignmentFinding"]["present"] is True
    assert report["scope"]["pullRequestMutationPerformed"] is False
    assert report["mutationCounters"]["pullRequestMutationRows"] == 0
    assert report["mutationCounters"]["vaultScanRows"] == 0
    assert report["privatePathLeakRows"] == 0
    assert validate_payload(report, TEXT_EVIDENCE_PR149_DISPOSITION_SCHEMA_ID, strict=True).ok


def test_pr149_writer_keeps_refs_sanitized(tmp_path: Path) -> None:
    _write_alignment_report(tmp_path)
    report = build_text_evidence_pr149_disposition_report(
        project_root=tmp_path,
        include_gh=False,
        generated_at="2026-05-26T00:00:00Z",
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_report(report, json_path=json_path, markdown_path=md_path)

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert json.loads(json_path.read_text(encoding="utf-8"))["privatePathLeakRows"] == 0
