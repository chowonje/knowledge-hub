from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_draft import (
    build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_human_review import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID,
    build_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_validation import (
    DECISION_APPROVE_SOURCE_URL,
    build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_operator_decision_packet import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_OPERATOR_DECISION_PACKET_SCHEMA_ID,
    build_parsed_artifact_manual_lookup_source_recovery_operator_decision_packet,
    write_parsed_artifact_manual_lookup_source_recovery_operator_decision_packet,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_plan import (
    build_parsed_artifact_manual_lookup_source_recovery_plan,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_review_pack import (
    build_parsed_artifact_manual_lookup_source_recovery_review_pack,
)
from knowledge_hub.papers.parsed_artifact_source_recovery_feasibility import (
    build_parsed_artifact_source_recovery_feasibility,
)


def _operator_report_root() -> Path:
    return Path.home() / ("." + "khub") / "reports" / "parsed-artifact-coverage" / "2026-05-21"


def _seed_paper(
    db: SQLiteDatabase,
    *,
    paper_id: str,
    title: str,
    pdf_path: str = "",
    text_path: str = "",
) -> None:
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": title,
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "",
            "pdf_path": pdf_path,
            "text_path": text_path,
            "translated_path": "",
        }
    )


def _packet_bundle(tmp_path: Path) -> tuple[dict, dict]:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    text_source = papers_dir / "source.txt"
    text_source.write_text("text source", encoding="utf-8")
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="custom-missing", title="Custom Missing")
    _seed_paper(
        db,
        paper_id="registered-missing",
        title="Registered Missing",
        pdf_path=str(papers_dir / "registered.pdf"),
    )
    _seed_paper(db, paper_id="text-only", title="Text Only", text_path=str(text_source))
    feasibility = build_parsed_artifact_source_recovery_feasibility(
        sqlite_db=db,
        papers_dir=papers_dir,
        generated_at="2026-05-21T00:00:00+00:00",
    )
    plan = build_parsed_artifact_manual_lookup_source_recovery_plan(
        feasibility_report=feasibility,
        feasibility_report_path=tmp_path / "feasibility.json",
        target_missing_parsed_artifacts=1,
        generated_at="2026-05-21T00:00:00+00:00",
    )
    review_pack = build_parsed_artifact_manual_lookup_source_recovery_review_pack(
        manual_lookup_plan_report=plan,
        manual_lookup_plan_report_path=tmp_path / "plan.json",
        generated_at="2026-05-21T00:00:00+00:00",
    )
    draft_report = build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft(
        review_pack_report=review_pack,
        review_pack_report_path=tmp_path / "review-pack.json",
        generated_at="2026-05-21T00:00:00+00:00",
    )
    decision_file = deepcopy(dict(draft_report.get("decisionFileDraft") or {}))
    validation_report = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        generated_at="2026-05-21T00:00:00+00:00",
        expected_input_rows=2,
    )
    human_review = build_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review(
        decision_file_validation_report=validation_report,
        decision_file=decision_file,
        review_pack_report=review_pack,
        source_recovery_feasibility_report=feasibility,
        expected_manual_lookup_rows=2,
        expected_text_source_holdout_rows=1,
    )
    return human_review, decision_file


def test_operator_packet_defaults_rows_to_needs_review(tmp_path: Path) -> None:
    human_review, decision_file = _packet_bundle(tmp_path)
    payload = build_parsed_artifact_manual_lookup_source_recovery_operator_decision_packet(
        human_review_report=human_review,
        decision_file=decision_file,
        expected_manual_lookup_rows=2,
    )

    assert payload["schema"] == PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_OPERATOR_DECISION_PACKET_SCHEMA_ID
    assert validate_payload(
        payload,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_OPERATOR_DECISION_PACKET_SCHEMA_ID,
        strict=True,
    ).ok
    assert payload["status"] == "operator_decision_packet_ready"
    assert payload["counts"]["operatorReviewRows"] == 2
    assert payload["counts"]["needsReviewRows"] == 2
    row = payload["operatorReviewRows"][0]
    assert row["currentDecision"] == "needs_review"
    assert row["currentMissingReason"] in {
        "registered_pdf_path_missing",
        "no_registered_source_artifact",
    }
    assert row["lookupQuery"]
    assert row["allowedDecisions"]
    assert row["riskNote"]
    assert "invent" in row["riskNote"].lower() or "explicit" in row["riskNote"].lower()


def test_operator_packet_preserves_explicit_approval(tmp_path: Path) -> None:
    human_review, decision_file = _packet_bundle(tmp_path)
    decision_file["decisions"][0]["decision"] = DECISION_APPROVE_SOURCE_URL
    decision_file["decisions"][0]["approvedSourceType"] = "url"
    decision_file["decisions"][0]["approvedSourceUrl"] = "https://example.com/paper.pdf"
    decision_file["decisions"][0]["approvedSourceContentHash"] = "a" * 64
    decision_file["decisions"][0]["approvedBy"] = "reviewer"
    decision_file["decisions"][0]["approvedAt"] = "2026-05-21T00:00:00+00:00"
    decision_file["decisions"][0]["notes"] = "verified publisher pdf"

    payload = build_parsed_artifact_manual_lookup_source_recovery_operator_decision_packet(
        human_review_report=human_review,
        decision_file=decision_file,
        expected_manual_lookup_rows=2,
    )

    assert payload["counts"]["approvedDecisionRows"] == 1
    assert payload["counts"]["needsReviewRows"] == 1
    assert payload["operatorReviewRows"][0]["currentDecision"] == DECISION_APPROVE_SOURCE_URL
    assert "approvedSourceUrl" in payload["operatorReviewRows"][0]["requiredApprovalFields"]
    assert "approvedSourceContentHash" in payload["operatorReviewRows"][0]["requiredApprovalFields"]


def test_operator_packet_writer_outputs_review_sheet(tmp_path: Path) -> None:
    human_review, decision_file = _packet_bundle(tmp_path)
    payload = build_parsed_artifact_manual_lookup_source_recovery_operator_decision_packet(
        human_review_report=human_review,
        decision_file=decision_file,
        expected_manual_lookup_rows=2,
    )
    paths = write_parsed_artifact_manual_lookup_source_recovery_operator_decision_packet(
        payload,
        tmp_path / "out",
    )

    sheet = json.loads(Path(paths["reviewSheetJsonPath"]).read_text(encoding="utf-8"))
    markdown = Path(paths["reviewSheetMarkdownPath"]).read_text(encoding="utf-8")
    assert len(sheet["operatorReviewRows"]) == 2
    assert "Fields The Human Must Fill" in markdown
    assert "risk note" in markdown.lower()


@pytest.mark.skip(reason="operator-local integration report is excluded from PR gate")
def test_operator_packet_integrated_measured_local_report() -> None:
    report_root = _operator_report_root()
    human_review = json.loads(
        (
            report_root
            / "parsed-artifact-manual-lookup-source-recovery-decision-file-human-review/"
            "01-parsed-artifact-manual-lookup-source-recovery-decision-file-human-review/"
            "parsed-artifact-manual-lookup-source-recovery-decision-file-human-review.json"
        ).read_text(encoding="utf-8")
    )
    decision_file = json.loads(
        (
            report_root
            / "parsed-artifact-manual-lookup-source-recovery-decision-file-draft/"
            "01-parsed-artifact-manual-lookup-source-recovery-decision-file-draft/"
            "manual-lookup-source-recovery-decisions.draft.json"
        ).read_text(encoding="utf-8")
    )

    payload = build_parsed_artifact_manual_lookup_source_recovery_operator_decision_packet(
        human_review_report=human_review,
        decision_file=decision_file,
    )

    assert payload["status"] in {"operator_decision_packet_ready", "blocked"}
    if payload["status"] == "blocked":
        assert payload["gate"]["operatorDecisionPacketReady"] is False
        assert payload["gate"]["schemaViolations"] or payload["counts"]["approvedDecisionRows"] > 0
        return
    assert payload["counts"]["operatorReviewRows"] == 15
    assert payload["counts"]["needsReviewRows"] + payload["counts"]["approvedDecisionRows"] == 15
    assert all(row["currentDecision"] in row["allowedDecisions"] for row in payload["operatorReviewRows"])
    assert validate_payload(
        payload,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_OPERATOR_DECISION_PACKET_SCHEMA_ID,
        strict=True,
    ).ok
    assert validate_payload(
        human_review,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
