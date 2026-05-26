from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_plan import (
    build_parsed_artifact_manual_lookup_source_recovery_plan,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_review_pack import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID,
    build_parsed_artifact_manual_lookup_source_recovery_review_pack,
    write_parsed_artifact_manual_lookup_source_recovery_review_pack,
)
from knowledge_hub.papers.parsed_artifact_source_recovery_feasibility import (
    build_parsed_artifact_source_recovery_feasibility,
)


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


def _plan_report(tmp_path: Path) -> dict:
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
    return build_parsed_artifact_manual_lookup_source_recovery_plan(
        feasibility_report=feasibility,
        feasibility_report_path=tmp_path / "feasibility.json",
        target_missing_parsed_artifacts=1,
        generated_at="2026-05-21T00:00:00+00:00",
    )


def test_review_pack_builds_inert_review_cards_and_allowlist_templates(tmp_path: Path) -> None:
    report = build_parsed_artifact_manual_lookup_source_recovery_review_pack(
        manual_lookup_plan_report=_plan_report(tmp_path),
        manual_lookup_plan_report_path=tmp_path / "plan.json",
        generated_at="2026-05-21T00:00:00+00:00",
    )

    assert report["schema"] == PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID
    assert validate_payload(report, PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert report["status"] == "manual_lookup_source_recovery_review_pack_candidate_only"
    assert report["counts"]["inputRows"] == 3
    assert report["counts"]["manualLookupReviewCardRows"] == 2
    assert report["counts"]["textSourceHoldoutRows"] == 1
    assert report["counts"]["approvalTemplateRows"] == 2
    assert report["counts"]["approvedSourceRows"] == 0
    assert report["counts"]["applyReadyRows"] == 0
    assert report["gate"]["reviewPackReady"] is True
    assert report["gate"]["applyReady"] is False
    assert report["gate"]["sourceRegistrationMutationReady"] is False
    assert report["gate"]["parsedArtifactMaterializationReady"] is False
    assert report["mutationCounters"]["sourceDownloadRows"] == 0
    assert report["mutationCounters"]["parsedArtifactWriteRows"] == 0
    assert all(card["operatorDecisionDefault"] == "needs_review" for card in report["reviewCards"])
    assert all(card["applyReady"] is False for card in report["reviewCards"])
    assert all(template["decision"] == "needs_review" for template in report["allowlistTemplateRows"])
    assert all(template["acceptedByThisTranche"] is False for template in report["allowlistTemplateRows"])


def test_review_pack_orders_high_priority_registered_missing_first(tmp_path: Path) -> None:
    report = build_parsed_artifact_manual_lookup_source_recovery_review_pack(
        manual_lookup_plan_report=_plan_report(tmp_path),
        generated_at="2026-05-21T00:00:00+00:00",
    )

    assert report["manualLookupReviewCardPaperIds"] == ["registered-missing", "custom-missing"]
    assert report["reviewCards"][0]["lookupPriority"] == "high"
    assert report["reviewCards"][0]["manualLookupMode"] == "registered_pdf_path_missing_manual_lookup"


def test_review_pack_blocks_invalid_input_schema() -> None:
    report = build_parsed_artifact_manual_lookup_source_recovery_review_pack(
        manual_lookup_plan_report={"schema": "wrong"},
        generated_at="2026-05-21T00:00:00+00:00",
    )

    assert report["status"] == "blocked_input_schema_violation"
    assert report["inputPlan"]["schemaValidation"]["ok"] is False
    assert report["counts"]["manualLookupReviewCardRows"] == 0
    assert report["gate"]["reviewPackReady"] is False
    assert validate_payload(report, PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID, strict=True).ok


def test_review_pack_writer_outputs_report_cards_allowlist_and_holdouts(tmp_path: Path) -> None:
    report = build_parsed_artifact_manual_lookup_source_recovery_review_pack(
        manual_lookup_plan_report=_plan_report(tmp_path),
        generated_at="2026-05-21T00:00:00+00:00",
    )

    paths = write_parsed_artifact_manual_lookup_source_recovery_review_pack(report, tmp_path / "reports")
    written = json.loads(Path(paths["reportJsonPath"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summaryJsonPath"]).read_text(encoding="utf-8"))
    cards = json.loads(Path(paths["reviewCardsPath"]).read_text(encoding="utf-8"))
    allowlist = json.loads(Path(paths["allowlistTemplatePath"]).read_text(encoding="utf-8"))

    assert validate_payload(written, PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert summary["reportSchema"] == PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID
    assert len(cards) == 2
    assert len(allowlist) == 2
    assert Path(paths["reportMarkdownPath"]).exists()
    assert Path(paths["textSourceHoldoutIdsPath"]).read_text(encoding="utf-8").strip() == "text-only"
