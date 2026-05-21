from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_apply import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_SCHEMA_ID,
    build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply,
)
import knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_apply as apply_module
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_DRY_RUN_SCHEMA_ID,
    build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_draft import (
    build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_validation import (
    DECISION_APPROVE_LOCAL_PDF,
    DECISION_APPROVE_SOURCE_URL,
    build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_paper(db: SQLiteDatabase, *, paper_id: str, title: str, pdf_path: str = "") -> None:
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
            "text_path": "",
            "translated_path": "",
        }
    )


def _approved_bundle(tmp_path: Path) -> tuple[dict, dict, dict, SQLiteDatabase, Path]:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="custom-missing", title="Custom Missing")
    _seed_paper(
        db,
        paper_id="registered-missing",
        title="Registered Missing",
        pdf_path=str(papers_dir / "registered.pdf"),
    )
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
    decision_file["decisions"][0]["decision"] = DECISION_APPROVE_SOURCE_URL
    decision_file["decisions"][0]["approvedSourceType"] = "url"
    decision_file["decisions"][0]["approvedSourceUrl"] = "https://example.com/paper.pdf"
    decision_file["decisions"][0]["approvedSourceContentHash"] = "a" * 64
    decision_file["decisions"][0]["approvedBy"] = "reviewer"
    decision_file["decisions"][0]["approvedAt"] = "2026-05-21T00:00:00+00:00"
    decision_file["decisions"][0]["notes"] = "approved for apply dry-run"
    validation = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        expected_input_rows=2,
    )
    return draft_report, validation, decision_file, db, papers_dir


def test_validation_blocks_approved_row_without_source_content_hash(tmp_path: Path) -> None:
    draft_report, _validation, decision_file, _db, _papers_dir = _approved_bundle(tmp_path)
    decision_file["decisions"][0]["approvedSourceContentHash"] = ""
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        expected_input_rows=2,
    )
    row = payload["validationRows"][0]
    assert payload["status"] == "blocked"
    assert payload["counts"]["applyReadyRows"] == 0
    assert row["validationStatus"] == "invalid_approval_fields"
    assert "approvedSourceContentHash_required" in row["validationBlockers"]


def test_apply_dry_run_blocks_without_apply_ready_rows(tmp_path: Path) -> None:
    draft_report, validation, decision_file, db, papers_dir = _approved_bundle(tmp_path)
    decision_file["decisions"][0]["decision"] = "needs_review"
    validation = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        expected_input_rows=2,
    )
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run(
        validation_report=validation,
        decision_file=decision_file,
        sqlite_db=db,
        papers_dir=papers_dir,
    )
    assert payload["status"] == "blocked"
    assert payload["counts"]["selectedRows"] == 0


def test_apply_dry_run_plans_apply_ready_row(tmp_path: Path) -> None:
    _draft_report, validation, decision_file, db, papers_dir = _approved_bundle(tmp_path)
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run(
        validation_report=validation,
        decision_file=decision_file,
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=1,
    )
    assert validate_payload(
        payload,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok
    assert payload["status"] == "ready"
    assert payload["counts"]["selectedRows"] == 1
    assert payload["items"][0]["approvedSourceContentHash"] == "a" * 64
    assert payload["items"][0]["sourceIntegrity"]["hashCheckRequiredAtApply"] is True
    assert payload["coverageTarget"]["minRecoveriesForTarget"] == 6


def test_apply_plan_from_ready_dry_run(tmp_path: Path) -> None:
    _draft_report, validation, decision_file, db, papers_dir = _approved_bundle(tmp_path)
    dry_run = build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run(
        validation_report=validation,
        decision_file=decision_file,
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=1,
    )
    apply_report = build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply(
        sqlite_db=db,
        papers_dir=papers_dir,
        dry_run_report=dry_run,
        apply=False,
    )
    assert validate_payload(
        apply_report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_SCHEMA_ID,
        strict=True,
    ).ok
    assert apply_report["status"] == "ready"
    assert apply_report["counts"]["planned"] == 1
    assert apply_report["mutationCounters"]["sourceDownloadRows"] == 0


def test_local_pdf_apply_requires_matching_source_content_hash(tmp_path: Path, monkeypatch) -> None:
    draft_report, _validation, decision_file, db, papers_dir = _approved_bundle(tmp_path)
    local_pdf = papers_dir / "approved.pdf"
    local_pdf.write_bytes(b"%PDF-1.4\napproved source\n%%EOF\n")
    decision_file["decisions"][0]["decision"] = DECISION_APPROVE_LOCAL_PDF
    decision_file["decisions"][0]["approvedSourceType"] = "local_pdf"
    decision_file["decisions"][0]["approvedSourceUrl"] = ""
    decision_file["decisions"][0]["approvedLocalPdfPath"] = str(local_pdf)
    decision_file["decisions"][0]["approvedSourceContentHash"] = _sha256(local_pdf)
    validation = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        expected_input_rows=2,
    )
    dry_run = build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run(
        validation_report=validation,
        decision_file=decision_file,
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=1,
    )

    def fake_materialize_parsed_artifacts(**kwargs):
        paper_ids = list(kwargs["paper_ids"])
        return {
            "schema": "knowledge-hub.paper.parsed-materialization.result.v1",
            "status": "ok",
            "counts": {"planned": 0, "materialized": len(paper_ids), "blocked": 0, "failed": 0, "skippedExisting": 0},
            "items": [{"paperId": paper_id, "status": "materialized"} for paper_id in paper_ids],
        }

    monkeypatch.setattr(apply_module, "materialize_parsed_artifacts", fake_materialize_parsed_artifacts)
    report = apply_module.build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply(
        sqlite_db=db,
        papers_dir=papers_dir,
        dry_run_report=dry_run,
        apply=True,
    )
    item = report["items"][0]
    assert report["status"] == "applied"
    assert item["sourceContentHashMatched"] is True
    assert item["sourceIntegrity"]["observedSourceContentHash"] == _sha256(local_pdf)


def test_local_pdf_cli_apply_does_not_require_allow_network(tmp_path: Path, monkeypatch) -> None:
    draft_report, _validation, decision_file, db, papers_dir = _approved_bundle(tmp_path)
    local_pdf = papers_dir / "approved.pdf"
    local_pdf.write_bytes(b"%PDF-1.4\napproved source\n%%EOF\n")
    decision_file["decisions"][0]["decision"] = DECISION_APPROVE_LOCAL_PDF
    decision_file["decisions"][0]["approvedSourceType"] = "local_pdf"
    decision_file["decisions"][0]["approvedSourceUrl"] = ""
    decision_file["decisions"][0]["approvedLocalPdfPath"] = str(local_pdf)
    decision_file["decisions"][0]["approvedSourceContentHash"] = _sha256(local_pdf)
    validation = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        expected_input_rows=2,
    )
    dry_run = build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run(
        validation_report=validation,
        decision_file=decision_file,
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=1,
    )
    dry_run_path = tmp_path / "dry-run.json"
    dry_run_path.write_text(json.dumps(dry_run), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "storage:\n"
        f"  sqlite: {str(tmp_path / 'knowledge.db')!r}\n"
        f"  papers_dir: {str(papers_dir)!r}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "apply-output"

    def fake_materialize_parsed_artifacts(**kwargs):
        paper_ids = list(kwargs["paper_ids"])
        return {
            "schema": "knowledge-hub.paper.parsed-materialization.result.v1",
            "status": "ok",
            "counts": {"planned": 0, "materialized": len(paper_ids), "blocked": 0, "failed": 0, "skippedExisting": 0},
            "items": [{"paperId": paper_id, "status": "materialized"} for paper_id in paper_ids],
        }

    monkeypatch.setattr(apply_module, "materialize_parsed_artifacts", fake_materialize_parsed_artifacts)
    db.close()

    exit_code = apply_module.main(
        [
            "--config",
            str(config_path),
            "--dry-run-report",
            str(dry_run_path),
            "--output-dir",
            str(output_dir),
            "--apply",
        ]
    )

    assert exit_code == 0
    report = json.loads(
        (output_dir / "parsed-artifact-manual-lookup-source-recovery-decision-file-apply.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "applied"
    assert report["items"][0]["sourceContentHashMatched"] is True
    assert report["mutationPolicy"]["networkAllowed"] is False
    assert report["mutationCounters"]["sourceDownloadRows"] == 0


def test_source_url_cli_apply_without_allow_network_blocks_without_download(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _draft_report, validation, decision_file, db, papers_dir = _approved_bundle(tmp_path)
    dry_run = build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run(
        validation_report=validation,
        decision_file=decision_file,
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=1,
    )
    dry_run_path = tmp_path / "dry-run.json"
    dry_run_path.write_text(json.dumps(dry_run), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "storage:\n"
        f"  sqlite: {str(tmp_path / 'knowledge.db')!r}\n"
        f"  papers_dir: {str(papers_dir)!r}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "apply-output"

    def fail_download(*_args, **_kwargs):
        raise AssertionError("download must not run without --allow-network")

    def fail_materialize(**_kwargs):
        raise AssertionError("materialization must not run when URL source is network-blocked")

    monkeypatch.setattr(apply_module, "_download_pdf", fail_download)
    monkeypatch.setattr(apply_module, "materialize_parsed_artifacts", fail_materialize)
    db.close()

    exit_code = apply_module.main(
        [
            "--config",
            str(config_path),
            "--dry-run-report",
            str(dry_run_path),
            "--output-dir",
            str(output_dir),
            "--apply",
        ]
    )

    assert exit_code == 0
    report = json.loads(
        (output_dir / "parsed-artifact-manual-lookup-source-recovery-decision-file-apply.json").read_text(
            encoding="utf-8"
        )
    )
    item = report["items"][0]
    assert report["status"] == "blocked"
    assert item["reason"] == "network_not_allowed_for_approved_source_url"
    assert item["sourceDownloadAttempted"] is False
    assert report["mutationPolicy"]["networkAllowed"] is False
    assert report["mutationCounters"]["sourceDownloadRows"] == 0


def test_local_pdf_apply_blocks_source_content_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    draft_report, _validation, decision_file, db, papers_dir = _approved_bundle(tmp_path)
    local_pdf = papers_dir / "approved.pdf"
    local_pdf.write_bytes(b"%PDF-1.4\nactual source\n%%EOF\n")
    decision_file["decisions"][0]["decision"] = DECISION_APPROVE_LOCAL_PDF
    decision_file["decisions"][0]["approvedSourceType"] = "local_pdf"
    decision_file["decisions"][0]["approvedSourceUrl"] = ""
    decision_file["decisions"][0]["approvedLocalPdfPath"] = str(local_pdf)
    decision_file["decisions"][0]["approvedSourceContentHash"] = "0" * 64
    validation = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        expected_input_rows=2,
    )
    dry_run = build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run(
        validation_report=validation,
        decision_file=decision_file,
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=1,
    )

    def fail_if_called(**_kwargs):
        raise AssertionError("materialization must not run after source hash mismatch")

    monkeypatch.setattr(apply_module, "materialize_parsed_artifacts", fail_if_called)
    report = apply_module.build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply(
        sqlite_db=db,
        papers_dir=papers_dir,
        dry_run_report=dry_run,
        apply=True,
    )
    item = report["items"][0]
    assert report["status"] == "blocked"
    assert report["counts"]["blocked"] == 1
    assert item["reason"] == "source_content_hash_mismatch"
    assert item["sourceContentHashMatched"] is False
