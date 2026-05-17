from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_recovery_review_pack import (
    SECTIONSPAN_PDF_OFFSET_RECOVERY_REVIEW_PACK_SCHEMA_ID,
    build_sectionspan_pdf_offset_recovery_review_pack,
    write_sectionspan_pdf_offset_recovery_review_pack_reports,
)


def _write(root: Path, payload: dict) -> Path:
    path = root / "sectionspan-pdf-offset-recovery-dry-run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _row(
    *,
    text: str = "1. Introduction",
    recovered: bool = True,
    canonical_page: int = 1,
    original_page: int = 1,
    canonical_hash: str = "hash-source",
    original_hash: str = "hash-source",
    method: str = "exact",
) -> dict:
    return {
        "recovery_plan_id": "sectionspan-pdf-offset-recovery:0001",
        "source_sectionspan_candidate_id": "sectionspan:paper-1:0001",
        "paper_id": "paper-1",
        "candidate_text": text,
        "section_type": "numbered_section",
        "section_level": 1,
        "canonical_span": {
            "chars_start": 10,
            "chars_end": 10 + len(text),
            "page": canonical_page,
            "sourceContentHash": canonical_hash,
            "locatorKind": "canonical_generated_markdown",
        },
        "original_pdf_span": {
            "originalPdfCharsStart": 20 if recovered else None,
            "originalPdfCharsEnd": 20 + len(text) if recovered else None,
            "page": original_page if recovered else None,
            "sourceContentHash": original_hash,
            "matchMethod": method if recovered else "",
            "matchConfidence": 1.0 if method == "exact" else 0.95,
        },
        "original_pdf_offset_recovered": recovered,
        "strict_eligible": False,
        "runtime_promotion_allowed": False,
        "citation_grade": False,
        "runtime_evidence": False,
    }


def _dry_run(rows: list[dict], *, unsafe: bool = False, status: str = "dry_run_complete") -> dict:
    return {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-recovery-dry-run.v1",
        "status": status,
        "counts": {
            "strictEligibleRows": 1 if unsafe else 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "dryRunOnly": True,
            "applyExecuted": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "recoveryRows": rows,
    }


def test_pdf_offset_recovery_review_pack_marks_recovered_rows_ready_but_non_strict(tmp_path: Path) -> None:
    payload = build_sectionspan_pdf_offset_recovery_review_pack(
        sectionspan_pdf_offset_recovery_dry_run_report=_write(tmp_path, _dry_run([_row(), _row(method="normalized_whitespace_case")]))
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_RECOVERY_REVIEW_PACK_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_RECOVERY_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "review_pack_ready"
    assert payload["counts"]["reviewCardRows"] == 2
    assert payload["counts"]["readyForHumanReviewRows"] == 2
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["counts"]["runtimeEvidenceRows"] == 0
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False
    assert all(card["strict_eligible"] is False for card in payload["reviewCards"])
    assert all(card["runtime_promotion_allowed"] is False for card in payload["reviewCards"])


def test_pdf_offset_recovery_review_pack_holds_out_page_and_hash_conflicts(tmp_path: Path) -> None:
    payload = build_sectionspan_pdf_offset_recovery_review_pack(
        sectionspan_pdf_offset_recovery_dry_run_report=_write(
            tmp_path,
            _dry_run(
                [
                    _row(original_page=2),
                    _row(original_hash="other-hash"),
                    _row(recovered=False),
                ]
            ),
        )
    )

    statuses = [card["review_status"] for card in payload["reviewCards"]]
    assert statuses == ["held_out_page_conflict", "held_out_source_hash_conflict", "held_out_recovery_blocked"]
    assert payload["counts"]["readyForHumanReviewRows"] == 0
    assert payload["counts"]["heldOutRows"] == 3


def test_pdf_offset_recovery_review_pack_blocks_unsafe_upstream_payload(tmp_path: Path) -> None:
    payload = build_sectionspan_pdf_offset_recovery_review_pack(
        sectionspan_pdf_offset_recovery_dry_run_report=_write(tmp_path, _dry_run([_row()], unsafe=True))
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["reviewPackReady"] is False
    assert "dryRun_strictEligibleRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]


def test_pdf_offset_recovery_review_pack_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    payload = build_sectionspan_pdf_offset_recovery_review_pack(
        sectionspan_pdf_offset_recovery_dry_run_report=_write(tmp_path / "input", _dry_run([_row()]))
    )

    paths = write_sectionspan_pdf_offset_recovery_review_pack_reports(payload, tmp_path / "reports")

    assert set(paths) == {"cards", "summary", "markdown"}
    cards = json.loads(Path(paths["cards"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(cards, SECTIONSPAN_PDF_OFFSET_RECOVERY_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, SECTIONSPAN_PDF_OFFSET_RECOVERY_REVIEW_PACK_SCHEMA_ID, strict=True).ok
    assert "review inputs only" in markdown
