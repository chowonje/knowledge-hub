from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_design_review import (
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID,
    REVIEW_STATUS_BLOCKED_NON_UNIQUE,
    REVIEW_STATUS_READY,
)
from knowledge_hub.papers.parsed_artifact_source_span_text_match_disambiguation_design import (
    DISAMBIGUATION_STATUS_BLOCKED_STILL_NON_UNIQUE,
    DISAMBIGUATION_STATUS_CANDIDATE,
    PARSED_ARTIFACT_SOURCE_SPAN_TEXT_MATCH_DISAMBIGUATION_DESIGN_SCHEMA_ID,
    build_parsed_artifact_source_span_text_match_disambiguation_design,
    write_parsed_artifact_source_span_text_match_disambiguation_design_reports,
)
from knowledge_hub.papers.source_text import source_hash_for_path


def _review_row(
    *,
    index: int = 1,
    review_status: str = REVIEW_STATUS_BLOCKED_NON_UNIQUE,
    text_surface: str = "Introduction",
    page: int = 1,
) -> dict:
    return {
        "review_row_id": f"review:{index:04d}",
        "design_row_id": f"offset-design:{index:04d}",
        "sourceSpanId": f"source-span:paper-1:section:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:section:{index}",
        "paper_id": "paper-1",
        "artifact_type": "section",
        "source_candidate_id": f"candidate-{index}",
        "sourceContentHash": "hash-paper-1",
        "source_file": "",
        "text_surface": text_surface,
        "design_status": "blocked_non_unique_text_match",
        "review_status": review_status,
        "review_blockers": ["non_unique_text_match_in_original_source_text"],
        "proposed_chars": {},
        "locator": {"page": page, "bbox": [], "blockIndexes": [], "chars": {"start": None, "end": None}},
        "recommended_action": "run_text_match_disambiguation_before_strict_evidence_design_review",
        "readyForStrictEvidenceDesignReview": False,
        "disambiguationRequired": review_status == REVIEW_STATUS_BLOCKED_NON_UNIQUE,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "sourceSpanUpdatedRows": 0,
    }


def _design_review_report(*rows: dict) -> dict:
    row_list = list(rows)
    ambiguous = [row for row in row_list if row.get("review_status") == REVIEW_STATUS_BLOCKED_NON_UNIQUE]
    ready = [row for row in row_list if row.get("review_status") == REVIEW_STATUS_READY]
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "offsetAuthorityDesignReportPath": "/tmp/offset.json",
            "offsetAuthorityDesignSchema": "knowledge-hub.paper.parsed-artifact-source-span-original-source-offset-authority-design.v1",
            "requestedPaperIds": [],
        },
        "counts": {
            "inputRows": len(row_list),
            "readyForStrictEvidenceDesignReviewRows": len(ready),
            "blockedNonUniqueTextMatchRows": len(ambiguous),
            "blockedMissingExpectedSubstringHashRows": 0,
            "blockedMissingCharsBasisRows": 0,
            "blockedMissingCharsOffsetsRows": 0,
            "blockedRequiresManualOrLaterExtractorReviewRows": 0,
            "blockedUnsupportedArtifactTypeRows": 0,
            "blockedInputSchemaViolationRows": 0,
            "sourceSpanUpdatedRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "schemaViolationCount": 0,
            "byPaperId": {},
            "byArtifactType": {},
            "byReviewStatus": {},
            "byRecommendedAction": {},
        },
        "gate": {
            "designReviewPacketReady": bool(ready),
            "disambiguationQueuePresent": bool(ambiguous),
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "parsed_artifact_source_span_strict_evidence_design_review_ready",
            "recommendedNextTranche": "parsed_artifact_source_span_text_match_disambiguation_design",
        },
        "policy": {
            "reportOnly": True,
            "sourceSpanStoreWrite": False,
            "strictEvidenceCreated": False,
            "strictEligibleMutation": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "warnings": [],
        "sampleBlockers": [],
        "readyDesignRows": ready,
        "ambiguousDisambiguationRows": ambiguous,
        "rows": row_list,
    }


def _paper_fixture(tmp_path: Path, *, pdf_text: str, page_count: int = 2) -> str:
    papers_dir = tmp_path / "papers"
    pdf_path = papers_dir / "paper-1.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_text.encode("utf-8"))
    manifest_dir = papers_dir / "parsed" / "paper-1"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.joinpath("manifest.json").write_text(
        json.dumps({"paper_id": "paper-1", "parser_meta": {"source_pdf": str(pdf_path)}}),
        encoding="utf-8",
    )
    return source_hash_for_path(str(pdf_path))


def test_disambiguation_design_page_unique_match(tmp_path: Path) -> None:
    source_hash = _paper_fixture(
        tmp_path,
        pdf_text="Page1 intro text.\n\nPage2 other Results text.",
    )
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            _design_review_report(
                _review_row(
                    index=1,
                    text_surface="Results",
                    page=2,
                    review_status=REVIEW_STATUS_BLOCKED_NON_UNIQUE,
                )
            )
        ),
        encoding="utf-8",
    )
    review = json.loads(review_path.read_text())
    review["ambiguousDisambiguationRows"][0]["sourceContentHash"] = source_hash
    review_path.write_text(json.dumps(review), encoding="utf-8")

    def page_loader(_path: str | Path) -> list[dict[str, Any]]:
        return [
            {"page": 1, "text": "Page1 intro text."},
            {"page": 2, "text": "Page2 other Results text."},
        ]

    report = build_parsed_artifact_source_span_text_match_disambiguation_design(
        design_review_report_path=review_path,
        papers_dir=tmp_path / "papers",
        parsed_root=tmp_path / "papers" / "parsed",
        page_loader=page_loader,
    )

    assert report["counts"]["disambiguationDesignCandidateOnlyRows"] == 1
    assert report["rows"][0]["disambiguation_status"] == DISAMBIGUATION_STATUS_CANDIDATE
    assert report["rows"][0]["proposed_chars"]["disambiguationMethod"] == "page_context_unique_match"
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_TEXT_MATCH_DISAMBIGUATION_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok


def test_disambiguation_design_still_ambiguous(tmp_path: Path) -> None:
    source_hash = _paper_fixture(tmp_path, pdf_text="Results on page one. Results on page two.")
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            _design_review_report(
                _review_row(index=1, text_surface="Results", page=1),
            )
        ),
        encoding="utf-8",
    )
    review = json.loads(review_path.read_text())
    review["ambiguousDisambiguationRows"][0]["sourceContentHash"] = source_hash
    review_path.write_text(json.dumps(review), encoding="utf-8")

    def page_loader(_path: str | Path) -> list[dict[str, Any]]:
        return [
            {"page": 1, "text": "Results on page one and another Results later."},
            {"page": 2, "text": "Results on page two."},
        ]

    report = build_parsed_artifact_source_span_text_match_disambiguation_design(
        design_review_report_path=review_path,
        papers_dir=tmp_path / "papers",
        parsed_root=tmp_path / "papers" / "parsed",
        page_loader=page_loader,
    )

    assert report["counts"]["blockedStillNonUniqueAfterLocatorContextRows"] == 1
    assert report["rows"][0]["disambiguation_status"] == DISAMBIGUATION_STATUS_BLOCKED_STILL_NON_UNIQUE
    assert len(report["rows"][0]["candidate_match_offsets"]) >= 2


def test_write_disambiguation_design_reports(tmp_path: Path) -> None:
    source_hash = _paper_fixture(tmp_path, pdf_text="Only Results here.")
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(_design_review_report(_review_row(index=1, text_surface="Results", page=1))),
        encoding="utf-8",
    )
    review = json.loads(review_path.read_text())
    review["ambiguousDisambiguationRows"][0]["sourceContentHash"] = source_hash
    review_path.write_text(json.dumps(review), encoding="utf-8")

    report = build_parsed_artifact_source_span_text_match_disambiguation_design(
        design_review_report_path=review_path,
        papers_dir=tmp_path / "papers",
        parsed_root=tmp_path / "papers" / "parsed",
        page_loader=lambda _path: [{"page": 1, "text": "Only Results here."}],
    )
    paths = write_parsed_artifact_source_span_text_match_disambiguation_design_reports(
        report,
        tmp_path / "out",
    )
    assert Path(paths["report"]).exists()
