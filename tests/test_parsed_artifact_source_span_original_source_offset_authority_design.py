from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.source_text import source_hash_for_path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_original_source_offset_authority_design import (
    DESIGN_STATUS_BLOCKED_MISSING_TEXT_SURFACE,
    DESIGN_STATUS_BLOCKED_NON_UNIQUE_TEXT_MATCH,
    DESIGN_STATUS_OFFSET_AUTHORITY_CANDIDATE,
    PARSED_ARTIFACT_SOURCE_SPAN_ORIGINAL_SOURCE_OFFSET_AUTHORITY_DESIGN_SCHEMA_ID,
    build_parsed_artifact_source_span_original_source_offset_authority_design,
    write_parsed_artifact_source_span_original_source_offset_authority_design_reports,
)
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed import (
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_V2_TYPED_SCHEMA_ID,
    POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY,
)


def _policy_gate_row(
    *,
    index: int = 1,
    artifact_type: str = "section",
    source_hash: str = "hash-paper-1",
    source_candidate_id: str = "candidate-1",
    text_key: str = "candidate-1",
) -> dict:
    return {
        "policy_gate_row_id": f"gate:{index:04d}",
        "readback_review_row_id": f"readback:{index:04d}",
        "sourceSpanId": f"source-span:paper-1:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:{artifact_type}:{index}",
        "runId": "run-1",
        "paper_id": "paper-1",
        "artifact_type": artifact_type,
        "source_candidate_id": source_candidate_id,
        "sourceContentHash": source_hash,
        "source_file": "",
        "locator": {"page": 1, "bbox": [], "blockIndexes": [], "chars": {"start": None, "end": None}},
        "idempotencyKey": f"idem-{index}",
        "source_span_store_path": "/tmp/paper-1.jsonl",
        "source_span_store_line": index,
        "policy_gate_status": POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY,
        "policy_blockers": ["chars_start_or_end_missing"],
        "readback_status": "readback_validated_source_span",
        "offset_authority_mode": "page_or_block_only",
        "strict_text_policy_candidate_only": False,
        "strict_caption_policy_candidate_only": False,
        "strict_structured_policy_candidate_only": False,
        "strictEvidenceDesignReviewReady": False,
        "strictEvidenceCreated": False,
        "sourceSpanCreated": False,
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "canonicalParsedArtifactsWritten": False,
        "recommended_action": "recover_original_source_char_offset_authority",
        "_text_key": text_key,
    }


def _policy_gate_report(*rows: dict) -> dict:
    clean_rows = []
    for row in rows:
        item = dict(row)
        item.pop("_text_key", None)
        clean_rows.append(item)
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_V2_TYPED_SCHEMA_ID,
        "status": "blocked",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "readbackReportPath": "/tmp/readback.json",
            "readbackSchema": "knowledge-hub.paper.parsed-artifact-source-span-promotion-readback-review.v1",
            "requestedPaperIds": [],
        },
        "counts": {
            "inputRows": len(clean_rows),
            "sourceSpanRecordRows": len(clean_rows),
            "readbackValidatedRows": len(clean_rows),
            "strictTextPolicyCandidateOnlyRows": 0,
            "strictCaptionPolicyCandidateOnlyRows": 0,
            "strictStructuredPolicyCandidateOnlyRows": 0,
            "blockedInputSchemaViolationRows": 0,
            "blockedReadbackNotReadyRows": 0,
            "blockedMissingSourceHashRows": 0,
            "blockedMissingLocatorRows": 0,
            "blockedMissingOffsetAuthorityRows": len(clean_rows),
            "blockedMissingStructuredAuthorityRows": 0,
            "blockedUnsupportedArtifactTypeRows": 0,
            "blockedRuntimeOrStrictFlagRows": 0,
            "blockedMissingRecordIdentityRows": 0,
            "blockedLocatorBasisUnknownRows": 0,
            "blockedNormalizationMismatchRows": 0,
            "blockedTypeAuthorityPolicyUndefinedRows": 0,
            "sourceSpanCreatedRows": 0,
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
            "byPolicyGateStatus": {},
            "byArtifactTypeByPolicyGateStatus": {},
            "byArtifactTypeByOffsetAuthorityMode": {},
            "byRecommendedAction": {},
        },
        "gate": {
            "readyForStrictEvidenceDesignReview": False,
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "blocked",
            "recommendedNextTranche": "parsed_artifact_source_span_original_source_offset_authority_design",
        },
        "policy": {
            "reportOnly": True,
            "strictEvidencePolicyGateTypedOnly": True,
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
        "rows": clean_rows,
    }


def _paper_fixture(tmp_path: Path, *, pdf_text: str) -> str:
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


def _candidate_reports(tmp_path: Path, *, surfaces: dict[str, str]) -> tuple[Path, Path]:
    sectionspan_path = tmp_path / "sectionspan.json"
    figure_path = tmp_path / "figure.json"
    sectionspan_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "source_candidate_id": key,
                        "paper_id": "paper-1",
                        "candidate_text": text,
                        "section_title": text,
                    }
                    for key, text in surfaces.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    figure_path.write_text(
        json.dumps({"candidates": []}),
        encoding="utf-8",
    )
    return sectionspan_path, figure_path


def test_offset_authority_design_unique_match(tmp_path: Path) -> None:
    source_hash = _paper_fixture(tmp_path, pdf_text="Abstract. Introduction to the topic.")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            _policy_gate_report(
                _policy_gate_row(
                    index=1,
                    source_candidate_id="section-1",
                    text_key="section-1",
                    source_hash=source_hash,
                )
            )
        ),
        encoding="utf-8",
    )
    sectionspan_path, figure_path = _candidate_reports(tmp_path, surfaces={"section-1": "Introduction"})

    def page_loader(_path: str | Path) -> list[dict[str, Any]]:
        return [{"page": 1, "text": "Abstract. Introduction to the topic."}]

    report = build_parsed_artifact_source_span_original_source_offset_authority_design(
        policy_gate_report_path=gate_path,
        papers_dir=tmp_path / "papers",
        parsed_root=tmp_path / "papers" / "parsed",
        sectionspan_candidate_report_path=sectionspan_path,
        figure_caption_candidate_report_path=figure_path,
        page_loader=page_loader,
    )

    assert report["status"] == "ok"
    assert report["counts"]["offsetAuthorityDesignCandidateOnlyRows"] == 1
    row = report["rows"][0]
    assert row["design_status"] == DESIGN_STATUS_OFFSET_AUTHORITY_CANDIDATE
    proposed = row["proposed_chars"]
    assert proposed["basis"] == "sourceContentHash"
    assert proposed["normalization"] == "nfkc_whitespace_casefold_v1"
    assert proposed["expectedSubstringSha256"]
    validation = validate_payload(report, PARSED_ARTIFACT_SOURCE_SPAN_ORIGINAL_SOURCE_OFFSET_AUTHORITY_DESIGN_SCHEMA_ID, strict=True)
    assert validation.ok


def test_offset_authority_design_ambiguous_match(tmp_path: Path) -> None:
    source_hash = _paper_fixture(tmp_path, pdf_text="the cat and the dog")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            _policy_gate_report(
                _policy_gate_row(
                    index=1,
                    source_candidate_id="section-1",
                    text_key="section-1",
                    source_hash=source_hash,
                )
            )
        ),
        encoding="utf-8",
    )
    sectionspan_path, figure_path = _candidate_reports(tmp_path, surfaces={"section-1": "the"})

    report = build_parsed_artifact_source_span_original_source_offset_authority_design(
        policy_gate_report_path=gate_path,
        papers_dir=tmp_path / "papers",
        parsed_root=tmp_path / "papers" / "parsed",
        sectionspan_candidate_report_path=sectionspan_path,
        figure_caption_candidate_report_path=figure_path,
        page_loader=lambda _path: [{"page": 1, "text": "the cat and the dog"}],
    )

    assert report["counts"]["blockedNonUniqueTextMatchRows"] == 1
    assert report["rows"][0]["design_status"] == DESIGN_STATUS_BLOCKED_NON_UNIQUE_TEXT_MATCH


def test_offset_authority_design_missing_text_surface(tmp_path: Path) -> None:
    source_hash = _paper_fixture(tmp_path, pdf_text="hello")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            _policy_gate_report(
                _policy_gate_row(index=1, source_candidate_id="missing-1", source_hash=source_hash)
            )
        ),
        encoding="utf-8",
    )
    sectionspan_path, figure_path = _candidate_reports(tmp_path, surfaces={})

    report = build_parsed_artifact_source_span_original_source_offset_authority_design(
        policy_gate_report_path=gate_path,
        papers_dir=tmp_path / "papers",
        parsed_root=tmp_path / "papers" / "parsed",
        sectionspan_candidate_report_path=sectionspan_path,
        figure_caption_candidate_report_path=figure_path,
        page_loader=lambda _path: [{"page": 1, "text": "hello"}],
    )

    assert report["counts"]["blockedMissingTextSurfaceRows"] == 1
    assert report["rows"][0]["design_status"] == DESIGN_STATUS_BLOCKED_MISSING_TEXT_SURFACE


def test_write_reports_round_trip(tmp_path: Path) -> None:
    source_hash = _paper_fixture(tmp_path, pdf_text="Intro section")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            _policy_gate_report(
                _policy_gate_row(index=1, source_candidate_id="section-1", source_hash=source_hash)
            )
        ),
        encoding="utf-8",
    )
    sectionspan_path, figure_path = _candidate_reports(tmp_path, surfaces={"section-1": "Intro"})

    report = build_parsed_artifact_source_span_original_source_offset_authority_design(
        policy_gate_report_path=gate_path,
        papers_dir=tmp_path / "papers",
        parsed_root=tmp_path / "papers" / "parsed",
        sectionspan_candidate_report_path=sectionspan_path,
        figure_caption_candidate_report_path=figure_path,
    )
    out_dir = tmp_path / "out"
    paths = write_parsed_artifact_source_span_original_source_offset_authority_design_reports(report, out_dir)
    assert Path(paths["report"]).exists()
    loaded = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert loaded["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_ORIGINAL_SOURCE_OFFSET_AUTHORITY_DESIGN_SCHEMA_ID
