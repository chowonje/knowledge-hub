from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.source_span_offset_authority_audit import (
    SOURCE_SPAN_OFFSET_AUTHORITY_AUDIT_SCHEMA_ID,
    build_source_span_offset_authority_audit,
    write_source_span_offset_authority_audit_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidate(
    *,
    candidate_id: str,
    candidate_type: str,
    chars_start: int | None,
    chars_end: int | None,
    page: int | None,
    locator_kind: str,
    source_hash: str = "hash-source",
    bbox: list[float] | None = None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "paper_id": "paper-1",
        "source_parser": "mineru+pymupdf_alignment",
        "candidate_text": "candidate text",
        "canonical_alignment_status": "aligned" if chars_start is not None else "failed",
        "alignment_method": "exact" if chars_start is not None else "none",
        "chars_start": chars_start,
        "chars_end": chars_end,
        "page": page,
        "sourceContentHash": source_hash,
        "source_span_locator": {"locatorKind": locator_kind},
        "layout_element_ids": ["mineru:1"] if bbox else [],
        "bbox": bbox,
        "strict_eligible": False,
        "citation_grade": False,
    }


def _reports(root: Path, *, wrong_schema: bool = False) -> dict[str, Path]:
    schemas = {
        "sectionspan": "knowledge-hub.paper.sectionspan-candidate-report.v1",
        "figure_caption": "knowledge-hub.paper.figure-caption-candidate-report.v1",
        "equation_quote": "knowledge-hub.paper.equation-quote-candidate-report.v1",
        "table_region": "knowledge-hub.paper.table-region-candidate-report.v1",
    }
    if wrong_schema:
        schemas = {key: f"example.wrong.{key}" for key in schemas}
    return {
        "sectionspan": _write(
            root,
            "section.json",
            {
                "schema": schemas["sectionspan"],
                "candidates": [
                    _candidate(
                        candidate_id="sectionspan:paper-1:0001",
                        candidate_type="section_span_candidate",
                        chars_start=10,
                        chars_end=20,
                        page=1,
                        locator_kind="canonical_generated_markdown",
                    )
                ],
            },
        ),
        "figure_caption": _write(
            root,
            "figure.json",
            {
                "schema": schemas["figure_caption"],
                "candidates": [
                    _candidate(
                        candidate_id="figurecaption:paper-1:0001",
                        candidate_type="figure_caption_candidate",
                        chars_start=None,
                        chars_end=None,
                        page=None,
                        locator_kind="generated_markdown",
                        bbox=[1.0, 2.0, 3.0, 4.0],
                    )
                ],
            },
        ),
        "equation_quote": _write(
            root,
            "equation.json",
            {
                "schema": schemas["equation_quote"],
                "candidates": [
                    _candidate(
                        candidate_id="equationquote:paper-1:0001",
                        candidate_type="equation_quote_candidate",
                        chars_start=None,
                        chars_end=None,
                        page=None,
                        locator_kind="",
                        bbox=[1.0, 2.0, 3.0, 4.0],
                    )
                ],
            },
        ),
        "table_region": _write(
            root,
            "table.json",
            {
                "schema": schemas["table_region"],
                "candidates": [
                    _candidate(
                        candidate_id="tableregion:paper-1:0001",
                        candidate_type="table_region_candidate",
                        chars_start=30,
                        chars_end=60,
                        page=2,
                        locator_kind="canonical_generated_markdown",
                        bbox=[1.0, 2.0, 3.0, 4.0],
                    )
                ],
            },
        ),
    }


def _build(paths: dict[str, Path]) -> dict:
    return build_source_span_offset_authority_audit(
        sectionspan_report=paths["sectionspan"],
        figure_caption_report=paths["figure_caption"],
        equation_quote_report=paths["equation_quote"],
        table_region_report=paths["table_region"],
    )


def test_source_span_offset_authority_audit_classifies_offsets_and_validates_schema(tmp_path: Path) -> None:
    paths = _reports(tmp_path)

    payload = _build(paths)

    assert payload["schema"] == SOURCE_SPAN_OFFSET_AUTHORITY_AUDIT_SCHEMA_ID
    assert validate_payload(payload, SOURCE_SPAN_OFFSET_AUTHORITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["auditedCandidateCount"] == 4
    assert payload["counts"]["canonicalParsedTextSpanCandidates"] == 2
    assert payload["counts"]["canonicalGeneratedMarkdownSpanCandidates"] == 2
    assert payload["counts"]["generatedMarkdownOnlyCandidates"] == 1
    assert payload["counts"]["layoutOrBboxOnlyCandidates"] == 1
    assert payload["counts"]["originalPdfOffsetCandidates"] == 0


def test_source_span_offset_authority_audit_never_promotes_to_strict_or_runtime(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path))

    assert payload["policy"]["auditOnly"] is True
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert payload["policy"]["reindexOrReembed"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False
    assert payload["gate"]["strictEvidenceReady"] is False
    assert payload["gate"]["parserRoutingReady"] is False
    for row in payload["rows"]:
        assert row["evidence_tier"] == "source_span_offset_authority_candidate_only"
        assert row["strict_eligible"] is False
        assert row["original_pdf_offset_available"] is False
        assert "source_span_offset_authority_audit_only" in row["strict_blockers"]


def test_source_span_offset_authority_audit_blocks_wrong_input_schema_ids(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path, wrong_schema=True))

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert set(payload["gate"]["schemaViolations"]) == {
        "sectionspan_schema_mismatch",
        "figure_caption_schema_mismatch",
        "equation_quote_schema_mismatch",
        "table_region_schema_mismatch",
    }


def test_source_span_offset_authority_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path / "input"))

    paths = write_source_span_offset_authority_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"audit", "markdown"}
    audit = json.loads(Path(paths["audit"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(audit, SOURCE_SPAN_OFFSET_AUTHORITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert "Canonical generated Markdown offsets" in markdown
    assert "Original PDF offset candidates" in markdown
