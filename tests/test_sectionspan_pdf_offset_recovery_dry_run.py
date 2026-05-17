from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_recovery_dry_run import (
    SECTIONSPAN_PDF_OFFSET_RECOVERY_DRY_RUN_SCHEMA_ID,
    build_sectionspan_pdf_offset_recovery_dry_run,
    write_sectionspan_pdf_offset_recovery_dry_run_reports,
)
from knowledge_hub.papers.source_text import source_hash_for_path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _source_file(root: Path, paper_id: str, content: str = "pdf bytes") -> Path:
    path = root / "sources" / f"{paper_id}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _manifest(parsed_root: Path, paper_id: str, source_pdf: Path) -> Path:
    return _write_json(
        parsed_root / paper_id / "manifest.json",
        {"paper_id": paper_id, "parser_meta": {"parser": "pymupdf", "source_pdf": str(source_pdf)}},
    )


def _row(paper_id: str, candidate_text: str, source_hash: str, *, index: int = 1) -> dict:
    return {
        "recovery_plan_id": f"sectionspan-pdf-offset-recovery:{index:04d}",
        "source_sectionspan_candidate_id": f"sectionspan:{paper_id}:{index:04d}",
        "paper_id": paper_id,
        "candidate_text": candidate_text,
        "section_type": "numbered_section",
        "section_level": 1,
        "canonical_span": {
            "chars_start": 10,
            "chars_end": 10 + len(candidate_text),
            "page": 1,
            "sourceContentHash": source_hash,
            "alignmentMethod": "exact",
            "alignmentStatus": "aligned",
            "locatorKind": "canonical_generated_markdown",
        },
    }


def _design_report(root: Path, rows: list[dict], *, status: str = "design_ready", schema: str | None = None) -> Path:
    return _write_json(
        root / "sectionspan-pdf-offset-recovery-design.json",
        {
            "schema": schema or "knowledge-hub.paper.sectionspan-pdf-offset-recovery-design.v1",
            "status": status,
            "recoveryPlanRows": rows,
        },
    )


def _build(root: Path, rows: list[dict], pages: dict[str, list[dict]], *, design_status: str = "design_ready") -> dict:
    parsed_root = root / "parsed"

    def loader(path: str | Path) -> list[dict]:
        return pages[Path(path).name]

    return build_sectionspan_pdf_offset_recovery_dry_run(
        sectionspan_pdf_offset_recovery_design_report=_design_report(root / "input", rows, status=design_status),
        pymupdf_parsed_root=parsed_root,
        pdf_page_text_loader=loader,
    )


def test_pdf_offset_recovery_dry_run_recovers_unique_exact_and_normalized_non_strict(tmp_path: Path) -> None:
    parsed_root = tmp_path / "parsed"
    source = _source_file(tmp_path, "paper-1")
    _manifest(parsed_root, "paper-1", source)
    source_hash = source_hash_for_path(str(source))
    rows = [
        _row("paper-1", "1. Introduction", source_hash, index=1),
        _row("paper-1", "2. Methods", source_hash, index=2),
    ]
    payload = _build(
        tmp_path,
        rows,
        {
            source.name: [
                {"page": 1, "text": "1. Introduction\nOpening text."},
                {"page": 2, "text": "2.\nMethods\nDetails."},
            ]
        },
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_RECOVERY_DRY_RUN_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_RECOVERY_DRY_RUN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "dry_run_complete"
    assert payload["counts"]["originalPdfOffsetRecoveredRows"] == 2
    assert payload["counts"]["exactRecoveredRows"] == 1
    assert payload["counts"]["normalizedRecoveredRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["applyExecuted"] is False
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False
    assert all(row["strict_eligible"] is False for row in payload["recoveryRows"])
    assert all(row["runtime_promotion_allowed"] is False for row in payload["recoveryRows"])
    assert sorted(path.relative_to(parsed_root).as_posix() for path in parsed_root.rglob("*") if path.is_file()) == [
        "paper-1/manifest.json"
    ]


def test_pdf_offset_recovery_dry_run_blocks_ambiguous_matches(tmp_path: Path) -> None:
    parsed_root = tmp_path / "parsed"
    source = _source_file(tmp_path, "paper-1")
    _manifest(parsed_root, "paper-1", source)
    payload = _build(
        tmp_path,
        [_row("paper-1", "Abstract", source_hash_for_path(str(source)))],
        {source.name: [{"page": 1, "text": "Abstract\nBody\nAbstract\nReferences"}]},
    )

    row = payload["recoveryRows"][0]
    assert row["recovery_status"] == "blocked_ambiguous_match"
    assert row["original_pdf_offset_recovered"] is False
    assert row["match_count"] == 2
    assert "original_pdf_offset_not_recovered" in row["strict_blockers"]


def test_pdf_offset_recovery_dry_run_blocks_source_hash_mismatch(tmp_path: Path) -> None:
    parsed_root = tmp_path / "parsed"
    source = _source_file(tmp_path, "paper-1")
    _manifest(parsed_root, "paper-1", source)
    payload = _build(
        tmp_path,
        [_row("paper-1", "1. Introduction", "wrong-hash")],
        {source.name: [{"page": 1, "text": "1. Introduction"}]},
    )

    row = payload["recoveryRows"][0]
    assert row["recovery_status"] == "blocked_source_hash_mismatch"
    assert row["recovery_failure_reason"] == "source_hash_mismatch"
    assert row["strict_eligible"] is False


def test_pdf_offset_recovery_dry_run_blocks_upstream_design_schema_or_status(tmp_path: Path) -> None:
    design = _design_report(
        tmp_path / "input",
        [],
        status="blocked",
        schema="example.wrong.schema",
    )
    payload = build_sectionspan_pdf_offset_recovery_dry_run(
        sectionspan_pdf_offset_recovery_design_report=design,
        pymupdf_parsed_root=tmp_path / "parsed",
        pdf_page_text_loader=lambda _path: [],
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["dryRunComplete"] is False
    assert "sectionspan_pdf_offset_recovery_design_schema_mismatch" in payload["gate"]["schemaViolations"]
    assert "sectionspan_pdf_offset_recovery_design_not_ready" in payload["gate"]["schemaViolations"]


def test_pdf_offset_recovery_dry_run_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    parsed_root = tmp_path / "parsed"
    source = _source_file(tmp_path, "paper-1")
    _manifest(parsed_root, "paper-1", source)
    payload = _build(
        tmp_path,
        [_row("paper-1", "1. Introduction", source_hash_for_path(str(source)))],
        {source.name: [{"page": 1, "text": "1. Introduction"}]},
    )

    paths = write_sectionspan_pdf_offset_recovery_dry_run_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, SECTIONSPAN_PDF_OFFSET_RECOVERY_DRY_RUN_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, SECTIONSPAN_PDF_OFFSET_RECOVERY_DRY_RUN_SCHEMA_ID, strict=True).ok
    assert "report-only dry-run" in markdown
