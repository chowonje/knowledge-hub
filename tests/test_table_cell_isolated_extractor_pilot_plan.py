from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.table_cell_isolated_extractor_pilot_plan import (
    TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_PLAN_SCHEMA_ID,
    build_table_cell_isolated_extractor_pilot_plan,
    write_table_cell_isolated_extractor_pilot_plan_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _next_action(*, unsafe: bool = False, wrong_schema: bool = False) -> dict:
    return {
        "schema": "wrong.schema" if wrong_schema else "knowledge-hub.paper.table-cell-next-action-gate.v1",
        "status": "next_action_ready",
        "counts": {
            "cellBboxTextPairingVerifiedRows": 1 if unsafe else 0,
            "cellSourceSpanCreatedRows": 0,
            "cellSourceHashLinkedRows": 0,
            "tableCellEvidenceCreatedRows": 0,
            "tableCellCitationGradeRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "dependencyApprovalRequired": True,
            "cellBboxTextPairingVerified": False,
            "cellSourceSpansCreated": False,
            "cellSourceHashLinked": False,
            "extractorChoiceMade": False,
            "tableCellEvidenceReady": False,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "extractorChoiceMade": False,
            "tableCellEvidenceCreated": False,
            "tableCellCitationGradeEvidenceCreated": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
    }


def _diagnostic(source_pdf: Path, *, wrong_schema: bool = False) -> dict:
    return {
        "schema": (
            "wrong.schema"
            if wrong_schema
            else "knowledge-hub.paper.table-cell-pymupdf-pairing-diagnostic.v1"
        ),
        "status": "diagnostic_ready",
        "counts": {
            "cellBboxTextPairingVerifiedRows": 0,
            "cellSourceSpanCreatedRows": 0,
            "cellSourceHashLinkedRows": 0,
            "tableCellEvidenceCreatedRows": 0,
            "tableCellCitationGradeRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "cellBboxTextPairingVerified": False,
            "cellSourceSpansCreated": False,
            "cellSourceHashLinked": False,
            "extractorChoiceMade": False,
            "tableCellEvidenceReady": False,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "extractorChoiceMade": False,
            "tableCellEvidenceCreated": False,
            "tableCellCitationGradeEvidenceCreated": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "diagnosticRows": [
            {
                "diagnostic_row_id": "table-cell-pymupdf-pairing-diagnostic:0001",
                "source_table_region_candidate_id": "tableregion:paper-1:0001",
                "paper_id": "paper-1",
                "table_label": "Table 1",
                "page": 9,
                "source_pdf_path": str(source_pdf),
                "selected_table_bbox": [1, 2, 3, 4],
                "cell_bbox_candidate_count": 4,
                "cell_text_candidate_count": 4,
                "diagnostic_unique_cell_text_matches": 1,
                "diagnostic_ambiguous_cell_text_matches": 2,
                "diagnostic_no_match_cell_texts": 1,
            }
        ],
    }


def _environment(*, pdfplumber: bool = False) -> dict:
    return {
        "pythonVersion": "Python 3.10.13",
        "venvAvailable": True,
        "modules": {
            "pdfplumber": {"available": pdfplumber, "origin": "/tmp/pdfplumber.py" if pdfplumber else ""},
            "camelot": {"available": False, "origin": ""},
            "tabula": {"available": False, "origin": ""},
            "pandas": {"available": True, "origin": "/tmp/pandas.py"},
            "fitz": {"available": True, "origin": "/tmp/fitz.py"},
        },
        "systemTools": {
            "java": {"available": False, "path": "/usr/bin/java", "returncode": 1, "output": "missing runtime"},
            "ghostscript": {"available": True, "path": "/opt/homebrew/bin/gs", "returncode": 0, "output": "10.0"},
            "pdftoppm": {"available": False, "path": "", "returncode": None, "output": ""},
            "tesseract": {"available": True, "path": "/opt/homebrew/bin/tesseract", "returncode": 0, "output": "5.0"},
        },
    }


def _paths(root: Path, *, source_exists: bool = True, unsafe: bool = False, wrong_schema: bool = False) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    source_pdf = root / "paper.pdf"
    if source_exists:
        source_pdf.write_bytes(b"%PDF-1.4\n")
    next_action = _write(root, "next-action.json", _next_action(unsafe=unsafe, wrong_schema=wrong_schema))
    diagnostic = _write(root, "diagnostic.json", _diagnostic(source_pdf, wrong_schema=wrong_schema))
    return next_action, diagnostic


def test_isolated_extractor_pilot_plan_requires_approval_without_installing_or_running(tmp_path: Path) -> None:
    next_action, diagnostic = _paths(tmp_path)

    payload = build_table_cell_isolated_extractor_pilot_plan(
        table_cell_next_action_gate=next_action,
        table_cell_pymupdf_pairing_diagnostic=diagnostic,
        environment=_environment(),
    )

    assert payload["schema"] == TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_PLAN_SCHEMA_ID
    assert validate_payload(payload, TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_PLAN_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "approval_required"
    assert payload["gate"]["approvalRequiredBeforeInstallOrRun"] is True
    assert payload["gate"]["globalInstallAllowed"] is False
    assert payload["gate"]["pilotExecuted"] is False
    assert payload["policy"]["isolatedPackageInstall"] is False
    assert payload["policy"]["pilotRunExecuted"] is False
    assert payload["counts"]["packagesInstalled"] == 0
    assert payload["counts"]["globalPackagesInstalled"] == 0


def test_plan_keeps_table_cell_evidence_and_runtime_gates_closed(tmp_path: Path) -> None:
    next_action, diagnostic = _paths(tmp_path)

    payload = build_table_cell_isolated_extractor_pilot_plan(
        table_cell_next_action_gate=next_action,
        table_cell_pymupdf_pairing_diagnostic=diagnostic,
        environment=_environment(pdfplumber=True),
    )

    assert payload["counts"]["availableExtractorModules"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["counts"]["runtimeEvidenceRows"] == 0
    assert payload["gate"]["extractorChoiceMade"] is False
    assert payload["gate"]["cellBboxTextPairingVerified"] is False
    assert payload["gate"]["tableCellEvidenceReady"] is False
    assert payload["gate"]["strictEvidenceReady"] is False
    assert payload["gate"]["parserRoutingReady"] is False
    assert payload["gate"]["answerIntegrationReady"] is False


def test_missing_source_pdf_blocks_the_pilot_plan(tmp_path: Path) -> None:
    next_action, diagnostic = _paths(tmp_path, source_exists=False)

    payload = build_table_cell_isolated_extractor_pilot_plan(
        table_cell_next_action_gate=next_action,
        table_cell_pymupdf_pairing_diagnostic=diagnostic,
        environment=_environment(),
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["sourcePdfsReady"] is False
    assert payload["gate"]["pilotPlanReady"] is False
    assert payload["counts"]["sourcePdfExistingRows"] == 0
    assert validate_payload(payload, TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_PLAN_SCHEMA_ID, strict=True).ok


def test_schema_or_unsafe_input_blocks_the_pilot_plan(tmp_path: Path) -> None:
    next_action, diagnostic = _paths(tmp_path, unsafe=True, wrong_schema=True)

    payload = build_table_cell_isolated_extractor_pilot_plan(
        table_cell_next_action_gate=next_action,
        table_cell_pymupdf_pairing_diagnostic=diagnostic,
        environment=_environment(),
    )

    assert payload["status"] == "blocked"
    assert "table_cell_next_action_gate_schema_mismatch" in payload["gate"]["schemaViolations"]
    assert "table_cell_pymupdf_pairing_diagnostic_schema_mismatch" in payload["gate"]["schemaViolations"]
    assert "next_action_cellBboxTextPairingVerifiedRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert validate_payload(payload, TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_PLAN_SCHEMA_ID, strict=True).ok


def test_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    next_action, diagnostic = _paths(tmp_path / "input")
    payload = build_table_cell_isolated_extractor_pilot_plan(
        table_cell_next_action_gate=next_action,
        table_cell_pymupdf_pairing_diagnostic=diagnostic,
        environment=_environment(),
    )

    paths = write_table_cell_isolated_extractor_pilot_plan_reports(payload, tmp_path / "reports")

    assert set(paths) == {"plan", "summary", "markdown"}
    plan = json.loads(Path(paths["plan"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(plan, TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_PLAN_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_PLAN_SCHEMA_ID, strict=True).ok
    assert "does not install packages" in markdown
