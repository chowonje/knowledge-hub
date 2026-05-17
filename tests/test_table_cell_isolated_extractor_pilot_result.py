from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.table_cell_isolated_extractor_pilot_result import (
    TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID,
    build_table_cell_isolated_extractor_pilot_result,
    write_table_cell_isolated_extractor_pilot_result_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _plan(*, wrong_schema: bool = False, unsafe: bool = False) -> dict:
    return {
        "schema": (
            "wrong.schema"
            if wrong_schema
            else "knowledge-hub.paper.table-cell-isolated-extractor-pilot-plan.v1"
        ),
        "status": "approval_required",
        "counts": {
            "plannedPilotRuns": 1 if unsafe else 0,
            "packagesInstalled": 0,
            "globalPackagesInstalled": 0,
            "tableCellEvidenceCreatedRows": 0,
            "tableCellCitationGradeRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "pilotExecuted": False,
            "extractorChoiceMade": False,
            "cellBboxTextPairingVerified": False,
            "cellSourceSpansCreated": False,
            "cellSourceHashLinked": False,
            "tableCellEvidenceReady": False,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "globalPackageInstall": False,
            "isolatedPackageInstall": False,
            "pilotRunExecuted": False,
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
        "targetRows": [
            {
                "paper_id": "paper-1",
                "table_label": "Table 1",
                "page": 3,
                "source_pdf_path": "/tmp/paper-1.pdf",
                "source_pdf_exists": True,
                "source_table_region_candidate_id": "tableregion:paper-1:0001",
            }
        ],
    }


def _path(root: Path, *, wrong_schema: bool = False, unsafe: bool = False) -> Path:
    return _write(root, "plan.json", _plan(wrong_schema=wrong_schema, unsafe=unsafe))


def _probe(_source_pdf: str | Path, page_number: int) -> dict[str, Any]:
    assert page_number == 3
    return {
        "status": "ok",
        "failureReason": "",
        "tables": [
            {
                "table_index": 1,
                "bbox": [10.0, 20.0, 100.0, 120.0],
                "row_count": 2,
                "column_count": 2,
                "cell_bbox_count": 4,
                "cell_bboxes_sample": [[10.0, 20.0, 50.0, 40.0]],
                "extracted_rows": [["Model", "Score"], ["A", "1.0"]],
            }
        ],
    }


def test_result_requires_explicit_approval_before_probe_attempt(tmp_path: Path) -> None:
    plan = _path(tmp_path)

    payload = build_table_cell_isolated_extractor_pilot_result(
        table_cell_isolated_extractor_pilot_plan=plan,
        extractor_available=True,
        extractor_probe_loader=lambda *_args: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    assert payload["schema"] == TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID
    assert validate_payload(payload, TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "approval_required"
    assert payload["counts"]["probeAttemptedRows"] == 0
    assert payload["counts"]["approvalRequiredRows"] == 1
    assert payload["gate"]["approvalRequiredBeforeInstallOrRun"] is True
    assert payload["policy"]["packageInstallAttempted"] is False


def test_approved_run_blocks_when_extractor_is_unavailable(tmp_path: Path) -> None:
    plan = _path(tmp_path)

    payload = build_table_cell_isolated_extractor_pilot_result(
        table_cell_isolated_extractor_pilot_plan=plan,
        approved_to_run=True,
        extractor_available=False,
        extractor_probe_loader=lambda *_args: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blockedRows"] == 1
    assert payload["counts"]["probeAttemptedRows"] == 0
    assert payload["gate"]["extractorAvailable"] is False
    assert payload["resultRows"][0]["probe_status"] == "blocked_extractor_unavailable"
    assert validate_payload(payload, TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID, strict=True).ok


def test_approved_available_extractor_runs_probe_but_stays_non_strict(tmp_path: Path) -> None:
    plan = _path(tmp_path)

    payload = build_table_cell_isolated_extractor_pilot_result(
        table_cell_isolated_extractor_pilot_plan=plan,
        approved_to_run=True,
        extractor_available=True,
        extractor_probe_loader=_probe,
    )

    assert payload["status"] == "pilot_complete_non_strict"
    assert payload["counts"]["probeAttemptedRows"] == 1
    assert payload["counts"]["cellBboxCandidateRows"] == 1
    assert payload["counts"]["selectedTableCellBboxCandidates"] == 4
    assert payload["counts"]["selectedTableCellTextCandidates"] == 4
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["pilotExecuted"] is True
    assert payload["gate"]["extractorChoiceMade"] is False
    assert payload["gate"]["tableCellEvidenceReady"] is False
    assert payload["gate"]["strictEvidenceReady"] is False
    assert payload["resultRows"][0]["strict_eligible"] is False
    assert validate_payload(payload, TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID, strict=True).ok


def test_wrong_or_unsafe_plan_blocks_without_probe(tmp_path: Path) -> None:
    plan = _path(tmp_path, wrong_schema=True, unsafe=True)

    payload = build_table_cell_isolated_extractor_pilot_result(
        table_cell_isolated_extractor_pilot_plan=plan,
        approved_to_run=True,
        extractor_available=True,
        extractor_probe_loader=lambda *_args: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    assert payload["status"] == "blocked"
    assert "table_cell_isolated_extractor_pilot_plan_schema_mismatch" in payload["gate"]["schemaViolations"]
    assert "plannedPilotRuns_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert payload["resultRows"][0]["probe_status"] == "blocked_input_plan"
    assert validate_payload(payload, TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID, strict=True).ok


def test_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    plan = _path(tmp_path / "input")
    payload = build_table_cell_isolated_extractor_pilot_result(
        table_cell_isolated_extractor_pilot_plan=plan,
        extractor_available=True,
    )

    paths = write_table_cell_isolated_extractor_pilot_result_reports(payload, tmp_path / "reports")

    assert set(paths) == {"result", "summary", "markdown"}
    result = json.loads(Path(paths["result"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(result, TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID, strict=True).ok
    assert "does not install packages" in markdown
