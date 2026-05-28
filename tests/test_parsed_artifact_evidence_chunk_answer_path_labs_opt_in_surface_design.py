from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_DEFAULT_OFF_NO_ANSWER_REGRESSION_SMOKE_SCHEMA_ID,
    READY_DECISION as DEFAULT_OFF_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_DESIGN_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design,
    write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design,
)


def _default_off_ready() -> dict[str, Any]:
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_DEFAULT_OFF_NO_ANSWER_REGRESSION_SMOKE_SCHEMA_ID,
        "status": "ready",
        "decision": DEFAULT_OFF_READY_DECISION,
        "counts": {
            "inputScenarioRows": 3,
            "passRows": 3,
            "failRows": 0,
            "noAnswerRows": 3,
            "answerableRows": 0,
            "evidencePacketContractAnswerableRows": 0,
            "adapterAppliedRows": 0,
            "adapterRowsAdded": 0,
            "selectedEvidenceCount": 0,
            "citationCount": 0,
            "evidencePacketContractSpanRows": 0,
            "localFakeLlmCallRows": 0,
            "externalLlmCallRows": 0,
            "modelApiCallRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "gate": {
            "readyForLabsOptInSurfaceDesign": True,
            "publicDefaultUnchanged": True,
        },
    }


def _surface_overrides(**updates: Any) -> dict[str, Any]:
    base = {
        "internalGenerateAnswerQueryPlanParam": True,
        "internalStreamAnswerQueryPlanParam": True,
        "khubAskPublicOptInFlagPresent": False,
        "khubAskQueryPlanPublicArgPresent": False,
        "defaultMcpAskAdapterArgPresent": False,
        "defaultMcpAskQueryPlanArgPresent": False,
        "defaultMcpAskToolInDefaultProfile": True,
        "futureLabsMcpToolAlreadyPresent": False,
        "futureLabsMcpToolInDefaultProfile": False,
        "labsMcpProfileAvailable": True,
        "defaultToolSetContainsOnlyAllowedDefaultNames": True,
        "paperLabsCliGroupImportable": True,
        "paperLabsCliRegisteredUnderLabs": True,
        "futureLabsCliCommandAlreadyPresent": False,
    }
    base.update(updates)
    return base


def test_labs_opt_in_surface_design_ready() -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design(
        default_off_report=_default_off_ready(),
        surface_observation_overrides=_surface_overrides(),
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["designRows"] == 10
    assert report["counts"]["designReadyRows"] == 10
    assert report["counts"]["plannedLabsCliRows"] == 1
    assert report["counts"]["plannedLabsMcpRows"] == 1
    assert report["counts"]["plannedPublicCliRows"] == 0
    assert report["counts"]["plannedDefaultMcpRows"] == 0
    assert report["counts"]["publicCliFlagRows"] == 0
    assert report["counts"]["defaultMcpSchemaChangeRows"] == 0
    assert report["gate"]["readyForLabsOptInSurfaceImplementation"] is True
    assert report["gate"]["publicKhubAskClosed"] is True
    assert report["gate"]["defaultMcpAskClosed"] is True
    assert report["design"]["futureLabsCliCommand"] == "khub labs paper evidence-chunk-ask"
    assert report["design"]["futureLabsMcpTool"] == "paper_evidence_chunk_answer_preview"
    assert report["design"]["defaultProfileAllowed"] is False
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_opt_in_surface_design_blocks_when_default_off_smoke_not_ready() -> None:
    source = _default_off_ready()
    source["status"] = "blocked"

    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design(
        default_off_report=source,
        surface_observation_overrides=_surface_overrides(),
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert "default_off_smoke_not_ready" in report["gate"]["semanticViolations"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_opt_in_surface_design_blocks_public_or_default_mcp_exposure() -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design(
        default_off_report=_default_off_ready(),
        surface_observation_overrides=_surface_overrides(
            khubAskPublicOptInFlagPresent=True,
            defaultMcpAskAdapterArgPresent=True,
        ),
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["gate"]["publicKhubAskClosed"] is False
    assert report["gate"]["defaultMcpAskClosed"] is False
    assert "public_khub_ask_stays_closed" in report["gate"]["semanticViolations"]
    assert "default_mcp_ask_stays_closed" in report["gate"]["semanticViolations"]


def test_labs_opt_in_surface_design_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design(
        default_off_report=_default_off_ready(),
        surface_observation_overrides=_surface_overrides(),
        generated_at="2026-05-29T00:00:00Z",
    )

    paths = write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in Surface Design"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok
