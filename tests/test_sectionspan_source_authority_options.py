from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_source_authority_options import (
    SECTIONSPAN_SOURCE_AUTHORITY_OPTIONS_SCHEMA_ID,
    build_sectionspan_source_authority_options,
    write_sectionspan_source_authority_options_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _design(*, unsafe: bool = False, status: str = "design_ready") -> dict:
    return {
        "schema": "knowledge-hub.paper.sectionspan-strict-promotion-design.v1",
        "status": status,
        "counts": {
            "inputReviewCards": 2,
            "designRowCount": 2,
            "candidateFormalizationReadyRows": 2,
            "strictPromotionReadyRows": 0,
            "runtimePromotionAllowedRows": 0,
            "heldOutCandidates": 1,
            "strictEligibleRows": 1 if unsafe else 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
            "byPromotionDesignStatus": {
                "blocked_original_pdf_offset_or_authority_decision_required": 2,
            },
        },
        "gate": {
            "strictPromotionDesignReady": status == "design_ready",
            "candidateFormalizationReady": status == "design_ready",
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "strictPromotionImplemented": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "designRows": [
            {"promotion_design_status": "blocked_original_pdf_offset_or_authority_decision_required"},
            {"promotion_design_status": "blocked_original_pdf_offset_or_authority_decision_required"},
        ],
    }


def _report_path(root: Path, payload: dict | None = None) -> Path:
    return _write(root, "sectionspan-strict-promotion-design.json", payload or _design())


def test_source_authority_options_report_lists_safe_options_and_validates_schema(tmp_path: Path) -> None:
    payload = build_sectionspan_source_authority_options(
        sectionspan_strict_promotion_design_report=_report_path(tmp_path)
    )

    assert payload["schema"] == SECTIONSPAN_SOURCE_AUTHORITY_OPTIONS_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_SOURCE_AUTHORITY_OPTIONS_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "options_ready"
    assert payload["counts"]["inputDesignRows"] == 2
    assert payload["counts"]["optionCount"] == 3
    assert payload["counts"]["strictPromotionReadyOptions"] == 0
    assert payload["gate"]["authorityDecisionMade"] is False
    assert payload["gate"]["strictEvidenceReady"] is False
    assert [item["option_id"] for item in payload["options"]] == [
        "keep_candidate_layer_only",
        "recover_original_pdf_offsets_first",
        "explicitly_authorize_canonical_generated_markdown_offsets",
    ]


def test_source_authority_options_do_not_make_policy_decision_or_runtime_promotion(tmp_path: Path) -> None:
    payload = build_sectionspan_source_authority_options(
        sectionspan_strict_promotion_design_report=_report_path(tmp_path)
    )

    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["authorityDecisionMade"] is False
    assert payload["policy"]["strictPromotionImplemented"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    for option in payload["options"]:
        assert option["authority_decision_made"] is False
        assert option["strict_promotion_ready"] is False
        assert option["runtime_promotion_allowed"] is False
        assert option["strict_eligible"] is False
        assert "runtime_answer_citation" in option["blocked_actions"]


def test_source_authority_options_block_unsafe_or_wrong_schema_input(tmp_path: Path) -> None:
    design = _design(unsafe=True)
    design["schema"] = "example.wrong.design"

    payload = build_sectionspan_source_authority_options(
        sectionspan_strict_promotion_design_report=_report_path(tmp_path, design)
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert payload["gate"]["schemaViolations"] == [
        "sectionspan_strict_promotion_design_schema_mismatch"
    ]
    assert "strictEligibleRows_nonzero" in payload["gate"]["unsafeUpstreamFlags"]
    assert validate_payload(payload, SECTIONSPAN_SOURCE_AUTHORITY_OPTIONS_SCHEMA_ID, strict=True).ok


def test_source_authority_options_block_when_design_not_ready(tmp_path: Path) -> None:
    payload = build_sectionspan_source_authority_options(
        sectionspan_strict_promotion_design_report=_report_path(tmp_path, _design(status="blocked"))
    )

    assert payload["status"] == "blocked"
    assert "sectionspan_strict_promotion_design_not_ready" in payload["gate"]["unsafeUpstreamFlags"]


def test_source_authority_options_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    payload = build_sectionspan_source_authority_options(
        sectionspan_strict_promotion_design_report=_report_path(tmp_path / "input")
    )

    paths = write_sectionspan_source_authority_options_reports(payload, tmp_path / "reports")

    assert set(paths) == {"options", "summary", "markdown"}
    options = json.loads(Path(paths["options"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(options, SECTIONSPAN_SOURCE_AUTHORITY_OPTIONS_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, SECTIONSPAN_SOURCE_AUTHORITY_OPTIONS_SCHEMA_ID, strict=True).ok
    assert "does not choose source authority" in markdown
