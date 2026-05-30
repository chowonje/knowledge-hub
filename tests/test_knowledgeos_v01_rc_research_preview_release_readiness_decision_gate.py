from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_research_preview_release_readiness_decision_gate import (
    BLOCKED_DECISION,
    DEFAULT_POSITIVE_COMPLETE_REVIEW_REPORT,
    DEFAULT_PRODUCT_DEFINITION_DOC,
    DEFAULT_RELEASE_NOTES_DOC,
    KNOWLEDGEOS_V01_RC_RESEARCH_PREVIEW_RELEASE_READINESS_DECISION_GATE_SCHEMA_ID,
    READY_DECISION,
    build_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate,
    write_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate,
)


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _positive_complete_report() -> dict[str, Any]:
    return _read(DEFAULT_POSITIVE_COMPLETE_REVIEW_REPORT)


def _product_definition_text() -> str:
    return Path(DEFAULT_PRODUCT_DEFINITION_DOC).read_text(encoding="utf-8")


def _release_notes_text() -> str:
    return Path(DEFAULT_RELEASE_NOTES_DOC).read_text(encoding="utf-8")


def _build(**updates: Any) -> dict[str, Any]:
    return build_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate(
        positive_complete_review_report=updates.pop("positive_complete_review_report", _positive_complete_report()),
        product_definition_text=updates.pop("product_definition_text", _product_definition_text()),
        release_notes_text=updates.pop("release_notes_text", _release_notes_text()),
        generated_at="2026-05-30T00:00:00Z",
        **updates,
    )


def test_release_readiness_gate_ready_for_controlled_research_preview() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["nextRecommendedTranche"] == "operator_release_package_handoff_or_branch_cleanup_decision"
    assert report["readinessDecision"]["researchPreviewDecision"] == "ready_for_controlled_research_preview_release"
    assert report["readinessDecision"]["publicDefaultDecision"] == "hold_public_default_promotion"
    assert report["readinessDecision"]["generalReleaseDecision"] == "not_ready_for_general_release"
    assert report["counts"]["researchPreviewReleaseReadyRows"] == 1
    assert report["counts"]["positiveSectionParagraphQualityCompleteRows"] == 1
    assert report["counts"]["positiveAnswerPassRows"] == 7
    assert report["counts"]["provenancePassRows"] == 7
    assert report["counts"]["strictProvenanceSpanRows"] == 20
    assert report["counts"]["sourceContentHashRows"] == 20
    assert report["counts"]["charsLocatorRows"] == 20
    assert report["counts"]["answerContractCitationProvenanceRows"] == 20
    assert report["counts"]["publicDefaultPromotionReadyRows"] == 0
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["generalRcReadyRows"] == 0
    assert report["counts"]["corpusScaleClaimProvenRows"] == 0
    assert report["counts"]["tableEquationFigureDefaultEvidenceRows"] == 0
    assert report["counts"]["visualHintAnswerEvidenceRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["counts"]["schemaViolationCount"] == 0
    assert report["gate"]["researchPreviewReleaseAllowed"] is True
    assert report["gate"]["publicDefaultPromotionAllowed"] is False
    assert report["gate"]["generalRcReady"] is False
    assert validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_RESEARCH_PREVIEW_RELEASE_READINESS_DECISION_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_release_readiness_gate_blocks_when_positive_complete_report_is_not_ready() -> None:
    positive = deepcopy(_positive_complete_report())
    positive["status"] = "blocked"

    report = _build(positive_complete_review_report=positive)

    assert report["status"] == "blocked"
    assert report["decision"] == BLOCKED_DECISION
    assert "positive_complete_not_ready" in report["gate"]["semanticViolations"]


def test_release_readiness_gate_blocks_when_release_notes_required_phrase_is_missing() -> None:
    release_text = _release_notes_text().replace("public/default promotion remains held", "public default remains held")

    report = _build(release_notes_text=release_text)

    assert report["status"] == "blocked"
    assert (
        "release_notes_missing_required_phrase:public/default promotion remains held"
        in report["gate"]["semanticViolations"]
    )


def test_release_readiness_gate_blocks_forbidden_release_language() -> None:
    report = _build(release_notes_text=_release_notes_text() + "\nThis is production-ready.\n")

    assert report["status"] == "blocked"
    assert "release_notes_forbidden_phrase:production-ready" in report["gate"]["semanticViolations"]


def test_release_readiness_gate_blocks_unexpected_public_default_promotion() -> None:
    positive = deepcopy(_positive_complete_report())
    positive["counts"]["publicDefaultPromotionReadyRows"] = 1

    report = _build(positive_complete_review_report=positive)

    assert report["status"] == "blocked"
    assert "positive_complete_public_default_ready_unexpected" in report["gate"]["semanticViolations"]
    assert report["counts"]["publicDefaultPromotionReadyRows"] == 0


def test_release_readiness_gate_blocks_private_path_marker() -> None:
    report = _build(product_definition_text=_product_definition_text() + "\n/" + "Users" + "/example/private\n")

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "research_preview_release_readiness_private_path_marker" in report["gate"]["semanticViolations"]


def test_release_readiness_gate_keeps_mutation_and_default_surface_counters_zero() -> None:
    report = _build()
    counts = report["counts"]

    for field in (
        "candidateStoreWriteRows",
        "sourceSpanCreatedRows",
        "runtimeEvidenceRows",
        "parserExecutionRows",
        "databaseMutationRows",
        "indexMutationRows",
        "vaultScanRows",
        "externalDownloadRows",
        "externalLlmCallRows",
        "modelApiCallRows",
        "judgeModelCallRows",
        "pushRows",
        "githubPrMutationRows",
        "mergeRows",
        "branchDeletionRows",
        "releaseTagRows",
        "packagePublishRows",
        "defaultOnRows",
        "defaultMcpToolRows",
        "defaultKhubAskRouteRows",
    ):
        assert counts[field] == 0
    assert counts["publicDefaultPromotionReadyRows"] == 0
    assert counts["publicDefaultPromotionHeldRows"] == 1


def test_release_readiness_gate_writer_outputs_schema_valid_report(tmp_path: Path) -> None:
    report = _build()

    paths = write_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = _read(paths["json"])
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert parsed["status"] == "ready"
    assert "researchPreviewReleaseReadyRows" in markdown
    assert "publicDefaultPromotionReadyRows" in markdown
    assert validate_payload(
        parsed,
        KNOWLEDGEOS_V01_RC_RESEARCH_PREVIEW_RELEASE_READINESS_DECISION_GATE_SCHEMA_ID,
        strict=True,
    ).ok
