from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review import (
    DEFAULT_PROMOTION_DECISION_GATE_REPORT,
    DEFAULT_RELEASE_NOTES_DOC,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RESEARCH_PREVIEW_RELEASE_NOTES_REVIEW_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review,
    write_parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review,
)


def _promotion_report(**updates: Any) -> dict[str, Any]:
    report = json.loads(DEFAULT_PROMOTION_DECISION_GATE_REPORT.read_text(encoding="utf-8"))
    for key, value in updates.items():
        report[key] = value
    return report


def _release_notes_text() -> str:
    return DEFAULT_RELEASE_NOTES_DOC.read_text(encoding="utf-8")


def _build(
    *,
    promotion_report: dict[str, Any] | None = None,
    release_notes_text: str | None = None,
) -> dict[str, Any]:
    return build_parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review(
        promotion_decision_gate_report=promotion_report or _promotion_report(),
        release_notes_text=release_notes_text if release_notes_text is not None else _release_notes_text(),
        generated_at="2026-05-29T00:00:00Z",
    )


def test_research_preview_release_notes_review_ready_but_default_promotion_held() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["releaseNotesDecision"]["researchPreviewReleaseNotesReady"] is True
    assert report["releaseNotesDecision"]["publicDefaultDecision"] == "hold_public_default_promotion"
    assert report["releaseNotesDecision"]["defaultSurfaceDecision"] == "do_not_enable_default_ask_or_default_mcp"
    assert report["counts"]["releaseNotesPathAllowedRows"] == 1
    assert report["counts"]["publicDefaultPromotionReadyRows"] == 0
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["generalRcReadyRows"] == 0
    assert report["counts"]["corpusScaleClaimProvenRows"] == 0
    assert report["counts"]["forbiddenPhraseRows"] == 0
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RESEARCH_PREVIEW_RELEASE_NOTES_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_research_preview_release_notes_review_blocks_when_promotion_gate_not_ready() -> None:
    promotion_report = _promotion_report(status="blocked")
    report = _build(promotion_report=promotion_report)

    assert report["status"] == "blocked"
    assert "promotion_gate_not_ready" in report["gate"]["semanticViolations"]


def test_research_preview_release_notes_review_blocks_missing_required_phrase() -> None:
    text = _release_notes_text().replace("public/default promotion remains held", "promotion is not expanded")
    report = _build(release_notes_text=text)

    assert report["status"] == "blocked"
    assert report["counts"]["missingRequiredPhraseRows"] == 1
    assert "missing_required_phrase:public/default promotion remains held" in report["gate"]["semanticViolations"]


def test_research_preview_release_notes_review_blocks_forbidden_promotion_phrase() -> None:
    report = _build(release_notes_text=_release_notes_text() + "\nThis is production-ready.\n")

    assert report["status"] == "blocked"
    assert report["counts"]["forbiddenPhraseRows"] == 1
    assert "forbidden_phrase:production-ready" in report["gate"]["semanticViolations"]


def test_research_preview_release_notes_review_blocks_private_path_marker() -> None:
    marker = "/" + "Users" + "/example/private"
    report = _build(release_notes_text=_release_notes_text() + f"\n{marker}\n")

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "release_notes_or_gate_private_path_marker" in report["gate"]["semanticViolations"]


def test_research_preview_release_notes_review_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build()

    paths = write_parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Research Preview Release Notes Review"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RESEARCH_PREVIEW_RELEASE_NOTES_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
