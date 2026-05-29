from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_vision_bottleneck_definition_review import (
    DEFAULT_DRAFT_PR_POST_OPEN_REVIEW_REPORT,
    DEFAULT_LABS_RELEASE_GATE_REPORT,
    DEFAULT_PRODUCT_DEFINITION_DOC,
    DEFAULT_PUBLIC_DEFAULT_PROMOTION_GATE_REPORT,
    KNOWLEDGEOS_V01_RC_VISION_BOTTLENECK_DEFINITION_REVIEW_SCHEMA_ID,
    READY_DECISION,
    build_knowledgeos_v01_rc_vision_bottleneck_definition_review,
    write_knowledgeos_v01_rc_vision_bottleneck_definition_review,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _product_text() -> str:
    return DEFAULT_PRODUCT_DEFINITION_DOC.read_text(encoding="utf-8")


def _pr_report(**updates: Any) -> dict[str, Any]:
    payload = _json(DEFAULT_DRAFT_PR_POST_OPEN_REVIEW_REPORT)
    for key, value in updates.items():
        payload[key] = value
    return payload


def _public_report(**updates: Any) -> dict[str, Any]:
    payload = _json(DEFAULT_PUBLIC_DEFAULT_PROMOTION_GATE_REPORT)
    for key, value in updates.items():
        payload[key] = value
    return payload


def _labs_report(**updates: Any) -> dict[str, Any]:
    payload = _json(DEFAULT_LABS_RELEASE_GATE_REPORT)
    for key, value in updates.items():
        payload[key] = value
    return payload


def _build(**updates: Any) -> dict[str, Any]:
    inputs = {
        "product_definition_text": _product_text(),
        "draft_pr_post_open_review_report": _pr_report(),
        "public_default_promotion_gate_report": _public_report(),
        "labs_release_gate_report": _labs_report(),
        "generated_at": "2026-05-29T00:00:00Z",
    }
    inputs.update(updates)
    return build_knowledgeos_v01_rc_vision_bottleneck_definition_review(**inputs)


def test_vision_bottleneck_definition_ready_with_narrow_scope_decision() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["productDecision"]["decisionLabel"] == "narrow_scope"
    assert report["productDecision"]["publicDefaultDecision"] == "hold_public_default_promotion"
    assert report["counts"]["readyForHumanReviewRows"] == 1
    assert report["counts"]["readyForMergeRows"] == 0
    assert report["counts"]["publicDefaultPromotionReadyRows"] == 0
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["corpusScaleClaimProvenRows"] == 0
    assert report["counts"]["v01RcBottleneckRows"] == 2
    assert report["counts"]["finalVisionBottleneckRows"] == 3
    assert report["gate"]["publicDefaultPromotionAllowed"] is False
    assert validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_VISION_BOTTLENECK_DEFINITION_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_vision_bottleneck_definition_blocks_when_product_definition_phrase_missing() -> None:
    report = _build(product_definition_text="KnowledgeOS")

    assert report["status"] == "blocked"
    assert any(
        blocker.startswith("missing_product_definition_phrase:")
        for blocker in report["gate"]["semanticViolations"]
    )


def test_vision_bottleneck_definition_blocks_when_pr_review_not_ready() -> None:
    report = _build(draft_pr_post_open_review_report=_pr_report(status="blocked"))

    assert report["status"] == "blocked"
    assert "draft_pr_post_open_review_not_ready" in report["gate"]["semanticViolations"]


def test_vision_bottleneck_definition_blocks_when_ci_not_green() -> None:
    pr_report = _pr_report()
    pr_report["counts"]["ciCheckSuccessRows"] = 6
    report = _build(draft_pr_post_open_review_report=pr_report)

    assert report["status"] == "blocked"
    assert "ci_checks_not_green" in report["gate"]["semanticViolations"]


def test_vision_bottleneck_definition_blocks_public_default_promotion_ready() -> None:
    public_report = _public_report()
    public_report["counts"]["publicDefaultPromotionReadyRows"] = 1
    report = _build(public_default_promotion_gate_report=public_report)

    assert report["status"] == "blocked"
    assert "public_default_promotion_ready_unexpected" in report["gate"]["semanticViolations"]


def test_vision_bottleneck_definition_blocks_labs_release_hygiene_failure() -> None:
    labs_report = _labs_report()
    labs_report["counts"]["publicHygieneIssueRows"] = 1
    report = _build(labs_release_gate_report=labs_report)

    assert report["status"] == "blocked"
    assert "public_hygiene_issues_present" in report["gate"]["semanticViolations"]


def test_vision_bottleneck_definition_blocks_private_path_marker() -> None:
    marker = "/" + "Users" + "/example/private"
    report = _build(product_definition_text=_product_text() + "\n" + marker)

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "vision_bottleneck_review_private_path_marker" in report["gate"]["semanticViolations"]


def test_vision_bottleneck_definition_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build()

    paths = write_knowledgeos_v01_rc_vision_bottleneck_definition_review(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# KnowledgeOS v0.1 RC Vision Bottleneck Definition Review"
    )
    assert validate_payload(
        parsed,
        KNOWLEDGEOS_V01_RC_VISION_BOTTLENECK_DEFINITION_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
