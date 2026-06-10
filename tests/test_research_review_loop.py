from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
from jsonschema import Draft202012Validator

from knowledge_hub.application.research_review_loop import RESEARCH_REVIEW_LOOP_SCHEMA
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.interfaces.cli.commands.review_loop_cmd import review_loop_group
from knowledge_hub.interfaces.cli.main import cli
from tests.research_review_loop_fixtures import (
    EVALUATION_PAPER_ID,
    SCHEMA_PATH,
    SUPPORTED_CLAIM,
    ReviewLoopDb,
    ReviewLoopKhub,
    claim_text_hash,
)


def test_research_review_loop_report_excludes_unreviewed_claims_from_context_pack(tmp_path: Path) -> None:
    # Given: an explicit paper scope with one reviewed claim and one unreviewed claim.
    decision_file = tmp_path / "decisions.json"
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-1",
                        "targetType": "claim",
                        "targetId": "claim-supported",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:00:00Z",
                        "reason": "Evidence span directly supports the claim.",
                        "claimTextHash": claim_text_hash(SUPPORTED_CLAIM),
                        "snippetHashes": ["hash-supported"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: the user drives the labs CLI through its real report surface.
    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--decision-file",
            str(decision_file),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: the payload is schema-valid and excludes unreviewed proposed claims.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(payload)
    assert validate_payload(payload, RESEARCH_REVIEW_LOOP_SCHEMA, strict=True).ok
    assert payload["schema"] == "knowledge-hub.research-review-loop.result.v1"
    assert payload["sourceScope"]["explicitSourceIds"] == [EVALUATION_PAPER_ID]
    assert payload["contextPackPreview"]["reviewedClaimIds"] == ["claim-supported"]
    assert payload["contextPackPreview"]["excludedUnreviewedClaimIds"] == [
        "claim-rejected",
        "claim-unsupported",
        "claim-unreviewed",
        "claim-missing-locator",
    ]
    assert payload["contextPackPreview"]["canonicalWriteAllowed"] is False
    assert payload["counts"]["unsupportedCanonicalRows"] == 0


def test_research_review_loop_requires_explicit_paper_scope() -> None:
    # Given: no explicit paper id.
    runner = CliRunner()

    # When: the report command is invoked without a source scope.
    result = runner.invoke(
        review_loop_group,
        ["report", "--json"],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: the command fails closed instead of broad-scanning papers.
    assert result.exit_code != 0
    assert "at least one --paper-id is required" in result.output


def test_labs_help_exposes_review_loop_without_default_promotion() -> None:
    # Given: the default CLI and labs surfaces.
    runner = CliRunner()

    # When: help text is rendered.
    default_help = runner.invoke(cli, ["--help"])
    labs_help = runner.invoke(cli, ["labs", "--help"])

    # Then: review-loop is labs-only.
    assert default_help.exit_code == 0
    assert labs_help.exit_code == 0
    assert "review-loop" not in default_help.output
    assert "review-loop" in labs_help.output
