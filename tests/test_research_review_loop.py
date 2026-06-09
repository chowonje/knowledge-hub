from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from click.testing import CliRunner
from jsonschema import Draft202012Validator

from knowledge_hub.application.research_review_loop import RESEARCH_REVIEW_LOOP_SCHEMA
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.interfaces.cli.commands.review_loop_cmd import review_loop_group
from knowledge_hub.interfaces.cli.main import cli


SCHEMA_PATH: Final = Path(__file__).resolve().parents[1] / "docs" / "schemas" / "research-review-loop-result.v1.json"


class _ReviewLoopDb:
    def list_claim_cards(self, *, source_kind: str, limit: int):
        assert source_kind == "paper"
        assert limit >= 1
        return [
            {
                "claim_card_id": "claim-card-v1:paper:claim-supported",
                "claim_id": "claim-supported",
                "claim_text": "Transformers use attention to relate token positions.",
                "paper_id": "1706.03762",
                "source_id": "1706.03762",
                "task_canonical": "sequence transduction",
                "dataset_canonical": "WMT14",
                "metric_canonical": "BLEU",
            },
            {
                "claim_card_id": "claim-card-v1:paper:claim-unreviewed",
                "claim_id": "claim-unreviewed",
                "claim_text": "The paper proves all sequence models are obsolete.",
                "paper_id": "1706.03762",
                "source_id": "1706.03762",
                "task_canonical": "sequence modeling",
            },
        ]

    def list_claim_card_source_refs(self, *, claim_card_id: str = ""):
        match claim_card_id:
            case "claim-card-v1:paper:claim-supported":
                return [{"source_card_id": "paper-card-v2:1706.03762"}]
            case "claim-card-v1:paper:claim-unreviewed":
                return [{"source_card_id": "paper-card-v2:1706.03762"}]
            case "":
                return []
            case unreachable:
                raise AssertionError(f"unexpected claim_card_id: {unreachable}")

    def list_evidence_anchors_v2(self, *, card_id: str, claim_ids: list[str]):
        assert card_id == "paper-card-v2:1706.03762"
        match claim_ids:
            case ["claim-supported"]:
                return [
                    {
                        "anchor_id": "span-supported",
                        "source_id": "1706.03762",
                        "locator": "section:3",
                        "snippet_hash": "hash-supported",
                        "quote": "Attention relates different positions in a sequence.",
                    }
                ]
            case ["claim-unreviewed"]:
                return []
            case unreachable:
                raise AssertionError(f"unexpected claim ids: {unreachable}")


class _ReviewLoopKhub:
    def __init__(self, db: _ReviewLoopDb) -> None:
        self._db = db

    def sqlite_db(self) -> _ReviewLoopDb:
        return self._db


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
            "1706.03762",
            "--decision-file",
            str(decision_file),
            "--json",
        ],
        obj={"khub": _ReviewLoopKhub(_ReviewLoopDb())},
    )

    # Then: the payload is schema-valid and excludes unreviewed proposed claims.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(payload)
    assert validate_payload(payload, RESEARCH_REVIEW_LOOP_SCHEMA, strict=True).ok
    assert payload["schema"] == "knowledge-hub.research-review-loop.result.v1"
    assert payload["sourceScope"]["explicitSourceIds"] == ["1706.03762"]
    assert payload["contextPackPreview"]["reviewedClaimIds"] == ["claim-supported"]
    assert payload["contextPackPreview"]["excludedUnreviewedClaimIds"] == ["claim-unreviewed"]
    assert payload["contextPackPreview"]["canonicalWriteAllowed"] is False
    assert payload["counts"]["unsupportedCanonicalRows"] == 0


def test_research_review_loop_requires_explicit_paper_scope() -> None:
    # Given: no explicit paper id.
    runner = CliRunner()

    # When: the report command is invoked without a source scope.
    result = runner.invoke(
        review_loop_group,
        ["report", "--json"],
        obj={"khub": _ReviewLoopKhub(_ReviewLoopDb())},
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
