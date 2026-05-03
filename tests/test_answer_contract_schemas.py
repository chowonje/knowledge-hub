from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from knowledge_hub.core.schema_validator import validate_payload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "docs" / "schemas" / "fixtures"

CONTRACTS = [
    (
        "knowledge-hub.evidence-packet.v1",
        "evidence-packet.v1.json",
        "evidence-packet.v1.fixture.json",
    ),
    (
        "knowledge-hub.answer-contract.v1",
        "answer-contract.v1.json",
        "answer-contract.v1.fixture.json",
    ),
    (
        "knowledge-hub.verification-verdict.v1",
        "verification-verdict.v1.json",
        "verification-verdict.v1.fixture.json",
    ),
]


def _load_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(("schema_id", "schema_name", "fixture_name"), CONTRACTS)
def test_answer_architecture_contract_fixtures_validate(
    schema_id: str,
    schema_name: str,
    fixture_name: str,
) -> None:
    assert schema_name
    fixture = _load_fixture(fixture_name)

    result = validate_payload(fixture, schema_id, strict=True)

    assert result.ok, result.errors


def test_verification_verdict_rejects_rewrite_and_retry_verdict() -> None:
    schema_id = "knowledge-hub.verification-verdict.v1"
    fixture = _load_fixture("verification-verdict.v1.fixture.json")
    payload = copy.deepcopy(fixture)
    payload["verdict"] = "rewrite_and_retry"

    result = validate_payload(payload, schema_id, strict=True)

    assert not result.ok
    assert any("/verdict" in error for error in result.errors)


def test_verification_verdict_rejects_rewrite_and_retry_action() -> None:
    schema_id = "knowledge-hub.verification-verdict.v1"
    fixture = _load_fixture("verification-verdict.v1.fixture.json")
    payload = copy.deepcopy(fixture)
    payload["rewrite_and_retry"] = {"attempt": 1}
    payload["recommended_action"] = "rewrite_and_retry"

    result = validate_payload(payload, schema_id, strict=True)

    assert not result.ok
    assert any("rewrite_and_retry" in error for error in result.errors)
