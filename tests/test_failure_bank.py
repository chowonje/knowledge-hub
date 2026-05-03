from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from knowledge_hub.application.failure_bank import (
    find_latest_answer_loop_failure_cards,
    link_failure_bank_to_eval_cases,
    list_failure_bank,
    sync_failure_bank_from_answer_loop,
)
from knowledge_hub.core.schema_validator import validate_payload


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def test_sync_failure_bank_imports_and_dedupes_answer_loop_cards(tmp_path: Path):
    cards_path = _write_jsonl(
        tmp_path / "runs" / "answer_loop" / "run_01" / "answer_loop_failure_cards.jsonl",
        [
            {
                "packetRef": "packet-1",
                "bucket": "groundedness_failure",
                "query": "What changed?",
                "answerBackend": "codex_mcp",
                "reason": "weak evidence",
                "predLabel": "good",
                "predGroundedness": "bad",
                "predUsefulness": "good",
                "predReadability": "good",
                "predSourceAccuracy": "bad",
                "predShouldAbstain": "0",
            }
        ],
    )
    bank_path = tmp_path / "failures" / "failure_bank.jsonl"

    first = sync_failure_bank_from_answer_loop(
        failure_cards_path=cards_path,
        bank_path=bank_path,
        now=datetime.fromisoformat("2026-04-26T00:00:00+00:00"),
    )
    second = sync_failure_bank_from_answer_loop(
        failure_cards_path=cards_path,
        bank_path=bank_path,
        now=datetime.fromisoformat("2026-04-27T00:00:00+00:00"),
    )

    assert validate_payload(first, first["schema"], strict=True).ok
    assert validate_payload(second, second["schema"], strict=True).ok
    assert first["createdCount"] == 1
    assert second["createdCount"] == 0
    assert second["updatedCount"] == 1
    assert second["recordCount"] == 1
    rows = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["status"] == "open"
    assert rows[0]["observationCount"] == 2
    assert rows[0]["bucket"] == "groundedness_failure"


def test_list_failure_bank_filters_by_status_and_bucket(tmp_path: Path):
    bank_path = _write_jsonl(
        tmp_path / "failure_bank.jsonl",
        [
            {"failureId": "fb_1", "status": "open", "bucket": "groundedness_failure"},
            {"failureId": "fb_2", "status": "fixed", "bucket": "source_accuracy_failure"},
        ],
    )

    payload = list_failure_bank(
        bank_path=bank_path,
        status="open",
        bucket="groundedness_failure",
    )

    assert validate_payload(payload, payload["schema"], strict=True).ok
    assert payload["status"] == "ok"
    assert payload["recordCount"] == 1
    assert payload["items"][0]["failureId"] == "fb_1"


def test_list_failure_bank_skips_malformed_jsonl_lines(tmp_path: Path):
    bank_path = tmp_path / "failure_bank.jsonl"
    bank_path.write_text(
        json.dumps({"failureId": "fb_1", "status": "open", "bucket": "groundedness_failure"})
        + "\n{bad-json}\n",
        encoding="utf-8",
    )

    payload = list_failure_bank(bank_path=bank_path)

    assert payload["status"] == "ok"
    assert payload["recordCount"] == 1
    assert payload["warnings"]
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_list_failure_bank_skips_invalid_utf8_lines(tmp_path: Path):
    bank_path = tmp_path / "failure_bank.jsonl"
    bank_path.write_bytes(
        b"\xff\n"
        + json.dumps({"failureId": "fb_1", "status": "open", "bucket": "groundedness_failure"}).encode("utf-8")
        + b"\n"
    )

    payload = list_failure_bank(bank_path=bank_path)

    assert payload["status"] == "ok"
    assert payload["recordCount"] == 1
    assert any("invalid utf-8" in warning for warning in payload["warnings"])
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_find_latest_answer_loop_failure_cards_prefers_latest_alias(tmp_path: Path):
    runs_root = tmp_path / "runs"
    older = _write_jsonl(
        runs_root / "answer_loop" / "run_01" / "answer_loop_failure_cards.jsonl",
        [{"packetRef": "old"}],
    )
    latest = _write_jsonl(
        runs_root / "answer_loop" / "latest" / "answer_loop_failure_cards.jsonl",
        [{"packetRef": "latest"}],
    )

    assert older.exists()
    assert find_latest_answer_loop_failure_cards(runs_root=runs_root) == latest


def test_link_failure_bank_to_eval_cases_matches_exact_query(tmp_path: Path):
    bank_path = _write_jsonl(
        tmp_path / "failure_bank.jsonl",
        [
            {
                "failureId": "fb_1",
                "status": "open",
                "bucket": "groundedness_failure",
                "query": "document memory와 chunk retrieval의 차이를 쉽게 설명해줘",
            }
        ],
    )
    registry_path = _write_jsonl(
        tmp_path / "eval_cases.jsonl",
        [
            {
                "schema": "knowledge-hub.eval-case.v1",
                "evalCaseId": "ec_1",
                "lane": "user_answer_eval_queries_v1",
                "sourceType": "vault",
                "scenarioType": "generic",
                "query": "document memory와 chunk retrieval의 차이를 쉽게 설명해줘",
                "expectedFamily": "",
                "expectedAnswerMode": "",
                "expectedSourceScope": "unspecified",
                "shouldAbstain": False,
                "expectedEvidencePolicy": "default",
                "successCriteria": {},
                "tags": [],
                "status": "active",
                "provenance": {},
                "createdAt": "2026-04-29T00:00:00+00:00",
                "updatedAt": "2026-04-29T00:00:00+00:00",
            }
        ],
    )

    payload = link_failure_bank_to_eval_cases(
        bank_path=bank_path,
        registry_path=registry_path,
        now=datetime.fromisoformat("2026-04-29T00:00:00+00:00"),
    )

    assert validate_payload(payload, payload["schema"], strict=True).ok
    assert payload["linkedCount"] == 1
    assert payload["statusCounts"] == {"linked": 1}
    rows = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["linkedEvalCaseId"] == "ec_1"
    assert rows[0]["linkedEvalCaseLane"] == "user_answer_eval_queries_v1"
    assert rows[0]["rootCauseHypothesis"]
    assert rows[0]["recommendedFix"]
