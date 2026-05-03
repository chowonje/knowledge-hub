from __future__ import annotations

import json
import os
from pathlib import Path

from knowledge_hub.application.eval_center import build_eval_center_summary
from knowledge_hub.core.schema_validator import validate_payload


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _seed_source_quality(runs_root: Path) -> Path:
    older = _write_json(
        runs_root / "source_quality_battery_20260425_010000" / "source_quality_battery_summary.json",
        {"created_at": "2026-04-25T01:00:00+00:00", "per_source": {}},
    )
    latest = _write_json(
        runs_root / "source_quality_battery_20260426_010143" / "source_quality_battery_summary.json",
        {
            "created_at": "2026-04-26T01:28:13+00:00",
            "gate_mode": "stub_hard",
            "retrieval_mode": "hybrid",
            "top_k": 6,
            "hard_gates": {"route_correctness": {"paper": {"value": 1.0, "passed": True}}},
            "soft_gates": {"paper_citation_correctness": 1.0},
            "per_source": {
                "paper": {
                    "rows": 24,
                    "route_correctness": 1.0,
                    "no_result_rate": 0.375,
                    "runtime_used_counts": {"ask_v2": 24},
                    "fallback_reason_counts": {"section_blocked_to_claim_cards": 5},
                    "legacy_runtime_rate": 0.0,
                    "capability_missing_rate": 0.0,
                    "forced_legacy_rate": 0.0,
                    "citation_correctness_soft": 1.0,
                }
            },
        },
    )
    os.utime(older, (1, 1))
    os.utime(latest, (2, 2))
    reports = runs_root / "reports"
    _write_json(
        reports / "source_quality_observation_latest.json",
        {
            "schema": "knowledge-hub.source-quality-observation.report.v1",
            "decision": "ready_for_hard_gate_review",
            "blockers": [],
            "run_count": 7,
            "required_runs": 7,
            "latest_run_dir": str(latest.parent),
        },
    )
    _write_json(
        reports / "source_quality_detail_observation_latest.json",
        {
            "schema": "knowledge-hub.source-quality-detail-observation.report.v1",
            "decision": "not_ready_for_detail_gate_review",
            "blockers": ["vault_abstention_correctness_need_7_numeric_points_have_6"],
        },
    )
    _write_json(reports / "source_quality_trend_latest.json", {"schema": "trend", "status": "ok"})
    _write_json(reports / "legacy_runtime_readiness_latest.json", {"schema": "legacy", "decision": "ready"})
    return latest


def _seed_answer_loop(runs_root: Path) -> Path:
    run_dir = runs_root / "answer_loop" / "manual-smoke-20260412" / "baseline_run_02"
    _write_json(
        run_dir / "answer_loop_collect_manifest.json",
        {
            "schema": "knowledge-hub.answer-loop.collect.result.v1",
            "status": "ok",
            "rowCount": 5,
            "packetCount": 5,
            "request": {
                "queriesPath": "eval/knowledgeos/runs/answer_loop/manual-smoke-20260412/queries_small_v1.csv",
                "answerBackends": ["codex_mcp"],
                "backendModels": {"codex_mcp": "gpt-5.4"},
                "retrievalMode": "hybrid",
                "topK": 8,
            },
            "artifactPaths": {},
        },
    )
    _write_json(
        run_dir / "answer_loop_judge_manifest.json",
        {
            "schema": "knowledge-hub.answer-loop.judge.result.v1",
            "status": "ok",
            "judgeProvider": "openai",
            "judgeModel": "gpt-4.1-nano",
            "rowCount": 5,
            "artifactPaths": {},
        },
    )
    return _write_json(
        run_dir / "answer_loop_summary.json",
        {
            "schema": "knowledge-hub.answer-loop.summary.result.v1",
            "status": "ok",
            "rowCount": 5,
            "overall": {
                "predLabelScore": 0.8,
                "predGroundednessScore": 0.8,
                "predSourceAccuracyScore": 0.8,
                "abstainAgreement": 1.0,
            },
            "backends": {},
            "failureBucketCounts": {"groundedness_failure": 1},
            "failureCardCount": 1,
        },
    )


def test_build_eval_center_summary_rolls_up_current_artifacts(tmp_path: Path):
    runs_root = tmp_path / "eval" / "knowledgeos" / "runs"
    queries_dir = tmp_path / "eval" / "knowledgeos" / "queries"
    latest_source_run = _seed_source_quality(runs_root)
    answer_summary = _seed_answer_loop(runs_root)
    _write_text(queries_dir / "paper_default_eval_queries_v1.csv", "query,source\np1,paper\n")
    _write_text(queries_dir / "user_answer_eval_queries_v1.csv", "query,source\n\"a,b\",paper,extra\n")

    payload = build_eval_center_summary(
        runs_root=runs_root,
        queries_dir=queries_dir,
        failure_bank_path=tmp_path / "missing_failure_bank.jsonl",
        repo_root=tmp_path,
        generated_at="2026-04-26T00:00:00+00:00",
    )

    assert payload["schema"] == "knowledge-hub.eval-center.summary.result.v1"
    assert payload["status"] == "warn"
    assert validate_payload(payload, payload["schema"], strict=True).ok
    assert payload["sourceQuality"]["latestRun"]["path"] == str(latest_source_run)
    assert payload["sourceQuality"]["latestRun"]["modifiedAt"]
    assert payload["sourceQuality"]["baseObservation"]["decision"] == "ready_for_hard_gate_review"
    assert payload["sourceQuality"]["perSource"][0]["routeCorrectness"] == 1.0
    assert payload["answerLoop"]["latestAlias"]["exists"] is False
    assert payload["answerLoop"]["summary"]["path"] == str(answer_summary)
    assert payload["answerLoop"]["summary"]["modifiedAt"]
    assert payload["answerLoop"]["collect"]["backendModels"] == {"codex_mcp": "gpt-5.4"}
    assert payload["answerLoop"]["judge"]["judgeModel"] == "gpt-4.1-nano"
    assert payload["queryInventory"]["count"] == 2
    assert any("extra field" in warning for warning in payload["warnings"])
    assert not any("missing answer-loop latest alias" in warning for warning in payload["warnings"])
    assert any(gap["id"] == "failure_bank" for gap in payload["gaps"])
    assert "answer_loop_latest_alias" not in {gap["id"] for gap in payload["gaps"]}
    assert payload["operatorBrief"]["summary"]["priority"] == "answer_loop_triage"
    assert any(section["id"] == "source_quality" for section in payload["operatorBrief"]["sections"])
    assert any(finding["part"] == "query_inventory" for finding in payload["operatorBrief"]["findings"])
    assert not any(
        finding["title"] == "Latest answer-loop alias is missing"
        for finding in payload["operatorBrief"]["findings"]
    )


def test_build_eval_center_summary_uses_configured_failure_bank_path(tmp_path: Path):
    runs_root = tmp_path / "eval" / "knowledgeos" / "runs"
    queries_dir = tmp_path / "eval" / "knowledgeos" / "queries"
    _seed_source_quality(runs_root)
    _seed_answer_loop(runs_root)
    _write_text(queries_dir / "paper_default_eval_queries_v1.csv", "query,source\np1,paper\n")
    bank_path = tmp_path / "outside" / "failure_bank.jsonl"
    _write_text(
        bank_path,
        json.dumps({"failureId": "fb_1", "status": "open", "bucket": "groundedness_failure"}) + "\n",
    )

    payload = build_eval_center_summary(
        runs_root=runs_root,
        queries_dir=queries_dir,
        failure_bank_path=bank_path,
        repo_root=tmp_path,
        generated_at="2026-04-26T00:00:00+00:00",
    )

    assert payload["failureBank"]["path"] == str(bank_path)
    assert payload["failureBank"]["recordCount"] == 1
    assert "failure_bank" not in {gap["id"] for gap in payload["gaps"]}
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_build_eval_center_summary_warns_on_stale_source_quality(tmp_path: Path):
    runs_root = tmp_path / "eval" / "knowledgeos" / "runs"
    queries_dir = tmp_path / "eval" / "knowledgeos" / "queries"
    _seed_source_quality(runs_root)
    _seed_answer_loop(runs_root)
    _write_text(queries_dir / "paper_default_eval_queries_v1.csv", "query,source\np1,paper\n")

    payload = build_eval_center_summary(
        runs_root=runs_root,
        queries_dir=queries_dir,
        failure_bank_path=tmp_path / "missing_failure_bank.jsonl",
        repo_root=tmp_path,
        generated_at="2026-04-27T00:00:00+00:00",
        expected_source_quality_local_date="2026-04-27",
        freshness_timezone="Asia/Seoul",
    )

    assert payload["status"] == "warn"
    assert payload["sourceQuality"]["freshness"]["status"] == "stale"
    assert payload["sourceQuality"]["freshness"]["expectedLocalDate"] == "2026-04-27"
    assert any("source-quality latest run is stale" in warning for warning in payload["warnings"])
    source_section = next(section for section in payload["operatorBrief"]["sections"] if section["id"] == "source_quality")
    assert source_section["status"] == "failed"
    assert any(
        finding["part"] == "source_quality" and finding["severity"] == "P1"
        for finding in payload["operatorBrief"]["findings"]
    )
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_build_eval_center_summary_reports_eval_case_registry(tmp_path: Path):
    runs_root = tmp_path / "eval" / "knowledgeos" / "runs"
    queries_dir = tmp_path / "eval" / "knowledgeos" / "queries"
    _seed_source_quality(runs_root)
    _seed_answer_loop(runs_root)
    _write_text(queries_dir / "paper_default_eval_queries_v1.csv", "query,source\np1,paper\n")
    bank_path = tmp_path / "failure_bank.jsonl"
    _write_text(bank_path, json.dumps({"failureId": "fb_1", "status": "open", "bucket": "groundedness_failure"}) + "\n")
    registry_path = tmp_path / "eval_cases.jsonl"
    _write_text(
        registry_path,
        json.dumps(
            {
                "schema": "knowledge-hub.eval-case.v1",
                "evalCaseId": "ec_1",
                "lane": "paper_default_eval",
                "sourceType": "paper",
                "scenarioType": "concept_explainer",
                "query": "CNN을 쉽게 설명해줘",
                "status": "active",
                "createdAt": "2026-04-29T00:00:00+00:00",
                "updatedAt": "2026-04-29T00:00:00+00:00",
            }
        )
        + "\n",
    )

    payload = build_eval_center_summary(
        runs_root=runs_root,
        queries_dir=queries_dir,
        failure_bank_path=bank_path,
        eval_case_registry_path=registry_path,
        repo_root=tmp_path,
        generated_at="2026-04-26T00:00:00+00:00",
    )

    assert payload["evalCases"]["exists"] is True
    assert payload["evalCases"]["recordCount"] == 1
    assert "eval_cases_store" not in {gap["id"] for gap in payload["gaps"]}
    eval_section = next(section for section in payload["operatorBrief"]["sections"] if section["id"] == "eval_cases")
    assert eval_section["status"] == "ok"
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_build_eval_center_summary_warns_on_malformed_failure_bank(tmp_path: Path):
    runs_root = tmp_path / "eval" / "knowledgeos" / "runs"
    queries_dir = tmp_path / "eval" / "knowledgeos" / "queries"
    _seed_source_quality(runs_root)
    _seed_answer_loop(runs_root)
    _write_text(queries_dir / "paper_default_eval_queries_v1.csv", "query,source\np1,paper\n")
    bank_path = tmp_path / "failure_bank.jsonl"
    _write_text(bank_path, "{bad-json}\n")

    payload = build_eval_center_summary(
        runs_root=runs_root,
        queries_dir=queries_dir,
        failure_bank_path=bank_path,
        repo_root=tmp_path,
        generated_at="2026-04-26T00:00:00+00:00",
    )

    assert payload["failureBank"]["exists"] is True
    assert payload["failureBank"]["recordCount"] == 0
    assert any("failed to read failure bank" in warning for warning in payload["warnings"])
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_build_eval_center_summary_warns_on_invalid_utf8_failure_bank(tmp_path: Path):
    runs_root = tmp_path / "eval" / "knowledgeos" / "runs"
    queries_dir = tmp_path / "eval" / "knowledgeos" / "queries"
    _seed_source_quality(runs_root)
    _seed_answer_loop(runs_root)
    _write_text(queries_dir / "paper_default_eval_queries_v1.csv", "query,source\np1,paper\n")
    bank_path = tmp_path / "failure_bank.jsonl"
    bank_path.write_bytes(
        b"\xff\n"
        + json.dumps({"failureId": "fb_1", "status": "open", "bucket": "groundedness_failure"}).encode("utf-8")
        + b"\n"
    )

    payload = build_eval_center_summary(
        runs_root=runs_root,
        queries_dir=queries_dir,
        failure_bank_path=bank_path,
        repo_root=tmp_path,
        generated_at="2026-04-26T00:00:00+00:00",
    )

    assert payload["failureBank"]["exists"] is True
    assert payload["failureBank"]["recordCount"] == 1
    assert any("invalid utf-8" in warning for warning in payload["warnings"])
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_build_eval_center_summary_reports_missing_roots(tmp_path: Path):
    payload = build_eval_center_summary(
        runs_root=tmp_path / "missing-runs",
        queries_dir=tmp_path / "missing-queries",
        failure_bank_path=tmp_path / "missing_failure_bank.jsonl",
        repo_root=tmp_path,
        generated_at="2026-04-26T00:00:00+00:00",
    )

    assert payload["status"] == "missing"
    assert validate_payload(payload, payload["schema"], strict=True).ok
    assert payload["queryInventory"]["exists"] is False
    assert payload["answerLoop"]["summary"]["exists"] is False
    assert any("missing queries dir" in warning for warning in payload["warnings"])


def test_build_eval_center_summary_warns_when_answer_loop_root_exists_without_summary(tmp_path: Path):
    runs_root = tmp_path / "eval" / "knowledgeos" / "runs"
    queries_dir = tmp_path / "eval" / "knowledgeos" / "queries"
    _seed_source_quality(runs_root)
    (runs_root / "answer_loop").mkdir(parents=True, exist_ok=True)
    _write_text(queries_dir / "paper_default_eval_queries_v1.csv", "query,source\np1,paper\n")

    payload = build_eval_center_summary(
        runs_root=runs_root,
        queries_dir=queries_dir,
        failure_bank_path=tmp_path / "missing_failure_bank.jsonl",
        repo_root=tmp_path,
        generated_at="2026-04-26T00:00:00+00:00",
    )

    assert payload["answerLoop"]["latestAlias"]["exists"] is False
    assert payload["answerLoop"]["summary"]["exists"] is False
    assert any("missing answer-loop summary under" in warning for warning in payload["warnings"])
    assert any(gap["id"] == "answer_loop_latest_alias" for gap in payload["gaps"])
    answer_loop_section = next(
        section for section in payload["operatorBrief"]["sections"] if section["id"] == "answer_loop"
    )
    assert "answer-loop latest alias is missing" in answer_loop_section["problem"]
    assert validate_payload(payload, payload["schema"], strict=True).ok
