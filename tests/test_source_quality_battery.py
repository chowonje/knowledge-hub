from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "eval/knowledgeos/scripts/run_source_quality_battery.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("source_quality_battery_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_source_rows_computes_per_source_metrics():
    module = _load_script()

    paper = module.summarize_source_rows(
        "paper",
        [
            {
                "family_match": "1",
                "no_result": "0",
                "citation_support_match": "1",
                "actual_runtime_used": "ask_v2",
                "actual_fallback_reason": "",
            },
            {
                "family_match": "0",
                "no_result": "1",
                "citation_support_match": "0",
                "actual_runtime_used": "legacy",
                "actual_fallback_reason": "ask_v2_not_used",
            },
        ],
    )
    vault = module.summarize_source_rows(
        "vault",
        [
            {
                "family_match": "1",
                "no_result": "0",
                "citation_count": "2",
                "stale_citation_count": "1",
                "expected_answer_mode": "concept_explainer",
                "actual_runtime_used": "legacy",
                "actual_fallback_reason": "ask_v2_capability_missing",
            },
            {
                "family_match": "1",
                "no_result": "1",
                "citation_count": "1",
                "stale_citation_count": "0",
                "expected_answer_mode": "abstain",
                "answer_mode_match": "1",
                "actual_runtime_used": "ask_v2",
                "actual_fallback_reason": "",
            },
        ],
    )
    web = module.summarize_source_rows(
        "web",
        [
            {
                "family_match": "1",
                "no_result": "0",
                "query_type": "temporal",
                "recency_violation": "1",
                "latest_source_age_days": "400",
                "actual_runtime_used": "ask_v2",
                "actual_fallback_reason": "",
            },
            {
                "family_match": "1",
                "no_result": "0",
                "query_type": "temporal",
                "recency_violation": "0",
                "latest_source_age_days": "20",
                "actual_runtime_used": "legacy",
                "actual_fallback_reason": "ask_v2_not_used",
            },
        ],
    )

    assert paper["route_correctness"] == 0.5
    assert paper["citation_correctness_soft"] == 0.5
    assert paper["legacy_runtime_rate"] == 0.5
    assert paper["forced_legacy_rate"] == 0.5
    assert vault["route_correctness"] == 1.0
    assert vault["stale_citation_rate"] == 0.333333
    assert vault["abstention_correctness_soft"] == 1.0
    assert vault["capability_missing_rate"] == 0.5
    assert web["route_correctness"] == 1.0
    assert web["recency_violation_soft"] == 0.5
    assert web["latest_source_age_days_p50"] == 400.0
    assert web["legacy_runtime_rate"] == 0.5


def test_hard_and_soft_gate_summary_shapes():
    module = _load_script()
    summaries = {
        "paper": {"route_correctness": 0.9, "citation_correctness_soft": 0.8},
        "vault": {"route_correctness": 0.95, "stale_citation_rate": 0.0, "abstention_correctness_soft": 1.0},
        "web": {"route_correctness": 0.75, "recency_violation_soft": 0.5},
    }

    hard = module._hard_gates(summaries, route_threshold=0.8, vault_stale_threshold=0.0)
    soft = module._soft_gates(summaries)

    assert hard["route_correctness"]["paper"]["passed"] is True
    assert hard["route_correctness"]["web"]["passed"] is False
    assert hard["vault_stale_citation_rate"]["vault"]["passed"] is True
    assert soft["paper_citation_correctness"] == 0.8
    assert soft["vault_abstention_correctness"] == 1.0
    assert soft["web_recency_violation"] == 0.5


def test_relative_output_path_falls_back_for_external_run_dirs(tmp_path: Path):
    module = _load_script()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external_output = tmp_path / "runs" / "battery" / "paper_default_eval.csv"
    external_output.parent.mkdir(parents=True)

    rendered = module._relative_output_path(external_output, repo_root)

    assert rendered == os.path.relpath(external_output, repo_root)
