from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path("eval/knowledgeos/scripts/check_answer_quality_gate.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("check_answer_quality_gate", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_answer_quality_gate_fixture_passes():
    module = _load_module()
    cases = module._read_cases(Path("eval/knowledgeos/fixtures/answer_quality_golden_cases.json"))

    payload = module.run_answer_quality_gate(
        cases,
        cases_path=Path("eval/knowledgeos/fixtures/answer_quality_golden_cases.json"),
        min_cases=4,
    )

    assert payload["status"] == "ok"
    assert payload["caseCount"] == 4
    assert payload["passedCaseCount"] == 4
    assert payload["failedCaseCount"] == 0


def test_answer_quality_gate_catches_unexpected_signal_citation():
    module = _load_module()
    case = {
        "case_id": "bad_signal_expectation",
        "answer": "Learning graph proves the answer.",
        "evidence": [
            {
                "title": "Learning edge",
                "excerpt": "Learning graph edge.",
                "source_id": "learning_edge:rag:1",
                "source_type": "learning_edge",
                "source_content_hash": "hash-edge",
                "span_locator": "chars:0-20",
            }
        ],
        "verification": {"status": "verified", "unsupportedClaimCount": 0, "needsCaution": False},
        "expected": {
            "status": "pass",
            "coverage_status": "complete",
            "citation_count": 1,
            "retrieval_signal_count": 0,
            "verification_verdict": "pass",
        },
    }

    result = module.evaluate_answer_quality_case(case)

    assert result["status"] == "fail"
    assert "coverage_status:none!=complete" in result["errors"]
    assert "citation_count:0!=1" in result["errors"]
