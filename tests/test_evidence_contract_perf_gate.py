from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from knowledge_hub.ai.answer_contracts import (
    build_answer_contract,
    build_evidence_packet_contract,
    build_verification_verdict,
)


SCRIPT_PATH = Path("eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("run_evidence_contract_perf_gate", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _case(
    case_id: str,
    *,
    query: str = "alpha?",
    source: str = "vault",
    expected_min_citation_count: int = 1,
    expected_abstain: bool = False,
) -> dict:
    return {
        "case_id": case_id,
        "query": query,
        "source": source,
        "expected_answer_mode": "abstain" if expected_abstain else "answer",
        "expected_min_citation_count": expected_min_citation_count,
        "expected_abstain": expected_abstain,
        "required_contracts": ["evidencePacketContract", "answerContract", "verificationVerdict"],
    }


def _contract_payload(
    *,
    query: str = "alpha?",
    answer: str = "Alpha evidence supports grounded answers.",
    source_type: str = "vault",
    answerable: bool = True,
    verification: dict | None = None,
    evidence: list[dict] | None = None,
    rewrite: dict | None = None,
) -> dict:
    evidence = evidence if evidence is not None else [
        {
            "title": "Alpha",
            "excerpt": "Alpha evidence supports grounded answers.",
            "citation_label": "S1",
            "citation_target": "vault:Alpha.md",
            "source_id": "vault:Alpha.md",
            "source_ref": "vault:Alpha.md",
            "source_type": source_type,
            "source_content_hash": "hash-alpha",
            "span_locator": "chars:10-52",
            "score": 0.9,
        }
    ]
    packet = SimpleNamespace(
        evidence=evidence,
        citations=[
            {"label": item.get("citation_label") or "S1", "target": item.get("citation_target"), "kind": "source"}
            for item in evidence
        ],
        evidence_packet={
            "answerable": bool(answerable),
            "answerableDecisionReason": "fixture answerable" if answerable else "fixture abstain",
        },
        evidence_policy={"policyKey": "test-policy", "classification": "P2"},
    )
    pipeline_result = SimpleNamespace(
        plan=SimpleNamespace(to_dict=lambda: {"queryFrame": {"source_type": source_type}}),
    )
    verification = verification or {
        "status": "verified",
        "unsupportedClaimCount": 0,
        "uncertainClaimCount": 0,
        "supportedClaimCount": 1,
        "needsCaution": False,
        "summary": "verified",
    }
    rewrite = rewrite or {"attempted": False, "applied": False, "finalAnswerSource": "original"}
    return {
        "answer": answer,
        "sources": evidence,
        "evidencePacketContract": build_evidence_packet_contract(
            query=query,
            retrieval_mode="hybrid",
            pipeline_result=pipeline_result,
            evidence_packet=packet,
        ),
        "answerContract": build_answer_contract(
            answer=answer,
            evidence_packet=packet,
            verification=verification,
            rewrite=rewrite,
            routing_meta={"provider": "fixture", "model": "test"},
        ),
        "verificationVerdict": build_verification_verdict(verification),
        "answerRewrite": rewrite,
    }


class _FakeSearcher:
    def __init__(self, payload_by_query: dict[str, dict]):
        self.payload_by_query = payload_by_query

    def generate_answer(self, query: str, **kwargs):  # noqa: ANN003
        _ = kwargs
        return self.payload_by_query[query]


def test_case_loader_rejects_missing_required_field(tmp_path):
    module = _load_module()
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [{"case_id": "bad", "query": "missing fields"}]}), encoding="utf-8")

    try:
        module.read_cases(path)
    except ValueError as error:
        assert "missing required fields" in str(error)
    else:
        raise AssertionError("read_cases should reject incomplete cases")


def test_latency_summary_uses_median_p50_and_nearest_rank_p95():
    module = _load_module()

    summary = module.latency_summary([10.0, 20.0, 30.0, 40.0])

    assert summary["p50"] == 25.0
    assert summary["p95"] == 40.0
    assert summary["avg"] == 25.0


def test_contract_validation_failure_marks_case_and_gate_failed():
    module = _load_module()
    cases = [_case("missing_contracts")]
    searcher = _FakeSearcher({"alpha?": {"answer": "no contracts"}})

    payload = module.run_gate(
        cases,
        searcher=searcher,
        cases_path=Path("cases.json"),
        timeout_sec=0,
        thresholds={"min_citation_grade_coverage_rate": 0.0},
    )

    assert payload["status"] == "failed"
    assert payload["failedCaseCount"] == 1
    assert "missing_contract:answerContract" in payload["cases"][0]["errors"]
    assert "failed_cases:1" in payload["errors"]


def test_expected_abstain_accepts_verification_recommended_abstain():
    module = _load_module()
    case = _case("abstain_by_verdict", expected_min_citation_count=0, expected_abstain=True)
    verification = {
        "status": "caution",
        "unsupportedClaimCount": 1,
        "uncertainClaimCount": 0,
        "supportedClaimCount": 0,
        "needsCaution": True,
        "summary": "unsupported",
    }
    searcher = _FakeSearcher({"alpha?": _contract_payload(verification=verification)})

    payload = module.run_gate([case], searcher=searcher, cases_path=Path("cases.json"), timeout_sec=0)

    assert payload["status"] == "ok"
    assert payload["cases"][0]["abstainObserved"] is True
    assert payload["abstainCorrectRate"] == 1.0


def test_stub_fixture_gate_summarizes_contract_metrics():
    module = _load_module()
    cases = [
        _case("ok_1", query="alpha?"),
        _case("ok_2", query="beta?"),
        _case("abstain_1", query="gamma?", expected_min_citation_count=0, expected_abstain=True),
    ]
    searcher = _FakeSearcher(
        {
            "alpha?": _contract_payload(query="alpha?"),
            "beta?": _contract_payload(query="beta?"),
            "gamma?": _contract_payload(
                query="gamma?",
                answer="",
                answerable=False,
                evidence=[],
                verification={"status": "abstain", "summary": "no evidence"},
            ),
        }
    )

    payload = module.run_gate(cases, searcher=searcher, cases_path=Path("cases.json"), timeout_sec=0, stub_llm=True)

    assert payload["status"] == "ok"
    assert payload["caseCount"] == 3
    assert payload["passedCaseCount"] == 3
    assert payload["contractValidRate"] == 1.0
    assert payload["citationGradeCoverageRate"] == 1.0
    assert payload["abstainCorrectRate"] == 1.0
    assert payload["timeoutCount"] == 0
