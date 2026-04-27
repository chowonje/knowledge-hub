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
            evidence_packet_contract=build_evidence_packet_contract(
                query=query,
                retrieval_mode="hybrid",
                pipeline_result=pipeline_result,
                evidence_packet=packet,
            ),
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
        self.calls: list[dict] = []

    def generate_answer(self, query: str, **kwargs):  # noqa: ANN003
        self.calls.append({"query": query, **kwargs})
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


def test_thermal_run_profile_selects_one_case_per_source():
    module = _load_module()
    cases = [
        _case("vault_1", source="vault"),
        _case("vault_2", source="vault"),
        _case("paper_1", source="paper"),
        _case("web_1", source="web"),
        _case("mixed_1", source="mixed"),
        _case("abstain_1", source="abstain"),
    ]

    selected = module.select_run_cases(cases, source_filter="all", run_profile="thermal")

    assert [case["case_id"] for case in selected] == ["vault_1", "paper_1", "web_1", "mixed_1", "abstain_1"]


def test_case_ask_source_type_uses_execution_source_override():
    module = _load_module()

    assert module.case_ask_source_type({"source": "abstain", "execution_source": "paper"}) == "paper"
    assert module.case_ask_source_type({"source": "abstain"}) is None


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
    assert payload["cases"][0]["failureCategories"] == ["contract_missing"]
    assert payload["failureCategories"] == {"contract_missing": 1}
    assert "failed_cases:1" in payload["errors"]


def test_answer_contract_abstain_is_source_of_truth_over_verdict_action():
    module = _load_module()
    case = _case("caution_answer", expected_min_citation_count=1, expected_abstain=False)
    verification = {
        "status": "caution",
        "unsupportedClaimCount": 1,
        "uncertainClaimCount": 0,
        "supportedClaimCount": 1,
        "needsCaution": True,
        "summary": "unsupported",
    }
    searcher = _FakeSearcher({"alpha?": _contract_payload(verification=verification)})

    payload = module.run_gate([case], searcher=searcher, cases_path=Path("cases.json"), timeout_sec=0)

    assert payload["cases"][0]["abstainObserved"] is False
    assert "abstain_mismatch:True!=False" not in payload["cases"][0]["errors"]


def test_expected_abstain_legacy_payload_accepts_verdict_action_without_answer_contract():
    module = _load_module()
    case = _case("legacy_abstain_by_verdict", expected_min_citation_count=0, expected_abstain=True)
    payload = {
        "verificationVerdict": build_verification_verdict(
            {
                "status": "abstain",
                "unsupportedClaimCount": 0,
                "uncertainClaimCount": 0,
                "supportedClaimCount": 0,
                "summary": "no evidence",
            }
        )
    }

    result = module.evaluate_case(case, payload, latency_ms=0.0)

    assert result["abstainObserved"] is True
    assert result["abstainOk"] is True


def test_expected_abstain_accepts_conservative_fallback_as_safe_refusal():
    module = _load_module()
    case = _case("fallback_abstain", expected_min_citation_count=0, expected_abstain=True)
    payload = _contract_payload(answer="Unsupported answer.")
    payload["answerRewrite"] = {"finalAnswerSource": "conservative_fallback"}
    payload["answerContract"]["rewrite"] = {"finalAnswerSource": "conservative_fallback"}

    result = module.evaluate_case(case, payload, latency_ms=0.0)

    assert result["hardAbstainObserved"] is False
    assert result["safeAbstainObserved"] is True
    assert result["abstainObserved"] is True
    assert result["abstainOk"] is True


def test_expected_answer_treats_conservative_fallback_as_hard_abstain():
    module = _load_module()
    case = _case("fallback_answer", expected_min_citation_count=1, expected_abstain=False)
    payload = _contract_payload(
        answer="Unsupported answer.",
        rewrite={"attempted": True, "applied": True, "finalAnswerSource": "conservative_fallback"},
    )

    result = module.evaluate_case(case, payload, latency_ms=0.0)

    assert result["hardAbstainObserved"] is True
    assert result["safeAbstainObserved"] is True
    assert result["abstainObserved"] is True
    assert result["abstainOk"] is False
    assert result["conservativeFallbackUsed"] is True


def test_temporal_hard_abstain_on_answer_case_is_corpus_dependency_when_citations_exist():
    module = _load_module()
    case = _case(
        "temporal_policy_gap",
        query="latest RAG evaluation article은 citation accuracy와 faithfulness를 어떻게 구분하나?",
        expected_min_citation_count=1,
        expected_abstain=False,
    )
    payload = _contract_payload(
        answer="The retrieved article discusses citation accuracy and faithfulness.",
        answerable=False,
        verification={"status": "abstain", "summary": "weak temporal grounding"},
    )
    payload["evidence_packet"] = {
        "answerable": False,
        "answerableDecisionReason": "weak_web_temporal_grounding",
        "insufficientEvidenceReasons": ["missing_temporal_grounding"],
    }
    payload["v2"] = {
        "evidenceVerification": {"unsupportedFields": ["temporal_version_grounding"]},
        "fallback": {"reason": "weak_web_temporal_grounding"},
    }

    result = module.evaluate_case(case, payload, latency_ms=0.0)

    assert result["citationGradeOk"] is True
    assert result["abstainObserved"] is True
    assert result["errors"] == ["provider_corpus_dependency:temporal_grounding"]
    assert result["failureCategories"] == ["provider/corpus_dependency"]


def test_soft_recency_temporal_abstain_is_not_hidden_as_corpus_dependency():
    module = _load_module()
    case = _case(
        "soft_recency_overroute",
        query="최근 RAG evaluation article은 citation accuracy와 faithfulness를 어떻게 구분하나?",
        expected_min_citation_count=1,
        expected_abstain=False,
    )
    payload = _contract_payload(
        answer="The retrieved article discusses citation accuracy and faithfulness.",
        answerable=False,
        verification={"status": "abstain", "summary": "weak temporal grounding"},
    )
    payload["evidence_packet"] = {
        "answerable": False,
        "answerableDecisionReason": "weak_web_temporal_grounding",
        "insufficientEvidenceReasons": ["missing_temporal_grounding"],
    }

    result = module.evaluate_case(case, payload, latency_ms=0.0)

    assert result["providerCorpusDependencyReason"] == ""
    assert result["errors"] == ["abstain_mismatch:True!=False"]
    assert result["failureCategories"] == ["abstain_mismatch"]


def test_ask_v2_weak_card_abstain_on_answer_case_is_corpus_dependency():
    module = _load_module()
    case = _case("weak_cards", expected_min_citation_count=1, expected_abstain=False)
    payload = _contract_payload(
        answer="",
        answerable=False,
        verification={"status": "abstain", "summary": "weak card evidence"},
    )
    payload["evidence_packet"] = {
        "answerable": False,
        "answerableDecisionReason": "ask_v2_unsupported_claim_cards",
        "insufficientEvidenceReasons": [],
    }
    payload["v2"] = {"fallback": {"used": True, "reason": "ask_v2_unsupported_claim_cards"}}

    result = module.evaluate_case(case, payload, latency_ms=0.0)

    assert result["citationGradeOk"] is True
    assert result["hardAbstainObserved"] is True
    assert result["providerCorpusDependencyReason"] == "retrieval_or_corpus_gap"
    assert result["errors"] == ["provider_corpus_dependency:retrieval_or_corpus_gap"]
    assert result["failureCategories"] == ["provider/corpus_dependency"]


def test_timeout_classification_does_not_count_derived_missing_contracts():
    module = _load_module()
    case = _case("timeout", expected_min_citation_count=0, expected_abstain=True)

    result = module.evaluate_case(case, {}, latency_ms=20_000.0, timeout=True)

    assert result["errors"][0] == "timeout"
    assert result["failureCategories"] == ["latency_timeout"]


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


def test_run_gate_records_external_codex_route_request():
    module = _load_module()
    cases = [_case("codex_route", query="alpha?")]
    searcher = _FakeSearcher({"alpha?": _contract_payload(query="alpha?")})

    payload = module.run_gate(
        cases,
        searcher=searcher,
        cases_path=Path("cases.json"),
        timeout_sec=0,
        allow_external=True,
        answer_route="codex",
    )

    assert payload["allowExternal"] is True
    assert payload["answerRouteRequested"] == "codex"
    assert payload["cases"][0]["status"] == "pass"
    assert searcher.calls[0]["allow_external"] is True
    assert searcher.calls[0]["answer_route_override"] == "codex"


def test_run_gate_can_label_live_stub_llm_mode_without_fixture_searcher():
    module = _load_module()
    cases = [_case("live_stub", query="alpha?")]
    searcher = _FakeSearcher({"alpha?": _contract_payload(query="alpha?")})

    payload = module.run_gate(
        cases,
        searcher=searcher,
        cases_path=Path("cases.json"),
        timeout_sec=0,
        stub_llm=True,
        mode_label="live_stub",
    )

    assert payload["mode"] == "live_stub"
    assert payload["llmStubbed"] is True
    assert payload["verificationPassRate"] is None
    assert payload["verificationPassRateRaw"] == 1.0
    assert payload["status"] == "ok"
