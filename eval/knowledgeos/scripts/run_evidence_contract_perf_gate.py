#!/usr/bin/env python3
"""Run the frontier Evidence-contract RAG local performance gate."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import signal
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_hub.ai.answer_contracts import (  # noqa: E402
    build_answer_contract,
    build_evidence_packet_contract,
    build_verification_verdict,
)
from knowledge_hub.application.ask_contracts import ensure_ask_contract_payload, external_policy_contract  # noqa: E402
from knowledge_hub.application.context import AppContextFactory  # noqa: E402
from knowledge_hub.core.schema_validator import validate_payload  # noqa: E402


SCHEMA = "knowledge-hub.evidence-contract-perf-gate.result.v1"
DEFAULT_CASES_PATH = "eval/knowledgeos/queries/evidence_contract_perf_gate_cases_v1.json"
DEFAULT_OUT_DIR = "eval/knowledgeos/runs"
REQUIRED_CASE_FIELDS = {
    "case_id",
    "query",
    "source",
    "expected_answer_mode",
    "expected_min_citation_count",
    "expected_abstain",
    "required_contracts",
}
CONTRACT_SCHEMA_BY_KEY = {
    "evidencePacketContract": "knowledge-hub.evidence-packet.v1",
    "answerContract": "knowledge-hub.answer-contract.v1",
    "verificationVerdict": "knowledge-hub.verification-verdict.v1",
}
SOURCE_FILTERS = {"all", "paper", "vault", "web", "mixed", "abstain"}
THERMAL_SOURCE_ORDER = ("vault", "paper", "web", "mixed", "abstain")
HARD_TEMPORAL_RE = re.compile(r"\b(latest|updated|newest|before|after|since)\b|최신|업데이트|이전|이후|오늘", re.IGNORECASE)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).casefold() in {"1", "true", "yes", "y", "on"}


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def read_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("cases") or []
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list or object with cases: {path}")
    cases: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"case #{index} is not an object")
        case = dict(item)
        missing = sorted(REQUIRED_CASE_FIELDS - set(case))
        if missing:
            raise ValueError(f"case #{index} missing required fields: {', '.join(missing)}")
        case_id = clean_text(case.get("case_id"))
        query = clean_text(case.get("query"))
        source = normalize_source_filter(case.get("source"))
        if not case_id:
            raise ValueError(f"case #{index} has empty case_id")
        if not query:
            raise ValueError(f"case {case_id} has empty query")
        if source not in SOURCE_FILTERS:
            raise ValueError(f"case {case_id} has unsupported source: {case.get('source')!r}")
        required_contracts = case.get("required_contracts")
        if not isinstance(required_contracts, list) or not required_contracts:
            raise ValueError(f"case {case_id} required_contracts must be a non-empty list")
        unsupported_contracts = sorted(
            clean_text(item) for item in required_contracts if clean_text(item) not in CONTRACT_SCHEMA_BY_KEY
        )
        if unsupported_contracts:
            raise ValueError(f"case {case_id} has unsupported contracts: {', '.join(unsupported_contracts)}")
        case["case_id"] = case_id
        case["query"] = query
        case["source"] = source
        case["expected_min_citation_count"] = as_int(case.get("expected_min_citation_count"), 0)
        case["expected_abstain"] = as_bool(case.get("expected_abstain"))
        case["required_contracts"] = [clean_text(item) for item in required_contracts]
        cases.append(case)
    return cases


def normalize_source_filter(value: Any) -> str:
    token = clean_text(value).casefold()
    if token in {"", "all", "any"}:
        return "all"
    if token in {"note", "notes"}:
        return "vault"
    return token


def ask_source_type(source: str) -> str | None:
    normalized = normalize_source_filter(source)
    if normalized in {"all", "mixed", "abstain"}:
        return None
    return normalized


def case_ask_source_type(case: dict[str, Any]) -> str | None:
    override = clean_text(case.get("execution_source") or case.get("ask_source_type"))
    return ask_source_type(override or clean_text(case.get("source")))


def select_cases(cases: list[dict[str, Any]], *, source_filter: str) -> list[dict[str, Any]]:
    normalized = normalize_source_filter(source_filter)
    if normalized == "all":
        return list(cases)
    return [case for case in cases if normalize_source_filter(case.get("source")) == normalized]


def select_run_cases(
    cases: list[dict[str, Any]],
    *,
    source_filter: str,
    run_profile: str,
    max_cases: int = 0,
    case_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected = select_cases(cases, source_filter=source_filter)
    wanted_ids = {clean_text(case_id) for case_id in list(case_ids or []) if clean_text(case_id)}
    if wanted_ids:
        return [case for case in selected if clean_text(case.get("case_id")) in wanted_ids]
    limit = max(0, int(max_cases or 0))
    if clean_text(run_profile).casefold() != "thermal":
        return selected[:limit] if limit else selected
    if normalize_source_filter(source_filter) != "all":
        return selected[: limit or 1]
    by_source: dict[str, dict[str, Any]] = {}
    for case in selected:
        source = normalize_source_filter(case.get("source"))
        by_source.setdefault(source, case)
    thermal_cases = [by_source[source] for source in THERMAL_SOURCE_ORDER if source in by_source]
    return thermal_cases[: limit or len(thermal_cases)]


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "avg": 0.0}
    ordered = sorted(float(item) for item in values)
    p95_index = max(0, min(len(ordered) - 1, int((0.95 * len(ordered) + 0.999999) - 1)))
    return {
        "p50": round(float(statistics.median(ordered)), 3),
        "p95": round(float(ordered[p95_index]), 3),
        "min": round(float(ordered[0]), 3),
        "max": round(float(ordered[-1]), 3),
        "avg": round(float(sum(ordered) / len(ordered)), 3),
    }


def contract_payload(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        return dict(value)
    if key == "verificationVerdict":
        nested = dict(payload.get("answerContract") or {}).get("verificationVerdict")
        if isinstance(nested, dict):
            return dict(nested)
    return {}


def validate_required_contracts(payload: dict[str, Any], required_contracts: list[str]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for key in required_contracts:
        current = contract_payload(payload, key)
        if not current:
            errors.append(f"missing_contract:{key}")
            continue
        schema_id = CONTRACT_SCHEMA_BY_KEY[key]
        result = validate_payload(current, schema_id, strict=True)
        if not result.ok:
            for error in result.errors:
                errors.append(f"{key}:{error}")
    return not errors, errors


def citation_grade_count(answer_contract: dict[str, Any]) -> int:
    count = 0
    for citation in list(answer_contract.get("citations") or []):
        if not isinstance(citation, dict):
            continue
        has_source = bool(clean_text(citation.get("source_id") or citation.get("sourceId")))
        has_hash = bool(clean_text(citation.get("content_hash") or citation.get("sourceContentHash")))
        has_start = citation.get("char_start") is not None or citation.get("charStart") is not None
        has_end = citation.get("char_end") is not None or citation.get("charEnd") is not None
        if has_source and has_hash and has_start and has_end:
            count += 1
    return count


def answer_abstains(payload: dict[str, Any]) -> bool:
    answer_contract = contract_payload(payload, "answerContract")
    verification_verdict = contract_payload(payload, "verificationVerdict")
    if answer_contract:
        return bool(answer_contract.get("abstain"))
    return clean_text(verification_verdict.get("recommended_action")).casefold() == "abstain"


def conservative_fallback_used(payload: dict[str, Any]) -> bool:
    rewrite = dict(payload.get("answerRewrite") or {})
    answer_contract = contract_payload(payload, "answerContract")
    contract_rewrite = dict(answer_contract.get("rewrite") or {})
    source = clean_text(rewrite.get("finalAnswerSource") or contract_rewrite.get("finalAnswerSource")).casefold()
    return source == "conservative_fallback"


def strict_evidence_answerable(payload: dict[str, Any]) -> bool | None:
    evidence_contract = contract_payload(payload, "evidencePacketContract")
    if "answerable" not in evidence_contract:
        return None
    return bool(evidence_contract.get("answerable"))


def safe_abstain_observed(payload: dict[str, Any]) -> bool:
    return answer_abstains(payload) or conservative_fallback_used(payload)


def provider_corpus_dependency_reason(payload: dict[str, Any], *, query: str = "") -> str:
    answer_contract = contract_payload(payload, "answerContract")
    evidence_packet = dict(payload.get("evidence_packet") or payload.get("evidencePacket") or {})
    v2 = dict(payload.get("v2") or {})
    verification = dict(v2.get("evidenceVerification") or {})
    fallback = dict(v2.get("fallback") or {})
    reason_text = " ".join(
        clean_text(value).casefold()
        for value in [
            answer_contract.get("abstainReason"),
            evidence_packet.get("answerableDecisionReason"),
            fallback.get("reason"),
            " ".join(clean_text(item) for item in list(evidence_packet.get("insufficientEvidenceReasons") or [])),
            " ".join(clean_text(item) for item in list(verification.get("unsupportedFields") or [])),
        ]
        if clean_text(value)
    )
    if any(token in reason_text for token in ("temporal_grounding", "weak_web_temporal_grounding", "missing_temporal_grounding")):
        if query and not HARD_TEMPORAL_RE.search(query):
            return ""
        return "temporal_grounding"
    if any(
        token in reason_text
        for token in (
            "no_evidence",
            "source_mismatch",
            "low_confidence_evidence",
            "weak_support_only",
            "direct_but_incomplete",
            "no_substantive_evidence",
            "non_substantive_evidence",
            "strict_abstention_threshold_not_met",
            "insufficient_for_latest_claim",
            "ask_v2_unsupported_claim_cards",
            "ask_v2_weak_evidence",
        )
    ):
        return "retrieval_or_corpus_gap"
    return ""


def classify_failure_categories(errors: list[str]) -> list[str]:
    categories: list[str] = []
    for error in errors:
        token = clean_text(error).casefold()
        if token == "timeout":
            categories.append("latency_timeout")
        elif token.startswith("missing_contract:") or token.startswith(("evidencepacketcontract:", "answercontract:", "verificationverdict:")):
            categories.append("contract_missing")
        elif token.startswith("citation_grade_below_min:"):
            categories.append("citation_grade")
        elif token.startswith("abstain_mismatch:"):
            categories.append("abstain_mismatch")
        elif token.startswith("provider_corpus_dependency:"):
            categories.append("provider/corpus_dependency")
    if errors and not categories:
        categories.append("provider/corpus_dependency")
    category_set = set(categories)
    if "latency_timeout" in category_set:
        return ["latency_timeout"]
    if "contract_missing" in category_set:
        category_set = {"contract_missing"}
    ordered = ["contract_missing", "citation_grade", "abstain_mismatch", "latency_timeout", "provider/corpus_dependency"]
    return [category for category in ordered if category in category_set]


def evaluate_case(
    case: dict[str, Any],
    payload: dict[str, Any],
    *,
    latency_ms: float,
    timeout: bool = False,
    llm_stubbed: bool = False,
) -> dict[str, Any]:
    required_contracts = [clean_text(item) for item in list(case.get("required_contracts") or [])]
    contracts_valid, contract_errors = validate_required_contracts(payload, required_contracts)
    answer_contract = contract_payload(payload, "answerContract")
    verification_verdict = contract_payload(payload, "verificationVerdict")
    citation_grade = citation_grade_count(answer_contract)
    expected_min_citations = as_int(case.get("expected_min_citation_count"), 0)
    citation_grade_ok = citation_grade >= expected_min_citations
    expected_abstain = as_bool(case.get("expected_abstain"))
    hard_abstain_observed = answer_abstains(payload)
    safe_abstain = safe_abstain_observed(payload)
    abstain_observed = safe_abstain if expected_abstain else hard_abstain_observed
    abstain_ok = abstain_observed if expected_abstain else not abstain_observed
    unsupported_count = as_int(verification_verdict.get("unsupportedClaimCount"), 0)
    verdict = clean_text(verification_verdict.get("verdict")).casefold()
    strict_answerable = strict_evidence_answerable(payload)
    conservative_fallback = conservative_fallback_used(payload)
    generation_dependency_reason = ""
    if (
        llm_stubbed
        and not expected_abstain
        and conservative_fallback
        and strict_answerable is True
        and citation_grade_ok
    ):
        generation_dependency_reason = "stubbed_generation_conservative_fallback"
        abstain_observed = False
        abstain_ok = True
    dependency_reason = (
        provider_corpus_dependency_reason(payload, query=clean_text(case.get("query")))
        if not expected_abstain and hard_abstain_observed and contracts_valid and citation_grade_ok
        else ""
    )
    errors: list[str] = []
    if timeout:
        errors.append("timeout")
    if not contracts_valid:
        errors.extend(contract_errors)
    if not citation_grade_ok:
        errors.append(f"citation_grade_below_min:{citation_grade}<{expected_min_citations}")
    if not abstain_ok:
        if dependency_reason:
            errors.append(f"provider_corpus_dependency:{dependency_reason}")
        else:
            errors.append(f"abstain_mismatch:{abstain_observed}!={expected_abstain}")
    failure_categories = classify_failure_categories(errors)
    return {
        "caseId": clean_text(case.get("case_id")),
        "query": clean_text(case.get("query")),
        "source": normalize_source_filter(case.get("source")),
        "expectedAnswerMode": clean_text(case.get("expected_answer_mode")),
        "expectedMinCitationCount": expected_min_citations,
        "expectedAbstain": expected_abstain,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "failureCategories": failure_categories,
        "contractsValid": contracts_valid,
        "contractErrors": contract_errors,
        "citationGradeCitationCount": citation_grade,
        "citationGradeOk": citation_grade_ok,
        "abstainObserved": abstain_observed,
        "hardAbstainObserved": hard_abstain_observed,
        "safeAbstainObserved": safe_abstain,
        "abstainOk": abstain_ok,
        "strictEvidenceAnswerable": strict_answerable,
        "generationDependencyReason": generation_dependency_reason,
        "providerCorpusDependencyReason": dependency_reason,
        "verificationVerdict": verdict,
        "verificationPass": verdict == "pass",
        "unsupportedClaimCount": unsupported_count,
        "conservativeFallbackUsed": conservative_fallback,
        "latencyMs": round(float(latency_ms), 3),
        "timeout": bool(timeout),
    }


def build_summary(case_results: list[dict[str, Any]], *, thresholds: dict[str, Any]) -> dict[str, Any]:
    case_count = len(case_results)
    passed = [item for item in case_results if item.get("status") == "pass"]
    failed = [item for item in case_results if item.get("status") == "fail"]
    citation_cases = [item for item in case_results if as_int(item.get("expectedMinCitationCount"), 0) > 0]
    abstain_cases = [item for item in case_results if bool(item.get("expectedAbstain"))]
    latencies = [as_float(item.get("latencyMs"), 0.0) for item in case_results if not bool(item.get("timeout"))]
    contract_valid_rate = _rate(sum(1 for item in case_results if item.get("contractsValid")), case_count)
    citation_grade_coverage_rate = _rate(sum(1 for item in citation_cases if item.get("citationGradeOk")), len(citation_cases))
    abstain_correct_rate = _rate(sum(1 for item in abstain_cases if item.get("abstainOk")), len(abstain_cases))
    verification_pass_rate = _rate(sum(1 for item in case_results if item.get("verificationPass")), case_count)
    unsupported_claim_rate = _rate(sum(1 for item in case_results if as_int(item.get("unsupportedClaimCount"), 0) > 0), case_count)
    conservative_fallback_rate = _rate(sum(1 for item in case_results if item.get("conservativeFallbackUsed")), case_count)
    generation_dependency_rate = _rate(sum(1 for item in case_results if clean_text(item.get("generationDependencyReason"))), case_count)
    failure_categories: dict[str, int] = {}
    for item in failed:
        for category in list(item.get("failureCategories") or []):
            key = clean_text(category) or "provider/corpus_dependency"
            failure_categories[key] = failure_categories.get(key, 0) + 1
    latency = latency_summary(latencies)
    timeout_count = sum(1 for item in case_results if item.get("timeout"))
    threshold_errors: list[str] = []
    if contract_valid_rate < float(thresholds["min_contract_valid_rate"]):
        threshold_errors.append(f"contractValidRate:{contract_valid_rate}<{thresholds['min_contract_valid_rate']}")
    if citation_grade_coverage_rate < float(thresholds["min_citation_grade_coverage_rate"]):
        threshold_errors.append(
            f"citationGradeCoverageRate:{citation_grade_coverage_rate}<{thresholds['min_citation_grade_coverage_rate']}"
        )
    if abstain_correct_rate < float(thresholds["min_abstain_correct_rate"]):
        threshold_errors.append(f"abstainCorrectRate:{abstain_correct_rate}<{thresholds['min_abstain_correct_rate']}")
    if timeout_count > int(thresholds["max_timeout_count"]):
        threshold_errors.append(f"timeoutCount:{timeout_count}>{thresholds['max_timeout_count']}")
    if latency["p95"] > float(thresholds["max_p95_latency_ms"]):
        threshold_errors.append(f"p95:{latency['p95']}>{thresholds['max_p95_latency_ms']}")
    return {
        "caseCount": case_count,
        "passedCaseCount": len(passed),
        "failedCaseCount": len(failed),
        "contractValidRate": contract_valid_rate,
        "citationGradeCoverageRate": citation_grade_coverage_rate,
        "abstainCorrectRate": abstain_correct_rate,
        "verificationPassRate": verification_pass_rate,
        "unsupportedClaimRate": unsupported_claim_rate,
        "conservativeFallbackRate": conservative_fallback_rate,
        "generationDependencyRate": generation_dependency_rate,
        "failureCategories": dict(sorted(failure_categories.items())),
        "latencyMs": latency,
        "timeoutCount": timeout_count,
        "thresholdErrors": threshold_errors,
    }


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(float(numerator) / float(denominator), 6)


def run_with_timeout(timeout_seconds: int, fn, *args, **kwargs):  # noqa: ANN001
    if timeout_seconds <= 0:
        return fn(*args, **kwargs)

    def _handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"evidence contract gate timeout after {timeout_seconds}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.alarm(int(timeout_seconds))
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


class _PerfGateStubLLM:
    def generate(self, prompt: str, context: str = "") -> str:  # noqa: ARG002
        return "근거가 있는 경우에만 답하고, 근거가 부족하면 abstain해야 합니다."

    def stream_generate(self, prompt: str, context: str = ""):
        _ = (prompt, context)
        yield self.generate(prompt, context)


class FixtureContractSearcher:
    """Deterministic searcher used by --stub-llm to avoid local corpus/provider variance."""

    def __init__(self, cases: list[dict[str, Any]]):
        self._case_by_query = {clean_text(case.get("query")): dict(case) for case in cases}

    def generate_answer(self, query: str, **kwargs):  # noqa: ANN003
        case = self._case_by_query.get(clean_text(query), {})
        source_type = clean_text(kwargs.get("source_type") or case.get("source") or "mixed")
        if source_type in {"", "all", "abstain"}:
            source_type = "mixed"
        case_id = clean_text(case.get("case_id")) or "fixture_case"
        expected_abstain = as_bool(case.get("expected_abstain"))
        if expected_abstain:
            answer = ""
            evidence: list[dict[str, Any]] = []
            verification = {
                "status": "abstain",
                "unsupportedClaimCount": 0,
                "uncertainClaimCount": 0,
                "supportedClaimCount": 0,
                "needsCaution": False,
                "summary": "fixture expected abstain",
            }
            answerable = False
        else:
            answer = f"Fixture evidence supports {clean_text(query)}."
            evidence = [
                {
                    "title": f"Fixture {case_id}",
                    "excerpt": f"Fixture evidence supports {clean_text(query)}.",
                    "citation_label": "S1",
                    "citation_target": f"{source_type}:{case_id}",
                    "source_id": f"{source_type}:{case_id}",
                    "source_ref": f"{source_type}:{case_id}",
                    "source_type": source_type,
                    "source_content_hash": f"hash-{case_id}",
                    "span_locator": "chars:0-80",
                    "score": 1.0,
                }
            ]
            verification = {
                "status": "verified",
                "unsupportedClaimCount": 0,
                "uncertainClaimCount": 0,
                "supportedClaimCount": 1,
                "needsCaution": False,
                "summary": "fixture verified",
            }
            answerable = True
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
            evidence_policy={"policyKey": "evidence-contract-perf-gate-fixture", "classification": "P2"},
        )
        pipeline_result = SimpleNamespace(
            plan=SimpleNamespace(to_dict=lambda: {"queryFrame": {"source_type": source_type}}),
        )
        rewrite = {"attempted": False, "applied": False, "finalAnswerSource": "original"}
        evidence_contract = build_evidence_packet_contract(
            query=clean_text(query),
            retrieval_mode=clean_text(kwargs.get("retrieval_mode")) or "hybrid",
            pipeline_result=pipeline_result,
            evidence_packet=packet,
        )
        return {
            "answer": answer,
            "sources": evidence,
            "evidence": evidence,
            "citations": list(packet.citations),
            "evidencePacketContract": evidence_contract,
            "answerContract": build_answer_contract(
                answer=answer,
                evidence_packet=packet,
                evidence_packet_contract=evidence_contract,
                verification=verification,
                rewrite=rewrite,
                routing_meta={"provider": "fixture", "model": "evidence-contract-perf-gate"},
            ),
            "verificationVerdict": build_verification_verdict(verification),
            "answerRewrite": rewrite,
        }


@contextmanager
def stubbed_answer_runtime(searcher: Any):
    llm = _PerfGateStubLLM()
    originals: dict[str, Any] = {}
    for name in (
        "llm",
        "_resolve_llm_for_request",
        "_record_answer_log",
    ):
        if hasattr(searcher, name):
            originals[name] = getattr(searcher, name)
    try:
        searcher.llm = llm
        searcher._resolve_llm_for_request = lambda **kwargs: (  # type: ignore[method-assign]
            llm,
            {"route": "local", "provider": "stub", "model": "evidence-contract-perf-gate"},
            [],
        )
        if hasattr(searcher, "_record_answer_log"):
            searcher._record_answer_log = lambda **kwargs: None  # type: ignore[method-assign]
        yield
    finally:
        for name, value in originals.items():
            setattr(searcher, name, value)


def run_ask_like_cli(
    searcher: Any,
    case: dict[str, Any],
    *,
    top_k: int,
    retrieval_mode: str,
    alpha: float,
    allow_external: bool,
    answer_route: str,
) -> dict[str, Any]:
    source_type = case_ask_source_type(case)
    result = searcher.generate_answer(
        clean_text(case.get("query")),
        top_k=int(top_k),
        source_type=source_type,
        retrieval_mode=str(retrieval_mode),
        alpha=float(alpha),
        allow_external=bool(allow_external),
        memory_route_mode="off",
        paper_memory_mode="off",
        answer_route_override=None if clean_text(answer_route).casefold() == "auto" else clean_text(answer_route).casefold(),
    )
    payload = ensure_ask_contract_payload(
        dict(result or {}),
        source_type=source_type,
        memory_route_mode="off",
        paper_memory_mode="off",
        external_policy=external_policy_contract(
            surface="evidence-contract-perf-gate",
            allow_external=bool(allow_external),
            requested=bool(allow_external),
            decision_source="frontier_external_gate" if bool(allow_external) else "frontier_local_gate",
        ),
    )
    payload["question"] = clean_text(case.get("query"))
    payload["schema"] = "knowledge-hub.ask.result.v1"
    payload["sourceType"] = source_type
    payload["retrievalMode"] = str(retrieval_mode)
    payload["alpha"] = float(alpha)
    payload["answerRouteRequested"] = clean_text(answer_route).casefold() or "auto"
    return payload


def run_gate(
    cases: list[dict[str, Any]],
    *,
    searcher: Any,
    cases_path: Path,
    top_k: int = 8,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.7,
    timeout_sec: int = 60,
    source_filter: str = "all",
    stub_llm: bool = False,
    run_profile: str = "full",
    max_cases: int = 0,
    case_ids: list[str] | None = None,
    allow_external: bool = False,
    answer_route: str = "auto",
    mode_label: str | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_cases = select_run_cases(
        cases,
        source_filter=source_filter,
        run_profile=run_profile,
        max_cases=max_cases,
        case_ids=case_ids,
    )
    thresholds = {
        "min_contract_valid_rate": 1.0,
        "min_citation_grade_coverage_rate": 0.8,
        "min_abstain_correct_rate": 0.8,
        "max_timeout_count": 0,
        "max_p95_latency_ms": 60_000.0,
        **dict(thresholds or {}),
    }
    case_results: list[dict[str, Any]] = []
    runtime_cm = stubbed_answer_runtime(searcher) if stub_llm else nullcontext()
    with runtime_cm:
        for case in selected_cases:
            started = time.perf_counter()
            timeout = False
            try:
                payload = run_with_timeout(
                    int(timeout_sec),
                    run_ask_like_cli,
                    searcher,
                    case,
                    top_k=int(top_k),
                    retrieval_mode=str(retrieval_mode),
                    alpha=float(alpha),
                    allow_external=bool(allow_external),
                    answer_route=str(answer_route),
                )
            except TimeoutError:
                timeout = True
                payload = {}
            latency_ms = (time.perf_counter() - started) * 1000.0
            case_results.append(
                evaluate_case(case, payload, latency_ms=latency_ms, timeout=timeout, llm_stubbed=bool(stub_llm))
            )
    summary = build_summary(case_results, thresholds=thresholds)
    if stub_llm:
        summary["verificationPassRateRaw"] = summary.get("verificationPassRate")
        summary["verificationPassRate"] = None
    errors = list(summary["thresholdErrors"])
    if not selected_cases:
        errors.append("no_cases_selected")
    if summary["failedCaseCount"]:
        errors.append(f"failed_cases:{summary['failedCaseCount']}")
    return {
        "schema": SCHEMA,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if not errors else "failed",
        "mode": mode_label or ("stub" if stub_llm else "live"),
        "llmStubbed": bool(stub_llm),
        "runProfile": clean_text(run_profile).casefold() or "full",
        "thermalFriendly": clean_text(run_profile).casefold() == "thermal",
        "casesPath": str(cases_path),
        "sourceFilter": normalize_source_filter(source_filter),
        "selectedCaseIds": [clean_text(case.get("case_id")) for case in selected_cases],
        "topK": int(top_k),
        "retrievalMode": str(retrieval_mode),
        "alpha": float(alpha),
        "allowExternal": bool(allow_external),
        "answerRouteRequested": clean_text(answer_route).casefold() or "auto",
        "thresholds": thresholds,
        **summary,
        "errors": errors,
        "cases": case_results,
    }


def render_markdown_report(payload: dict[str, Any]) -> str:
    latency = dict(payload.get("latencyMs") or {})
    lines = [
        "# Evidence-contract RAG Local Performance Gate",
        "",
        f"- status: `{payload.get('status')}`",
        f"- mode: `{payload.get('mode')}`",
        f"- run profile: `{payload.get('runProfile')}`",
        f"- LLM stubbed: `{payload.get('llmStubbed')}`",
        f"- allow external: `{payload.get('allowExternal')}`",
        f"- answer route: `{payload.get('answerRouteRequested')}`",
        f"- cases: `{payload.get('passedCaseCount')}/{payload.get('caseCount')}` passed",
        f"- contract valid rate: `{payload.get('contractValidRate')}`",
        f"- citation-grade coverage rate: `{payload.get('citationGradeCoverageRate')}`",
        f"- abstain correct rate: `{payload.get('abstainCorrectRate')}`",
        f"- verification pass rate: `{payload.get('verificationPassRate')}`",
        f"- unsupported claim rate: `{payload.get('unsupportedClaimRate')}`",
        f"- conservative fallback rate: `{payload.get('conservativeFallbackRate')}`",
        f"- generation dependency rate: `{payload.get('generationDependencyRate')}`",
        f"- failure categories: `{json.dumps(payload.get('failureCategories') or {}, ensure_ascii=False)}`",
        f"- latency p50/p95: `{latency.get('p50', 0)}` / `{latency.get('p95', 0)}` ms",
        f"- timeout count: `{payload.get('timeoutCount')}`",
        "",
        "## Failed Cases",
        "",
    ]
    failed = [case for case in list(payload.get("cases") or []) if dict(case).get("status") == "fail"]
    if not failed:
        lines.append("- none")
    else:
        for case in failed:
            item = dict(case or {})
            lines.append(f"- `{item.get('caseId')}`: {', '.join(list(item.get('errors') or []))}")
    lines.append("")
    return "\n".join(lines)


def write_reports(payload: dict[str, Any], *, out_dir: Path, timestamp: str | None = None) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"evidence_contract_perf_gate_{stamp}.json"
    md_path = out_dir / f"evidence_contract_perf_gate_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_report(payload), encoding="utf-8")
    return {"jsonPath": str(json_path), "markdownPath": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Evidence-contract RAG performance gate.")
    parser.add_argument("--cases", default=DEFAULT_CASES_PATH)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--source", default="all", choices=sorted(SOURCE_FILTERS))
    parser.add_argument("--timeout-sec", "--timeout-seconds", type=int, default=None)
    parser.add_argument("--run-profile", choices=["auto", "thermal", "full"], default="auto")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--answer-route", default="auto")
    parser.add_argument("--allow-external", action="store_true", default=False)
    parser.add_argument("--stub-llm", action="store_true")
    parser.add_argument("--live-stub-llm", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json", default=False)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    if bool(args.stub_llm) and bool(args.live_stub_llm):
        parser.error("--stub-llm and --live-stub-llm are mutually exclusive")

    cases_path = Path(args.cases).expanduser().resolve()
    run_profile = str(args.run_profile or "auto").strip().lower()
    if run_profile == "auto":
        run_profile = "full" if bool(args.stub_llm) else "thermal"
    timeout_sec = int(args.timeout_sec) if args.timeout_sec is not None else (10 if run_profile == "thermal" else 60)
    mode_label = "stub" if bool(args.stub_llm) else ("live_stub" if bool(args.live_stub_llm) else "live")
    try:
        cases = read_cases(cases_path)
        app = None if bool(args.stub_llm) else AppContextFactory().build(require_search=True)
        searcher = FixtureContractSearcher(cases) if bool(args.stub_llm) else app.searcher
        payload = run_gate(
            cases,
            searcher=searcher,
            cases_path=cases_path,
            top_k=int(args.top_k),
            retrieval_mode=str(args.mode),
            alpha=float(args.alpha),
            timeout_sec=timeout_sec,
            source_filter=str(args.source),
            stub_llm=bool(args.stub_llm or args.live_stub_llm),
            run_profile=run_profile,
            max_cases=int(args.max_cases),
            case_ids=list(args.case_id or []),
            allow_external=bool(args.allow_external),
            answer_route=str(args.answer_route or "auto"),
            mode_label=mode_label,
        )
    except Exception as exc:  # noqa: BLE001
        payload = {
            "schema": SCHEMA,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "mode": mode_label,
            "llmStubbed": bool(args.stub_llm or args.live_stub_llm),
            "runProfile": run_profile,
            "thermalFriendly": run_profile == "thermal",
            "casesPath": str(cases_path),
            "sourceFilter": normalize_source_filter(args.source),
            "selectedCaseIds": [],
            "allowExternal": bool(args.allow_external),
            "answerRouteRequested": clean_text(args.answer_route).casefold() or "auto",
            "errors": [f"evidence_contract_perf_gate_failed:{exc}"],
            "caseCount": 0,
            "passedCaseCount": 0,
            "failedCaseCount": 0,
            "contractValidRate": 0.0,
            "citationGradeCoverageRate": 0.0,
            "abstainCorrectRate": 0.0,
            "verificationPassRate": None if bool(args.stub_llm or args.live_stub_llm) else 0.0,
            "verificationPassRateRaw": 0.0,
            "unsupportedClaimRate": 0.0,
            "conservativeFallbackRate": 0.0,
            "generationDependencyRate": 0.0,
            "failureCategories": {"provider/corpus_dependency": 1},
            "latencyMs": latency_summary([]),
            "timeoutCount": 0,
            "cases": [],
        }
    report_paths = write_reports(payload, out_dir=Path(args.out_dir).expanduser())
    payload["reportPaths"] = report_paths
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"evidence-contract perf gate: {payload.get('status')} ({report_paths['jsonPath']})")
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
