#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_hub.ai.answer_contracts import build_answer_contract


SCHEMA = "knowledge-hub.answer-quality-gate.result.v1"
DEFAULT_CASES_PATH = "eval/knowledgeos/fixtures/answer_quality_golden_cases.json"


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _read_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("cases") or []
    if not isinstance(payload, list):
        raise ValueError(f"expected list or object with cases: {path}")
    return [dict(item) for item in payload if isinstance(item, dict)]


def _packet_from_case(case: dict[str, Any]) -> SimpleNamespace:
    evidence = [dict(item or {}) for item in list(case.get("evidence") or [])]
    citations = [
        {
            "label": _clean_text(item.get("citation_label") or f"S{index}"),
            "target": _clean_text(item.get("citation_target") or item.get("source_id")),
            "kind": "source",
        }
        for index, item in enumerate(evidence, start=1)
    ]
    answerable = bool(evidence) or bool(str(case.get("answer") or "").strip())
    return SimpleNamespace(
        evidence=evidence,
        citations=citations,
        evidence_packet={
            "answerable": answerable,
            "answerableDecisionReason": "fixture evidence present" if answerable else "no fixture evidence",
        },
        evidence_policy={"policyKey": "answer-quality-golden"},
    )


def evaluate_answer_quality_case(case: dict[str, Any]) -> dict[str, Any]:
    expected = dict(case.get("expected") or {})
    contract = build_answer_contract(
        answer=str(case.get("answer") or ""),
        evidence_packet=_packet_from_case(case),
        verification=dict(case.get("verification") or {}),
        rewrite=dict(case.get("rewrite") or {}),
        routing_meta={"provider": "fixture", "model": "answer-quality-gate"},
    )
    coverage = dict(contract.get("coverage") or {})
    verdict = dict(contract.get("verificationVerdict") or {})
    errors: list[str] = []

    expected_status = _clean_text(expected.get("status") or "pass").lower()
    expected_coverage_status = _clean_text(expected.get("coverage_status"))
    if expected_coverage_status and coverage.get("status") != expected_coverage_status:
        errors.append(f"coverage_status:{coverage.get('status')}!={expected_coverage_status}")
    min_coverage = float(expected.get("min_coverage_ratio", 0.0))
    if float(contract.get("coverageRatio") or 0.0) < min_coverage:
        errors.append(f"coverage_ratio_below:{contract.get('coverageRatio')}<{min_coverage}")
    if "abstain" in expected and bool(contract.get("abstain")) is not bool(expected.get("abstain")):
        errors.append(f"abstain:{contract.get('abstain')}!={expected.get('abstain')}")
    if "citation_count" in expected and len(contract.get("citations") or []) != int(expected.get("citation_count") or 0):
        errors.append(f"citation_count:{len(contract.get('citations') or [])}!={expected.get('citation_count')}")
    if "retrieval_signal_count" in expected and len(contract.get("retrievalSignals") or []) != int(expected.get("retrieval_signal_count") or 0):
        errors.append(
            f"retrieval_signal_count:{len(contract.get('retrievalSignals') or [])}!={expected.get('retrieval_signal_count')}"
        )
    if "verification_verdict" in expected and verdict.get("verdict") != expected.get("verification_verdict"):
        errors.append(f"verification_verdict:{verdict.get('verdict')}!={expected.get('verification_verdict')}")

    failed = bool(errors)
    if expected_status == "fail" and not failed:
        errors.append("expected_failure_but_case_passed")
    elif expected_status == "pass" and failed:
        pass
    elif expected_status == "fail" and failed:
        errors = []

    return {
        "caseId": _clean_text(case.get("case_id")),
        "status": "pass" if not errors else "fail",
        "expectedStatus": expected_status,
        "errors": errors,
        "coverageStatus": coverage.get("status"),
        "coverageRatio": contract.get("coverageRatio"),
        "citationCount": len(contract.get("citations") or []),
        "retrievalSignalCount": len(contract.get("retrievalSignals") or []),
        "verificationVerdict": verdict.get("verdict"),
        "abstain": bool(contract.get("abstain")),
    }


def run_answer_quality_gate(cases: list[dict[str, Any]], *, cases_path: Path, min_cases: int = 1) -> dict[str, Any]:
    results = [evaluate_answer_quality_case(case) for case in cases]
    passed = [item for item in results if item["status"] == "pass"]
    failed = [item for item in results if item["status"] == "fail"]
    errors: list[str] = []
    if len(results) < int(min_cases):
        errors.append(f"insufficient_cases:{len(results)}/{int(min_cases)}")
    if failed:
        errors.append(f"failed_cases:{len(failed)}")
    return {
        "schema": SCHEMA,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if not errors else "failed",
        "casesPath": str(cases_path),
        "caseCount": len(cases),
        "passedCaseCount": len(passed),
        "failedCaseCount": len(failed),
        "errors": errors,
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check answer quality against deterministic AnswerContract fixtures.")
    parser.add_argument("--cases", default=DEFAULT_CASES_PATH)
    parser.add_argument("--min-cases", type=int, default=1)
    parser.add_argument("--json", action="store_true", dest="as_json", default=False)
    args = parser.parse_args(argv)
    cases_path = Path(args.cases).expanduser().resolve()
    try:
        payload = run_answer_quality_gate(_read_cases(cases_path), cases_path=cases_path, min_cases=int(args.min_cases))
    except Exception as exc:  # noqa: BLE001
        payload = {
            "schema": SCHEMA,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "casesPath": str(cases_path),
            "errors": [f"answer_quality_gate_failed:{exc}"],
            "cases": [],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["status"])
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
