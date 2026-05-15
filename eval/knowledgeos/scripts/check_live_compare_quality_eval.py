#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_hub.ai.answer_contracts import NON_EVIDENCE_SOURCE_SCHEMES, NON_EVIDENCE_SOURCE_TYPES


SCHEMA = "knowledge-hub.live-compare-quality-eval.result.v1"
DEFAULT_CASES_PATH = "eval/knowledgeos/queries/live_compare_quality_eval_cases.local.json"
VALID_STATUSES = {"supported", "conflict", "unknown", "insufficient"}
PUBLIC_SOURCE_TYPES = {"paper", "vault", "web", "concept"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize(value: Any) -> str:
    return _clean_text(value).casefold()


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return bool(default)
    if isinstance(value, bool):
        return value
    return _normalize(value) in {"1", "true", "yes", "y", "on"}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, tuple):
        return [_clean_text(item) for item in value if _clean_text(item)]
    text = _clean_text(value)
    if not text:
        return []
    delimiter = "|" if "|" in text else ","
    return [_clean_text(item) for item in text.split(delimiter) if _clean_text(item)]


def _read_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("cases") or []
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list or object with cases: {path}")
    cases: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"case #{index} is not an object")
        if _as_bool(item.get("template"), False):
            continue
        query = _clean_text(item.get("query"))
        if not query:
            continue
        case = dict(item)
        case.setdefault("case_id", f"case_{index}")
        cases.append(case)
    return cases


def _source_scheme(source_id: Any) -> str:
    text = _normalize(source_id)
    return text.split(":", 1)[0] if ":" in text else ""


def _is_non_evidence_ref(item: dict[str, Any]) -> bool:
    source_type = _normalize(item.get("source_type") or item.get("sourceType"))
    if source_type in NON_EVIDENCE_SOURCE_TYPES:
        return True
    scheme = _source_scheme(item.get("source_id") or item.get("sourceId") or item.get("spanRef"))
    return bool(scheme and scheme in NON_EVIDENCE_SOURCE_SCHEMES)


def _compare_packet(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload.get("comparePacket") or payload.get("comparePacketContract") or {})


def _dimensions(packet: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item or {}) for item in list(packet.get("dimensions") or []) if isinstance(item, dict)]


def _supporting_spans(dimensions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for dimension in dimensions:
        spans.extend(dict(item or {}) for item in list(dimension.get("supportingSpans") or []) if isinstance(item, dict))
    return spans


def _source_id_from_item(item: dict[str, Any]) -> str:
    return _clean_text(item.get("source_id") or item.get("sourceId") or item.get("target") or item.get("id"))


def _collect_source_ids(payload: dict[str, Any], spans: list[dict[str, Any]]) -> set[str]:
    source_ids: set[str] = set()
    for item in spans:
        source_id = _source_id_from_item(item)
        if source_id:
            source_ids.add(source_id)
    return source_ids


def _collect_payload_source_ids(payload: dict[str, Any]) -> set[str]:
    source_ids: set[str] = set()
    trace = dict(payload.get("trace") or {})
    for item in list(payload.get("citations") or trace.get("citations") or []):
        if isinstance(item, dict):
            source_id = _source_id_from_item(item)
            if source_id:
                source_ids.add(source_id)
    for item in list(payload.get("sources") or trace.get("sources") or []):
        if isinstance(item, dict):
            source_id = _source_id_from_item(item)
            if source_id:
                source_ids.add(source_id)
    return source_ids


def _term_hits(text: str, terms: list[str]) -> list[str]:
    normalized_text = _normalize(text)
    return [term for term in terms if _normalize(term) and _normalize(term) in normalized_text]


def _ratio(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(count / total, 6)


def evaluate_case(
    case: dict[str, Any],
    payload: dict[str, Any],
    *,
    default_min_supporting_span_count: int = 1,
) -> dict[str, Any]:
    case_id = _clean_text(case.get("case_id"))
    expected_sources = _as_list(case.get("expected_source_ids") or case.get("expected_source_id"))
    expected_terms = _as_list(case.get("expected_dimension_terms") or case.get("expected_terms"))
    expected_statuses = {_normalize(item) for item in _as_list(case.get("expected_statuses"))}
    expected_statuses = {item for item in expected_statuses if item in VALID_STATUSES} or set(VALID_STATUSES)
    min_spans = _as_int(case.get("expected_min_supporting_span_count"), default_min_supporting_span_count)
    expected_answerable = _as_bool(case.get("expected_answerable"), True)
    require_trace_citations = _as_bool(case.get("require_trace_citations"), True)
    forbid_non_evidence = _as_bool(case.get("forbidden_non_evidence_support"), True)

    packet = _compare_packet(payload)
    dimensions = _dimensions(packet)
    spans = _supporting_spans(dimensions)
    coverage = dict(packet.get("coverage") or {})
    trace = dict(payload.get("trace") or {})
    citations = list(payload.get("citations") or trace.get("citations") or [])
    source_ids = _collect_source_ids(payload, spans)
    payload_source_ids = _collect_payload_source_ids(payload)
    dimension_text = "\n".join(
        "\n".join(
            [
                _clean_text(dimension.get("label")),
                _clean_text(dimension.get("leftClaim")),
                _clean_text(dimension.get("rightClaim")),
                _clean_text(dimension.get("notes")),
            ]
        )
        for dimension in dimensions
    )
    term_hits = _term_hits(dimension_text, expected_terms)
    statuses = [_normalize(item.get("comparisonStatus") or item.get("status")) for item in dimensions]
    invalid_statuses = [status for status in statuses if status not in expected_statuses]
    covered_sources = [source_id for source_id in expected_sources if source_id in source_ids]
    expected_source_coverage = 1.0 if not expected_sources else round(len(covered_sources) / len(expected_sources), 6)
    dimension_coverage = 1.0 if not expected_terms else round(len(term_hits) / len(expected_terms), 6)
    supporting_span_coverage = 1.0 if len(spans) >= min_spans else round(len(spans) / max(1, min_spans), 6)
    trace_citation_coverage = 1.0 if (not require_trace_citations or citations) else 0.0
    non_evidence_spans = [span for span in spans if _is_non_evidence_ref(span)]
    answerable = bool(coverage.get("answerable")) if "answerable" in coverage else bool(dimensions and spans)

    errors: list[str] = []
    if not packet:
        errors.append("compare_packet_missing")
    if expected_answerable and not answerable:
        errors.append("compare_packet_not_answerable")
    if expected_sources and expected_source_coverage < 1.0:
        errors.append("expected_source_coverage_incomplete")
    if expected_terms and dimension_coverage < 1.0:
        errors.append("dimension_terms_missing")
    if invalid_statuses:
        errors.append("unexpected_dimension_status")
    if len(spans) < min_spans:
        errors.append(f"supporting_span_count_below_min:{len(spans)}<{min_spans}")
    if require_trace_citations and not citations:
        errors.append("trace_citations_missing")
    if forbid_non_evidence and non_evidence_spans:
        errors.append("non_evidence_supporting_span_leak")
    payload_status = str(payload.get("status") or "").lower()
    if payload_status == "failed" or (
        expected_answerable and payload_status in {"insufficient_evidence", "insufficient_compare_contract"}
    ):
        errors.append(f"compare_status_not_ok:{payload.get('status')}")

    return {
        "caseId": case_id,
        "query": _clean_text(case.get("query")),
        "source": _clean_text(case.get("source") or case.get("source_type")),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "compareStatus": _clean_text(payload.get("status")),
        "comparePacketPresent": bool(packet),
        "answerable": answerable,
        "expectedSourceIds": expected_sources,
        "coveredExpectedSourceIds": covered_sources,
        "payloadSourceIds": sorted(payload_source_ids),
        "expectedSourceCoverage": expected_source_coverage,
        "expectedDimensionTerms": expected_terms,
        "matchedDimensionTerms": term_hits,
        "dimensionCoverage": dimension_coverage,
        "dimensionStatuses": statuses,
        "expectedStatuses": sorted(expected_statuses),
        "supportingSpanCount": len(spans),
        "expectedMinSupportingSpanCount": min_spans,
        "supportingSpanCoverage": supporting_span_coverage,
        "traceCitationCount": len(citations),
        "traceCitationCoverage": trace_citation_coverage,
        "nonEvidenceLeakCount": len(non_evidence_spans),
        "warnings": [str(item) for item in list(payload.get("warnings") or [])],
    }


def build_summary(
    cases: list[dict[str, Any]],
    case_results: list[dict[str, Any]],
    *,
    cases_path: Path,
    min_cases: int,
    min_compare_packet_present_rate: float,
    min_answerable_rate: float,
    min_expected_source_coverage_rate: float,
    min_dimension_coverage_rate: float,
    min_supporting_span_coverage_rate: float,
    min_trace_citation_coverage_rate: float,
    fail_on_insufficient: bool,
) -> dict[str, Any]:
    evaluated = [item for item in case_results if item.get("status") != "skipped"]
    failures = [item for item in evaluated if item.get("status") == "fail"]
    passes = [item for item in evaluated if item.get("status") == "pass"]
    compare_packet_present_rate = _ratio(sum(1 for item in evaluated if item.get("comparePacketPresent")), len(evaluated))
    answerable_rate = _ratio(sum(1 for item in evaluated if item.get("answerable")), len(evaluated))
    expected_source_coverage_rate = _ratio(
        sum(1 for item in evaluated if float(item.get("expectedSourceCoverage") or 0.0) >= 1.0),
        len(evaluated),
    )
    dimension_coverage_rate = _ratio(
        sum(1 for item in evaluated if float(item.get("dimensionCoverage") or 0.0) >= 1.0),
        len(evaluated),
    )
    supporting_span_coverage_rate = _ratio(
        sum(1 for item in evaluated if float(item.get("supportingSpanCoverage") or 0.0) >= 1.0),
        len(evaluated),
    )
    trace_citation_coverage_rate = _ratio(
        sum(1 for item in evaluated if float(item.get("traceCitationCoverage") or 0.0) >= 1.0),
        len(evaluated),
    )
    non_evidence_leak_count = sum(int(item.get("nonEvidenceLeakCount") or 0) for item in evaluated)

    errors: list[str] = []
    insufficient = len(evaluated) < int(min_cases)
    if insufficient:
        errors.append(f"insufficient_evaluable_cases:{len(evaluated)}/{int(min_cases)}")
    thresholds = [
        ("compare_packet_present_rate", compare_packet_present_rate, min_compare_packet_present_rate),
        ("answerable_rate", answerable_rate, min_answerable_rate),
        ("expected_source_coverage_rate", expected_source_coverage_rate, min_expected_source_coverage_rate),
        ("dimension_coverage_rate", dimension_coverage_rate, min_dimension_coverage_rate),
        ("supporting_span_coverage_rate", supporting_span_coverage_rate, min_supporting_span_coverage_rate),
        ("trace_citation_coverage_rate", trace_citation_coverage_rate, min_trace_citation_coverage_rate),
    ]
    for key, value, minimum in thresholds:
        if value is not None and float(value) < float(minimum):
            errors.append(f"{key}_below_threshold:{value}<{minimum}")
    if non_evidence_leak_count:
        errors.append(f"non_evidence_leaks:{non_evidence_leak_count}")
    if failures:
        errors.append(f"failed_cases:{len(failures)}")

    if insufficient and not fail_on_insufficient:
        status = "skipped"
    else:
        status = "ok" if not errors else "failed"

    return {
        "schema": SCHEMA,
        "createdAt": _now_iso(),
        "status": status,
        "casesPath": str(cases_path),
        "caseCount": len(cases),
        "evaluatedCaseCount": len(evaluated),
        "skippedCaseCount": len(case_results) - len(evaluated),
        "passedCaseCount": len(passes),
        "failedCaseCount": len(failures),
        "comparePacketPresentRate": compare_packet_present_rate,
        "answerableRate": answerable_rate,
        "expectedSourceCoverageRate": expected_source_coverage_rate,
        "dimensionCoverageRate": dimension_coverage_rate,
        "supportingSpanCoverageRate": supporting_span_coverage_rate,
        "traceCitationCoverageRate": trace_citation_coverage_rate,
        "nonEvidenceLeakCount": non_evidence_leak_count,
        "minCases": int(min_cases),
        "thresholds": {
            "comparePacketPresentRate": float(min_compare_packet_present_rate),
            "answerableRate": float(min_answerable_rate),
            "expectedSourceCoverageRate": float(min_expected_source_coverage_rate),
            "dimensionCoverageRate": float(min_dimension_coverage_rate),
            "supportingSpanCoverageRate": float(min_supporting_span_coverage_rate),
            "traceCitationCoverageRate": float(min_trace_citation_coverage_rate),
        },
        "failOnInsufficient": bool(fail_on_insufficient),
        "errors": errors,
        "cases": case_results,
    }


def _parse_json_output(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def run_compare_command(
    case: dict[str, Any],
    *,
    timeout_seconds: int,
    allow_external: bool,
) -> dict[str, Any]:
    query = _clean_text(case.get("query"))
    command = [sys.executable, "-m", "knowledge_hub.interfaces.cli.main", "compare", query, "--json"]
    source = _clean_text(case.get("source") or case.get("source_type"))
    if source and _normalize(source) in PUBLIC_SOURCE_TYPES:
        command.extend(["--source", source])
    command.extend(["--top-k", str(_as_int(case.get("top_k"), 8))])
    mode = _normalize(case.get("mode") or "hybrid")
    if mode in {"semantic", "keyword", "hybrid"}:
        command.extend(["--mode", mode])
    command.extend(["--alpha", str(_as_float(case.get("alpha"), 0.7))])
    if allow_external and _as_bool(case.get("allow_external"), False):
        command.append("--allow-external")
    else:
        command.append("--no-allow-external")

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=max(1, int(timeout_seconds)),
        check=False,
    )
    if completed.returncode != 0:
        return {
            "schema": "knowledge-hub.compare.result.v1",
            "status": "failed",
            "query": query,
            "comparePacket": {},
            "trace": {},
            "citations": [],
            "sources": [],
            "warnings": [completed.stderr.strip() or f"khub compare exited {completed.returncode}"],
        }
    return _parse_json_output(completed.stdout)


def run_live_compare_quality_eval(
    *,
    cases: list[dict[str, Any]],
    compare_runner: Callable[[dict[str, Any]], dict[str, Any]],
    cases_path: Path,
    min_cases: int,
    min_compare_packet_present_rate: float,
    min_answerable_rate: float,
    min_expected_source_coverage_rate: float,
    min_dimension_coverage_rate: float,
    min_supporting_span_coverage_rate: float,
    min_trace_citation_coverage_rate: float,
    fail_on_insufficient: bool,
) -> dict[str, Any]:
    case_results: list[dict[str, Any]] = []
    for case in cases:
        try:
            payload = compare_runner(case)
            case_results.append(evaluate_case(case, payload))
        except Exception as error:  # pragma: no cover - operator resilience
            case_results.append(
                {
                    "caseId": _clean_text(case.get("case_id")),
                    "query": _clean_text(case.get("query")),
                    "status": "fail",
                    "errors": [f"case_execution_failed:{error}"],
                    "comparePacketPresent": False,
                    "answerable": False,
                    "expectedSourceCoverage": 0.0,
                    "dimensionCoverage": 0.0,
                    "supportingSpanCoverage": 0.0,
                    "traceCitationCoverage": 0.0,
                    "nonEvidenceLeakCount": 0,
                }
            )
    return build_summary(
        cases,
        case_results,
        cases_path=cases_path,
        min_cases=min_cases,
        min_compare_packet_present_rate=min_compare_packet_present_rate,
        min_answerable_rate=min_answerable_rate,
        min_expected_source_coverage_rate=min_expected_source_coverage_rate,
        min_dimension_coverage_rate=min_dimension_coverage_rate,
        min_supporting_span_coverage_rate=min_supporting_span_coverage_rate,
        min_trace_citation_coverage_rate=min_trace_citation_coverage_rate,
        fail_on_insufficient=fail_on_insufficient,
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Live Compare Quality Eval",
        "",
        f"- status: `{payload.get('status')}`",
        f"- cases: `{payload.get('evaluatedCaseCount')}` evaluated / `{payload.get('caseCount')}` loaded",
        f"- compare packet present: `{payload.get('comparePacketPresentRate')}`",
        f"- answerable: `{payload.get('answerableRate')}`",
        f"- expected source coverage: `{payload.get('expectedSourceCoverageRate')}`",
        f"- dimension coverage: `{payload.get('dimensionCoverageRate')}`",
        f"- supporting span coverage: `{payload.get('supportingSpanCoverageRate')}`",
        f"- trace citation coverage: `{payload.get('traceCitationCoverageRate')}`",
        f"- non-evidence leaks: `{payload.get('nonEvidenceLeakCount')}`",
        f"- cases path: `{payload.get('casesPath')}`",
    ]
    errors = list(payload.get("errors") or [])
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in errors)
    failed_cases = [item for item in list(payload.get("cases") or []) if item.get("status") == "fail"]
    if failed_cases:
        lines.extend(["", "## Failed Cases", ""])
        for item in failed_cases:
            lines.append(f"- `{item.get('caseId')}`: {', '.join(item.get('errors') or [])}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check live local-corpus compare quality against curated cases.")
    parser.add_argument("--cases", default=DEFAULT_CASES_PATH)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--min-cases", type=int, default=1)
    parser.add_argument("--min-compare-packet-present-rate", type=float, default=1.0)
    parser.add_argument("--min-answerable-rate", type=float, default=1.0)
    parser.add_argument("--min-expected-source-coverage-rate", type=float, default=1.0)
    parser.add_argument("--min-dimension-coverage-rate", type=float, default=1.0)
    parser.add_argument("--min-supporting-span-coverage-rate", type=float, default=1.0)
    parser.add_argument("--min-trace-citation-coverage-rate", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--allow-external", action="store_true", default=False)
    parser.add_argument("--fail-on-insufficient", action="store_true", default=False)
    parser.add_argument("--json", action="store_true", dest="as_json", default=False)
    args = parser.parse_args(argv)

    cases_path = Path(args.cases).expanduser().resolve()
    cases = _read_cases(cases_path)
    payload = run_live_compare_quality_eval(
        cases=cases,
        compare_runner=lambda case: run_compare_command(
            case,
            timeout_seconds=int(args.timeout_seconds),
            allow_external=bool(args.allow_external),
        ),
        cases_path=cases_path,
        min_cases=int(args.min_cases),
        min_compare_packet_present_rate=float(args.min_compare_packet_present_rate),
        min_answerable_rate=float(args.min_answerable_rate),
        min_expected_source_coverage_rate=float(args.min_expected_source_coverage_rate),
        min_dimension_coverage_rate=float(args.min_dimension_coverage_rate),
        min_supporting_span_coverage_rate=float(args.min_supporting_span_coverage_rate),
        min_trace_citation_coverage_rate=float(args.min_trace_citation_coverage_rate),
        fail_on_insufficient=bool(args.fail_on_insufficient),
    )
    if args.out_json:
        out_json = Path(args.out_json).expanduser()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md).expanduser()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_markdown(payload), encoding="utf-8")
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_markdown(payload), end="")
    return 0 if payload.get("status") in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
