#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_hub.ai.answer_contracts import NON_EVIDENCE_SOURCE_SCHEMES, NON_EVIDENCE_SOURCE_TYPES
from knowledge_hub.application.corpus_artifacts import (
    corpus_entry_ref,
    find_corpus_entry_for_source,
    inspect_corpus_requirement,
    load_corpus_manifest,
    public_corpus_artifact_diagnostic,
)
from knowledge_hub.core.config import Config
from knowledge_hub.domain.source_identity import (
    alias_groups_for_items,
    aliases_with_groups,
    source_identity_aliases,
)


SCHEMA = "knowledge-hub.live-compare-quality-eval.result.v1"
DEFAULT_CASES_PATH = "eval/knowledgeos/queries/live_compare_quality_eval_cases.local.json"
VALID_STATUSES = {"supported", "conflict", "unknown", "insufficient"}
PUBLIC_SOURCE_TYPES = {"paper", "vault", "web", "concept"}
OFFSET_LOCATOR_RE = re.compile(r"^chars:\d+-\d+$")


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


def _case_corpus_requirements(case: dict[str, Any]) -> list[dict[str, Any]]:
    raw = case.get("corpusRequirements") or case.get("corpus_requirements") or []
    if not isinstance(raw, list):
        return []
    return [dict(item or {}) for item in raw if isinstance(item, dict)]


def _case_expected_source_ids(case: dict[str, Any]) -> list[str]:
    return _as_list(case.get("expected_source_ids") or case.get("expected_source_id"))


def _case_corpus_requirements_or_derived(
    case: dict[str, Any],
    *,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    requirements = _case_corpus_requirements(case)
    if requirements:
        return requirements, [], 0

    expected_sources = _case_expected_source_ids(case)
    if not expected_sources:
        return [], [], 0

    derived: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    derived_by_artifact: dict[str, dict[str, Any]] = {}
    for source_id in expected_sources:
        entry = find_corpus_entry_for_source(source_id, manifest)
        if entry is None:
            missing.append(
                {
                    "sourceId": source_id,
                    "status": "missing_manifest_entry",
                    "reason": "expected source id has no corpus manifest entry",
                }
            )
            continue
        artifact_id = corpus_entry_ref(entry)
        key = artifact_id.casefold()
        if key in derived_by_artifact:
            derived_by_artifact[key].setdefault("derivedFromExpectedSourceIds", []).append(source_id)
            continue
        requirement: dict[str, Any] = {
            "artifactId": artifact_id,
            "derivedFromExpectedSourceIds": [source_id],
        }
        if entry.get("corpusTier") not in (None, ""):
            requirement["corpusTier"] = entry["corpusTier"]
        if entry.get("minOffsetsRequired") not in (None, ""):
            requirement["minOffsetsRequired"] = entry["minOffsetsRequired"]
        derived_by_artifact[key] = requirement
        derived.append(requirement)
    return derived, missing, len(derived)


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


def _collect_source_ids(spans: list[dict[str, Any]]) -> set[str]:
    source_ids: set[str] = set()
    for item in spans:
        source_id = _source_id_from_item(item)
        if source_id:
            source_ids.add(source_id)
    return source_ids


def _collect_payload_source_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    trace = dict(payload.get("trace") or {})
    for raw in [
        *list(payload.get("citations") or trace.get("citations") or []),
        *list(payload.get("sources") or trace.get("sources") or []),
    ]:
        if isinstance(raw, dict):
            items.append(dict(raw))
    return items


def _coverage_for_expected_sources(
    expected_sources: list[str],
    spans: list[dict[str, Any]],
    *,
    alias_items: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    if not expected_sources:
        return [], [], []
    groups = alias_groups_for_items([*spans, *list(alias_items or [])])
    covered: list[str] = []
    alias_resolved: list[str] = []
    unresolved: list[str] = []
    raw_source_ids = _collect_source_ids(spans)
    expanded_span_aliases = [aliases_with_groups(span, groups) for span in spans]
    for expected in expected_sources:
        expected_aliases = source_identity_aliases(expected)
        raw_match = expected in raw_source_ids
        alias_match = bool(expected_aliases and any(expected_aliases & aliases for aliases in expanded_span_aliases))
        if raw_match or alias_match:
            covered.append(expected)
            if alias_match and not raw_match:
                alias_resolved.append(expected)
        else:
            unresolved.append(expected)
    return covered, alias_resolved, unresolved


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


def _span_locator(item: dict[str, Any]) -> str:
    return _clean_text(item.get("spanLocator") or item.get("span_locator") or item.get("locator"))


def _source_content_hash(item: dict[str, Any]) -> str:
    return _clean_text(item.get("sourceContentHash") or item.get("source_content_hash"))


def _has_strict_chars_locator(item: dict[str, Any]) -> bool:
    return bool(OFFSET_LOCATOR_RE.match(_span_locator(item)))


def _is_strict_source_span(item: dict[str, Any]) -> bool:
    if not _as_bool(item.get("strictSpanBacked"), False):
        return False
    if _as_bool(item.get("fallbackSpan"), False):
        return False
    return bool(_source_content_hash(item) and _has_strict_chars_locator(item))


def _corpus_skip_reason(requirement_results: list[dict[str, Any]]) -> str:
    statuses = {_clean_text(item.get("status")) for item in requirement_results}
    if "hash_mismatch" in statuses:
        return "skipped_hash_mismatch"
    if "missing_manifest_entry" in statuses:
        return "skipped_missing_manifest_entry"
    if "missing_artifact" in statuses:
        return "skipped_missing_corpus"
    return "skipped_corpus_unavailable"


def _evaluate_corpus_requirements(
    case: dict[str, Any],
    *,
    manifest: dict[str, Any],
    config: Any,
    derive_requirements: bool = False,
) -> dict[str, Any]:
    if derive_requirements:
        requirements, missing_requirements, derived_requirement_count = _case_corpus_requirements_or_derived(
            case,
            manifest=manifest,
        )
    else:
        requirements = _case_corpus_requirements(case)
        missing_requirements = []
        derived_requirement_count = 0
    results = [
        public_corpus_artifact_diagnostic(inspect_corpus_requirement(requirement, manifest=manifest, config=config))
        for requirement in requirements
    ]
    if missing_requirements:
        return {
            "evaluable": False,
            "hardFailure": True,
            "requirements": results,
            "missingCorpusRequirements": missing_requirements,
            "derivedCorpusRequirementCount": derived_requirement_count,
            "skipReason": "missing_corpus_requirement",
        }
    if not requirements:
        return {
            "evaluable": True,
            "requirements": [],
            "derivedCorpusRequirementCount": 0,
            "missingCorpusRequirements": [],
        }
    missing_or_mismatch = [item for item in results if _clean_text(item.get("status")) != "ok"]
    if not missing_or_mismatch:
        return {
            "evaluable": True,
            "requirements": results,
            "derivedCorpusRequirementCount": derived_requirement_count,
            "missingCorpusRequirements": [],
        }
    hard_fail = any(_clean_text(item.get("corpusTier")) == "repo_fixture" for item in missing_or_mismatch)
    blocking = [
        item
        for item in missing_or_mismatch
        if _clean_text(item.get("corpusTier")) in {"local_corpus", "repo_fixture", ""}
    ]
    if not hard_fail and not blocking:
        return {
            "evaluable": True,
            "requirements": results,
            "derivedCorpusRequirementCount": derived_requirement_count,
            "missingCorpusRequirements": [],
        }
    return {
        "evaluable": False,
        "hardFailure": hard_fail,
        "requirements": results,
        "missingCorpusRequirements": [],
        "derivedCorpusRequirementCount": derived_requirement_count,
        "skipReason": _corpus_skip_reason(blocking or missing_or_mismatch),
    }


def _has_offset_locator(item: dict[str, Any]) -> bool:
    locator = _span_locator(item)
    if locator and OFFSET_LOCATOR_RE.match(locator):
        return True
    try:
        int(item.get("charStart") if item.get("charStart") is not None else item.get("char_start"))
        int(item.get("charEnd") if item.get("charEnd") is not None else item.get("char_end"))
        return True
    except (TypeError, ValueError):
        return False


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = _clean_text(value)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _failure_categories(errors: list[str], diagnostics: list[str]) -> list[str]:
    categories: set[str] = set()
    for error in errors:
        if error == "compare_packet_missing":
            categories.add("compare_packet_missing")
        elif error == "compare_packet_not_answerable":
            categories.add("not_answerable")
        elif error == "compare_packet_unexpectedly_answerable":
            categories.add("false_positive_answerable")
        elif error.startswith("strict_span_count_below_min"):
            categories.add("strict_span_gap")
        elif error == "expected_strict_source_coverage_below_min":
            categories.add("strict_source_coverage_gap")
        elif error == "expected_source_coverage_incomplete":
            categories.add("expected_source_coverage_gap")
        elif error == "dimension_terms_missing":
            categories.add("dimension_gap")
        elif error == "unexpected_dimension_status":
            categories.add("unexpected_dimension_status")
        elif error.startswith("supporting_span_count_below_min"):
            categories.add("supporting_span_gap")
        elif error == "trace_citations_missing":
            categories.add("citation_gap")
        elif error == "non_evidence_supporting_span_leak":
            categories.add("non_evidence_leak")
        elif error.startswith("compare_status_not_ok"):
            categories.add("compare_status_not_ok")
        elif error.startswith("case_execution_failed"):
            categories.add("compare_runtime_failed")
    if errors:
        if "fallback_only_support" in diagnostics:
            categories.add("fallback_only")
        if "locator_only_spans_present" in diagnostics:
            categories.add("locator_only_anchor")
        if "trace_without_strict_spans" in diagnostics:
            categories.add("trace_without_strict_spans")
        if "unresolved_expected_source_alias" in diagnostics:
            categories.add("unresolved_expected_source_alias")
    return sorted(categories)


def evaluate_case(
    case: dict[str, Any],
    payload: dict[str, Any],
    *,
    default_min_supporting_span_count: int = 1,
) -> dict[str, Any]:
    case_id = _clean_text(case.get("case_id"))
    expected_sources = _case_expected_source_ids(case)
    expected_terms = _as_list(case.get("expected_dimension_terms") or case.get("expected_terms"))
    expected_statuses = {_normalize(item) for item in _as_list(case.get("expected_statuses"))}
    expected_statuses = {item for item in expected_statuses if item in VALID_STATUSES} or set(VALID_STATUSES)
    min_spans = _as_int(case.get("expected_min_supporting_span_count"), default_min_supporting_span_count)
    expected_answerable = _as_bool(case.get("expected_answerable"), True)
    min_strict_spans = _as_int(
        case.get("expected_min_strict_span_count"),
        min_spans if expected_answerable else 0,
    )
    min_strict_source_coverage = _as_float(
        case.get("expected_min_strict_source_coverage"),
        1.0 if expected_answerable and expected_sources else 0.0,
    )
    require_trace_citations = _as_bool(case.get("require_trace_citations"), True)
    forbid_non_evidence = _as_bool(case.get("forbidden_non_evidence_support"), True)

    packet = _compare_packet(payload)
    dimensions = _dimensions(packet)
    spans = _supporting_spans(dimensions)
    strict_spans = [span for span in spans if _is_strict_source_span(span)]
    fallback_spans = [span for span in spans if _as_bool(span.get("fallbackSpan"), False)]
    locator_only_spans = [
        span
        for span in spans
        if _span_locator(span)
        and not _as_bool(span.get("strictSpanBacked"), False)
        and not _as_bool(span.get("fallbackSpan"), False)
    ]
    offset_backed_spans = [span for span in spans if _has_offset_locator(span)]
    coverage = dict(packet.get("coverage") or {})
    trace = dict(payload.get("trace") or {})
    citations = list(payload.get("citations") or trace.get("citations") or [])
    payload_source_items = _collect_payload_source_items(payload)
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
    covered_sources, alias_resolved_sources, unresolved_source_aliases = _coverage_for_expected_sources(
        expected_sources,
        spans,
        alias_items=payload_source_items,
    )
    strict_covered_sources, strict_alias_resolved_sources, strict_unresolved_source_aliases = _coverage_for_expected_sources(
        expected_sources,
        strict_spans,
        alias_items=payload_source_items,
    )
    fallback_covered_sources, fallback_alias_resolved_sources, fallback_unresolved_source_aliases = _coverage_for_expected_sources(
        expected_sources,
        fallback_spans,
        alias_items=payload_source_items,
    )
    expected_source_coverage = 1.0 if not expected_sources else round(len(covered_sources) / len(expected_sources), 6)
    expected_strict_source_coverage = 1.0 if not expected_sources else round(len(strict_covered_sources) / len(expected_sources), 6)
    expected_fallback_source_coverage = 0.0 if not expected_sources else round(len(fallback_covered_sources) / len(expected_sources), 6)
    dimension_coverage = 1.0 if not expected_terms else round(len(term_hits) / len(expected_terms), 6)
    supporting_span_coverage = 1.0 if len(spans) >= min_spans else round(len(spans) / max(1, min_spans), 6)
    strict_span_coverage = 1.0 if len(strict_spans) >= min_strict_spans else round(len(strict_spans) / max(1, min_strict_spans), 6)
    fallback_span_share = 0.0 if not spans else round(len(fallback_spans) / len(spans), 6)
    trace_citation_coverage = 1.0 if (not require_trace_citations or citations) else 0.0
    non_evidence_spans = [span for span in spans if _is_non_evidence_ref(span)]
    answerable = bool(coverage.get("answerable")) if "answerable" in coverage else bool(dimensions and spans)

    provenance_diagnostics: list[str] = []
    if spans and not strict_spans:
        provenance_diagnostics.append("fallback_only_support" if fallback_spans else "no_strict_spans")
    if fallback_spans:
        provenance_diagnostics.append("fallback_spans_present")
    if locator_only_spans:
        provenance_diagnostics.append("locator_only_spans_present")
    if expected_sources and expected_strict_source_coverage < 1.0:
        provenance_diagnostics.append("strict_source_coverage_gap")
    if alias_resolved_sources:
        provenance_diagnostics.append("alias_resolved_expected_source")
    if unresolved_source_aliases:
        provenance_diagnostics.append("unresolved_expected_source_alias")
    if citations and not strict_spans:
        provenance_diagnostics.append("trace_without_strict_spans")

    errors: list[str] = []
    if not packet:
        errors.append("compare_packet_missing")
    if expected_answerable and not answerable:
        errors.append("compare_packet_not_answerable")
    if not expected_answerable and answerable:
        errors.append("compare_packet_unexpectedly_answerable")
    if len(strict_spans) < min_strict_spans:
        errors.append(f"strict_span_count_below_min:{len(strict_spans)}<{min_strict_spans}")
    if expected_sources and expected_strict_source_coverage < min_strict_source_coverage:
        errors.append("expected_strict_source_coverage_below_min")
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
    failure_categories = _failure_categories(errors, provenance_diagnostics)

    return {
        "caseId": case_id,
        "query": _clean_text(case.get("query")),
        "source": _clean_text(case.get("source") or case.get("source_type")),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "compareStatus": _clean_text(payload.get("status")),
        "comparePacketPresent": bool(packet),
        "answerable": answerable,
        "expectedAnswerable": expected_answerable,
        "expectedSourceIds": expected_sources,
        "coveredExpectedSourceIds": covered_sources,
        "aliasResolvedExpectedSourceIds": alias_resolved_sources,
        "unresolvedExpectedSourceAliases": unresolved_source_aliases,
        "strictCoveredExpectedSourceIds": strict_covered_sources,
        "strictAliasResolvedExpectedSourceIds": strict_alias_resolved_sources,
        "strictUnresolvedExpectedSourceAliases": strict_unresolved_source_aliases,
        "fallbackCoveredExpectedSourceIds": fallback_covered_sources,
        "fallbackAliasResolvedExpectedSourceIds": fallback_alias_resolved_sources,
        "fallbackUnresolvedExpectedSourceAliases": fallback_unresolved_source_aliases,
        "payloadSourceIds": sorted(payload_source_ids),
        "expectedSourceCoverage": expected_source_coverage,
        "expectedStrictSourceCoverage": expected_strict_source_coverage,
        "expectedFallbackSourceCoverage": expected_fallback_source_coverage,
        "expectedMinStrictSourceCoverage": min_strict_source_coverage,
        "expectedDimensionTerms": expected_terms,
        "matchedDimensionTerms": term_hits,
        "dimensionCoverage": dimension_coverage,
        "dimensionStatuses": statuses,
        "expectedStatuses": sorted(expected_statuses),
        "supportingSpanCount": len(spans),
        "expectedMinSupportingSpanCount": min_spans,
        "supportingSpanCoverage": supporting_span_coverage,
        "strictSpanBackedCount": len(strict_spans),
        "fallbackSpanCount": len(fallback_spans),
        "locatorOnlySpanCount": len(locator_only_spans),
        "offsetBackedSpanCount": len(offset_backed_spans),
        "expectedMinStrictSpanCount": min_strict_spans,
        "strictSpanCoverage": strict_span_coverage,
        "fallbackSpanShare": fallback_span_share,
        "traceCitationCount": len(citations),
        "traceCitationCoverage": trace_citation_coverage,
        "nonEvidenceLeakCount": len(non_evidence_spans),
        "provenanceDiagnostics": provenance_diagnostics,
        "failureCategories": failure_categories,
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
    min_expected_answerable_strict_source_coverage_rate: float,
    min_dimension_coverage_rate: float,
    min_supporting_span_coverage_rate: float,
    min_trace_citation_coverage_rate: float,
    min_corpus_coverage_rate: float,
    fail_on_insufficient: bool,
) -> dict[str, Any]:
    evaluated = [item for item in case_results if item.get("status") != "skipped"]
    skipped_missing_corpus = [item for item in case_results if item.get("skipReason") == "skipped_missing_corpus"]
    skipped_hash_mismatch = [item for item in case_results if item.get("skipReason") == "skipped_hash_mismatch"]
    skipped_missing_manifest_entry = [item for item in case_results if item.get("skipReason") == "skipped_missing_manifest_entry"]
    failures = [item for item in evaluated if item.get("status") == "fail"]
    passes = [item for item in evaluated if item.get("status") == "pass"]
    expected_answerable_cases = [item for item in evaluated if bool(item.get("expectedAnswerable"))]
    expected_no_answer_cases = [item for item in evaluated if not bool(item.get("expectedAnswerable"))]
    compare_packet_present_rate = _ratio(sum(1 for item in evaluated if item.get("comparePacketPresent")), len(evaluated))
    answerable_rate = _ratio(sum(1 for item in evaluated if item.get("answerable")), len(evaluated))
    expected_answerable_pass_rate = _ratio(sum(1 for item in expected_answerable_cases if item.get("status") == "pass"), len(expected_answerable_cases))
    expected_no_answer_pass_rate = _ratio(sum(1 for item in expected_no_answer_cases if item.get("status") == "pass"), len(expected_no_answer_cases))
    expected_source_coverage_rate = _ratio(
        sum(1 for item in evaluated if float(item.get("expectedSourceCoverage") or 0.0) >= 1.0),
        len(evaluated),
    )
    expected_answerable_strict_source_coverage_rate = _ratio(
        sum(1 for item in expected_answerable_cases if float(item.get("expectedStrictSourceCoverage") or 0.0) >= 1.0),
        len(expected_answerable_cases),
    )
    dimension_coverage_rate = _ratio(
        sum(1 for item in evaluated if float(item.get("dimensionCoverage") or 0.0) >= 1.0),
        len(evaluated),
    )
    supporting_span_coverage_rate = _ratio(
        sum(1 for item in evaluated if float(item.get("supportingSpanCoverage") or 0.0) >= 1.0),
        len(evaluated),
    )
    strict_span_coverage_rate = _ratio(
        sum(1 for item in evaluated if float(item.get("strictSpanCoverage") or 0.0) >= 1.0),
        len(evaluated),
    )
    fallback_span_case_rate = _ratio(sum(1 for item in evaluated if int(item.get("fallbackSpanCount") or 0) > 0), len(evaluated))
    fallback_only_case_rate = _ratio(
        sum(1 for item in evaluated if "fallback_only_support" in list(item.get("provenanceDiagnostics") or [])),
        len(evaluated),
    )
    locator_only_case_rate = _ratio(
        sum(1 for item in evaluated if int(item.get("locatorOnlySpanCount") or 0) > 0),
        len(evaluated),
    )
    trace_citation_coverage_rate = _ratio(
        sum(1 for item in evaluated if float(item.get("traceCitationCoverage") or 0.0) >= 1.0),
        len(evaluated),
    )
    non_evidence_leak_count = sum(int(item.get("nonEvidenceLeakCount") or 0) for item in evaluated)
    coverage_pct = _ratio(len(evaluated), len(cases))
    failure_category_counts = _count_values(
        [
            category
            for item in failures
            for category in list(item.get("failureCategories") or [])
        ]
    )
    provenance_diagnostic_counts = _count_values(
        [
            diagnostic
            for item in evaluated
            for diagnostic in list(item.get("provenanceDiagnostics") or [])
        ]
    )
    missing_corpus_requirement_count = sum(
        len(list(item.get("missingCorpusRequirements") or []))
        for item in case_results
    )
    derived_corpus_requirement_count = sum(int(item.get("derivedCorpusRequirementCount") or 0) for item in case_results)

    errors: list[str] = []
    insufficient = len(evaluated) < int(min_cases)
    if insufficient:
        errors.append(f"insufficient_evaluable_cases:{len(evaluated)}/{int(min_cases)}")
    thresholds = [
        ("compare_packet_present_rate", compare_packet_present_rate, min_compare_packet_present_rate),
        ("expected_answerable_pass_rate", expected_answerable_pass_rate, min_answerable_rate),
        ("expected_source_coverage_rate", expected_source_coverage_rate, min_expected_source_coverage_rate),
        (
            "expected_answerable_strict_source_coverage_rate",
            expected_answerable_strict_source_coverage_rate,
            min_expected_answerable_strict_source_coverage_rate,
        ),
        ("dimension_coverage_rate", dimension_coverage_rate, min_dimension_coverage_rate),
        ("supporting_span_coverage_rate", supporting_span_coverage_rate, min_supporting_span_coverage_rate),
        ("trace_citation_coverage_rate", trace_citation_coverage_rate, min_trace_citation_coverage_rate),
        ("corpus_coverage_rate", coverage_pct, min_corpus_coverage_rate),
    ]
    for key, value, minimum in thresholds:
        if value is not None and float(value) < float(minimum):
            errors.append(f"{key}_below_threshold:{value}<{minimum}")
    if non_evidence_leak_count:
        errors.append(f"non_evidence_leaks:{non_evidence_leak_count}")
    if missing_corpus_requirement_count:
        errors.append(f"missing_corpus_requirements:{missing_corpus_requirement_count}")
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
        "declaredCaseCount": len(cases),
        "evaluatedCaseCount": len(evaluated),
        "evaluableCaseCount": len(evaluated),
        "skippedCaseCount": len(case_results) - len(evaluated),
        "coveragePct": coverage_pct,
        "coverage_pct": coverage_pct,
        "skippedForMissingCorpus": len(skipped_missing_corpus),
        "skipped_for_missing_corpus": len(skipped_missing_corpus),
        "skippedForHashMismatch": len(skipped_hash_mismatch),
        "skippedForMissingManifestEntry": len(skipped_missing_manifest_entry),
        "missingCorpusRequirementCount": missing_corpus_requirement_count,
        "derivedCorpusRequirementCount": derived_corpus_requirement_count,
        "coverageReport": {
            "passed": len(passes),
            "evaluable": len(evaluated),
            "declared": len(cases),
            "coverage_pct": coverage_pct,
            "skipped_for_missing_corpus": len(skipped_missing_corpus),
            "skipped_for_hash_mismatch": len(skipped_hash_mismatch),
            "missing_corpus_requirements": missing_corpus_requirement_count,
            "derived_corpus_requirements": derived_corpus_requirement_count,
        },
        "passedCaseCount": len(passes),
        "failedCaseCount": len(failures),
        "comparePacketPresentRate": compare_packet_present_rate,
        "answerableRate": answerable_rate,
        "expectedAnswerableCaseCount": len(expected_answerable_cases),
        "expectedNoAnswerCaseCount": len(expected_no_answer_cases),
        "expectedAnswerablePassRate": expected_answerable_pass_rate,
        "expectedNoAnswerPassRate": expected_no_answer_pass_rate,
        "expectedSourceCoverageRate": expected_source_coverage_rate,
        "expectedAnswerableStrictSourceCoverageRate": expected_answerable_strict_source_coverage_rate,
        "dimensionCoverageRate": dimension_coverage_rate,
        "supportingSpanCoverageRate": supporting_span_coverage_rate,
        "strictSpanCoverageRate": strict_span_coverage_rate,
        "fallbackSpanCaseRate": fallback_span_case_rate,
        "fallbackOnlyCaseRate": fallback_only_case_rate,
        "locatorOnlyCaseRate": locator_only_case_rate,
        "traceCitationCoverageRate": trace_citation_coverage_rate,
        "nonEvidenceLeakCount": non_evidence_leak_count,
        "failureCategoryCounts": failure_category_counts,
        "provenanceDiagnosticCounts": provenance_diagnostic_counts,
        "minCases": int(min_cases),
        "thresholds": {
            "comparePacketPresentRate": float(min_compare_packet_present_rate),
            "expectedAnswerablePassRate": float(min_answerable_rate),
            "expectedSourceCoverageRate": float(min_expected_source_coverage_rate),
            "expectedAnswerableStrictSourceCoverageRate": float(min_expected_answerable_strict_source_coverage_rate),
            "dimensionCoverageRate": float(min_dimension_coverage_rate),
            "supportingSpanCoverageRate": float(min_supporting_span_coverage_rate),
            "traceCitationCoverageRate": float(min_trace_citation_coverage_rate),
            "corpusCoverageRate": float(min_corpus_coverage_rate),
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
    min_expected_answerable_strict_source_coverage_rate: float,
    min_dimension_coverage_rate: float,
    min_supporting_span_coverage_rate: float,
    min_trace_citation_coverage_rate: float,
    fail_on_insufficient: bool,
    min_corpus_coverage_rate: float = 1.0,
    corpus_manifest_path: str | Path | None = None,
    config: Any | None = None,
    derive_corpus_requirements: bool = False,
) -> dict[str, Any]:
    case_results: list[dict[str, Any]] = []
    manifest = load_corpus_manifest(corpus_manifest_path)
    active_config = config or Config()
    for case in cases:
        try:
            corpus = _evaluate_corpus_requirements(
                case,
                manifest=manifest,
                config=active_config,
                derive_requirements=derive_corpus_requirements,
            )
            if not corpus.get("evaluable"):
                hard_failure = bool(corpus.get("hardFailure"))
                skip_reason = _clean_text(corpus.get("skipReason"))
                missing_requirements = list(corpus.get("missingCorpusRequirements") or [])
                error_reason = "missing_corpus_requirement" if missing_requirements else skip_reason
                case_results.append(
                    {
                        "caseId": _clean_text(case.get("case_id")),
                        "query": _clean_text(case.get("query")),
                        "source": _clean_text(case.get("source") or case.get("source_type")),
                        "status": "fail" if hard_failure else "skipped",
                        "skipReason": skip_reason if not hard_failure else "",
                        "corpusRequirements": corpus.get("requirements") or [],
                        "missingCorpusRequirements": missing_requirements,
                        "derivedCorpusRequirementCount": int(corpus.get("derivedCorpusRequirementCount") or 0),
                        "errors": [f"corpus_requirement_failed:{error_reason}"] if hard_failure else [],
                        "comparePacketPresent": False,
                        "answerable": False,
                        "expectedAnswerable": _as_bool(case.get("expected_answerable"), True),
                        "nonEvidenceLeakCount": 0,
                        "failureCategories": ["corpus_requirement_failed"] if hard_failure else [],
                        "provenanceDiagnostics": [],
                    }
                )
                continue
            payload = compare_runner(case)
            result = evaluate_case(case, payload)
            result["corpusRequirements"] = corpus.get("requirements") or []
            result["missingCorpusRequirements"] = list(corpus.get("missingCorpusRequirements") or [])
            result["derivedCorpusRequirementCount"] = int(corpus.get("derivedCorpusRequirementCount") or 0)
            case_results.append(result)
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
                    "expectedStrictSourceCoverage": 0.0,
                    "traceCitationCoverage": 0.0,
                    "nonEvidenceLeakCount": 0,
                    "failureCategories": ["compare_runtime_failed"],
                    "provenanceDiagnostics": [],
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
        min_expected_answerable_strict_source_coverage_rate=min_expected_answerable_strict_source_coverage_rate,
        min_dimension_coverage_rate=min_dimension_coverage_rate,
        min_supporting_span_coverage_rate=min_supporting_span_coverage_rate,
        min_trace_citation_coverage_rate=min_trace_citation_coverage_rate,
        min_corpus_coverage_rate=min_corpus_coverage_rate,
        fail_on_insufficient=fail_on_insufficient,
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Live Compare Quality Eval",
        "",
        f"- status: `{payload.get('status')}`",
        f"- cases: `{payload.get('evaluatedCaseCount')}` evaluated / `{payload.get('declaredCaseCount', payload.get('caseCount'))}` declared",
        f"- coverage: `{payload.get('coveragePct')}`",
        f"- skipped missing corpus: `{payload.get('skippedForMissingCorpus')}`",
        f"- skipped hash mismatch: `{payload.get('skippedForHashMismatch')}`",
        f"- derived corpus requirements: `{payload.get('derivedCorpusRequirementCount')}`",
        f"- missing corpus requirements: `{payload.get('missingCorpusRequirementCount')}`",
        f"- compare packet present: `{payload.get('comparePacketPresentRate')}`",
        f"- answerable: `{payload.get('answerableRate')}`",
        f"- expected answerable pass: `{payload.get('expectedAnswerablePassRate')}`",
        f"- expected no-answer pass: `{payload.get('expectedNoAnswerPassRate')}`",
        f"- expected source coverage: `{payload.get('expectedSourceCoverageRate')}`",
        f"- expected answerable strict source coverage: `{payload.get('expectedAnswerableStrictSourceCoverageRate')}`",
        f"- dimension coverage: `{payload.get('dimensionCoverageRate')}`",
        f"- supporting span coverage: `{payload.get('supportingSpanCoverageRate')}`",
        f"- strict span coverage: `{payload.get('strictSpanCoverageRate')}`",
        f"- fallback span case rate: `{payload.get('fallbackSpanCaseRate')}`",
        f"- fallback-only case rate: `{payload.get('fallbackOnlyCaseRate')}`",
        f"- locator-only case rate: `{payload.get('locatorOnlyCaseRate')}`",
        f"- trace citation coverage: `{payload.get('traceCitationCoverageRate')}`",
        f"- non-evidence leaks: `{payload.get('nonEvidenceLeakCount')}`",
        f"- cases path: `{payload.get('casesPath')}`",
    ]
    failure_category_counts = dict(payload.get("failureCategoryCounts") or {})
    if failure_category_counts:
        lines.extend(["", "## Failure Categories", ""])
        lines.extend(f"- `{key}`: `{value}`" for key, value in sorted(failure_category_counts.items()))
    provenance_diagnostic_counts = dict(payload.get("provenanceDiagnosticCounts") or {})
    if provenance_diagnostic_counts:
        lines.extend(["", "## Provenance Diagnostics", ""])
        lines.extend(f"- `{key}`: `{value}`" for key, value in sorted(provenance_diagnostic_counts.items()))
    errors = list(payload.get("errors") or [])
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in errors)
    failed_cases = [item for item in list(payload.get("cases") or []) if item.get("status") == "fail"]
    if failed_cases:
        lines.extend(["", "## Failed Cases", ""])
        for item in failed_cases:
            lines.append(f"- `{item.get('caseId')}`: {', '.join(item.get('errors') or [])}")
    skipped_cases = [item for item in list(payload.get("cases") or []) if item.get("status") == "skipped"]
    if skipped_cases:
        lines.extend(["", "## Skipped Cases", ""])
        for item in skipped_cases:
            lines.append(f"- `{item.get('caseId')}`: `{item.get('skipReason')}`")
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
    parser.add_argument("--min-expected-answerable-strict-source-coverage-rate", type=float, default=1.0)
    parser.add_argument("--min-dimension-coverage-rate", type=float, default=1.0)
    parser.add_argument("--min-supporting-span-coverage-rate", type=float, default=1.0)
    parser.add_argument("--min-trace-citation-coverage-rate", type=float, default=1.0)
    parser.add_argument("--min-corpus-coverage-rate", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--allow-external", action="store_true", default=False)
    parser.add_argument("--fail-on-insufficient", action="store_true", default=False)
    parser.add_argument("--corpus-manifest", default="")
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
        min_expected_answerable_strict_source_coverage_rate=float(args.min_expected_answerable_strict_source_coverage_rate),
        min_dimension_coverage_rate=float(args.min_dimension_coverage_rate),
        min_supporting_span_coverage_rate=float(args.min_supporting_span_coverage_rate),
        min_trace_citation_coverage_rate=float(args.min_trace_citation_coverage_rate),
        min_corpus_coverage_rate=float(args.min_corpus_coverage_rate),
        fail_on_insufficient=bool(args.fail_on_insufficient),
        corpus_manifest_path=args.corpus_manifest or None,
        derive_corpus_requirements=True,
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
