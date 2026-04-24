#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_hub.ai.answer_contracts import NON_EVIDENCE_SOURCE_SCHEMES, NON_EVIDENCE_SOURCE_TYPES


SCHEMA = "knowledge-hub.live-retrieval-span-eval.result.v1"
DEFAULT_CASES_PATH = "eval/knowledgeos/queries/live_retrieval_span_eval_cases.local.json"
PUBLIC_SOURCE_TYPES = {"paper", "vault", "web", "concept"}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize(value: Any) -> str:
    return _clean_text(value).casefold()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _normalize(value) in {"1", "true", "yes", "y", "on"}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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
        query = _clean_text(item.get("query"))
        if not query:
            continue
        case = dict(item)
        case.setdefault("case_id", f"case_{index}")
        cases.append(case)
    return cases


def _result_metadata(result: Any) -> dict[str, Any]:
    return dict(getattr(result, "metadata", None) or {})


def _result_source_id(result: Any) -> str:
    metadata = _result_metadata(result)
    return _clean_text(
        getattr(result, "document_id", "")
        or metadata.get("source_id")
        or metadata.get("sourceId")
        or metadata.get("id")
    )


def _result_source_type(result: Any) -> str:
    metadata = _result_metadata(result)
    return _normalize(metadata.get("source_type") or metadata.get("sourceType"))


def _source_scheme(source_id: str) -> str:
    text = _clean_text(source_id)
    return text.split(":", 1)[0].lower() if ":" in text else ""


def _is_non_evidence_result(result: Any) -> bool:
    source_type = _result_source_type(result)
    if source_type in NON_EVIDENCE_SOURCE_TYPES:
        return True
    scheme = _source_scheme(_result_source_id(result))
    return bool(scheme and scheme in NON_EVIDENCE_SOURCE_SCHEMES)


def _result_text(result: Any) -> str:
    metadata = _result_metadata(result)
    parts = [
        _result_source_id(result),
        metadata.get("title", ""),
        metadata.get("file_path", ""),
        metadata.get("source_url", ""),
        getattr(result, "document", ""),
    ]
    return "\n".join(_clean_text(part) for part in parts if _clean_text(part))


def _serialize_result(result: Any, *, rank: int) -> dict[str, Any]:
    metadata = _result_metadata(result)
    return {
        "rank": int(rank),
        "sourceId": _result_source_id(result),
        "sourceType": _result_source_type(result),
        "title": _clean_text(metadata.get("title")),
        "score": getattr(result, "score", None),
        "semanticScore": getattr(result, "semantic_score", None),
        "lexicalScore": getattr(result, "lexical_score", None),
        "distance": getattr(result, "distance", None),
        "nonEvidence": _is_non_evidence_result(result),
        "textPreview": _result_text(result)[:500],
    }


def _term_hits(text: str, terms: list[str]) -> list[str]:
    normalized_text = _normalize(text)
    return [term for term in terms if _normalize(term) and _normalize(term) in normalized_text]


def _search_case(
    searcher: Any,
    case: dict[str, Any],
    *,
    default_top_k: int,
    retrieval_mode: str,
    alpha: float,
    use_ontology_expansion: bool,
) -> list[Any]:
    source_type = _normalize(case.get("source_type"))
    top_k = _as_int(case.get("top_k"), default_top_k)
    if source_type in PUBLIC_SOURCE_TYPES:
        return list(
            searcher.search(
                _clean_text(case.get("query")),
                top_k=top_k,
                source_type=source_type,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                use_ontology_expansion=use_ontology_expansion,
            )
            or []
        )
    metadata_filter = {"source_type": source_type} if source_type else None
    return list(
        searcher.search(
            _clean_text(case.get("query")),
            top_k=top_k,
            source_type=None,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            use_ontology_expansion=use_ontology_expansion,
            metadata_filter=metadata_filter,
        )
        or []
    )


def evaluate_case(
    case: dict[str, Any],
    results: list[Any],
    *,
    default_top_k: int = 10,
    default_min_rank: int = 3,
    min_term_overlap_ratio: float = 1.0,
) -> dict[str, Any]:
    case_id = _clean_text(case.get("case_id"))
    expected_source_ids = _as_list(case.get("expected_source_ids") or case.get("expected_source_id"))
    expected_terms = _as_list(case.get("expected_text_terms"))
    expected_role = _normalize(case.get("expected_evidence_role") or "citation")
    must_abstain = _as_bool(case.get("must_abstain"))
    min_rank = _as_int(case.get("min_rank"), default_min_rank)
    top_k = _as_int(case.get("top_k"), default_top_k)

    serialized = [_serialize_result(result, rank=index) for index, result in enumerate(results[:top_k], start=1)]
    source_rank = 0
    matched_result: Any | None = None
    for index, result in enumerate(results[:top_k], start=1):
        if _result_source_id(result) in expected_source_ids:
            source_rank = index
            matched_result = result
            break

    if matched_result is not None:
        term_text = _result_text(matched_result)
    else:
        term_text = "\n".join(_result_text(result) for result in results[:top_k])
    hits = _term_hits(term_text, expected_terms)
    term_overlap_ratio = 1.0 if not expected_terms else round(len(hits) / len(expected_terms), 6)
    source_hit_at_k = bool(source_rank)
    source_hit_within_rank = bool(source_rank and source_rank <= min_rank)
    matched_non_evidence = bool(matched_result is not None and _is_non_evidence_result(matched_result))
    non_evidence_result_count = sum(1 for result in results[:top_k] if _is_non_evidence_result(result))
    citation_grade_result_count = max(0, min(len(results), top_k) - non_evidence_result_count)

    skipped_reason = ""
    if not expected_source_ids and not must_abstain:
        skipped_reason = "missing_expected_source_ids"

    errors: list[str] = []
    if skipped_reason:
        status = "skipped"
    elif expected_role == "retrieval_signal_only":
        if not source_hit_at_k:
            errors.append("expected_signal_source_not_found")
        if not matched_non_evidence:
            errors.append("expected_signal_was_not_classified_non_evidence")
        if term_overlap_ratio < float(min_term_overlap_ratio):
            errors.append(f"term_overlap_below_threshold:{term_overlap_ratio}<{min_term_overlap_ratio}")
        status = "pass" if not errors else "fail"
    elif must_abstain:
        if source_hit_at_k:
            errors.append("unexpected_expected_source_hit")
        if hits:
            errors.append("unexpected_expected_term_overlap")
        status = "pass" if not errors else "fail"
    else:
        if not source_hit_at_k:
            errors.append("expected_source_not_found")
        if source_hit_at_k and not source_hit_within_rank:
            errors.append(f"expected_source_rank_too_low:{source_rank}>{min_rank}")
        if term_overlap_ratio < float(min_term_overlap_ratio):
            errors.append(f"term_overlap_below_threshold:{term_overlap_ratio}<{min_term_overlap_ratio}")
        if matched_non_evidence:
            errors.append("matched_source_is_non_evidence")
        status = "pass" if not errors else "fail"

    return {
        "caseId": case_id,
        "query": _clean_text(case.get("query")),
        "sourceType": _normalize(case.get("source_type")),
        "expectedSourceIds": expected_source_ids,
        "expectedTextTerms": expected_terms,
        "expectedEvidenceRole": expected_role,
        "mustAbstain": must_abstain,
        "minRank": min_rank,
        "topK": top_k,
        "status": status,
        "skippedReason": skipped_reason,
        "errors": errors,
        "sourceHitAtK": source_hit_at_k,
        "sourceHitWithinMinRank": source_hit_within_rank,
        "sourceRank": source_rank,
        "termHits": hits,
        "missingTerms": [term for term in expected_terms if term not in hits],
        "termOverlapRatio": term_overlap_ratio,
        "matchedSourceNonEvidence": matched_non_evidence,
        "nonEvidenceResultCount": non_evidence_result_count,
        "citationGradeResultCount": citation_grade_result_count,
        "topResults": serialized,
    }


def build_summary(
    cases: list[dict[str, Any]],
    case_results: list[dict[str, Any]],
    *,
    cases_path: Path,
    retrieval_mode: str,
    top_k: int,
    alpha: float,
    min_cases: int,
    min_source_hit_rate: float,
    min_term_overlap_ratio: float,
    fail_on_insufficient: bool,
) -> dict[str, Any]:
    evaluated = [item for item in case_results if item.get("status") != "skipped"]
    failures = [item for item in evaluated if item.get("status") == "fail"]
    passed = [item for item in evaluated if item.get("status") == "pass"]
    source_cases = [item for item in evaluated if item.get("expectedEvidenceRole") != "retrieval_signal_only" and not item.get("mustAbstain")]
    source_hits = [item for item in source_cases if item.get("sourceHitAtK")]
    rank_hits = [item for item in source_cases if item.get("sourceHitWithinMinRank")]
    term_passes = [item for item in source_cases if float(item.get("termOverlapRatio") or 0.0) >= min_term_overlap_ratio]

    def _ratio(count: int, total: int) -> float | None:
        if total <= 0:
            return None
        return round(count / total, 6)

    source_hit_rate = _ratio(len(source_hits), len(source_cases))
    rank_hit_rate = _ratio(len(rank_hits), len(source_cases))
    term_overlap_pass_rate = _ratio(len(term_passes), len(source_cases))
    insufficient = len(evaluated) < int(min_cases)
    errors: list[str] = []
    if insufficient:
        errors.append(f"insufficient_evaluable_cases:{len(evaluated)}/{int(min_cases)}")
    if source_cases and source_hit_rate is not None and source_hit_rate < min_source_hit_rate:
        errors.append(f"source_hit_rate_below_threshold:{source_hit_rate}<{min_source_hit_rate}")
    if source_cases and term_overlap_pass_rate is not None and term_overlap_pass_rate < min_term_overlap_ratio:
        errors.append(f"term_overlap_pass_rate_below_threshold:{term_overlap_pass_rate}<{min_term_overlap_ratio}")
    if failures:
        errors.append(f"failed_cases:{len(failures)}")

    if insufficient and not fail_on_insufficient:
        status = "skipped"
    else:
        status = "ok" if not errors else "failed"

    return {
        "schema": SCHEMA,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "casesPath": str(cases_path),
        "retrievalMode": retrieval_mode,
        "topK": int(top_k),
        "alpha": float(alpha),
        "caseCount": len(cases),
        "evaluatedCaseCount": len(evaluated),
        "skippedCaseCount": len(case_results) - len(evaluated),
        "passedCaseCount": len(passed),
        "failedCaseCount": len(failures),
        "sourceCaseCount": len(source_cases),
        "sourceHitAtKRate": source_hit_rate,
        "sourceHitWithinMinRankRate": rank_hit_rate,
        "termOverlapPassRate": term_overlap_pass_rate,
        "minCases": int(min_cases),
        "minSourceHitRate": float(min_source_hit_rate),
        "minTermOverlapRatio": float(min_term_overlap_ratio),
        "failOnInsufficient": bool(fail_on_insufficient),
        "errors": errors,
        "cases": case_results,
    }


def run_live_retrieval_span_eval(
    *,
    searcher: Any,
    cases: list[dict[str, Any]],
    cases_path: Path,
    top_k: int,
    retrieval_mode: str,
    alpha: float,
    use_ontology_expansion: bool,
    min_cases: int,
    min_source_hit_rate: float,
    min_term_overlap_ratio: float,
    fail_on_insufficient: bool,
) -> dict[str, Any]:
    case_results: list[dict[str, Any]] = []
    for case in cases:
        results = _search_case(
            searcher,
            case,
            default_top_k=top_k,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            use_ontology_expansion=use_ontology_expansion,
        )
        case_results.append(
            evaluate_case(
                case,
                results,
                default_top_k=top_k,
                min_term_overlap_ratio=min_term_overlap_ratio,
            )
        )
    return build_summary(
        cases,
        case_results,
        cases_path=cases_path,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        alpha=alpha,
        min_cases=min_cases,
        min_source_hit_rate=min_source_hit_rate,
        min_term_overlap_ratio=min_term_overlap_ratio,
        fail_on_insufficient=fail_on_insufficient,
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Live Retrieval Span Eval",
        "",
        f"- status: `{payload.get('status')}`",
        f"- cases: `{payload.get('evaluatedCaseCount')}` evaluated / `{payload.get('caseCount')}` loaded",
        f"- source hit@K: `{payload.get('sourceHitAtKRate')}`",
        f"- source hit within min rank: `{payload.get('sourceHitWithinMinRankRate')}`",
        f"- term overlap pass rate: `{payload.get('termOverlapPassRate')}`",
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


def _build_default_searcher() -> Any:
    from knowledge_hub.application.context import AppContextFactory

    app = AppContextFactory().build(require_search=True)
    return app.searcher


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check live local-corpus retrieval against curated source/span cases.")
    parser.add_argument("--cases", default=DEFAULT_CASES_PATH)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", default="hybrid", choices=["semantic", "keyword", "hybrid"])
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--use-ontology-expansion", action="store_true", default=False)
    parser.add_argument("--min-cases", type=int, default=1)
    parser.add_argument("--min-source-hit-rate", type=float, default=1.0)
    parser.add_argument("--min-term-overlap-ratio", type=float, default=1.0)
    parser.add_argument("--fail-on-insufficient", action="store_true", default=False)
    parser.add_argument("--json", action="store_true", dest="as_json", default=False)
    args = parser.parse_args(argv)

    cases_path = Path(args.cases).expanduser().resolve()
    try:
        cases = _read_cases(cases_path)
        payload = run_live_retrieval_span_eval(
            searcher=_build_default_searcher(),
            cases=cases,
            cases_path=cases_path,
            top_k=int(args.top_k),
            retrieval_mode=str(args.mode),
            alpha=float(args.alpha),
            use_ontology_expansion=bool(args.use_ontology_expansion),
            min_cases=int(args.min_cases),
            min_source_hit_rate=float(args.min_source_hit_rate),
            min_term_overlap_ratio=float(args.min_term_overlap_ratio),
            fail_on_insufficient=bool(args.fail_on_insufficient),
        )
    except Exception as exc:  # noqa: BLE001
        payload = {
            "schema": SCHEMA,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "casesPath": str(cases_path),
            "errors": [f"live_retrieval_span_eval_failed:{exc}"],
            "cases": [],
        }

    if args.out_json:
        out_json = Path(args.out_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_markdown(payload), encoding="utf-8")

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(payload))
    return 0 if payload.get("status") in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
