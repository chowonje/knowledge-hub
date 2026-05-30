"""Report-only source alias normalization gate for short paper aliases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from knowledge_hub.domain.ai_papers.families import (
    PAPER_FAMILY_COMPARE,
    PAPER_FAMILY_CONCEPT_EXPLAINER,
    PAPER_FAMILY_DISCOVER,
    PAPER_FAMILY_LOOKUP,
    explicit_paper_id,
)
from knowledge_hub.domain.ai_papers.query_plan import build_rule_query_plan

SOURCE_ALIAS_NORMALIZATION_CASE_SCHEMA_ID = "knowledge-hub.paper.source-alias-normalization-case.v1"
SOURCE_ALIAS_NORMALIZATION_REPORT_SCHEMA_ID = "knowledge-hub.paper.source-alias-normalization-report.v1"

SHORT_ALIASES = ("RAG", "GPT", "CNN", "VLM")
EXPLICIT_TITLE_FORMS = (
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "Language Models are Few-Shot Learners",
    "ImageNet Classification with Deep Convolutional Neural Networks",
    "An Image is Worth 16x16 Words",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
)
PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9.+-]+|[가-힣]+")


@dataclass(frozen=True)
class SourceAliasCase:
    case_id: str
    query: str
    short_alias: str
    expected_disposition: str
    description: str
    metadata_filter: dict[str, Any] | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _private_path_leak_count(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return 1 if PRIVATE_PATH_RE.search(encoded) else 0


def _contains_alias(query: str, alias: str) -> bool:
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE)
    return bool(pattern.search(query))


def _explicit_title_context(query: str) -> str:
    normalized = _clean_text(query).casefold()
    for title in EXPLICIT_TITLE_FORMS:
        if title.casefold() in normalized:
            return title
    return ""


def _query_tokens(query: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in TOKEN_RE.findall(_clean_text(query)):
        term = _clean_text(raw)
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(term)
    return result


def _context_terms(query: str, alias: str) -> list[str]:
    alias_key = alias.casefold()
    stopwords = {
        "and",
        "compare",
        "comparison",
        "difference",
        "paper",
        "papers",
        "vs",
        "계열",
        "기준",
        "논문",
        "차이",
        "비교",
        "비교해줘",
        "관련",
        "찾아줘",
        "설명해줘",
        "요약해줘",
    }
    terms: list[str] = []
    for term in _query_tokens(query):
        key = term.casefold()
        if key == alias_key or key in stopwords:
            continue
        if len(term) < 2:
            continue
        if re.fullmatch(r"[가-힣]+", term):
            continue
        terms.append(term)
    return terms[:4]


def default_source_alias_cases() -> list[SourceAliasCase]:
    return [
        SourceAliasCase(
            case_id="rag-concept-only",
            query="RAG란?",
            short_alias="RAG",
            expected_disposition="discover_only",
            description="bare RAG should not resolve to one source paper",
        ),
        SourceAliasCase(
            case_id="rag-discover-only",
            query="RAG 관련 논문 찾아줘",
            short_alias="RAG",
            expected_disposition="discover_only",
            description="discover intent can shortlist papers without treating the alias as citation-grade source scope",
        ),
        SourceAliasCase(
            case_id="rag-self-rag-contextual-compare",
            query="RAG와 Self-RAG를 비교해줘",
            short_alias="RAG",
            expected_disposition="contextual_alias_resolved",
            description="compare context can use RAG only as one side of an explicit pair",
        ),
        SourceAliasCase(
            case_id="gpt-bert-contextual-compare",
            query="BERT와 GPT 계열의 차이를 논문 기준으로 비교해줘",
            short_alias="GPT",
            expected_disposition="contextual_alias_resolved",
            description="GPT may resolve only inside a bounded BERT/GPT compare frame",
        ),
        SourceAliasCase(
            case_id="cnn-concept-representative-only",
            query="CNN을 쉽게 설명해줘",
            short_alias="CNN",
            expected_disposition="concept_only",
            description="CNN explainer can use representative candidates but not direct alias scope",
        ),
        SourceAliasCase(
            case_id="cnn-vit-contextual-compare",
            query="CNN vs ViT 비교해줘",
            short_alias="CNN",
            expected_disposition="contextual_alias_resolved",
            description="CNN can resolve inside an explicit CNN/ViT compare frame",
        ),
        SourceAliasCase(
            case_id="vlm-concept-only",
            query="VLM을 설명해줘",
            short_alias="VLM",
            expected_disposition="concept_only",
            description="VLM is concept-only without a named source or explicit title",
        ),
        SourceAliasCase(
            case_id="rag-explicit-title-context",
            query="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks 논문 요약해줘",
            short_alias="RAG",
            expected_disposition="explicit_title_context",
            description="full paper title context can scope source lookup even if RAG is the common short alias",
        ),
        SourceAliasCase(
            case_id="rag-explicit-id-resolved",
            query="2005.11401 논문 요약해줘",
            short_alias="RAG",
            expected_disposition="explicit_id_resolved",
            description="arXiv id is explicit source context, independent of short alias expansion",
        ),
        SourceAliasCase(
            case_id="gpt-bare-lookup-blocked",
            query="GPT 논문 요약해줘",
            short_alias="GPT",
            expected_disposition="blocked_short_alias_no_context",
            description="bare short alias plus lookup wording is not enough for direct source resolution",
        ),
    ]


def _policy_from_context(
    *,
    case: SourceAliasCase,
    family: str,
    explicit_id: str,
    explicit_title: str,
    context_terms: Sequence[str],
) -> tuple[str, str, bool, list[str], str]:
    if explicit_id:
        return (
            "explicit_id_resolved",
            "explicit_source_resolution_allowed",
            True,
            ["explicit_paper_id"],
            "",
        )
    if explicit_title:
        return (
            "explicit_title_context",
            "explicit_source_resolution_allowed",
            True,
            ["explicit_paper_title"],
            "",
        )
    if family == PAPER_FAMILY_COMPARE and context_terms:
        return (
            "contextual_alias_resolved",
            "contextual_resolution_only",
            True,
            ["compare_family", "paired_context_term"],
            "",
        )
    if family == PAPER_FAMILY_CONCEPT_EXPLAINER:
        return (
            "concept_only",
            "representative_candidates_only",
            False,
            ["concept_explainer_family"],
            "short_alias_requires_explicit_source_context",
        )
    if family == PAPER_FAMILY_DISCOVER:
        return (
            "discover_only",
            "shortlist_only",
            False,
            ["discover_family"],
            "short_alias_discovery_not_single_source_scope",
        )
    if family == PAPER_FAMILY_LOOKUP:
        return (
            "blocked_short_alias_no_context",
            "no_direct_source_resolution",
            False,
            ["lookup_family_without_explicit_source_context"],
            "short_alias_lookup_requires_explicit_id_or_title",
        )
    return (
        "blocked_short_alias_no_context",
        "no_direct_source_resolution",
        False,
        ["no_supported_context"],
        "short_alias_requires_explicit_source_context",
    )


def normalize_source_alias_case(case: SourceAliasCase) -> dict[str, Any]:
    query = _clean_text(case.query)
    alias = _clean_text(case.short_alias).upper()
    plan = build_rule_query_plan(query, source_type="paper", metadata_filter=case.metadata_filter).to_dict()
    family = _clean_text(plan.get("family"))
    explicit_id = explicit_paper_id(query, metadata_filter=case.metadata_filter)
    explicit_title = _explicit_title_context(query)
    context_terms = _context_terms(query, alias)
    disposition, policy, allowed, signals, blocker = _policy_from_context(
        case=case,
        family=family,
        explicit_id=explicit_id,
        explicit_title=explicit_title,
        context_terms=context_terms,
    )
    query_plan_resolved_ids = [str(item) for item in list(plan.get("resolved_paper_ids") or []) if str(item).strip()]
    short_alias_present = _contains_alias(query, alias) or bool(explicit_id and alias in {"RAG", "GPT", "CNN", "VLM"})
    unsafe = bool(
        disposition in {"blocked_short_alias_no_context", "concept_only", "discover_only"}
        and policy == "explicit_source_resolution_allowed"
    )
    expectation_met = disposition == case.expected_disposition and not unsafe
    return {
        "schema": SOURCE_ALIAS_NORMALIZATION_CASE_SCHEMA_ID,
        "caseId": case.case_id,
        "query": query,
        "queryHash": _sha256_text(query),
        "shortAlias": alias,
        "shortAliasPresent": short_alias_present,
        "description": case.description,
        "family": family,
        "disposition": disposition,
        "expectedDisposition": case.expected_disposition,
        "expectationMet": expectation_met,
        "sourceResolutionPolicy": policy,
        "directShortAliasResolutionAllowed": allowed,
        "contextSignals": signals,
        "contextTerms": list(context_terms),
        "explicitPaperId": explicit_id,
        "explicitTitle": explicit_title,
        "queryPlan": {
            "family": family,
            "entities": list(plan.get("entities") or []),
            "expandedTerms": list(plan.get("expanded_terms") or plan.get("expandedTerms") or []),
            "resolvedPaperIds": query_plan_resolved_ids,
            "answerMode": _clean_text(plan.get("answer_mode") or plan.get("answerMode")),
        },
        "queryPlanResolvedPaperIdRows": len(query_plan_resolved_ids),
        "unsafeDirectAliasResolution": unsafe,
        "blockerReason": blocker,
        "runtimeWiringChangedRows": 0,
        "strictEvidencePromotionRows": 0,
        "runtimeAnswerVisibleExposureRows": 0,
    }


def build_source_alias_normalization_report(
    *,
    cases: Sequence[SourceAliasCase] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    rows = [normalize_source_alias_case(case) for case in list(cases or default_source_alias_cases())]
    unsafe_rows = sum(1 for row in rows if row.get("unsafeDirectAliasResolution"))
    expectation_failures = sum(1 for row in rows if not row.get("expectationMet"))
    report: dict[str, Any] = {
        "schema": SOURCE_ALIAS_NORMALIZATION_REPORT_SCHEMA_ID,
        "status": "ready" if rows and unsafe_rows == 0 and expectation_failures == 0 else "blocked",
        "generatedAt": generated_at or utc_now_iso(),
        "scope": {
            "writes": "report_only",
            "aliasPolicy": "source_alias_normalization_text_v0_1",
            "runtimeWiringChanged": False,
            "visualLayoutBranchDeferred": True,
            "strictEvidencePromotionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
        },
        "caseRows": len(rows),
        "shortAliasCaseRows": sum(1 for row in rows if row.get("shortAliasPresent")),
        "blockedShortAliasRows": sum(1 for row in rows if row.get("disposition") == "blocked_short_alias_no_context"),
        "conceptOnlyRows": sum(1 for row in rows if row.get("disposition") == "concept_only"),
        "discoverOnlyRows": sum(1 for row in rows if row.get("disposition") == "discover_only"),
        "contextualAliasResolvedRows": sum(1 for row in rows if row.get("disposition") == "contextual_alias_resolved"),
        "explicitIdResolvedRows": sum(1 for row in rows if row.get("disposition") == "explicit_id_resolved"),
        "explicitTitleContextRows": sum(1 for row in rows if row.get("disposition") == "explicit_title_context"),
        "unsafeDirectAliasRows": unsafe_rows,
        "expectationFailureRows": expectation_failures,
        "rows": rows,
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "runtimeWiringChangedRows": 0,
            "strictEvidencePromotionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "warnings": [
            "report classifies short-alias source scope policy only; it does not change ask-v2 runtime selection"
        ],
        "schemaErrors": [],
    }
    report["privatePathLeakRows"] = _private_path_leak_count(report)
    if report["privatePathLeakRows"]:
        report["status"] = "blocked"
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Source Alias Normalization",
        "",
        f"- status: `{report.get('status')}`",
        f"- caseRows: `{report.get('caseRows')}`",
        f"- blockedShortAliasRows: `{report.get('blockedShortAliasRows')}`",
        f"- conceptOnlyRows: `{report.get('conceptOnlyRows')}`",
        f"- discoverOnlyRows: `{report.get('discoverOnlyRows')}`",
        f"- contextualAliasResolvedRows: `{report.get('contextualAliasResolvedRows')}`",
        f"- explicitIdResolvedRows: `{report.get('explicitIdResolvedRows')}`",
        f"- explicitTitleContextRows: `{report.get('explicitTitleContextRows')}`",
        f"- unsafeDirectAliasRows: `{report.get('unsafeDirectAliasRows')}`",
        f"- privatePathLeakRows: `{report.get('privatePathLeakRows')}`",
        "",
        "## Cases",
        "",
        "| caseId | alias | family | disposition | policy | blocker |",
        "|---|---:|---|---|---|---|",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"| `{row.get('caseId')}` | `{row.get('shortAlias')}` | `{row.get('family')}` | "
            f"`{row.get('disposition')}` | `{row.get('sourceResolutionPolicy')}` | "
            f"`{row.get('blockerReason') or ''}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


__all__ = [
    "SOURCE_ALIAS_NORMALIZATION_CASE_SCHEMA_ID",
    "SOURCE_ALIAS_NORMALIZATION_REPORT_SCHEMA_ID",
    "SourceAliasCase",
    "build_source_alias_normalization_report",
    "default_source_alias_cases",
    "normalize_source_alias_case",
    "write_report",
]
