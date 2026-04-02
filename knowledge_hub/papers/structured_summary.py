"""Labs-only structured paper summary helpers.

This module adds a summary-first reading path for papers without changing the
default `khub paper summarize` runtime. It reuses parser/document-memory
artifacts, keeps outputs inspectable, and degrades conservatively when the LLM
or parser path is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from knowledge_hub.document_memory import DocumentMemoryBuilder, DocumentMemoryRetriever
from knowledge_hub.document_memory.payloads import semantic_units_payload
from knowledge_hub.learning.task_router import get_llm_for_task
from knowledge_hub.knowledge.claim_normalization import ClaimNormalizationService
from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.memory_payloads import card_payload

_NEGATIVE_HINTS = (
    "limitation",
    "한계",
    "failure",
    "fail",
    "fails",
    "drop",
    "degrade",
    "bound",
    "missing",
    "low recall",
    "domain shift",
    "not fully address",
    "weakness",
    "constraint",
)
_RESULT_HINTS = ("result", "results", "evaluation", "experiment", "score", "f1", "exact match", "accuracy", "bleu")
_METHOD_HINTS = ("method", "approach", "framework", "system", "model", "architecture", "방법", "구성")
_PROBLEM_HINTS = ("abstract", "introduction", "background", "overview", "motivation", "요약", "소개", "배경")
_NOVELTY_HINTS = ("new", "novel", "contribution", "기여", "제안", "improve", "improves", "향상", "outperform")
_DATASET_HINTS = ("benchmark", "dataset", "evalset", "triviaqa", "natural questions", "nq", "hotpotqa", "ms marco")
_COMPARATOR_HINTS = ("baseline", "over", "compared to", "vs", "outperform", "outperforms", "beats")
_WRAPPER_HINTS = ("\\begin{figure", "\\begin{subfigure", "\\includegraphics", "\\caption{", "label{fig:", "figure[", "table[")
_LATEX_HINTS = ("\\documentclass", "\\usepackage", "\\begin{table", "\\begin{figure", "\\section{", "\\subsection{")
_METADATA_HINTS = ("arxiv id", "status:", "translated_path", "pdf_path", "metadata", "번역 완료", "논문 키워드")
_DEPTH_MAP = {
    "oneLine": "shallow",
    "coreIdea": "shallow",
    "whenItMatters": "shallow",
    "problem": "medium",
    "methodSteps": "medium",
    "keyResults": "deep",
    "limitations": "deep",
    "whatIsNew": "deep",
}
_DEPTH_COMPONENTS = {
    "shallow": ["paper_memory", "document_memory"],
    "medium": ["paper_memory", "document_memory", "claim_evidence"],
    "deep": ["paper_memory", "document_memory", "claim_evidence", "chunk"],
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        raw = values
    elif isinstance(values, tuple):
        raw = list(values)
    else:
        raw = [values]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = _clean_text(item)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
    return result


def _excerpt(text: Any, *, limit: int = 240) -> str:
    body = _clean_text(text)
    if len(body) <= limit:
        return body
    return body[: max(0, limit - 3)].rstrip() + "..."


def _extract_json_object(text: str) -> dict[str, Any]:
    body = str(text or "").strip()
    if not body:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, flags=re.DOTALL)
    if fenced:
        body = fenced.group(1)
    else:
        start = body.find("{")
        end = body.rfind("}")
        if start >= 0 and end > start:
            body = body[start : end + 1]
    body = re.sub(r",(\s*[}\]])", r"\1", body)
    try:
        value = json.loads(body)
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _artifact_root(config: Any, *, paper_id: str) -> Path:
    papers_dir = Path(str(getattr(config, "papers_dir", "") or "")).expanduser()
    return papers_dir / "summaries" / str(paper_id).strip()


def _parsed_artifact_root(config: Any, *, paper_id: str) -> Path:
    papers_dir = Path(str(getattr(config, "papers_dir", "") or "")).expanduser()
    return papers_dir / "parsed" / str(paper_id).strip()


def _classify_parser_failure(error: Any) -> str:
    message = _clean_text(error)
    lowered = message.casefold()
    if "pdf not found" in lowered or "source missing" in lowered:
        return "source missing"
    if "not installed" in lowered or "cli not found" in lowered:
        return "parser unavailable"
    if "empty" in lowered or "0 element" in lowered or "no element" in lowered:
        return "parser produced empty artifact"
    return "parser execution failed"


def _source_signature(document: dict[str, Any], claim_hints: list[dict[str, Any]], *, paper_parser: str, quick: bool) -> str:
    payload = {
        "documentId": str(document.get("documentId") or ""),
        "documentTitle": str(document.get("documentTitle") or ""),
        "summary": document.get("summary") or {},
        "units": [
            {
                "unitId": unit.get("unitId"),
                "unitType": unit.get("unitType"),
                "title": unit.get("title"),
                "sectionPath": unit.get("sectionPath"),
                "contextualSummary": unit.get("contextualSummary"),
                "sourceExcerpt": unit.get("sourceExcerpt"),
                "provenance": unit.get("provenance") or {},
            }
            for unit in list(document.get("units") or [])
        ],
        "claimHints": claim_hints,
        "paperParser": str(paper_parser or "raw"),
        "quick": bool(quick),
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _source_signature_with_memory(
    document: dict[str, Any],
    claim_hints: list[dict[str, Any]],
    paper_memory: dict[str, Any],
    *,
    paper_parser: str,
    quick: bool,
) -> str:
    payload = {
        "base": _source_signature(document, claim_hints, paper_parser=paper_parser, quick=quick),
        "paperMemory": paper_memory,
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _unit_text(unit: dict[str, Any]) -> str:
    return " ".join(
        part for part in (
            unit.get("title"),
            unit.get("sectionPath"),
            unit.get("contextualSummary"),
            unit.get("sourceExcerpt"),
            unit.get("documentThesis"),
        ) if _clean_text(part)
    )


def _is_placeholder_unit(unit: dict[str, Any]) -> bool:
    haystack = _unit_text(unit).casefold()
    title = _clean_text(unit.get("title")).casefold()
    return bool(
        "pending_summary" in haystack
        or "요약본/번역본이 아직 등록되지 않았습니다" in haystack
        or title in {"metadata", "메타데이터"}
    )


def _has_digit_signal(text: str) -> bool:
    return bool(re.search(r"\d", str(text or "")))


def _has_negative_signal(text: str) -> bool:
    haystack = str(text or "").casefold()
    return any(token in haystack for token in _NEGATIVE_HINTS)


def _has_dataset_signal(text: str) -> bool:
    haystack = str(text or "").casefold()
    return any(token in haystack for token in _DATASET_HINTS)


def _has_comparator_signal(text: str) -> bool:
    haystack = str(text or "").casefold()
    return any(token in haystack for token in _COMPARATOR_HINTS)


def _has_novelty_signal(text: str) -> bool:
    haystack = str(text or "").casefold()
    return any(token in haystack for token in _NOVELTY_HINTS)


def _looks_wrapper_text(text: str) -> bool:
    haystack = str(text or "").casefold()
    return any(token in haystack for token in _WRAPPER_HINTS)


def _looks_latex_heavy(text: str) -> bool:
    haystack = str(text or "").casefold()
    latex_hits = sum(1 for token in _LATEX_HINTS if token in haystack)
    return latex_hits >= 2


def _looks_metadata_text(text: str) -> bool:
    haystack = str(text or "").casefold()
    return any(token in haystack for token in _METADATA_HINTS)


def _claim_hint_texts(claim_hints: list[dict[str, Any]], *, kind: str) -> list[str]:
    out: list[str] = []
    for hint in claim_hints:
        if kind == "limitations":
            text = _clean_text(hint.get("limitationText") or hint.get("claimText"))
        elif kind == "results":
            text = _clean_text(hint.get("claimText") or hint.get("evidenceSummary"))
        else:
            text = _clean_text(hint.get("claimText"))
        if text:
            out.append(text.casefold())
    return out


def _bundle_score(unit: dict[str, Any], kind: str, *, claim_hints: list[dict[str, Any]], parser_used: str) -> float:
    if _is_placeholder_unit(unit):
        return -9.0
    unit_type = _clean_text(unit.get("unitType") or unit.get("unit_type")).casefold()
    text = _unit_text(unit).casefold()
    score = float(unit.get("confidence") or 0.0)
    claim_texts = _claim_hint_texts(claim_hints, kind=kind)
    overlaps_claim_hint = bool(claim_texts and any(token and token in text for token in claim_texts))
    if _looks_metadata_text(text):
        score -= 6.0
    if parser_used == "raw" and _looks_latex_heavy(text):
        score -= 5.0
    if _looks_wrapper_text(text):
        score -= 7.0
    if kind == "problem":
        if unit_type in {"summary", "background"}:
            score += 4.0
        if any(token in text for token in _PROBLEM_HINTS):
            score += 2.0
        if unit_type in {"table_block", "image_block"} or _looks_wrapper_text(text):
            score -= 10.0
    elif kind == "method":
        if unit_type == "method":
            score += 4.0
        if any(token in text for token in _METHOD_HINTS):
            score += 2.0
        if _has_digit_signal(text) and _has_comparator_signal(text):
            score -= 2.5
    elif kind == "results":
        if unit_type in {"result", "table_block", "image_block"}:
            score += 4.0
        if any(token in text for token in _RESULT_HINTS):
            score += 2.0
        if _has_digit_signal(text):
            score += 1.5
        if _has_dataset_signal(text):
            score += 1.5
        if _has_comparator_signal(text):
            score += 1.5
    elif kind == "limitations":
        if unit_type == "limitation":
            score += 4.0
        if _has_negative_signal(text):
            score += 2.5
        if "ablation" in text:
            score += 1.0
        if overlaps_claim_hint:
            score += 1.5
    elif kind == "novelty":
        if unit_type in {"method", "result", "summary"}:
            score += 2.0
        if _has_novelty_signal(text):
            score += 2.0
        if not _has_comparator_signal(text) and unit_type == "result":
            score -= 1.0
    return score


def _choose_units(
    units: list[dict[str, Any]],
    kind: str,
    *,
    limit: int,
    claim_hints: list[dict[str, Any]],
    parser_used: str,
) -> list[dict[str, Any]]:
    ranked = sorted(
        units,
        key=lambda unit: (
            _bundle_score(unit, kind, claim_hints=claim_hints, parser_used=parser_used),
            _has_digit_signal(_unit_text(unit)),
            len(_clean_text(unit.get("contextualSummary"))),
        ),
        reverse=True,
    )
    return [
        unit
        for unit in ranked
        if _bundle_score(unit, kind, claim_hints=claim_hints, parser_used=parser_used) > -5.0
    ][: max(1, int(limit))]


def _compose_one_line_candidate(
    *,
    problem_units: list[dict[str, Any]],
    method_units: list[dict[str, Any]],
    result_units: list[dict[str, Any]],
) -> str:
    problem = _clean_text((problem_units[:1] or [{}])[0].get("contextualSummary") or "")
    core_idea = _clean_text((method_units[:1] or [{}])[0].get("contextualSummary") or "")
    result = _clean_text((result_units[:1] or [{}])[0].get("contextualSummary") or "")
    pieces = [piece for piece in (problem, core_idea, result) if piece]
    if not pieces:
        return ""
    return _excerpt(" / ".join(pieces), limit=220)


def _claim_hint_rows_with_coverage(sqlite_db: Any, config: Any, *, paper_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    note_id = f"paper:{paper_id}"
    rows = list(sqlite_db.list_claims_by_note(note_id, limit=24))
    if not rows:
        return [], {
            "totalClaims": 0,
            "normalizedClaims": 0,
            "failedClaims": 0,
            "usedClaimHints": 0,
            "resultHintsAvailable": False,
            "limitationHintsAvailable": False,
            "status": "low",
        }
    normalizer = ClaimNormalizationService(sqlite_db, config)
    out: list[dict[str, Any]] = []
    normalized_count = 0
    failed_count = 0
    for row in rows:
        try:
            normalized = normalizer.normalize_claim(row, persist=False, allow_external=False, llm_mode="fallback-only")
        except Exception:
            failed_count += 1
            continue
        if _clean_text(normalized.get("status")) == "failed":
            failed_count += 1
            continue
        normalized_count += 1
        out.append(
            {
                "claimId": _clean_text(row.get("claim_id")),
                "claimText": _clean_text(row.get("claim_text")),
                "task": _clean_text(normalized.get("task")),
                "dataset": _clean_text(normalized.get("dataset")),
                "metric": _clean_text(normalized.get("metric")),
                "comparator": _clean_text(normalized.get("comparator")),
                "resultDirection": _clean_text(normalized.get("resultDirection")),
                "limitationText": _clean_text(normalized.get("limitationText")),
                "evidenceSummary": _clean_text(normalized.get("evidenceSummary")),
            }
        )
    trimmed = out[:8]
    result_hints_available = any(
        _clean_text(item.get("dataset"))
        or _clean_text(item.get("metric"))
        or _clean_text(item.get("comparator"))
        or _clean_text(item.get("resultDirection"))
        or _clean_text(item.get("evidenceSummary"))
        for item in trimmed
    )
    limitation_hints_available = any(_clean_text(item.get("limitationText")) for item in trimmed)
    status = "good" if (normalized_count > 0 and (result_hints_available or limitation_hints_available)) else "low"
    return trimmed, {
        "totalClaims": len(rows),
        "normalizedClaims": normalized_count,
        "failedClaims": failed_count,
        "usedClaimHints": len(trimmed),
        "resultHintsAvailable": result_hints_available,
        "limitationHintsAvailable": limitation_hints_available,
        "status": status,
    }


def _claim_hint_rows(sqlite_db: Any, config: Any, *, paper_id: str) -> list[dict[str, Any]]:
    hints, _ = _claim_hint_rows_with_coverage(sqlite_db, config, paper_id=paper_id)
    return hints


def _claim_text_block(claims: list[dict[str, Any]], *, kind: str, limit: int) -> list[str]:
    out: list[str] = []
    for claim in claims:
        if kind == "results" and not (_clean_text(claim.get("metric")) or _clean_text(claim.get("resultDirection"))):
            continue
        if kind == "limitations" and not (_clean_text(claim.get("limitationText")) or _has_negative_signal(claim.get("claimText"))):
            continue
        evidence = _clean_text(claim.get("evidenceSummary"))
        claim_text = _clean_text(claim.get("claimText"))
        text = claim_text
        if evidence and evidence.casefold() != claim_text.casefold():
            text = f"{claim_text} | evidence={evidence}"
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _paper_memory_payload(sqlite_db: Any, *, paper_id: str) -> tuple[dict[str, Any], list[str]]:
    row = sqlite_db.get_paper_memory_card(str(paper_id).strip())
    warnings: list[str] = []
    if not row:
        try:
            row = PaperMemoryBuilder(sqlite_db).build_and_store(paper_id=str(paper_id).strip())
        except Exception as error:
            warnings.append(f"paper memory unavailable: {error}")
            row = None
    return card_payload(row), warnings


def _build_context(
    document: dict[str, Any],
    bundles: dict[str, list[dict[str, Any]]],
    claim_hints: list[dict[str, Any]],
    paper_memory: dict[str, Any],
    evidence_summaries: dict[str, dict[str, Any]],
    supplemental_context_packet: dict[str, Any],
    *,
    quick: bool,
) -> str:
    summary = dict(document.get("summary") or {})
    title = _clean_text(document.get("documentTitle"))
    thesis = _clean_text(summary.get("documentThesis"))
    sections: list[str] = [
        f"TITLE: {title}",
        f"DOCUMENT_THESIS: {thesis}",
        f"DOCUMENT_SUMMARY: {_clean_text(summary.get('contextualSummary'))}",
    ]
    if paper_memory:
        sections.append(
            "PAPER_MEMORY:\n"
            + "\n".join(
                [
                    f"- PAPER_CORE: {_clean_text(paper_memory.get('paperCore'))}",
                    f"- PROBLEM_CONTEXT: {_clean_text(paper_memory.get('problemContext'))}",
                    f"- METHOD_CORE: {_clean_text(paper_memory.get('methodCore'))}",
                    f"- EVIDENCE_CORE: {_clean_text(paper_memory.get('evidenceCore'))}",
                    f"- LIMITATIONS: {_clean_text(paper_memory.get('limitations'))}",
                ]
            )
        )
    one_line_candidate = _compose_one_line_candidate(
        problem_units=bundles.get("problem", []),
        method_units=bundles.get("method", []),
        result_units=bundles.get("results", []),
    )
    if one_line_candidate:
        sections.append(f"ONE_LINE_CANDIDATE: {one_line_candidate}")

    def _bundle_block(label: str, items: list[dict[str, Any]], *, limit_chars: int) -> str:
        lines: list[str] = []
        used = 0
        for unit in items:
            unit_text = _excerpt(
                f"{unit.get('sectionPath') or unit.get('title')}: {unit.get('contextualSummary') or unit.get('sourceExcerpt')}",
                limit=320 if quick else 520,
            )
            if not unit_text:
                continue
            if used + len(unit_text) > limit_chars and lines:
                break
            lines.append(f"- {unit_text}")
            used += len(unit_text)
        return f"{label}:\n" + ("\n".join(lines) if lines else "-")

    sections.append(_bundle_block("PROBLEM_UNITS", bundles.get("problem", []), limit_chars=700 if quick else 1400))
    sections.append(_bundle_block("METHOD_UNITS", bundles.get("method", []), limit_chars=1100 if quick else 2200))
    sections.append(_bundle_block("RESULT_UNITS", bundles.get("results", []), limit_chars=1100 if quick else 2200))
    sections.append(_bundle_block("LIMITATION_UNITS", bundles.get("limitations", []), limit_chars=800 if quick else 1600))
    result_claims = _claim_text_block(claim_hints, kind="results", limit=3 if quick else 5)
    limitation_claims = _claim_text_block(claim_hints, kind="limitations", limit=2 if quick else 4)
    if result_claims:
        sections.append("RESULT_CLAIMS:\n" + "\n".join(f"- {item}" for item in result_claims))
    if limitation_claims:
        sections.append("LIMITATION_CLAIMS:\n" + "\n".join(f"- {item}" for item in limitation_claims))
    for field in ("keyResults", "limitations", "whatIsNew"):
        packet = dict(evidence_summaries.get(field) or {})
        lines = [item for item in list(packet.get("summaryLines") or []) if _clean_text(item)]
        if lines:
            sections.append(f"{field.upper()}_EVIDENCE_SUMMARY:\n" + "\n".join(f"- {item}" for item in lines))
    supplemental_block = _supplemental_context_block(supplemental_context_packet)
    if supplemental_block:
        sections.append(supplemental_block)
    return "\n\n".join(part for part in sections if _clean_text(part))


def _has_result_claim_hints(claim_hints: list[dict[str, Any]]) -> bool:
    for claim in claim_hints:
        if _clean_text(claim.get("metric")) or _clean_text(claim.get("resultDirection")) or _clean_text(claim.get("comparator")):
            return True
    return False


def _has_limitation_claim_hints(claim_hints: list[dict[str, Any]]) -> bool:
    for claim in claim_hints:
        if _clean_text(claim.get("limitationText")) or _has_negative_signal(_clean_text(claim.get("claimText"))):
            return True
    return False


def _has_novelty_claim_hints(claim_hints: list[dict[str, Any]]) -> bool:
    for claim in claim_hints:
        if _clean_text(claim.get("comparator")) or _clean_text(claim.get("resultDirection")):
            return True
    return False


def _field_route(
    *,
    field: str,
    primary_form: str,
    supporting_forms: list[str],
    selection_reason: str,
    retrieval_paths: list[str],
) -> dict[str, Any]:
    depth = _DEPTH_MAP.get(field, "medium")
    return {
        "field": field,
        "primaryForm": primary_form,
        "supportingForms": supporting_forms,
        "verifierForm": "chunk",
        "depth": depth,
        "depthComponents": list(_DEPTH_COMPONENTS.get(depth, [])),
        "retrievalPaths": retrieval_paths,
        "selectionReason": selection_reason,
    }


def _depth_plan_payload(field_routes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "strategy": "memory_form_first",
        "fieldDepths": [
            {
                "field": str(route.get("field") or ""),
                "depth": str(route.get("depth") or _DEPTH_MAP.get(str(route.get("field") or ""), "medium")),
                "memoryForms": list(route.get("depthComponents") or _DEPTH_COMPONENTS.get(str(route.get("depth") or ""), [])),
                "reason": str(route.get("selectionReason") or ""),
            }
            for route in field_routes
        ],
    }


def _evidence_summary_for_field(
    *,
    field: str,
    units: list[dict[str, Any]],
    claim_hints: list[dict[str, Any]],
    paper_memory: dict[str, Any],
) -> dict[str, Any]:
    claim_lines: list[str] = []
    if field == "keyResults":
        claim_lines = _claim_text_block(claim_hints, kind="results", limit=4)
    elif field == "limitations":
        claim_lines = _claim_text_block(claim_hints, kind="limitations", limit=3)
    elif field == "whatIsNew":
        claim_lines = [
            text
            for text in _claim_text_block(claim_hints, kind="results", limit=4)
            if _has_novelty_signal(text) or _has_comparator_signal(text)
        ][:3]

    unit_lines = [
        _excerpt(
            f"{_clean_text(unit.get('sectionPath') or unit.get('title'))}: "
            f"{_clean_text(unit.get('contextualSummary') or unit.get('sourceExcerpt'))}",
            limit=220,
        )
        for unit in list(units or [])
        if _clean_text(unit.get("contextualSummary") or unit.get("sourceExcerpt"))
    ]
    unit_lines = [line for line in unit_lines if line]
    summary_lines = list(dict.fromkeys([*claim_lines, *unit_lines]))[:4]
    if field == "whatIsNew" and not summary_lines:
        summary_lines = [
            _clean_text(paper_memory.get("methodCore")) or _clean_text(paper_memory.get("paperCore"))
        ]
    evidence_refs = [
        {
            "unitId": _clean_text(unit.get("unitId")),
            "sectionPath": _clean_text(unit.get("sectionPath")),
            "excerpt": _excerpt(unit.get("sourceExcerpt") or unit.get("contextualSummary"), limit=160),
        }
        for unit in list(units or [])[:2]
    ]
    return {
        "field": field,
        "summaryLines": [line for line in summary_lines if line],
        "claimHintsUsed": len(claim_lines),
        "unitCount": len(list(units or [])),
        "evidenceRefs": evidence_refs,
    }


def _evidence_summaries_payload(
    *,
    bundles: dict[str, list[dict[str, Any]]],
    claim_hints: list[dict[str, Any]],
    paper_memory: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "keyResults": _evidence_summary_for_field(
            field="keyResults",
            units=bundles.get("results", []),
            claim_hints=claim_hints,
            paper_memory=paper_memory,
        ),
        "limitations": _evidence_summary_for_field(
            field="limitations",
            units=bundles.get("limitations", []),
            claim_hints=claim_hints,
            paper_memory=paper_memory,
        ),
        "whatIsNew": _evidence_summary_for_field(
            field="whatIsNew",
            units=bundles.get("novelty", []) or bundles.get("results", []),
            claim_hints=claim_hints,
            paper_memory=paper_memory,
        ),
    }


def _supplemental_context_packet(
    *,
    sqlite_db: Any,
    paper_id: str,
    bundles: dict[str, list[dict[str, Any]]],
    claim_hints: list[dict[str, Any]],
    weak_coverage: bool,
) -> dict[str, Any]:
    if not weak_coverage:
        return {
            "eligible": False,
            "used": False,
            "mode": "supplemental_only",
            "sources": ["ontology", "cluster"],
            "relations": [],
            "linkedEntities": [],
            "conceptAliases": [],
        }

    relations: list[dict[str, str]] = []
    for claim in claim_hints:
        dataset = _clean_text(claim.get("dataset"))
        metric = _clean_text(claim.get("metric"))
        comparator = _clean_text(claim.get("comparator"))
        if dataset or metric or comparator:
            relations.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "comparator": comparator,
                }
            )
    concepts = list(sqlite_db.get_paper_concepts(str(paper_id).strip()) or [])
    concept_aliases = [
        _clean_text(item.get("canonical_name") or item.get("entity_id"))
        for item in concepts[:5]
        if _clean_text(item.get("canonical_name") or item.get("entity_id"))
    ]
    linked_entities = [
        _clean_text(unit.get("title") or unit.get("sectionPath"))
        for group in (bundles.get("results", []), bundles.get("novelty", []))
        for unit in group
        if _clean_text(unit.get("unitType")) in {"table_block", "image_block"}
    ][:4]
    used = bool(relations or concept_aliases or linked_entities)
    return {
        "eligible": True,
        "used": used,
        "mode": "supplemental_only",
        "sources": ["ontology", "cluster"],
        "relations": relations[:4],
        "linkedEntities": linked_entities,
        "conceptAliases": concept_aliases,
    }


def _supplemental_context_block(packet: dict[str, Any]) -> str:
    if not bool(packet.get("used")):
        return ""
    lines: list[str] = []
    for relation in list(packet.get("relations") or []):
        pieces = [
            f"dataset={_clean_text(relation.get('dataset'))}" if _clean_text(relation.get("dataset")) else "",
            f"metric={_clean_text(relation.get('metric'))}" if _clean_text(relation.get("metric")) else "",
            f"comparator={_clean_text(relation.get('comparator'))}" if _clean_text(relation.get("comparator")) else "",
        ]
        token = ", ".join(part for part in pieces if part)
        if token:
            lines.append(f"- relation: {token}")
    for alias in list(packet.get("conceptAliases") or [])[:4]:
        lines.append(f"- concept_alias: {alias}")
    for entity in list(packet.get("linkedEntities") or [])[:3]:
        lines.append(f"- linked_entity: {entity}")
    return "SUPPLEMENTAL_CONTEXT:\n" + ("\n".join(lines) if lines else "-")


def _memory_route_payload(
    *,
    bundles: dict[str, list[dict[str, Any]]],
    claim_hints: list[dict[str, Any]],
    paper_memory: dict[str, Any],
) -> dict[str, Any]:
    claim_results = _has_result_claim_hints(claim_hints)
    claim_limits = _has_limitation_claim_hints(claim_hints)
    claim_novelty = _has_novelty_claim_hints(claim_hints)
    weak_problem = not bool(bundles.get("problem"))
    weak_method = not bool(bundles.get("method"))
    weak_sections = weak_problem or weak_method
    has_paper_memory = bool(_clean_text(paper_memory.get("paperCore")) or _clean_text(paper_memory.get("methodCore")))

    field_routes = [
        _field_route(
            field="oneLine",
            primary_form="paper_memory" if has_paper_memory else "document_memory",
            supporting_forms=["document_memory"],
            selection_reason="논문 대표 카드로 framing하고 summary/result section으로 한줄 요약을 다듬습니다.",
            retrieval_paths=["paper_memory_prefilter", "document_memory_summary", "vector", "lexical"],
        ),
        _field_route(
            field="problem",
            primary_form="document_memory" if not weak_problem else ("paper_memory" if has_paper_memory else "document_memory"),
            supporting_forms=["paper_memory"] + (["ontology_cluster"] if weak_problem else []),
            selection_reason="문제 정의는 summary/background section이 가장 직접적이고, 약할 때만 paper framing을 보조로 씁니다.",
            retrieval_paths=["document_memory_summary", "lexical", "vector"],
        ),
        _field_route(
            field="coreIdea",
            primary_form="paper_memory" if has_paper_memory else "document_memory",
            supporting_forms=["document_memory"] + (["ontology_cluster"] if weak_method else []),
            selection_reason="핵심 아이디어는 논문 카드의 압축 설명을 먼저 쓰고 method/summary section으로 보강합니다.",
            retrieval_paths=["paper_memory_prefilter", "document_memory_summary", "lexical", "vector"],
        ),
        _field_route(
            field="methodSteps",
            primary_form="document_memory",
            supporting_forms=[],
            selection_reason="방법 단계는 section-aware method units가 가장 직접적입니다.",
            retrieval_paths=["document_memory_summary", "lexical", "vector"],
        ),
        _field_route(
            field="keyResults",
            primary_form="claim_evidence" if claim_results else "document_memory",
            supporting_forms=["document_memory"],
            selection_reason="결과는 normalized claim/evidence가 있으면 먼저 쓰고, 없으면 result/table units로 내려갑니다.",
            retrieval_paths=["claim_compare", "document_memory_summary", "vector", "lexical"],
        ),
        _field_route(
            field="limitations",
            primary_form="claim_evidence" if claim_limits else "document_memory",
            supporting_forms=["document_memory"],
            selection_reason="한계는 negative claim hint가 있으면 우선 쓰고, 없으면 limitation/discussion section을 사용합니다.",
            retrieval_paths=["claim_compare", "document_memory_summary", "vector", "lexical"],
        ),
        _field_route(
            field="whenItMatters",
            primary_form="paper_memory" if has_paper_memory else "document_memory",
            supporting_forms=["document_memory"],
            selection_reason="적용 맥락은 논문 카드의 framing과 result summary를 함께 씁니다.",
            retrieval_paths=["paper_memory_prefilter", "document_memory_summary", "vector", "lexical"],
        ),
        _field_route(
            field="whatIsNew",
            primary_form="claim_evidence" if claim_novelty else ("paper_memory" if has_paper_memory else "document_memory"),
            supporting_forms=["paper_memory", "document_memory"],
            selection_reason="새로움은 comparator/result-direction claim이 있으면 우선 사용하고, 없으면 paper/method summary로 복원합니다.",
            retrieval_paths=["claim_compare", "paper_memory_prefilter", "document_memory_summary", "vector", "lexical"],
        ),
    ]

    disabled_forms = [
        {
            "name": "ontology_cluster",
            "reason": "single-paper summary에서는 추상화·오염 위험이 있어 기본 primary input에서 제외합니다.",
        },
        {
            "name": "chunk_as_primary",
            "reason": "chunk는 읽기 품질보다 grounding 검증에 적합하므로 verifier/fallback 전용입니다.",
        },
    ]
    route_warnings: list[str] = []
    if not claim_results or not claim_limits:
        route_warnings.append("claim coverage low")
    if weak_sections:
        route_warnings.append("section coverage weak")
    route_warnings.append("ontology disabled for single-paper summary")
    return {
        "decisionOrder": "memory_form_first",
        "fieldRoutes": field_routes,
        "disabledForms": disabled_forms,
        "routeWarnings": route_warnings,
        "strategyComparison": [
            {
                "name": "memory_form_first",
                "pros": [
                    "summary 오염을 줄이고 problem/result/limitation 분리를 더 명확하게 합니다.",
                    "reasoning trace와 verifier handoff를 field별로 명시하기 쉽습니다.",
                ],
                "cons": ["routing 규칙을 유지해야 합니다."],
            },
            {
                "name": "retrieval_path_first",
                "pros": ["구현은 단순합니다."],
                "cons": [
                    "cluster/graph/vector hit가 곧 기억 선택처럼 동작해 품질 일관성이 약해집니다.",
                ],
            },
        ],
    }


def _primary_house_for_field(
    *,
    field: str,
    paper_memory: dict[str, Any],
    claim_hints: list[dict[str, Any]],
) -> str:
    if field in {"problem", "methodSteps", "whenItMatters"}:
        return "reading_house"
    if field in {"oneLine", "coreIdea"}:
        return "reading_house"
    if field == "whatIsNew":
        return "evidence_house" if (_has_novelty_claim_hints(claim_hints) or bool(paper_memory)) else "reading_house"
    if field == "keyResults":
        return "evidence_house" if _has_result_claim_hints(claim_hints) else "reading_house"
    if field == "limitations":
        return "evidence_house" if _has_limitation_claim_hints(claim_hints) else "reading_house"
    return "reading_house"


def _house_field_route(
    *,
    field: str,
    primary_house: str,
    selection_reason: str,
    retrieval_paths: list[str],
    paper_memory: dict[str, Any],
    claim_hints: list[dict[str, Any]],
) -> dict[str, Any]:
    supporting_houses: list[str] = []
    if primary_house == "reading_house" and (bool(paper_memory) or field in {"oneLine", "coreIdea"}):
        supporting_houses.append("evidence_house")
    if primary_house == "evidence_house":
        supporting_houses.append("reading_house")
    if field in {"problem", "coreIdea"}:
        supporting_houses.append("context_house")
    seen: set[str] = set()
    normalized_support: list[str] = []
    for house in supporting_houses:
        if house in seen or house == primary_house:
            continue
        if house == "evidence_house" and not (bool(paper_memory) or claim_hints):
            continue
        seen.add(house)
        normalized_support.append(house)
    return {
        "field": field,
        "primaryHouse": primary_house,
        "supportingHouses": normalized_support,
        "verifierForm": "chunk",
        "retrievalPaths": retrieval_paths,
        "selectionReason": selection_reason,
    }


def _reading_core_payload(
    *,
    document: dict[str, Any],
    bundles: dict[str, list[dict[str, Any]]],
    claim_hints: list[dict[str, Any]],
    paper_memory: dict[str, Any],
    diagnostics: dict[str, Any],
    memory_route: dict[str, Any],
    supplemental_context_packet: dict[str, Any],
) -> dict[str, Any]:
    semantic_units = semantic_units_payload(document)
    weak_problem = not bool(bundles.get("problem"))
    weak_method = not bool(bundles.get("method"))
    weak_coverage = weak_problem or weak_method
    has_paper_memory = bool(_clean_text(paper_memory.get("paperCore")) or _clean_text(paper_memory.get("methodCore")))
    has_claim_evidence = bool(claim_hints)
    supplemental_context_used = bool(supplemental_context_packet.get("used"))

    degrade_reasons: list[str] = []
    parser_fallback_reason = _clean_text(diagnostics.get("parserFallbackReason"))
    if parser_fallback_reason:
        degrade_reasons.append(f"parser degraded: {parser_fallback_reason}")
    if not has_paper_memory:
        degrade_reasons.append("paper_memory unavailable; reading house handled framing fallback")
    if not _has_result_claim_hints(claim_hints):
        degrade_reasons.append("result claim_evidence unavailable; document_memory fallback active")
    if not _has_limitation_claim_hints(claim_hints):
        degrade_reasons.append("limitation claim_evidence unavailable; document_memory fallback active")
    if weak_coverage:
        degrade_reasons.append("section coverage weak; context house remains supplemental-only")

    field_routes = [
        _house_field_route(
            field=route["field"],
            primary_house=_primary_house_for_field(
                field=route["field"],
                paper_memory=paper_memory,
                claim_hints=claim_hints,
            ),
            selection_reason=str(route.get("selectionReason") or ""),
            retrieval_paths=list(route.get("retrievalPaths") or []),
            paper_memory=paper_memory,
            claim_hints=claim_hints,
        )
        for route in list(memory_route.get("fieldRoutes") or [])
    ]
    depth_plan = _depth_plan_payload(field_routes)
    primary_house_by_field = {str(route["field"]): str(route["primaryHouse"]) for route in field_routes}
    houses_used = {
        "readingHouse": True,
        "evidenceHouse": bool(has_paper_memory or has_claim_evidence),
        "contextHouse": supplemental_context_used,
    }
    disabled_houses = [
        {
            "name": "context_house",
            "reason": "single-paper summary에서는 ontology/cluster를 기본 primary input으로 승격하지 않습니다.",
        }
    ]
    if not (has_paper_memory or has_claim_evidence):
        disabled_houses.append(
            {
                "name": "evidence_house",
                "reason": "paper_memory와 claim_evidence가 모두 비어 있어 reading house fallback만 사용합니다.",
            }
        )
    reading_packet = {
        "documentId": str(document.get("documentId") or ""),
        "sourceOfTruth": "document_memory",
        "bundleCounts": {
            "problem": len(bundles.get("problem", [])),
            "method": len(bundles.get("method", [])),
            "results": len(bundles.get("results", [])),
            "limitations": len(bundles.get("limitations", [])),
            "novelty": len(bundles.get("novelty", [])),
        },
        "semanticUnitsCounts": dict(semantic_units.get("counts") or {}),
    }
    evidence_packet = {
        "paperMemoryAvailable": has_paper_memory,
        "claimHintCount": len(claim_hints),
        "resultHintsAvailable": _has_result_claim_hints(claim_hints),
        "limitationHintsAvailable": _has_limitation_claim_hints(claim_hints),
        "noveltyHintsAvailable": _has_novelty_claim_hints(claim_hints),
    }
    paper_summary_assembly = {
        "fields": field_routes,
        "verifierForm": "chunk",
    }
    route_warnings = list(memory_route.get("routeWarnings") or [])
    if weak_coverage and not supplemental_context_used:
        route_warnings = [*route_warnings, "context house kept disabled despite weak section coverage"]
    return {
        "decisionOrder": "memory_form_first",
        "housesUsed": houses_used,
        "primaryHouseByField": primary_house_by_field,
        "supplementalContextUsed": supplemental_context_used,
        "disabledHouses": disabled_houses,
        "degradeReasons": degrade_reasons,
        "routeWarnings": list(dict.fromkeys(route_warnings)),
        "depthPlan": depth_plan,
        "fieldRoutes": field_routes,
        "packets": {
            "readingPacket": reading_packet,
            "evidencePacket": evidence_packet,
            "supplementalContextPacket": supplemental_context_packet,
            "paperSummaryAssembly": paper_summary_assembly,
        },
    }


def _summary_prompt(*, language: str = "ko") -> str:
    return f"""당신은 AI/ML 논문을 읽고 구조화된 독해 결과를 만드는 분석가다.
아래 context를 읽고 JSON 객체 하나만 출력하라. 마크다운, 코드펜스, 설명문을 추가하지 마라.

필수 규칙:
- 출력 언어는 {language}
- 사실을 과장하지 말고 context에 있는 정보만 사용
- 모호한 항목은 빈 문자열 또는 짧은 주의 문구로 처리
- methodSteps, keyResults, limitations, confidenceNotes는 배열
- keyResults는 수치/데이터셋/비교 기준이 있으면 최대한 포함
- limitations는 실패 조건/적용 한계를 우선
- problem에는 figure/table wrapper 텍스트를 넣지 말고 문제 정의 문장만 사용
- methodSteps에는 결과 비교 문장을 넣지 말고 방법 설명만 사용
- keyResults에는 가능한 경우 benchmark/dataset/comparator/수치를 포함
- whatIsNew에는 baseline 대비 차이 또는 새로운 설계 포인트를 직접 써라
- context 안의 ONE_LINE_CANDIDATE가 있으면 다듬어서 쓰되 그대로 복사하지는 마라

JSON schema:
{{
  "oneLine": "string",
  "problem": "string",
  "coreIdea": "string",
  "methodSteps": ["string"],
  "keyResults": ["string"],
  "limitations": ["string"],
  "whenItMatters": "string",
  "whatIsNew": "string",
  "confidenceNotes": ["string"]
}}

예시:
{{
  "oneLine": "질문응답에서 검색 기반 메모리 카드를 재사용해 성능을 높이는 구조를 제안한다.",
  "problem": "긴 세션에서 과거 정보를 다시 찾아 쓰기 어렵다는 문제를 다룬다.",
  "coreIdea": "세션을 메모리 카드로 압축하고 이후 작업에서 관련 카드를 검색해 재사용한다.",
  "methodSteps": ["세션을 메모리 카드로 압축한다.", "관련 카드를 검색하고 재정렬한다."],
  "keyResults": ["개발자 에이전트 벤치마크에서 baseline 대비 성공률이 8.2포인트 향상된다."],
  "limitations": ["retrieval recall이 낮거나 domain shift가 크면 성능이 떨어진다."],
  "whenItMatters": "긴 세션을 유지하는 작업형 에이전트에서 특히 중요하다.",
  "whatIsNew": "로그 자체가 아니라 검색 가능한 memory card 단위를 중심으로 설계한다.",
  "confidenceNotes": ["results와 limitations는 선택된 근거 unit과 claim hint를 함께 참고했다."]
}}"""


def _normalize_summary(value: dict[str, Any], *, fallback: dict[str, Any]) -> dict[str, Any]:
    out = {
        "oneLine": _clean_text(value.get("oneLine")) or _clean_text(fallback.get("oneLine")),
        "problem": _clean_text(value.get("problem")) or _clean_text(fallback.get("problem")),
        "coreIdea": _clean_text(value.get("coreIdea")) or _clean_text(fallback.get("coreIdea")),
        "methodSteps": _clean_list(value.get("methodSteps")) or _clean_list(fallback.get("methodSteps")),
        "keyResults": _clean_list(value.get("keyResults")) or _clean_list(fallback.get("keyResults")),
        "limitations": _clean_list(value.get("limitations")) or _clean_list(fallback.get("limitations")),
        "whenItMatters": _clean_text(value.get("whenItMatters")) or _clean_text(fallback.get("whenItMatters")),
        "whatIsNew": _clean_text(value.get("whatIsNew")) or _clean_text(fallback.get("whatIsNew")),
        "confidenceNotes": _clean_list(value.get("confidenceNotes")) or _clean_list(fallback.get("confidenceNotes")),
    }
    return out


def _field_evidence(field: str, units: list[dict[str, Any]], *, document_id: str, limit: int = 2) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for unit in units:
        unit_id = _clean_text(unit.get("unitId"))
        if unit_id and unit_id in seen:
            continue
        if unit_id:
            seen.add(unit_id)
        provenance = dict(unit.get("provenance") or {})
        out.append(
            {
                "field": field,
                "documentId": document_id,
                "unitId": unit_id,
                "sectionPath": _clean_text(unit.get("sectionPath")),
                "page": provenance.get("page"),
                "bbox": provenance.get("bbox"),
                "excerpt": _excerpt(unit.get("sourceExcerpt") or unit.get("contextualSummary"), limit=220),
            }
        )
        if len(out) >= max(1, int(limit)):
            break
    return out


def _render_markdown(summary: dict[str, Any], evidence_map: list[dict[str, Any]]) -> str:
    evidence_by_field: dict[str, list[dict[str, Any]]] = {}
    for item in evidence_map:
        evidence_by_field.setdefault(str(item.get("field") or ""), []).append(item)

    def _marker(field: str) -> str:
        refs = list(evidence_by_field.get(field) or [])
        if not refs:
            return ""
        first = refs[0]
        section = _clean_text(first.get("sectionPath")) or _clean_text(first.get("unitId"))
        page = first.get("page")
        if section and page is not None:
            return f" [{section} p.{page}]"
        if section:
            return f" [{section}]"
        return ""

    lines = [
        "# Structured Paper Summary",
        "",
        "## 한줄 요약",
        "",
        f"{summary.get('oneLine') or ''}{_marker('oneLine')}",
        "",
        "## 문제",
        "",
        f"{summary.get('problem') or ''}{_marker('problem')}",
        "",
        "## 핵심 아이디어",
        "",
        f"{summary.get('coreIdea') or ''}{_marker('coreIdea')}",
        "",
        "## 방법",
        "",
    ]
    for item in list(summary.get("methodSteps") or []):
        lines.append(f"- {item}{_marker('methodSteps')}")
    lines.extend(["", "## 주요 결과", ""])
    for item in list(summary.get("keyResults") or []):
        lines.append(f"- {item}{_marker('keyResults')}")
    lines.extend(["", "## 한계", ""])
    for item in list(summary.get("limitations") or []):
        lines.append(f"- {item}{_marker('limitations')}")
    lines.extend(
        [
            "",
            "## 언제 중요한가",
            "",
            f"{summary.get('whenItMatters') or ''}{_marker('whenItMatters')}",
            "",
            "## 새로움",
            "",
            f"{summary.get('whatIsNew') or ''}{_marker('whatIsNew')}",
            "",
            "## 주의 사항",
            "",
        ]
    )
    for item in list(summary.get("confidenceNotes") or []):
        lines.append(f"- {item}")
    lines.extend(["", "## 부록: Evidence Map", ""])
    for item in evidence_map:
        lines.append(
            f"- `{item.get('field')}` :: {item.get('sectionPath') or item.get('unitId')} "
            f"(page={item.get('page')}) :: {item.get('excerpt')}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _build_user_card(
    *,
    paper_id: str,
    paper_title: str,
    parser_used: str,
    fallback_used: bool,
    llm_route: str,
    summary: dict[str, Any],
    evidence_map: list[dict[str, Any]],
    evidence_summaries: dict[str, Any],
    warnings: list[str],
    claim_coverage: dict[str, Any],
    summary_md_path: Path,
) -> dict[str, Any]:
    evidence_summary = {
        "keyResults": list(((evidence_summaries.get("keyResults") or {}).get("summaryLines") or [])),
        "limitations": list(((evidence_summaries.get("limitations") or {}).get("summaryLines") or [])),
        "whatIsNew": list(((evidence_summaries.get("whatIsNew") or {}).get("summaryLines") or [])),
    }
    return {
        "schema": "knowledge-hub.paper-summary.user-card.result.v1",
        "status": "ok" if all(summary.get(key) for key in ("oneLine", "problem", "coreIdea")) else "partial",
        "paperId": paper_id,
        "paperTitle": paper_title,
        "parserUsed": parser_used,
        "fallbackUsed": bool(fallback_used),
        "llmRoute": llm_route,
        "summary": summary,
        "evidenceSummary": evidence_summary,
        "evidenceMap": evidence_map,
        "claimCoverage": dict(claim_coverage),
        "warnings": warnings,
        "artifactPaths": {
            "summaryMdPath": str(summary_md_path),
        },
    }


@dataclass(slots=True)
class SummaryBuildArtifacts:
    artifact_dir: str
    summary_json_path: str
    summary_md_path: str
    manifest_path: str


class StructuredPaperSummaryService:
    def __init__(
        self,
        sqlite_db: Any,
        config: Any,
        *,
        document_memory_builder: DocumentMemoryBuilder | None = None,
        document_memory_retriever: DocumentMemoryRetriever | None = None,
    ):
        self.sqlite_db = sqlite_db
        self.config = config
        self._builder = document_memory_builder or DocumentMemoryBuilder(sqlite_db, config=config)
        self._retriever = document_memory_retriever or DocumentMemoryRetriever(sqlite_db)

    def artifact_dir_for(self, *, paper_id: str) -> Path:
        return _artifact_root(self.config, paper_id=str(paper_id).strip())

    def load_artifact(self, *, paper_id: str) -> dict[str, Any] | None:
        path = self.artifact_dir_for(paper_id=paper_id) / "summary.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return dict(payload) if isinstance(payload, dict) else None

    def _document_memory_payload(
        self,
        *,
        paper_id: str,
        paper_parser: str,
        refresh_parse: bool,
        opendataloader_options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        warnings: list[str] = []
        requested_parser = str(paper_parser or "auto").strip().lower() or "auto"
        attempted_parser = requested_parser
        parser_used = requested_parser
        parser_fallback_reason = ""
        if requested_parser == "auto":
            attempted: list[str] = []
            for candidate in ("mineru", "opendataloader"):
                attempted.append(candidate)
                try:
                    self._builder.build_and_store_paper(
                        paper_id=paper_id,
                        paper_parser=candidate,
                        refresh_parse=bool(refresh_parse),
                        opendataloader_options=opendataloader_options,
                    )
                    parser_used = candidate
                    attempted_parser = ",".join(attempted)
                    break
                except Exception as error:
                    parser_fallback_reason = _classify_parser_failure(error)
                    warnings.append(f"paper parser auto fallback from {candidate}: {parser_fallback_reason} ({error})")
            else:
                parser_used = "raw"
                attempted_parser = ",".join([*attempted, "raw"])
                self._builder.build_and_store_paper(
                    paper_id=paper_id,
                    paper_parser="raw",
                    refresh_parse=False,
                )
        else:
            self._builder.build_and_store_paper(
                paper_id=paper_id,
                paper_parser=requested_parser,
                refresh_parse=bool(refresh_parse),
                opendataloader_options=opendataloader_options,
            )
            parser_used = requested_parser
        document_id = f"paper:{paper_id}"
        document = self._retriever.get_document(document_id) or {}
        if not document:
            raise RuntimeError(f"document memory not found after build: {document_id}")
        units = list(document.get("units") or [])
        summary = dict(document.get("summary") or {})
        diagnostics = {
            "documentId": document_id,
            "unitCount": len(units),
            "structuredSectionsDetected": sum(
                1 for unit in units if list((unit.get("provenance") or {}).get("heading_path") or [])
            ),
            "elementsImported": int((summary.get("provenance") or {}).get("elements_imported") or 0),
            "parserUsed": parser_used,
            "parserAttempted": attempted_parser,
            "parserFallbackReason": parser_fallback_reason,
            "parseArtifactPath": str(_parsed_artifact_root(self.config, paper_id=paper_id))
            if parser_used in {"opendataloader", "mineru"}
            else "",
        }
        return document, diagnostics, warnings

    def _fallback_summary(
        self,
        document: dict[str, Any],
        bundles: dict[str, list[dict[str, Any]]],
        paper_memory: dict[str, Any],
        *,
        paper_id: str,
        parser_used: str,
        llm_warning: str = "",
    ) -> dict[str, Any]:
        summary = dict(document.get("summary") or {})
        thesis = _clean_text(summary.get("documentThesis") or summary.get("contextualSummary"))
        problem = (
            _clean_text((bundles.get("problem") or [{}])[0].get("contextualSummary"))
            if bundles.get("problem")
            else _clean_text(paper_memory.get("problemContext")) or thesis
        )
        core_idea = (
            _clean_text(paper_memory.get("paperCore"))
            or _clean_text(paper_memory.get("methodCore"))
            or (_clean_text((bundles.get("method") or [{}])[0].get("contextualSummary")) if bundles.get("method") else thesis)
        )
        method_steps = [
            _clean_text(unit.get("contextualSummary") or unit.get("sourceExcerpt"))
            for unit in list(bundles.get("method") or [])[:4]
            if _clean_text(unit.get("contextualSummary") or unit.get("sourceExcerpt"))
        ]
        key_results = [
            _clean_text(unit.get("contextualSummary") or unit.get("sourceExcerpt"))
            for unit in list(bundles.get("results") or [])[:4]
            if _clean_text(unit.get("contextualSummary") or unit.get("sourceExcerpt"))
        ]
        limitations = [
            _clean_text(unit.get("contextualSummary") or unit.get("sourceExcerpt"))
            for unit in list(bundles.get("limitations") or [])[:3]
            if _clean_text(unit.get("contextualSummary") or unit.get("sourceExcerpt"))
        ]
        confidence_notes = []
        if llm_warning:
            confidence_notes.append(llm_warning)
        if parser_used == "raw":
            confidence_notes.append("구조 provenance는 raw parser 기준이라 page/bbox 정보가 비어 있을 수 있습니다.")
        if not limitations:
            confidence_notes.append("한계 섹션 신호가 약해 limitation coverage가 제한적일 수 있습니다.")
        return {
            "oneLine": _clean_text(paper_memory.get("paperCore")) or thesis or f"{paper_id} 논문의 구조화 요약",
            "problem": problem or thesis,
            "coreIdea": core_idea or thesis,
            "methodSteps": method_steps,
            "keyResults": key_results,
            "limitations": limitations,
            "whenItMatters": key_results[0] if key_results else (_clean_text(paper_memory.get("evidenceCore")) or thesis),
            "whatIsNew": _clean_text(paper_memory.get("methodCore")) or core_idea or thesis,
            "confidenceNotes": confidence_notes,
        }

    def _llm_summary(self, *, title: str, context: str) -> tuple[dict[str, Any], dict[str, Any], list[str], bool]:
        llm, decision, warnings = get_llm_for_task(
            self.config,
            task_type="materialization_summary",
            allow_external=True,
            query=title,
            context=context[:12000],
            source_count=1,
            timeout_sec=90,
        )
        if llm is None:
            return {}, decision.to_dict(), [*warnings, "요약 LLM을 사용할 수 없어 deterministic fallback으로 요약했습니다."], True
        try:
            text = llm.generate(_summary_prompt(language="ko"), context=context, max_tokens=2200)
        except Exception as error:
            return {}, decision.to_dict(), [*warnings, f"요약 LLM 호출 실패: {error}"], True
        parsed = _extract_json_object(text)
        decision_reasons = list(getattr(decision, "reasons", []) or [])
        fallback_used = any(str(reason).startswith("fallback_from_") for reason in decision_reasons)
        if not parsed:
            warnings = [*warnings, "LLM structured output을 JSON으로 파싱하지 못해 fallback으로 보정했습니다."]
            fallback_used = True
        return parsed, decision.to_dict(), warnings, fallback_used

    def build(
        self,
        *,
        paper_id: str,
        paper_parser: str = "raw",
        refresh_parse: bool = False,
        quick: bool = False,
        opendataloader_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        token = str(paper_id).strip()
        paper = self.sqlite_db.get_paper(token)
        if not paper:
            raise ValueError(f"paper not found: {paper_id}")

        requested_parser = str(paper_parser or "auto").strip().lower() or "auto"
        try:
            document, diagnostics, parser_warnings = self._document_memory_payload(
                paper_id=token,
                paper_parser=requested_parser,
                refresh_parse=bool(refresh_parse),
                opendataloader_options=opendataloader_options,
            )
        except Exception as error:
            parser_failure = _classify_parser_failure(error)
            if requested_parser in {"opendataloader", "mineru"}:
                blocked_memory_route = {
                    "decisionOrder": "memory_form_first",
                    "fieldRoutes": [],
                    "disabledForms": [
                        {"name": "ontology_cluster", "reason": "single-paper summary에서는 primary input으로 쓰지 않습니다."},
                        {"name": "chunk_as_primary", "reason": "chunk는 verifier/fallback 전용입니다."},
                    ],
                    "routeWarnings": ["paper summary blocked before reading-core assembly"],
                }
                blocked_reading_core = {
                    "decisionOrder": "memory_form_first",
                    "housesUsed": {
                        "readingHouse": False,
                        "evidenceHouse": False,
                        "contextHouse": False,
                    },
                    "primaryHouseByField": {},
                    "supplementalContextUsed": False,
                    "disabledHouses": [
                        {"name": "reading_house", "reason": "document-memory artifact를 만들지 못했습니다."},
                        {"name": "evidence_house", "reason": "paper summary assembly 전에 blocked 상태가 발생했습니다."},
                        {"name": "context_house", "reason": "single-paper summary에서는 기본 비활성입니다."},
                    ],
                    "degradeReasons": [str(error)],
                    "routeWarnings": ["paper summary blocked before reading-core assembly"],
                    "fieldRoutes": [],
                    "packets": {
                        "readingPacket": {"sourceOfTruth": "document_memory", "bundleCounts": {}, "semanticUnitsCounts": {}},
                        "evidencePacket": {"paperMemoryAvailable": False, "claimHintCount": 0},
                        "supplementalContextPacket": {"eligible": False, "used": False, "mode": "supplemental_only", "sources": ["ontology", "cluster"]},
                        "paperSummaryAssembly": {"fields": [], "verifierForm": "chunk"},
                    },
                }
                return {
                    "schema": "knowledge-hub.paper-summary.build.result.v1",
                    "status": "blocked",
                    "paperId": token,
                    "paperTitle": _clean_text(paper.get("title")),
                    "parserUsed": requested_parser,
                    "parserAttempted": requested_parser,
                    "parserFallbackReason": parser_failure,
                    "fallbackUsed": True,
                    "llmRoute": "fallback-only",
                    "contextStats": {
                        "problemUnits": 0,
                        "methodUnits": 0,
                        "resultUnits": 0,
                        "limitationUnits": 0,
                        "claimHintsUsed": 0,
                        "paperMemoryAvailable": False,
                        "contextChars": 0,
                    },
                    "documentMemoryDiagnostics": {
                        "documentId": f"paper:{token}",
                        "unitCount": 0,
                        "structuredSectionsDetected": 0,
                        "elementsImported": 0,
                        "parserUsed": requested_parser,
                        "parserAttempted": requested_parser,
                        "parserFallbackReason": parser_failure,
                        "parseArtifactPath": str(_parsed_artifact_root(self.config, paper_id=token)),
                    },
                    "memoryRoute": blocked_memory_route,
                    "readingCore": blocked_reading_core,
                    "evidenceSummaries": {},
                    "summary": {},
                    "evidenceMap": [],
                    "warnings": [str(error)],
                }
            raise

        all_units = [
            unit for unit in list(document.get("units") or [])
            if _clean_text(unit.get("unitType")) != "document_summary"
        ]
        parser_used = str(diagnostics.get("parserUsed") or requested_parser or "raw")
        bundle_limits = {"problem": 1, "method": 2, "results": 2, "limitations": 1, "novelty": 1} if quick else {
            "problem": 2,
            "method": 4,
            "results": 4,
            "limitations": 2,
            "novelty": 2,
        }
        paper_memory, paper_memory_warnings = _paper_memory_payload(self.sqlite_db, paper_id=token)
        claim_hints, claim_coverage = _claim_hint_rows_with_coverage(self.sqlite_db, self.config, paper_id=token)
        bundles = {
            "problem": _choose_units(all_units, "problem", limit=bundle_limits["problem"], claim_hints=claim_hints, parser_used=parser_used),
            "method": _choose_units(all_units, "method", limit=bundle_limits["method"], claim_hints=claim_hints, parser_used=parser_used),
            "results": _choose_units(all_units, "results", limit=bundle_limits["results"], claim_hints=claim_hints, parser_used=parser_used),
            "limitations": _choose_units(all_units, "limitations", limit=bundle_limits["limitations"], claim_hints=claim_hints, parser_used=parser_used),
            "novelty": _choose_units(all_units, "novelty", limit=bundle_limits["novelty"], claim_hints=claim_hints, parser_used=parser_used),
        }
        memory_route = _memory_route_payload(bundles=bundles, claim_hints=claim_hints, paper_memory=paper_memory)
        supplemental_context_packet = _supplemental_context_packet(
            sqlite_db=self.sqlite_db,
            paper_id=token,
            bundles=bundles,
            claim_hints=claim_hints,
            weak_coverage=(not bool(bundles.get("problem")) or not bool(bundles.get("method"))),
        )
        reading_core = _reading_core_payload(
            document=document,
            bundles=bundles,
            claim_hints=claim_hints,
            paper_memory=paper_memory,
            diagnostics=diagnostics,
            memory_route=memory_route,
            supplemental_context_packet=supplemental_context_packet,
        )
        evidence_summaries = _evidence_summaries_payload(
            bundles=bundles,
            claim_hints=claim_hints,
            paper_memory=paper_memory,
        )
        context = _build_context(
            document,
            bundles,
            claim_hints,
            paper_memory,
            evidence_summaries,
            supplemental_context_packet,
            quick=bool(quick),
        )
        llm_value, llm_decision, warnings, llm_fallback_used = self._llm_summary(title=_clean_text(paper.get("title")), context=context)
        warnings = [*parser_warnings, *paper_memory_warnings, *warnings]
        fallback = self._fallback_summary(
            document,
            bundles,
            paper_memory,
            paper_id=token,
            parser_used=parser_used,
            llm_warning=warnings[0] if warnings else "",
        )
        summary = _normalize_summary(llm_value, fallback=fallback)
        evidence_map = []
        evidence_map.extend(_field_evidence("oneLine", (bundles.get("problem", [])[:1] or bundles.get("novelty", [])[:1] or bundles.get("results", [])[:1]), document_id=f"paper:{token}", limit=1))
        evidence_map.extend(_field_evidence("problem", bundles.get("problem", [])[:2], document_id=f"paper:{token}", limit=2))
        evidence_map.extend(_field_evidence("coreIdea", bundles.get("method", [])[:2], document_id=f"paper:{token}", limit=2))
        evidence_map.extend(_field_evidence("methodSteps", bundles.get("method", [])[: min(2, len(bundles.get('method', [])))], document_id=f"paper:{token}", limit=2))
        evidence_map.extend(_field_evidence("keyResults", bundles.get("results", [])[: min(2, len(bundles.get('results', [])))], document_id=f"paper:{token}", limit=2))
        evidence_map.extend(_field_evidence("limitations", bundles.get("limitations", [])[: min(2, len(bundles.get('limitations', [])))], document_id=f"paper:{token}", limit=2))
        evidence_map.extend(_field_evidence("whenItMatters", bundles.get("results", [])[:1], document_id=f"paper:{token}", limit=1))
        evidence_map.extend(_field_evidence("whatIsNew", bundles.get("novelty", [])[:1] or bundles.get("method", [])[:1], document_id=f"paper:{token}", limit=1))

        artifact_dir = self.artifact_dir_for(paper_id=token)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        summary_json_path = artifact_dir / "summary.json"
        card_json_path = artifact_dir / "card.json"
        summary_md_path = artifact_dir / "summary.md"
        manifest_path = artifact_dir / "manifest.json"
        evidence_summaries_path = artifact_dir / "evidence-summaries.json"
        user_card = _build_user_card(
            paper_id=token,
            paper_title=_clean_text(paper.get("title")),
            parser_used=parser_used,
            fallback_used=bool(llm_fallback_used or bool(diagnostics.get("parserFallbackReason"))),
            llm_route=_clean_text(llm_decision.get("route")),
            summary=summary,
            evidence_map=evidence_map,
            evidence_summaries=evidence_summaries,
            warnings=list(dict.fromkeys(warnings)),
            claim_coverage=claim_coverage,
            summary_md_path=summary_md_path,
        )
        manifest = {
            "paper_id": token,
            "paper_title": _clean_text(paper.get("title")),
            "parser_used": parser_used,
            "parser_attempted": str(diagnostics.get("parserAttempted") or requested_parser),
            "parser_fallback_reason": _clean_text(diagnostics.get("parserFallbackReason")),
            "document_memory_document_id": f"paper:{token}",
            "source_hash": _source_signature_with_memory(document, claim_hints, paper_memory, paper_parser=parser_used, quick=bool(quick)),
            "llm": {
                "provider": _clean_text(llm_decision.get("provider")),
                "model": _clean_text(llm_decision.get("model")),
                "route": _clean_text(llm_decision.get("route")),
                "timeoutSec": llm_decision.get("timeoutSec"),
            },
            "fallback_used": bool(llm_fallback_used or bool(diagnostics.get("parserFallbackReason"))),
            "llm_route": _clean_text(llm_decision.get("route")),
            "context_stats": {
                "problemUnits": len(bundles.get("problem", [])),
                "methodUnits": len(bundles.get("method", [])),
                "resultUnits": len(bundles.get("results", [])),
                "limitationUnits": len(bundles.get("limitations", [])),
                "claimHintsUsed": len(claim_hints),
                "claimCoverage": {**dict(claim_coverage), "paperMemoryAvailable": bool(paper_memory)},
                "paperMemoryAvailable": bool(paper_memory),
                "contextChars": len(context),
            },
            "claim_coverage": claim_coverage,
            "user_card_path": str(card_json_path),
            "evidence_summaries_path": str(evidence_summaries_path),
            "evidence_summaries": evidence_summaries,
            "memory_route": memory_route,
            "reading_core": reading_core,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "warnings": list(dict.fromkeys(warnings)),
        }
        payload = {
            "schema": "knowledge-hub.paper-summary.build.result.v1",
            "status": "ok" if all(summary.get(key) for key in ("oneLine", "problem", "coreIdea")) else "partial",
            "paperId": token,
            "paperTitle": _clean_text(paper.get("title")),
            "parserUsed": parser_used,
            "parserAttempted": str(diagnostics.get("parserAttempted") or requested_parser),
            "parserFallbackReason": _clean_text(diagnostics.get("parserFallbackReason")),
            "fallbackUsed": bool(llm_fallback_used or bool(diagnostics.get("parserFallbackReason"))),
            "llmRoute": _clean_text(llm_decision.get("route")),
            "contextStats": {
                "problemUnits": len(bundles.get("problem", [])),
                "methodUnits": len(bundles.get("method", [])),
                "resultUnits": len(bundles.get("results", [])),
                "limitationUnits": len(bundles.get("limitations", [])),
                "claimHintsUsed": len(claim_hints),
                "claimCoverage": dict(claim_coverage),
                "paperMemoryAvailable": bool(paper_memory),
                "contextChars": len(context),
            },
            "claimCoverage": {**dict(claim_coverage), "paperMemoryAvailable": bool(paper_memory)},
            "documentMemoryDiagnostics": diagnostics,
            "memoryRoute": memory_route,
            "readingCore": reading_core,
            "evidenceSummaries": evidence_summaries,
            "summary": summary,
            "evidenceMap": evidence_map,
            "warnings": list(dict.fromkeys(warnings)),
            "artifactPaths": {
                "artifactDir": str(artifact_dir),
                "cardJsonPath": str(card_json_path),
                "evidenceSummariesPath": str(evidence_summaries_path),
                "summaryJsonPath": str(summary_json_path),
                "summaryMdPath": str(summary_md_path),
                "manifestPath": str(manifest_path),
            },
        }
        card_json_path.write_text(json.dumps(user_card, ensure_ascii=False, indent=2), encoding="utf-8")
        evidence_summaries_path.write_text(json.dumps(evidence_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_md_path.write_text(_render_markdown(summary, evidence_map), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def show(self, *, paper_id: str) -> dict[str, Any]:
        token = str(paper_id).strip()
        payload = self.load_artifact(paper_id=token)
        if not payload:
            return {
                "schema": "knowledge-hub.paper-summary.card.result.v1",
                "status": "failed",
                "paperId": token,
                "summary": {},
                "evidenceMap": [],
                "warnings": [f"paper summary artifact not found: {token}"],
            }
        payload = dict(payload)
        payload["schema"] = "knowledge-hub.paper-summary.card.result.v1"
        return payload


__all__ = ["StructuredPaperSummaryService", "SummaryBuildArtifacts"]
