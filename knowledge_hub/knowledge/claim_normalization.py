"""Claim normalization and comparison helpers for labs surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from knowledge_hub.learning.task_router import get_llm_for_task
from knowledge_hub.knowledge.semantic_units import evidence_links_from_claim


CLAIM_NORMALIZATION_VERSION = "claim-v2"
_UNKNOWN = "unknown"

_TASK_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("open_domain_qa", ("open-domain question answering", "open-domain qa", "natural questions", "triviaqa")),
    ("question_answering", ("question answering", "qa", "reading comprehension")),
    ("retrieval_augmented_generation", ("retrieval-augmented generation", "retrieval augmented generation", "rag")),
    ("retrieval", ("retrieval", "retriever", "retrieval recall", "passage retrieval")),
    ("faithfulness_evaluation", ("faithfulness", "grounding", "grounded answer", "hallucination")),
)

_DATASET_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Natural Questions", ("natural questions", "nq")),
    ("TriviaQA", ("triviaqa",)),
    ("HotpotQA", ("hotpotqa",)),
    ("SQuAD", ("squad",)),
)

_METRIC_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("exact_match", ("exact match", "em")),
    ("f1", ("f1", "f1 score")),
    ("recall", ("recall", "retrieval recall")),
    ("faithfulness", ("faithfulness", "grounding quality")),
    ("accuracy", ("accuracy",)),
    ("precision", ("precision",)),
)

_SCOPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("open-domain QA", ("open-domain question answering", "open-domain qa")),
    ("single-hop", ("single-hop", "single hop")),
    ("multi-hop", ("multi-hop", "multi hop")),
    ("low-recall regime", ("low recall", "retrieval recall is low")),
    ("domain shift", ("domain shift",)),
    ("english-only", ("english-only", "english only", "english corpus", "english data")),
    ("multilingual", ("multilingual", "cross-lingual", "cross lingual")),
    ("few-shot", ("few-shot", "few shot")),
    ("zero-shot", ("zero-shot", "zero shot")),
    ("low-resource", ("low-resource", "low resource", "data-scarce")),
    ("biomedical", ("biomedical",)),
    ("legal", ("legal domain", "legal")),
    ("financial", ("financial domain", "financial")),
)

_LIMITATION_CUES = (
    "however",
    "but",
    "limitation",
    "limitations",
    "degrade",
    "drops",
    "drop",
    "low recall",
    "domain shift",
    "not address",
    "unresolved",
    "한계",
    "제약",
)

_BETTER_CUES = (
    "improve",
    "improves",
    "improved",
    "better",
    "outperform",
    "outperforms",
    "gain",
    "gains",
    "increase",
    "increases",
    "higher",
    "향상",
    "개선",
)

_WORSE_CUES = (
    "worse",
    "degrade",
    "degrades",
    "degraded",
    "decrease",
    "decreases",
    "decreased",
    "drop",
    "drops",
    "lower",
    "reduce",
    "reduces",
    "reduced",
    "hurt",
    "harms",
    "저하",
    "감소",
    "악화",
)

_NEUTRAL_CUES = (
    "depends on",
    "bounded by",
    "associated with",
    "linked to",
    "requires",
    "requires",
    "depends",
)

_COMPARATOR_PATTERNS = (
    r"(?:over|than|versus|vs\.?|compared to|compared with)\s+(?:a\s+|an\s+|the\s+)?([a-z0-9][^.,;]+?)(?:\s+(?:on|when|while|under)\b|$)",
)

_VALUE_RE = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>points?|%|percent|x|times?)?", re.IGNORECASE)
_WHEN_RE = re.compile(r"\b(?:when|if|under|in)\s+([^.;]+)")
_TOPK_RE = re.compile(r"\btop-?k\s*(?:=|of|set to)?\s*(\d+)", re.IGNORECASE)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_token(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _safe_json_loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _first_nonempty(*values: Any) -> str:
    for value in values:
        token = _clean_text(value)
        if token:
            return token
    return ""


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = _clean_text(item)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(token)
    return out


def _limited_text(value: str, limit: int = 240) -> str:
    token = _clean_text(value)
    if len(token) <= limit:
        return token
    return token[: max(0, limit - 3)].rstrip() + "..."


def _evidence_summary(claim: dict[str, Any], normalized: dict[str, Any], context: dict[str, Any]) -> str:
    parts = [
        _first_nonempty(normalized.get("result_value_text"), normalized.get("metric"), normalized.get("task")),
        _clean_text(context.get("paper_title") or ""),
        _clean_text(claim.get("claim_text") or ""),
    ]
    return _limited_text(" | ".join(part for part in parts if part), limit=220)


def _match_patterns(text: str, patterns: tuple[tuple[str, tuple[str, ...]], ...]) -> str:
    lowered = _normalize_token(text)
    for label, hints in patterns:
        if any(hint in lowered for hint in hints):
            return label
    return ""


def _all_pattern_matches(text: str, patterns: tuple[tuple[str, tuple[str, ...]], ...]) -> list[str]:
    lowered = _normalize_token(text)
    out: list[str] = []
    for label, hints in patterns:
        if any(hint in lowered for hint in hints):
            out.append(label)
    return _unique(out)


def _extract_numeric_value(text: str) -> tuple[str, float | None]:
    for match in _VALUE_RE.finditer(str(text or "")):
        number = match.group("num")
        unit = _clean_text(match.group("unit"))
        if not number:
            continue
        try:
            parsed = float(number)
        except Exception:
            continue
        text_value = f"{number} {unit}".strip()
        return text_value, parsed
    return "", None


def _extract_comparator(text: str, fallback: str = "") -> str:
    body = _clean_text(text)
    for pattern in _COMPARATOR_PATTERNS:
        match = re.search(pattern, body, flags=re.IGNORECASE)
        if match:
            return _clean_text(match.group(1))
    fallback_token = _clean_text(fallback)
    return fallback_token


def _extract_conditions(text: str) -> str:
    body = _clean_text(text)
    matches: list[str] = []
    for item in _WHEN_RE.findall(body):
        token = _clean_text(item)
        if token:
            matches.append(token)
    for item in _TOPK_RE.findall(body):
        matches.append(f"top-k={item}")
    for cue in ("retrieval recall is low", "low recall", "domain shift", "ablations where relevant passages are missing"):
        if cue in body.lower():
            matches.append(cue)
    for cue in ("few-shot", "few shot", "zero-shot", "zero shot", "low-resource", "low resource", "english-only", "multilingual"):
        if cue in body.lower():
            matches.append(cue)
    return "; ".join(_unique(matches[:3]))


def _extract_direction(predicate: str, text: str) -> str:
    haystack = f"{predicate} {text}".lower()
    if re.search(r"\bdecreas(?:e|es|ed|ing)\b", haystack):
        return "worse"
    if any(cue in haystack for cue in _BETTER_CUES):
        return "better"
    if any(cue in haystack for cue in _WORSE_CUES):
        return "worse"
    if any(cue in haystack for cue in _NEUTRAL_CUES):
        return "neutral"
    return _UNKNOWN


def _extract_limitation_text(text: str) -> str:
    lowered = text.lower()
    if any(cue in lowered for cue in _LIMITATION_CUES):
        return _limited_text(text, limit=240)
    return ""


def _extract_negative_scope_text(text: str) -> str:
    body = _clean_text(text)
    patterns = (
        r"\b(?:does not|do not|did not|not evaluated on|not applicable to|only evaluated on|limited to)\b[^.;]+",
        r"\b(?:적용되지 않|다루지 않|평가하지 않|제한된다)\b[^.;]+",
    )
    for pattern in patterns:
        match = re.search(pattern, body, flags=re.IGNORECASE)
        if match:
            return _limited_text(match.group(0), limit=240)
    return ""


def _extract_evidence_strength(text: str, *, dataset: str, metric: str, value_numeric: float | None) -> str:
    lowered = text.lower()
    if value_numeric is not None and (dataset or metric):
        return "strong"
    if any(token in lowered for token in ("ablation", "benchmark", "experiment", "evaluat", "measured")) and (dataset or metric):
        return "strong"
    if dataset or metric or any(token in lowered for token in ("benchmark", "experiment", "results", "evaluation")):
        return "medium"
    return "weak"


def _status_for(
    task: str,
    metric: str,
    dataset: str,
    comparator: str,
    evidence_strength: str,
    limitation_text: str,
) -> str:
    if task and metric and dataset and comparator:
        return "normalized"
    if task or metric or limitation_text or evidence_strength != "weak":
        return "partial"
    return "failed"


def _load_note_row(db: Any, note_id: str) -> dict[str, Any] | None:
    if not note_id:
        return None
    try:
        return db.get_note(note_id)
    except Exception:
        return None


def _paper_by_path(db: Any, path_value: str) -> dict[str, Any] | None:
    token = str(path_value or "").strip()
    if not token:
        return None
    try:
        papers = db.list_papers(limit=5000)
    except Exception:
        return None
    for paper in papers:
        for key in ("text_path", "translated_path", "pdf_path"):
            if str(paper.get(key) or "").strip() == token:
                return paper
    return None


def _paper_id_from_claim(claim: dict[str, Any], evidence_ptrs: list[dict[str, Any]], db: Any) -> str:
    claim_id = str(claim.get("claim_id") or "").strip()
    match = re.match(r"^claim:([0-9]{4}\.[0-9]{4,5})[:_]", claim_id)
    if match:
        return match.group(1)
    for key in ("subject_entity_id", "object_entity_id"):
        token = str(claim.get(key) or "").strip()
        if token.startswith("paper:"):
            return token.split("paper:", 1)[1]
    for ptr in evidence_ptrs:
        if not isinstance(ptr, dict):
            continue
        note_id = str(ptr.get("note_id") or "").strip()
        if note_id.startswith("paper:"):
            return note_id.split("paper:", 1)[1]
        paper_id = str(ptr.get("paper_id") or ptr.get("arxiv_id") or "").strip()
        if paper_id:
            return paper_id
        paper = _paper_by_path(db, str(ptr.get("path") or ""))
        if paper:
            return str(paper.get("arxiv_id") or "").strip()
    return ""


def _claim_context(db: Any, claim: dict[str, Any]) -> dict[str, Any]:
    evidence_ptrs = list(claim.get("evidence_ptrs") or [])
    note_titles: list[str] = []
    note_ids: list[str] = []
    paper_id = _paper_id_from_claim(claim, evidence_ptrs, db)
    paper_title = ""
    paper_text = ""
    if paper_id:
        paper = db.get_paper(paper_id)
        if paper:
            paper_title = str(paper.get("title") or "").strip()
            paper_text = _first_nonempty(paper.get("notes"))
            if not paper_text:
                for key in ("translated_path", "text_path"):
                    raw_path = Path(str(paper.get(key) or "").strip())
                    if raw_path.exists():
                        try:
                            paper_text = raw_path.read_text(encoding="utf-8")[:3000]
                        except Exception:
                            paper_text = ""
                        if paper_text:
                            break
    for ptr in evidence_ptrs:
        if not isinstance(ptr, dict):
            continue
        note_id = str(ptr.get("note_id") or "").strip()
        if note_id:
            note = _load_note_row(db, note_id)
            if note:
                note_ids.append(note_id)
                note_titles.append(str(note.get("title") or note_id))
    if paper_id:
        note_id = f"paper:{paper_id}"
        note = _load_note_row(db, note_id)
        if note:
            note_ids.append(note_id)
            note_titles.append(str(note.get("title") or note_id))
            if not paper_text:
                paper_text = str(note.get("content") or "")[:3000]
    return {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "paper_text": paper_text,
        "note_ids": _unique(note_ids),
        "note_titles": _unique(note_titles),
    }


def _is_accepted_claim(claim: dict[str, Any]) -> bool:
    evidence_ptrs = list(claim.get("evidence_ptrs") or [])
    accepted_seen = False
    for ptr in evidence_ptrs:
        if not isinstance(ptr, dict):
            continue
        decision = _normalize_token(ptr.get("claim_decision"))
        if decision == "pending":
            return False
        if decision == "accepted":
            accepted_seen = True
    return accepted_seen or not evidence_ptrs


def _llm_fill_fields(
    db: Any,
    config: Any,
    claim: dict[str, Any],
    context: dict[str, Any],
    *,
    allow_external: bool,
    llm_mode: str,
) -> tuple[dict[str, str], list[str]]:
    llm, decision, warnings = get_llm_for_task(
        config,
        task_type="claim_extraction",
        allow_external=bool(allow_external),
        query=str(claim.get("claim_text") or ""),
        context=_first_nonempty(context.get("paper_title"), context.get("paper_text"))[:2500],
        source_count=max(1, len(context.get("note_titles") or [])),
        force_route=llm_mode if llm_mode in {"fallback-only", "local", "mini", "strong", "auto"} else None,
    )
    if llm is None:
        return {}, list(warnings or [])
    prompt = (
        "Extract missing claim-normalization fields as compact JSON.\n"
        "Return only valid JSON with keys: task, dataset, metric, comparator, condition_text, scope_text, limitation_text, negative_scope_text.\n"
        "Use empty strings when unknown.\n\n"
        f"Claim: {claim.get('claim_text')}\n"
        f"Predicate: {claim.get('predicate')}\n"
        f"Paper title: {context.get('paper_title')}\n"
        f"Note titles: {', '.join(context.get('note_titles') or [])}\n"
        f"Context:\n{_limited_text(str(context.get('paper_text') or ''), limit=2400)}\n"
    )
    try:
        raw = llm.generate(prompt).strip()
    except Exception as error:
        return {}, [*list(warnings or []), f"claim normalization llm failed: {error}"]
    if hasattr(decision, "to_dict"):
        warnings = [*list(warnings or []), f"claim-normalization route={decision.to_dict().get('route', '')}"]
    body = raw
    if "```" in body:
        body = re.sub(r"^```(?:json)?\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    try:
        parsed = json.loads(body)
    except Exception:
        return {}, [*list(warnings or []), "claim normalization llm returned invalid json"]
    if not isinstance(parsed, dict):
        return {}, [*list(warnings or []), "claim normalization llm returned non-object json"]
    return {key: _clean_text(parsed.get(key)) for key in parsed}, list(warnings or [])


def claim_normalization_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row.get("normalized_payload") or {})
    return {
        "claimId": str(row.get("claim_id") or ""),
        "version": str(row.get("normalization_version") or CLAIM_NORMALIZATION_VERSION),
        "status": str(row.get("status") or "failed"),
        "task": str(row.get("task") or ""),
        "dataset": str(row.get("dataset") or ""),
        "metric": str(row.get("metric") or ""),
        "comparator": str(row.get("comparator") or ""),
        "resultDirection": str(row.get("result_direction") or _UNKNOWN),
        "resultValueText": str(row.get("result_value_text") or ""),
        "resultValueNumeric": row.get("result_value_numeric"),
        "conditionText": str(row.get("condition_text") or ""),
        "scopeText": str(row.get("scope_text") or ""),
        "limitationText": str(row.get("limitation_text") or ""),
        "negativeScopeText": str(row.get("negative_scope_text") or ""),
        "evidenceStrength": str(row.get("evidence_strength") or "weak"),
        "paperId": str(payload.get("paper_id") or ""),
        "paperTitle": str(payload.get("paper_title") or ""),
        "noteIds": list(payload.get("note_ids") or []),
        "noteTitles": list(payload.get("note_titles") or []),
        "evidenceSummary": str(payload.get("evidence_summary") or ""),
        "warnings": list(payload.get("warnings") or []),
        "createdAt": str(row.get("created_at") or ""),
        "updatedAt": str(row.get("updated_at") or ""),
    }


@dataclass(slots=True)
class ClaimNormalizationService:
    db: Any
    config: Any

    def _candidate_claims(
        self,
        *,
        claim_ids: list[str] | None = None,
        paper_ids: list[str] | None = None,
        include_all: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if claim_ids:
            result: list[dict[str, Any]] = []
            for claim_id in claim_ids:
                item = self.db.get_claim(str(claim_id).strip())
                if item and _is_accepted_claim(item):
                    result.append(item)
            return result[: max(1, int(limit))]
        rows = self.db.list_ontology_claims(limit=max(1, int(limit * 8 if paper_ids else limit)))
        if include_all or not paper_ids:
            return [row for row in rows if _is_accepted_claim(row)][: max(1, int(limit))]
        wanted = {str(item).strip() for item in paper_ids if str(item).strip()}
        result = []
        for row in rows:
            if not _is_accepted_claim(row):
                continue
            context = _claim_context(self.db, row)
            paper_id = str(context.get("paper_id") or "")
            if paper_id and paper_id in wanted:
                result.append(row)
            if len(result) >= max(1, int(limit)):
                break
        return result

    def normalize_claim(
        self,
        claim: dict[str, Any],
        *,
        allow_external: bool = False,
        llm_mode: str = "auto",
        persist: bool = True,
    ) -> dict[str, Any]:
        claim_id = str(claim.get("claim_id") or "").strip()
        context = _claim_context(self.db, claim)
        combined_text = " ".join(
            part
            for part in (
                claim.get("claim_text"),
                claim.get("predicate"),
                context.get("paper_title"),
                " ".join(context.get("note_titles") or []),
                context.get("paper_text"),
            )
            if _clean_text(part)
        )
        task = _match_patterns(combined_text, _TASK_PATTERNS)
        dataset = "; ".join(_all_pattern_matches(combined_text, _DATASET_PATTERNS))
        metric = _match_patterns(combined_text, _METRIC_PATTERNS)
        comparator = _extract_comparator(combined_text, str(claim.get("object_literal") or ""))
        result_value_text, result_value_numeric = _extract_numeric_value(combined_text)
        condition_text = _extract_conditions(combined_text)
        scope_text = "; ".join(_all_pattern_matches(combined_text, _SCOPE_PATTERNS))
        result_direction = _extract_direction(str(claim.get("predicate") or ""), combined_text)
        limitation_text = _extract_limitation_text(str(claim.get("claim_text") or ""))
        negative_scope_text = _extract_negative_scope_text(combined_text)
        evidence_strength = _extract_evidence_strength(
            combined_text,
            dataset=dataset,
            metric=metric,
            value_numeric=result_value_numeric,
        )
        warnings: list[str] = []

        if any(not value for value in (task, dataset, metric, comparator, condition_text, scope_text, limitation_text, negative_scope_text)):
            llm_values, llm_warnings = _llm_fill_fields(
                self.db,
                self.config,
                claim,
                context,
                allow_external=allow_external,
                llm_mode=llm_mode,
            )
            warnings.extend(llm_warnings)
            task = task or str(llm_values.get("task") or "")
            dataset = dataset or str(llm_values.get("dataset") or "")
            metric = metric or str(llm_values.get("metric") or "")
            comparator = comparator or str(llm_values.get("comparator") or "")
            condition_text = condition_text or str(llm_values.get("condition_text") or "")
            scope_text = scope_text or str(llm_values.get("scope_text") or "")
            limitation_text = limitation_text or str(llm_values.get("limitation_text") or "")
            negative_scope_text = negative_scope_text or str(llm_values.get("negative_scope_text") or "")

        status = _status_for(
            task,
            metric,
            dataset,
            comparator,
            evidence_strength,
            limitation_text,
        )
        normalized_payload = {
            "paper_id": str(context.get("paper_id") or ""),
            "paper_title": str(context.get("paper_title") or ""),
            "note_ids": list(context.get("note_ids") or []),
            "note_titles": list(context.get("note_titles") or []),
            "evidence_summary": _evidence_summary(
                claim,
                {
                    "task": task,
                    "metric": metric,
                    "result_value_text": result_value_text,
                },
                context,
            ),
            "warnings": list(dict.fromkeys(warnings)),
        }
        if persist:
            self.db.upsert_claim_normalization(
                claim_id=claim_id,
                normalization_version=CLAIM_NORMALIZATION_VERSION,
                status=status,
                task=task,
                dataset=dataset,
                metric=metric,
                comparator=comparator,
                result_direction=result_direction,
                result_value_text=result_value_text,
                result_value_numeric=result_value_numeric,
                condition_text=condition_text,
                scope_text=scope_text,
                limitation_text=limitation_text,
                negative_scope_text=negative_scope_text,
                evidence_strength=evidence_strength,
                normalized_payload=normalized_payload,
            )
            stored = self.db.get_claim_normalization(
                claim_id,
                normalization_version=CLAIM_NORMALIZATION_VERSION,
            ) or {}
        else:
            stored = {
                "claim_id": claim_id,
                "normalization_version": CLAIM_NORMALIZATION_VERSION,
                "status": status,
                "task": task,
                "dataset": dataset,
                "metric": metric,
                "comparator": comparator,
                "result_direction": result_direction,
                "result_value_text": result_value_text,
                "result_value_numeric": result_value_numeric,
                "condition_text": condition_text,
                "scope_text": scope_text,
                "limitation_text": limitation_text,
                "negative_scope_text": negative_scope_text,
                "evidence_strength": evidence_strength,
                "normalized_payload": normalized_payload,
                "created_at": "",
                "updated_at": "",
            }
        return claim_normalization_payload(stored)

    def normalize_claims(
        self,
        *,
        claim_ids: list[str] | None = None,
        paper_ids: list[str] | None = None,
        include_all: bool = False,
        limit: int = 100,
        allow_external: bool = False,
        llm_mode: str = "auto",
        persist: bool = True,
    ) -> dict[str, Any]:
        claims = self._candidate_claims(
            claim_ids=claim_ids,
            paper_ids=paper_ids,
            include_all=include_all,
            limit=limit,
        )
        items = [
            self.normalize_claim(
                claim,
                allow_external=allow_external,
                llm_mode=llm_mode,
                persist=persist,
            )
            for claim in claims
        ]
        return {
            "claims": claims,
            "items": items,
        }


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9가-힣]+", _normalize_token(text))
        if len(token) >= 2
    }


def _materially_different(left: str, right: str) -> bool:
    left_token = _clean_text(left)
    right_token = _clean_text(right)
    if not left_token or not right_token:
        return False
    if left_token.casefold() == right_token.casefold():
        return False
    left_numbers = tuple(re.findall(r"\d+(?:\.\d+)?", left_token))
    right_numbers = tuple(re.findall(r"\d+(?:\.\d+)?", right_token))
    if left_numbers or right_numbers:
        return left_numbers != right_numbers or left_token.casefold() != right_token.casefold()
    if left_token.casefold() in right_token.casefold() or right_token.casefold() in left_token.casefold():
        return False
    left_terms = _token_set(left_token)
    right_terms = _token_set(right_token)
    if not left_terms or not right_terms:
        return left_token.casefold() != right_token.casefold()
    overlap = len(left_terms & right_terms) / max(1, len(left_terms | right_terms))
    return overlap < 0.5


def _has_mixed_constraint(values: list[str]) -> bool:
    normalized = [_clean_text(value) for value in values]
    present = [value for value in normalized if value]
    missing_count = len([value for value in normalized if not value])
    if present and missing_count:
        return True
    return any(_materially_different(left, right) for left in present for right in present if left != right)


def _claim_compare_item(claim: dict[str, Any], normalization: dict[str, Any]) -> dict[str, Any]:
    evidence_summary = str(normalization.get("evidenceSummary") or "")
    if not evidence_summary:
        evidence_summary = _limited_text(str(claim.get("claim_text") or ""), limit=180)
    return {
        "claimId": str(claim.get("claim_id") or ""),
        "paperId": str(normalization.get("paperId") or ""),
        "paperTitle": str(normalization.get("paperTitle") or ""),
        "claimText": str(claim.get("claim_text") or ""),
        "predicate": str(claim.get("predicate") or ""),
        "confidence": float(claim.get("confidence") or 0.0),
        "evidenceSummary": evidence_summary,
        "normalizationStatus": str(normalization.get("status") or "failed"),
        "evidenceLinks": evidence_links_from_claim(
            claim,
            paper_id=str(normalization.get("paperId") or ""),
            paper_title=str(normalization.get("paperTitle") or ""),
        ),
        "normalized": normalization,
    }


@dataclass(slots=True)
class ClaimComparisonService:
    db: Any
    config: Any

    def compare(
        self,
        *,
        claim_ids: list[str] | None = None,
        paper_ids: list[str] | None = None,
        task: str = "",
        dataset: str = "",
        metric: str = "",
        limit: int = 200,
    ) -> dict[str, Any]:
        normalizer = ClaimNormalizationService(self.db, self.config)
        selected = normalizer._candidate_claims(claim_ids=claim_ids, paper_ids=paper_ids, limit=limit)
        items: list[tuple[dict[str, Any], dict[str, Any]]] = []
        skipped_claims: list[dict[str, Any]] = []
        for claim in selected:
            claim_id = str(claim.get("claim_id") or "").strip()
            normalization = self.db.get_claim_normalization(
                claim_id,
                normalization_version=CLAIM_NORMALIZATION_VERSION,
            )
            normalized_payload = (
                claim_normalization_payload(normalization)
                if normalization
                else normalizer.normalize_claim(claim, persist=True, allow_external=False, llm_mode="fallback-only")
            )
            if task and str(normalized_payload.get("task") or "") != task:
                continue
            if dataset and str(normalized_payload.get("dataset") or "") != dataset:
                continue
            if metric and str(normalized_payload.get("metric") or "") != metric:
                continue
            if not normalized_payload.get("task") or not normalized_payload.get("metric"):
                skipped_claims.append(
                    {
                        "claimId": claim_id,
                        "reason": "missing_comparison_axes",
                        "normalizationStatus": str(normalized_payload.get("status") or "failed"),
                    }
                )
                continue
            items.append((claim, normalized_payload))

        grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for claim, normalized in items:
            grouped.setdefault(
                (str(normalized.get("task") or ""), str(normalized.get("metric") or "")),
                [],
            ).append((claim, normalized))

        aligned_groups: list[dict[str, Any]] = []
        conflict_candidates: list[dict[str, Any]] = []
        incomparable_groups: list[dict[str, Any]] = []
        for (group_task, group_metric), rows in grouped.items():
            if len(rows) < 2:
                skipped_claims.append(
                    {
                        "claimId": str(rows[0][0].get("claim_id") or ""),
                        "reason": "insufficient_comparison_peers",
                        "normalizationStatus": str(rows[0][1].get("status") or "failed"),
                    }
                )
                continue
            compare_items = [_claim_compare_item(claim, normalized) for claim, normalized in rows]
            datasets = {str(item["normalized"].get("dataset") or "") for item in compare_items if str(item["normalized"].get("dataset") or "")}
            comparators = {str(item["normalized"].get("comparator") or "") for item in compare_items if str(item["normalized"].get("comparator") or "")}
            conditions = [str(item["normalized"].get("conditionText") or "") for item in compare_items]
            scopes = [str(item["normalized"].get("scopeText") or "") for item in compare_items]
            directions = {str(item["normalized"].get("resultDirection") or _UNKNOWN) for item in compare_items}
            if len(datasets) > 1:
                incomparable_groups.append(
                    {
                        "comparisonKey": {"task": group_task, "metric": group_metric},
                        "reason": "dataset_mismatch",
                        "claims": compare_items,
                    }
                )
                continue
            if len(comparators) > 1:
                incomparable_groups.append(
                    {
                        "comparisonKey": {"task": group_task, "metric": group_metric},
                        "reason": "comparator_mismatch",
                        "claims": compare_items,
                    }
                )
                continue
            if _has_mixed_constraint(conditions):
                incomparable_groups.append(
                    {
                        "comparisonKey": {"task": group_task, "metric": group_metric},
                        "reason": "condition_mismatch",
                        "claims": compare_items,
                    }
                )
                continue
            if _has_mixed_constraint(scopes):
                incomparable_groups.append(
                    {
                        "comparisonKey": {"task": group_task, "metric": group_metric},
                        "reason": "scope_mismatch",
                        "claims": compare_items,
                    }
                )
                continue
            if {"better", "worse"} <= directions:
                severity = (
                    "low"
                    if any(str(item["normalized"].get("evidenceStrength") or "") == "weak" for item in compare_items)
                    else "high"
                )
                conflict_candidates.append(
                    {
                        "comparisonKey": {"task": group_task, "metric": group_metric},
                        "severity": severity,
                        "reason": "direction_conflict",
                        "claims": compare_items,
                    }
                )
                continue
            aligned_groups.append(
                {
                    "comparisonKey": {"task": group_task, "metric": group_metric},
                    "reason": "same_direction",
                    "claims": compare_items,
                }
            )

        return {
            "alignedGroups": aligned_groups,
            "conflictCandidates": conflict_candidates,
            "incomparableGroups": incomparable_groups,
            "skippedClaims": skipped_claims,
            "selectedCount": len(selected),
        }
