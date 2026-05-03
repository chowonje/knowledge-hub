from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _append_unique(values: list[str], candidate: str, *, limit: int | None = None) -> None:
    token = _clean_text(candidate)
    if not token:
        return
    lowered = token.casefold()
    if any(existing.casefold() == lowered for existing in values):
        return
    values.append(token)
    if limit is not None and len(values) > limit:
        del values[limit:]


@dataclass(frozen=True)
class _InternalWebReferenceSpec:
    key: str
    title: str
    file_path: str
    query_patterns: tuple[str, ...]
    anchor_patterns: tuple[str, ...]
    query_terms: tuple[str, ...]
    before: int = 0
    after: int = 1


_INTERNAL_REFERENCE_SPECS: tuple[_InternalWebReferenceSpec, ...] = (
    _InternalWebReferenceSpec(
        key="web_rerank_role",
        title="Vector search quality notes: rerank as a post-retrieval precision layer",
        file_path="docs/PROJECT_STATE.md",
        query_patterns=(r"\brerank(?:er)?\b", r"벡터 검색", r"품질 개선"),
        anchor_patterns=(r"fusion/rerank", r"reranker rollout", r"cross-encoder reranker"),
        query_terms=("rerank", "vector search", "precision layer", "candidate merge"),
        after=2,
    ),
    _InternalWebReferenceSpec(
        key="web_version_grounding",
        title="Web card v2 notes: version grounding needs explicit temporal markers",
        file_path="docs/PROJECT_STATE.md",
        query_patterns=(r"version grounding", r"web card v2"),
        anchor_patterns=(r"version grounding", r"web temporal boosts are weaker when only `observed_at` is available"),
        query_terms=("version grounding", "document_date", "event_date", "observed_at"),
        after=2,
    ),
    _InternalWebReferenceSpec(
        key="web_observed_at_guard",
        title="Temporal guard notes: observed_at-only evidence should not drive latest answers",
        file_path="docs/PROJECT_STATE.md",
        query_patterns=(r"observed_at", r"강한 최신 답변", r"latest"),
        anchor_patterns=(r"weak `observed_at`-only evidence", r"observed_at-only evidence"),
        query_terms=("observed_at", "latest", "temporal grounding", "guard"),
        after=2,
    ),
    _InternalWebReferenceSpec(
        key="web_ontology_routing",
        title="Web ask v2 notes: ontology-first routing helps entity-heavy web queries",
        file_path="docs/PROJECT_STATE.md",
        query_patterns=(r"ontology-first", r"온톨로지", r"routing"),
        anchor_patterns=(r"ontology-routed", r"ontology-first routing"),
        query_terms=("ontology-first routing", "ontology", "routing", "entity-heavy"),
        after=2,
    ),
    _InternalWebReferenceSpec(
        key="web_evidence_anchor_fields",
        title="Web evidence anchors carry document_date, event_date, observed_at, and updated_at markers",
        file_path="knowledge_hub/ai/ask_v2_verification.py",
        query_patterns=(r"evidence anchor", r"\banchor\b", r"필드"),
        anchor_patterns=(r"document_date", r"event_date", r"observed_at", r"temporal_version_grounding"),
        query_terms=("evidence anchor", "document_date", "event_date", "observed_at", "updated_at"),
        after=3,
    ),
)


def internal_reference_keys(query: str) -> list[str]:
    text = _clean_text(query).casefold()
    matched: list[str] = []
    for spec in _INTERNAL_REFERENCE_SPECS:
        if all(re.search(pattern, text, re.IGNORECASE) for pattern in spec.query_patterns):
            _append_unique(matched, spec.key)
            continue
        hits = sum(1 for pattern in spec.query_patterns if re.search(pattern, text, re.IGNORECASE))
        if hits >= 2:
            _append_unique(matched, spec.key)
    return matched


def internal_reference_requested(query: str) -> bool:
    return bool(internal_reference_keys(query))


def _file_snippet(*, file_path: Path, patterns: tuple[str, ...], before: int, after: int) -> tuple[int, str]:
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return 1, ""
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for index, line in enumerate(lines):
            if not regex.search(line):
                continue
            start = max(0, index - before)
            end = min(len(lines), index + after + 1)
            snippet = " ".join(_clean_text(item) for item in lines[start:end] if _clean_text(item))
            return index + 1, snippet
    fallback = " ".join(_clean_text(item) for item in lines[: max(1, after + 1)] if _clean_text(item))
    return 1, fallback


def build_internal_reference_cards(query: str, *, limit: int = 3) -> list[dict[str, Any]]:
    keys = set(internal_reference_keys(query))
    if not keys:
        return []
    query_text = _clean_text(query)
    cards: list[dict[str, Any]] = []
    for spec in _INTERNAL_REFERENCE_SPECS:
        if spec.key not in keys:
            continue
        file_path = _REPO_ROOT / spec.file_path
        line_number, snippet = _file_snippet(
            file_path=file_path,
            patterns=spec.anchor_patterns,
            before=spec.before,
            after=spec.after,
        )
        if not snippet:
            continue
        try:
            updated_at = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc).isoformat()
        except Exception:
            updated_at = ""
        source_url = f"internal://{spec.file_path}#L{line_number}"
        card_id = f"web-internal-ref:{spec.key}"
        document_id = f"web_internal_ref:{spec.key}"
        anchor_excerpt = snippet
        card = {
            "card_id": card_id,
            "document_id": document_id,
            "canonical_url": source_url,
            "title": spec.title,
            "page_core": snippet,
            "topic_core": snippet,
            "result_core": "",
            "limitations_core": "Internal project reference rather than an external web page.",
            "version_core": updated_at or "internal reference",
            "when_not_to_use": "Use this only for project-specific web-runtime questions.",
            "search_text": _clean_text(
                " ".join(
                    [
                        query_text,
                        spec.title,
                        snippet,
                        *spec.query_terms,
                        spec.file_path,
                    ]
                )
            ),
            "quality_flag": "ok",
            "document_date": updated_at,
            "event_date": "",
            "observed_at": updated_at,
            "source_url": source_url,
            "version": "web-internal-reference-v1",
            "updated_at": updated_at,
            "diagnostics": {
                "internalReference": True,
                "internalReferenceKey": spec.key,
                "filePath": spec.file_path,
                "line": line_number,
            },
            "anchors": [
                {
                    "anchor_id": f"{card_id}:anchor",
                    "card_id": card_id,
                    "excerpt": anchor_excerpt,
                    "score": 0.99,
                    "evidence_role": "supporting",
                    "document_id": document_id,
                    "section_path": f"{spec.file_path}:L{line_number}",
                    "source_url": source_url,
                    "document_date": updated_at,
                    "event_date": "",
                    "observed_at": updated_at,
                    "updated_at_marker": updated_at,
                    "title": spec.title,
                }
            ],
        }
        cards.append(card)
        if len(cards) >= limit:
            break
    return cards


__all__ = [
    "build_internal_reference_cards",
    "internal_reference_keys",
    "internal_reference_requested",
]
