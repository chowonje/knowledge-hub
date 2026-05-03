from __future__ import annotations

import re
from typing import Any


_CURATED_REPRESENTATIVE_ANCHORS: dict[str, dict[str, str]] = {
    "bert": {
        "paperId": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    },
    "cnn": {
        "paperId": "alexnet-2012",
        "title": "ImageNet Classification with Deep Convolutional Neural Networks",
    },
    "dqn": {
        "paperId": "1312.5602",
        "title": "Playing Atari with Deep Reinforcement Learning",
    },
    "rag": {
        "paperId": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    },
    "transformer": {
        "paperId": "1706.03762",
        "title": "Attention Is All You Need",
    },
    "fid": {
        "paperId": "2007.01282",
        "title": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
    },
    "vit": {
        "paperId": "2010.11929",
        "title": "An Image is Worth 16x16 Words",
    },
}
_CURATED_REPRESENTATIVE_TITLES_BY_ID = {
    value["paperId"]: value["title"] for value in _CURATED_REPRESENTATIVE_ANCHORS.values()
}
_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _dedupe_lines(values: list[Any], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = _clean_text(raw)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
        if limit is not None and len(result) >= limit:
            break
    return result


def _compact_alnum(value: Any) -> str:
    return "".join(_TITLE_TOKEN_RE.findall(_clean_text(value).casefold()))


def _title_prefix_match(entity: str, title: str) -> bool:
    token = _compact_alnum(entity)
    if len(token) < 3:
        return False
    title_tokens = _TITLE_TOKEN_RE.findall(_clean_text(title).casefold())
    if not title_tokens:
        return False
    first_token = title_tokens[0]
    if first_token.startswith(token):
        return True
    compact_title = "".join(title_tokens)
    return bool(compact_title) and compact_title.startswith(token)


def _dedupe_hint_rows(hints: list[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in hints:
        paper_id = _clean_text(item.get("paperId"))
        title = _clean_text(item.get("title"))
        key = (paper_id.casefold(), title.casefold())
        if not paper_id and not title:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "entity": _clean_text(item.get("entity")),
                "conceptId": _clean_text(item.get("conceptId")),
                "paperId": paper_id,
                "title": title,
            }
        )
        if limit is not None and len(result) >= limit:
            break
    return result


def local_title_prefix_hints(entities: list[str], *, sqlite_db: Any | None = None) -> list[dict[str, Any]]:
    if not sqlite_db or not entities:
        return []
    search_papers = getattr(sqlite_db, "search_papers", None)
    if not callable(search_papers):
        return []

    hints: list[dict[str, Any]] = []
    for entity in entities[:3]:
        token = _clean_text(entity)
        if not token:
            continue
        try:
            rows = list(search_papers(token, limit=5) or [])
        except Exception:
            continue
        for row in rows[:5]:
            title = _clean_text((row or {}).get("title"))
            if not title or not _title_prefix_match(token, title):
                continue
            paper_id = _clean_text((row or {}).get("arxiv_id") or (row or {}).get("paper_id"))
            hints.append(
                {
                    "entity": token,
                    "conceptId": "",
                    "paperId": paper_id,
                    "title": title,
                }
            )
    return _dedupe_hint_rows(hints, limit=6)


def local_title_prefix_rescue_forms(entities: list[str], *, sqlite_db: Any | None = None) -> list[str]:
    return _dedupe_lines(
        [item.get("title") for item in local_title_prefix_hints(entities, sqlite_db=sqlite_db)],
        limit=6,
    )


def expand_concept_terms(entities: list[str], *, sqlite_db: Any | None = None) -> list[str]:
    expanded: list[str] = list(entities[:4])
    if not sqlite_db or not entities:
        return _dedupe_lines(expanded, limit=6)

    try:
        from knowledge_hub.learning.resolver import EntityResolver

        resolver = EntityResolver(sqlite_db)
        get_concept_papers = getattr(sqlite_db, "get_concept_papers", None)
        for entity in entities[:3]:
            identity = resolver.resolve(entity, entity_type="concept")
            if identity is None:
                continue
            expanded.append(str(identity.display_name or ""))
            for alias in list(identity.aliases or [])[:3]:
                expanded.append(str(alias or ""))
            if callable(get_concept_papers):
                for row in list(get_concept_papers(identity.canonical_id) or [])[:2]:
                    title = _clean_text((row or {}).get("title"))
                    if title:
                        expanded.append(title)
    except Exception:
        pass
    return _dedupe_lines(expanded, limit=6)


def representative_hint(entities: list[str], *, sqlite_db: Any | None = None) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for entity in entities[:3]:
        curated = _CURATED_REPRESENTATIVE_ANCHORS.get(_clean_text(entity).casefold())
        if curated:
            hints.append(
                {
                    "entity": entity,
                    "conceptId": "",
                    "paperId": _clean_text(curated.get("paperId")),
                    "title": _clean_text(curated.get("title")),
                }
            )
    hints.extend(local_title_prefix_hints(entities, sqlite_db=sqlite_db))
    if not sqlite_db or not entities:
        return _dedupe_hint_rows(hints, limit=8)
    try:
        from knowledge_hub.learning.resolver import EntityResolver

        resolver = EntityResolver(sqlite_db)
        get_concept_papers = getattr(sqlite_db, "get_concept_papers", None)
        if not callable(get_concept_papers):
            return hints
        for entity in entities[:3]:
            identity = resolver.resolve(entity, entity_type="concept")
            if identity is None:
                continue
            for row in list(get_concept_papers(identity.canonical_id) or [])[:3]:
                paper_id = _clean_text((row or {}).get("paper_id") or (row or {}).get("arxiv_id"))
                title = _clean_text((row or {}).get("title"))
                if not paper_id and not title:
                    continue
                hints.append(
                    {
                        "entity": entity,
                        "conceptId": _clean_text(identity.canonical_id),
                        "paperId": paper_id,
                        "title": title,
                    }
                )
    except Exception:
        return _dedupe_hint_rows(hints, limit=8)
    return _dedupe_hint_rows(hints, limit=8)


def curated_representative_title(paper_id: str) -> str:
    return _clean_text(_CURATED_REPRESENTATIVE_TITLES_BY_ID.get(_clean_text(paper_id), ""))


__all__ = [
    "curated_representative_title",
    "expand_concept_terms",
    "local_title_prefix_hints",
    "local_title_prefix_rescue_forms",
    "representative_hint",
]
