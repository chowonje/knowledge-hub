from __future__ import annotations

import re
from typing import Any

from knowledge_hub.core.keywords import STOP_ENGLISH, STOP_KOREAN
from knowledge_hub.core.models import SearchResult


GENERIC_CONCEPT_TOKENS = {
    "개념",
    "concept",
    "approach",
    "method",
    "methods",
    "system",
    "systems",
    "overview",
    "framework",
    "theory",
}

GENERIC_VAULT_TOKENS = {
    "개념",
    "기초",
    "기본",
    "노트",
    "데이터",
    "디자인",
    "방법",
    "분석",
    "요약",
    "용어",
    "원칙",
    "자료",
    "정리",
    "참고",
    "web",
    "glossary",
    "note",
    "notes",
    "overview",
    "principles",
    "reference",
    "references",
    "summary",
    "terms",
}

SOURCE_PRIORS_BY_INTENT: dict[str, dict[str, float]] = {
    "definition": {"paper": 0.03, "concept": 0.02, "vault": -0.005, "web": 0.005},
    "comparison": {"paper": 0.03, "concept": 0.01, "vault": -0.005, "web": 0.005},
    "implementation": {"paper": 0.02, "vault": 0.005, "web": 0.01, "concept": -0.005},
    "paper_lookup": {"paper": 0.04, "vault": 0.0, "web": -0.005, "concept": -0.01},
    "paper_topic": {"paper": 0.045, "concept": 0.015, "vault": -0.005, "web": 0.005},
    "evaluation": {"paper": 0.03, "vault": 0.0, "web": 0.01, "concept": 0.0},
    "howto": {"paper": 0.015, "vault": 0.005, "web": 0.015, "concept": 0.0},
    "topic_lookup": {"paper": 0.02, "concept": 0.015, "vault": 0.0, "web": 0.005},
}

_ONTOLOGY_QUERY_TOKEN_LIMIT = 6
_ONTOLOGY_QUERY_TEXT_LIMIT = 72

_META_RESPONSE_PATTERNS = [
    r"i(?:'m| am) unable to access",
    r"i cannot access",
    r"i do not have access",
    r"논문 전문을 직접 열람할 수 없",
    r"외부 문서.*읽을 수 없",
    r"원문을 업로드해",
    r"원문.*필요",
    r"본문.*필요",
    r"현재 대화에는 논문 본문이 포함되어 있지 않",
    r"요청하신 요약을 만들려면",
    r"링크.*알려주시면",
    r"링크/p?df.*필요",
]

_VAULT_HUB_PATTERNS = [
    r"아틀라스",
    r"마인드맵",
    r"백링크",
    r"체크리스트",
    r"\b(?:hub|index|overview|atlas|mindmap|backlink|checklist)\b",
    r"전체\s*정리",
    r"전체\s*구조",
    r"대표\s*개념",
]

_HUB_NAVIGATION_QUERY_PATTERNS = [
    r"아틀라스",
    r"마인드맵",
    r"백링크",
    r"전체\s*정리",
    r"\b(?:hub|index|overview|atlas|mindmap|backlink|navigation|map)\b",
]

_DIRECT_ANSWER_MARKERS_BY_INTENT: dict[str, tuple[str, ...]] = {
    "definition": ("definition", "define", "meaning", "overview", "summary", "설명", "정의", "의미", "개요", "요약", "목적"),
    "comparison": ("compare", "comparison", "versus", "vs", "tradeoff", "difference", "비교", "차이", "장단점", "트레이드오프"),
    "implementation": ("implementation", "implement", "architecture", "pipeline", "service", "orchestrator", "구현", "아키텍처", "파이프라인", "서비스"),
    "evaluation": ("evaluation", "benchmark", "metric", "result", "results", "finding", "experiment", "평가", "벤치마크", "지표", "결과", "실험"),
    "howto": ("how to", "setup", "guide", "guideline", "steps", "usage", "방법", "설정", "가이드", "사용법"),
    "paper_lookup": ("abstract", "summary", "paper", "논문", "초록", "요약"),
    "paper_topic": ("papers", "paper", "논문", "논문들", "survey", "overview", "related", "representative", "찾아", "정리", "묶어", "계열"),
    "topic_lookup": ("overview", "summary", "concept", "설명", "개요", "요약", "개념"),
}

_DIRECT_UNIT_TYPE_BONUS: dict[str, dict[str, float]] = {
    "definition": {"summary": 0.012, "document_summary": 0.012, "background": 0.005},
    "comparison": {"summary": 0.006, "result": 0.012},
    "implementation": {"method": 0.016, "summary": 0.004},
    "evaluation": {"result": 0.016, "summary": 0.006},
    "howto": {"method": 0.014, "summary": 0.004},
    "paper_lookup": {"summary": 0.01, "document_summary": 0.01},
    "paper_topic": {"summary": 0.01, "document_summary": 0.01, "background": 0.006, "result": 0.004},
    "topic_lookup": {"summary": 0.008, "background": 0.004},
}


def normalize_source_type(source_type: str | None) -> str:
    source = str(source_type or "").strip().lower()
    if source in {"", "all", "*"}:
        return ""
    if source == "note":
        return "vault"
    if source in {"repo", "repository", "workspace"}:
        return "project"
    if source == "youtube":
        return "web"
    return source


def classify_query_intent(query: str) -> str:
    text = str(query or "").strip().lower()
    if not text:
        return "definition"
    has_explicit_paper_id = bool(re.search(r"\b(?:arxiv|doi)\b", text) or re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", text))
    paper_terms = {"paper", "papers", "논문", "논문들", "논문을", "논문들에", "논문들을"}
    topic_paper_markers = {
        "find",
        "search",
        "collect",
        "curate",
        "organize",
        "overview",
        "survey",
        "related",
        "representative",
        "alternatives",
        "alternative",
        "papers about",
        "찾아",
        "찾아줘",
        "정리",
        "정리해줘",
        "묶어",
        "모아",
        "추천",
        "관련",
        "대표",
        "대체",
        "차세대",
        "계열",
    }
    if any(token in text for token in {"compare", "comparison", "vs", "versus", "차이", "비교", "장단점", "tradeoff", "trade-off", "pros", "cons"}):
        return "comparison"
    if has_explicit_paper_id:
        return "paper_lookup"
    if any(token in text for token in paper_terms) and any(marker in text for marker in topic_paper_markers):
        return "paper_topic"
    if any(token in text for token in paper_terms | {"citation", "citations"}):
        return "paper_lookup"
    if any(token in text for token in {"implement", "implementation", "code", "patch", "fix", "구현", "코드", "수정"}):
        return "implementation"
    if any(token in text for token in {"evaluate", "evaluation", "benchmark", "metric", "metrics", "score", "scores", "measure", "measurement", "assess", "assessment", "test", "testing", "experiment", "experiments", "실험", "평가", "지표"}):
        return "evaluation"
    if any(token in text for token in {"how to", "how do i", "usage", "use ", "setup", "install", "configure", "tutorial", "guide", "example", "examples", "사용법", "어떻게", "설정", "설치", "방법"}):
        return "howto"
    if any(token in text for token in {"what is", "define", "definition", "meaning", "뜻", "무엇", "무슨", "설명"}):
        return "definition"
    if len(_tokenize(text)) <= 4:
        return "topic_lookup"
    return "definition"


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _query_forms(query: str, query_forms: list[str] | None = None) -> list[str]:
    values = [_clean_text(query)]
    for raw in list(query_forms or []):
        token = _clean_text(raw)
        if not token:
            continue
        if any(existing.lower() == token.lower() for existing in values):
            continue
        values.append(token)
    return [value for value in values if value]


def is_non_substantive_text(text: str) -> bool:
    lowered = _clean_text(text).lower()
    if not lowered:
        return True
    if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in _META_RESPONSE_PATTERNS):
        return True
    if len(lowered) <= 64 and re.search(r"\bcitations?\s*:\s*\d+\b", lowered, re.IGNORECASE):
        return True
    return False


def is_vault_hub_query(query: str) -> bool:
    text = _clean_text(query).lower()
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in _HUB_NAVIGATION_QUERY_PATTERNS)


def is_vault_hub_note(*, title: str = "", file_path: str = "", document: str = "", query: str = "") -> bool:
    if is_vault_hub_query(query):
        return False
    body = " ".join(part for part in (title, file_path, document[:240]) if str(part or "").strip())
    lowered = _clean_text(body).lower()
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in _VAULT_HUB_PATTERNS)


def _tokenize(text: str) -> set[str]:
    body = str(text or "").lower()
    en = [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", body) if len(t) >= 3 and t not in STOP_ENGLISH]
    ko = [t for t in re.findall(r"[가-힣]{2,}", body) if t not in STOP_KOREAN]
    return set(en + ko)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if parsed < 0:
            return 0.0
        if parsed > 1:
            return 1.0
        return parsed
    except Exception:
        return default


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1.0, len(left | right))


def _title_text(metadata: dict[str, Any]) -> str:
    return str(metadata.get("title") or metadata.get("canonical_name") or "").strip()


def _alias_text(metadata: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("aliases", "alias", "entity_aliases", "keywords", "related_concepts"):
        raw = metadata.get(key)
        if isinstance(raw, list):
            values.extend(str(item).strip() for item in raw if str(item).strip())
        elif isinstance(raw, str) and raw.strip():
            values.extend(part.strip() for part in re.split(r"[,;|]", raw) if part.strip())
    canonical = str(metadata.get("canonical_name") or "").strip()
    if canonical:
        values.append(canonical)
    return " ".join(values)


def _ontology_text(metadata: dict[str, Any], sqlite_db: Any | None = None) -> str:
    values: list[str] = []
    entity_id = str(metadata.get("entity_id") or metadata.get("concept_id") or "").strip()
    if entity_id:
        values.append(entity_id)
    for key in ("canonical_name", "entity_type", "title", "aliases", "alias", "entity_aliases", "related_concepts", "keywords"):
        raw = metadata.get(key)
        if isinstance(raw, list):
            values.extend(str(item).strip() for item in raw if str(item).strip())
        elif isinstance(raw, str) and raw.strip():
            values.extend(part.strip() for part in re.split(r"[,;|]", raw) if part.strip())
    if sqlite_db and entity_id:
        try:
            entity = sqlite_db.get_ontology_entity(entity_id) or {}
            canonical_name = str(entity.get("canonical_name") or "").strip()
            if canonical_name:
                values.append(canonical_name)
            entity_type = str(entity.get("entity_type") or "").strip()
            if entity_type:
                values.append(entity_type)
            values.extend(str(alias).strip() for alias in sqlite_db.get_entity_aliases(entity_id) if str(alias).strip())
        except Exception:
            pass
    return " ".join(values)


def _ontology_primary_label(metadata: dict[str, Any]) -> str:
    entity_id = str(metadata.get("entity_id") or metadata.get("concept_id") or "").strip()
    return str(metadata.get("canonical_name") or metadata.get("title") or entity_id or "").strip()


def _metadata_text_values(metadata: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for key in keys:
        raw = metadata.get(key)
        if isinstance(raw, list):
            values.extend(str(item).strip() for item in raw if str(item).strip())
        elif isinstance(raw, str) and raw.strip():
            values.extend(part.strip() for part in re.split(r"[,;|]", raw) if part.strip())
    return values


def _metadata_identity_values(metadata: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "title",
        "canonical_name",
        "file_path",
        "url",
        "parent_id",
        "resolved_parent_id",
        "document_scope_id",
        "section_scope_id",
        "stable_scope_id",
        "record_id",
        "note_id",
        "arxiv_id",
        "entity_id",
        "concept_id",
    ):
        raw = metadata.get(key)
        if isinstance(raw, str) and raw.strip():
            values.add(_clean_text(raw).lower())
    values.update(_clean_text(item).lower() for item in _metadata_text_values(metadata, ("aliases", "alias", "entity_aliases", "keywords", "related_concepts")) if _clean_text(item))
    return {value for value in values if value}


def _metadata_link_values(metadata: dict[str, Any]) -> list[str]:
    return _metadata_text_values(metadata, ("links", "related_notes", "references", "related_concepts"))


def _metadata_mentions_other(base_metadata: dict[str, Any], other_metadata: dict[str, Any]) -> bool:
    other_values = _metadata_identity_values(other_metadata)
    if not other_values:
        return False
    for link in _metadata_link_values(base_metadata):
        token = _clean_text(link).lower()
        if not token:
            continue
        if token in other_values:
            return True
        if any(value in token or token in value for value in other_values):
            return True
    return False


def _result_scope_signature(result: SearchResult) -> dict[str, str]:
    metadata = result.metadata or {}
    source_type = normalize_source_type(metadata.get("source_type"))
    return {
        "source_type": source_type,
        "title": _title_text(metadata),
        "file_path": str(metadata.get("file_path") or "").strip(),
        "parent_id": str(metadata.get("resolved_parent_id") or metadata.get("parent_id") or "").strip(),
        "document_scope_id": str(metadata.get("document_scope_id") or metadata.get("document_id") or "").strip(),
        "section_scope_id": str(metadata.get("section_scope_id") or "").strip(),
        "stable_scope_id": str(metadata.get("stable_scope_id") or "").strip(),
        "cluster_id": str(metadata.get("cluster_id") or "").strip(),
        "url": str(metadata.get("url") or "").strip(),
        "arxiv_id": str(metadata.get("arxiv_id") or "").strip(),
    }


def _related_note_relation(base: SearchResult, candidate: SearchResult) -> tuple[float, list[str]]:
    base_meta = base.metadata or {}
    cand_meta = candidate.metadata or {}
    if base_meta is cand_meta:
        return 0.0, []

    score = 0.0
    reasons: list[str] = []
    base_sig = _result_scope_signature(base)
    cand_sig = _result_scope_signature(candidate)

    if base_sig["document_scope_id"] and base_sig["document_scope_id"] == cand_sig["document_scope_id"]:
        score += 0.42
        reasons.append("same_document_scope")
    elif base_sig["stable_scope_id"] and base_sig["stable_scope_id"] == cand_sig["stable_scope_id"]:
        score += 0.38
        reasons.append("same_stable_scope")

    if base_sig["section_scope_id"] and base_sig["section_scope_id"] == cand_sig["section_scope_id"]:
        score += 0.24
        reasons.append("same_section_scope")

    if base_sig["parent_id"] and base_sig["parent_id"] == cand_sig["parent_id"]:
        score += 0.3
        reasons.append("same_parent")

    if base_sig["cluster_id"] and base_sig["cluster_id"] == cand_sig["cluster_id"]:
        score += 0.22
        reasons.append("same_cluster")

    title_overlap = _jaccard(_tokenize(base_sig["title"]), _tokenize(cand_sig["title"]))
    if title_overlap > 0.0:
        score += min(0.18, 0.3 * title_overlap)
        reasons.append("title_overlap")

    path_overlap = _jaccard(_tokenize(base_sig["file_path"].replace("/", " ")), _tokenize(cand_sig["file_path"].replace("/", " ")))
    if path_overlap > 0.0:
        score += min(0.12, 0.2 * path_overlap)
        reasons.append("path_overlap")

    if _metadata_mentions_other(base_meta, cand_meta) or _metadata_mentions_other(cand_meta, base_meta):
        score += 0.25
        reasons.append("linked_note")

    if base_sig["source_type"] == "vault" and cand_sig["source_type"] == "vault" and not cand_sig["document_scope_id"]:
        score += 0.02
        reasons.append("same_source_type")

    return round(score, 6), reasons


def build_related_note_suggestions(results: list[SearchResult], *, limit: int = 3) -> dict[str, list[dict[str, Any]]]:
    if not results:
        return {}

    representatives: dict[str, SearchResult] = {}
    for item in results:
        key = _parent_identity(item)
        if not key:
            continue
        current = representatives.get(key)
        if current is None or _safe_float(item.score, 0.0) > _safe_float(current.score, 0.0):
            representatives[key] = item

    suggestions: dict[str, list[dict[str, Any]]] = {}
    for base_key, base_result in representatives.items():
        scored: list[dict[str, Any]] = []
        for candidate_key, candidate in representatives.items():
            if candidate_key == base_key:
                continue
            relation_score, reasons = _related_note_relation(base_result, candidate)
            if relation_score <= 0.0:
                continue
            cand_meta = candidate.metadata or {}
            payload = {
                "note_id": candidate_key,
                "title": _title_text(cand_meta) or candidate_key,
                "source_type": normalize_source_type(cand_meta.get("source_type")),
                "score": round(relation_score, 6),
                "reasons": list(reasons),
                "parent_id": str(cand_meta.get("resolved_parent_id") or cand_meta.get("parent_id") or ""),
                "document_scope_id": str(cand_meta.get("document_scope_id") or cand_meta.get("document_id") or ""),
                "section_scope_id": str(cand_meta.get("section_scope_id") or ""),
                "stable_scope_id": str(cand_meta.get("stable_scope_id") or ""),
                "cluster_id": str(cand_meta.get("cluster_id") or ""),
            }
            scored.append(payload)
        scored.sort(key=lambda item: (float(item["score"]), str(item["title"])), reverse=True)
        suggestions[base_key] = scored[: max(1, int(limit or 3))]

    return suggestions


def _query_ontology_tokens(query: str, sqlite_db: Any | None = None) -> set[str]:
    query_tokens = _tokenize(query)
    if not sqlite_db or not query_tokens:
        return query_tokens
    compact_query = _clean_text(query)
    # Ontology rescue is valuable for short concept phrases, but long title-heavy
    # compare forms spend disproportionate time in fuzzy entity resolution.
    if len(query_tokens) > _ONTOLOGY_QUERY_TOKEN_LIMIT or len(compact_query) > _ONTOLOGY_QUERY_TEXT_LIMIT:
        return query_tokens
    try:
        from knowledge_hub.learning.resolver import EntityResolver

        resolver = EntityResolver(sqlite_db)
    except Exception:
        return query_tokens

    resolved_terms: list[str] = []
    for token in sorted(query_tokens):
        identity = resolver.resolve(token)
        if not identity:
            continue
        resolved_terms.append(str(identity.display_name or "").strip())
        resolved_terms.extend(str(alias).strip() for alias in (identity.aliases or []) if str(alias).strip())
        resolved_terms.append(str(identity.canonical_id or "").strip())
    return query_tokens | _tokenize(" ".join(resolved_terms))


def _parent_identity(result: SearchResult) -> str:
    metadata = result.metadata or {}
    for candidate in (
        metadata.get("document_scope_id"),
        metadata.get("resolved_document_scope_id"),
        metadata.get("resolved_parent_id"),
        metadata.get("arxiv_id"),
        metadata.get("url"),
        metadata.get("file_path"),
        metadata.get("parent_id"),
        metadata.get("title"),
        result.document_id,
    ):
        token = str(candidate or "").strip()
        if token:
            return token
    return _clean_text(result.document)[:120]


def _collect_contributions(
    *,
    result: SearchResult,
    query: str,
    intent: str,
    sqlite_db: Any | None = None,
    query_forms: list[str] | None = None,
) -> dict[str, float]:
    metadata = result.metadata or {}
    query_form_values = _query_forms(query, query_forms)
    query_clean = _clean_text(query).lower()
    query_tokens: set[str] = set()
    for form in query_form_values:
        query_tokens |= _tokenize(form)
    query_ontology_tokens: set[str] = set()
    for form in query_form_values:
        query_ontology_tokens |= _query_ontology_tokens(form, sqlite_db)
    title = _title_text(metadata)
    title_clean = _clean_text(title).lower()
    title_tokens = _tokenize(title)
    path = str(metadata.get("file_path") or "").strip()
    path_tokens = _tokenize(path.replace("/", " "))
    alias_tokens = _tokenize(_alias_text(metadata))
    ontology_text = _ontology_text(metadata, sqlite_db)
    ontology_label = _ontology_primary_label(metadata).lower()
    ontology_tokens = _tokenize(ontology_text)
    normalized_source = normalize_source_type(metadata.get("source_type"))
    body_text = _clean_text(getattr(result, "document", "")).lower()
    unit_type = str(metadata.get("unit_type") or "").strip().lower()
    section_path = str(metadata.get("section_path") or "").strip().lower()

    exact_title_match = 0.06 if title_clean and any(form.lower() == title_clean for form in query_form_values) else 0.0
    title_overlap = min(0.035, 0.07 * _jaccard(query_tokens, title_tokens))
    path_overlap = min(0.025, 0.05 * _jaccard(query_tokens, path_tokens))
    alias_overlap = min(0.02, 0.04 * _jaccard(query_tokens, alias_tokens))
    ontology_exact_match = 0.05 if ontology_label and any(form.lower() == ontology_label for form in query_form_values) else 0.0
    ontology_overlap = min(0.04, 0.08 * _jaccard(query_ontology_tokens, ontology_tokens))
    source_prior = SOURCE_PRIORS_BY_INTENT.get(intent, SOURCE_PRIORS_BY_INTENT["definition"]).get(normalized_source, 0.0)
    paper_source_mismatch_penalty = 0.0
    if intent == "paper_lookup" and normalized_source and normalized_source != "paper":
        paper_source_mismatch_penalty = 0.035

    refusal_excerpt_penalty = 0.0
    if body_text and is_non_substantive_text(body_text):
        refusal_excerpt_penalty = 0.18

    citation_only_penalty = 0.0
    compact_body = body_text.replace("\n", " ").strip()
    if compact_body and len(compact_body) <= 64 and re.search(r"\bcitations?\s*:\s*\d+\b", compact_body, re.IGNORECASE):
        citation_only_penalty = 0.07

    generic_concept_penalty = 0.0
    generic_vault_penalty = 0.0
    vault_hub_penalty = 0.0
    direct_answer_bonus = 0.0
    concept_rescue_bonus = 0.0
    if normalized_source == "concept":
        weak_query_fit = (
            exact_title_match == 0.0
            and ontology_exact_match == 0.0
            and title_overlap < 0.01
            and alias_overlap < 0.01
            and ontology_overlap < 0.01
            and path_overlap < 0.01
        )
        generic_title = bool(title_tokens) and title_tokens <= GENERIC_CONCEPT_TOKENS
        if weak_query_fit or generic_title:
            generic_concept_penalty = 0.02
        elif intent in {"paper_topic", "topic_lookup", "evaluation", "howto"}:
            if title_overlap < 0.01 and alias_overlap < 0.01 and ontology_overlap < 0.01:
                generic_concept_penalty = 0.025
    elif normalized_source == "vault":
        weak_query_fit = (
            exact_title_match == 0.0
            and ontology_exact_match == 0.0
            and title_overlap < 0.01
            and alias_overlap < 0.01
            and ontology_overlap < 0.01
            and path_overlap < 0.01
        )
        generic_title = bool(title_tokens) and title_tokens <= GENERIC_VAULT_TOKENS
        short_title = 0 < len(title_tokens) <= 2
        if weak_query_fit:
            generic_vault_penalty = 0.01
            if intent in {"definition", "comparison", "evaluation", "paper_lookup", "paper_topic", "topic_lookup"}:
                generic_vault_penalty += 0.01
            if generic_title or short_title:
                generic_vault_penalty += 0.015
        if intent in {"definition", "comparison", "implementation", "evaluation", "paper_topic", "topic_lookup", "howto"} and is_vault_hub_note(
            title=title,
            file_path=path,
            document=body_text,
            query=query,
        ):
            vault_hub_penalty = 0.085
            if weak_query_fit:
                vault_hub_penalty += 0.025

    marker_text = " ".join(part for part in (title_clean, section_path, body_text[:400]) if part)
    direct_markers = _DIRECT_ANSWER_MARKERS_BY_INTENT.get(intent, ())
    if direct_markers and any(marker in marker_text for marker in direct_markers):
        direct_answer_bonus += 0.012
    if title_overlap >= 0.018 or alias_overlap >= 0.015:
        direct_answer_bonus += 0.006
    if path_overlap >= 0.012 and intent in {"implementation", "howto"}:
        direct_answer_bonus += 0.004
    direct_answer_bonus += _DIRECT_UNIT_TYPE_BONUS.get(intent, {}).get(unit_type, 0.0)
    if normalized_source == "web" and intent in {"definition", "comparison", "evaluation"} and any(
        marker in marker_text
        for marker in ("guide", "guideline", "reference", "watchlist", "가이드", "레퍼런스", "참고")
    ):
        direct_answer_bonus += 0.005
    if normalized_source == "paper" and intent in {"paper_lookup", "paper_topic", "evaluation"} and unit_type in {"summary", "document_summary", "result"}:
        direct_answer_bonus += 0.005
    if normalized_source == "paper" and intent == "definition" and query_forms:
        if exact_title_match > 0.0:
            concept_rescue_bonus = 0.08
        elif title_overlap > 0.0 or ontology_overlap > 0.0:
            concept_rescue_bonus = 0.018

    return {
        "exact_title_match_boost": round(exact_title_match, 6),
        "near_title_overlap_boost": round(title_overlap, 6),
        "path_token_overlap_boost": round(path_overlap, 6),
        "alias_overlap_boost": round(alias_overlap, 6),
        "ontology_entity_exact_match_boost": round(ontology_exact_match, 6),
        "ontology_entity_overlap_boost": round(ontology_overlap, 6),
        "source_prior_boost": round(source_prior, 6),
        "direct_answer_bonus": round(direct_answer_bonus, 6),
        "concept_rescue_bonus": round(concept_rescue_bonus, 6),
        "paper_source_mismatch_penalty": round(paper_source_mismatch_penalty, 6),
        "refusal_excerpt_penalty": round(refusal_excerpt_penalty, 6),
        "citation_only_penalty": round(citation_only_penalty, 6),
        "generic_concept_penalty": round(generic_concept_penalty, 6),
        "generic_vault_penalty": round(generic_vault_penalty, 6),
        "vault_hub_penalty": round(vault_hub_penalty, 6),
    }


def _top_signal_summary(contributions: dict[str, float], limit: int = 5) -> list[dict[str, Any]]:
    allowed = {
        "feature_boost",
        "quality_boost",
        "source_trust_boost",
        "reference_prior_boost",
        "contradiction_penalty",
        "exact_title_match_boost",
        "near_title_overlap_boost",
        "path_token_overlap_boost",
        "alias_overlap_boost",
        "ontology_entity_exact_match_boost",
        "ontology_entity_overlap_boost",
        "source_prior_boost",
        "direct_answer_bonus",
        "concept_rescue_bonus",
        "paper_source_mismatch_penalty",
        "refusal_excerpt_penalty",
        "citation_only_penalty",
        "generic_concept_penalty",
        "generic_vault_penalty",
        "vault_hub_penalty",
        "duplicate_exposure_penalty",
    }
    ranked: list[dict[str, Any]] = []
    for key, value in contributions.items():
        if key not in allowed:
            continue
        try:
            parsed = float(value)
        except Exception:
            continue
        if abs(parsed) <= 0.000001:
            continue
        ranked.append({"name": key, "value": round(parsed, 6)})
    ranked.sort(key=lambda item: abs(float(item["value"])), reverse=True)
    return ranked[:limit]


def _diversify_by_parent(results: list[SearchResult]) -> list[SearchResult]:
    if len(results) <= 2:
        return results

    parent_order: list[str] = []
    grouped: dict[str, list[SearchResult]] = {}
    for item in results:
        parent_key = _parent_identity(item)
        if parent_key not in grouped:
            parent_order.append(parent_key)
            grouped[parent_key] = []
        grouped[parent_key].append(item)

    if len(grouped) == len(results):
        return results

    diversified: list[SearchResult] = []
    depth = 0
    while len(diversified) < len(results):
        progressed = False
        for parent_key in parent_order:
            siblings = grouped[parent_key]
            if depth < len(siblings):
                diversified.append(siblings[depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    return diversified


def apply_query_fit_reranking(
    results: list[SearchResult],
    *,
    query: str,
    sqlite_db: Any | None = None,
    query_forms: list[str] | None = None,
) -> list[SearchResult]:
    if not results:
        return results

    intent = classify_query_intent(query)
    provisional: list[SearchResult] = []
    contributions_by_item_id: dict[int, dict[str, float]] = {}
    sort_scores_by_result: dict[int, float] = {}

    for item in results:
        metadata = dict(item.metadata or {})
        normalized_source = normalize_source_type(metadata.get("source_type"))
        metadata["source_type"] = normalized_source or str(metadata.get("source_type") or "")
        item.metadata = metadata

        contributions = _collect_contributions(
            result=item,
            query=query,
            intent=intent,
            sqlite_db=sqlite_db,
            query_forms=query_forms,
        )
        total_boost = (
            _safe_float(contributions.get("exact_title_match_boost"))
            + _safe_float(contributions.get("near_title_overlap_boost"))
            + _safe_float(contributions.get("path_token_overlap_boost"))
            + _safe_float(contributions.get("alias_overlap_boost"))
            + _safe_float(contributions.get("ontology_entity_exact_match_boost"))
            + _safe_float(contributions.get("ontology_entity_overlap_boost"))
            + _safe_float(contributions.get("source_prior_boost"))
            + _safe_float(contributions.get("direct_answer_bonus"))
            + _safe_float(contributions.get("concept_rescue_bonus"))
            - _safe_float(contributions.get("paper_source_mismatch_penalty"))
            - _safe_float(contributions.get("refusal_excerpt_penalty"))
            - _safe_float(contributions.get("citation_only_penalty"))
            - _safe_float(contributions.get("generic_concept_penalty"))
            - _safe_float(contributions.get("generic_vault_penalty"))
            - _safe_float(contributions.get("vault_hub_penalty"))
        )
        raw_sort_score = _safe_float(item.score, 0.0) + total_boost
        sort_scores_by_result[id(item)] = raw_sort_score
        item.score = max(0.0, min(1.0, raw_sort_score))
        contributions_by_item_id[id(item)] = dict(contributions)
        provisional.append(item)

    provisional.sort(
        key=lambda r: (
            sort_scores_by_result.get(id(r), r.score),
            r.score,
            r.semantic_score,
            r.lexical_score,
        ),
        reverse=True,
    )
    related_notes_by_parent = build_related_note_suggestions(provisional)

    seen_parents: dict[str, int] = {}
    for item in provisional:
        key = _parent_identity(item)
        duplicate_rank = seen_parents.get(key, 0)
        duplicate_penalty = min(0.45, 0.32 * duplicate_rank)
        seen_parents[key] = duplicate_rank + 1

        contributions = dict(contributions_by_item_id.get(id(item)) or {})
        contributions["duplicate_exposure_penalty"] = round(duplicate_penalty, 6)
        raw_sort_score = sort_scores_by_result.get(id(item), _safe_float(item.score, 0.0)) - duplicate_penalty
        sort_scores_by_result[id(item)] = raw_sort_score
        item.score = max(0.0, min(1.0, raw_sort_score))

        extras = dict(item.lexical_extras or {})
        ranking_signals = dict(extras.get("ranking_signals") or {})
        ranking_signals.update(contributions)
        ranking_signals["normalized_source_type"] = normalize_source_type((item.metadata or {}).get("source_type"))
        ranking_signals["query_intent"] = intent
        ranking_signals["ranking_signal_total"] = round(raw_sort_score, 6)
        ranking_signals["duplicate_collapsed"] = duplicate_rank > 0
        ranking_signals["duplicate_parent_rank"] = duplicate_rank
        ranking_signals["top_ranking_signals"] = _top_signal_summary(
            {
                **ranking_signals,
                "feature_boost": _safe_float(extras.get("feature_boost"), 0.0),
                "quality_boost": _safe_float(extras.get("quality_boost"), 0.0),
                "source_trust_boost": _safe_float(extras.get("source_trust_boost"), 0.0),
                "reference_prior_boost": _safe_float(extras.get("reference_prior_boost"), 0.0),
                "contradiction_penalty": -_safe_float(ranking_signals.get("contradiction_penalty"), 0.0),
            }
        )

        extras["normalized_source_type"] = ranking_signals["normalized_source_type"]
        extras["query_intent"] = intent
        extras["duplicate_collapsed"] = ranking_signals["duplicate_collapsed"]
        extras["duplicate_parent_rank"] = duplicate_rank
        extras["top_ranking_signals"] = list(ranking_signals["top_ranking_signals"])
        extras["ranking_signals"] = ranking_signals
        extras["retrieval_adjusted_score"] = round(item.score, 6)
        extras["retrieval_sort_score"] = round(raw_sort_score, 6)
        extras["related_note_suggestions"] = list(related_notes_by_parent.get(key, []))
        extras["related_notes"] = [str(entry.get("title") or entry.get("note_id") or "") for entry in related_notes_by_parent.get(key, []) if str(entry.get("title") or entry.get("note_id") or "").strip()]
        item.lexical_extras = extras

    provisional.sort(
        key=lambda r: (
            sort_scores_by_result.get(id(r), r.score),
            r.score,
            r.semantic_score,
            r.lexical_score,
        ),
        reverse=True,
    )
    return _diversify_by_parent(provisional)


__all__ = [
    "apply_query_fit_reranking",
    "build_related_note_suggestions",
    "classify_query_intent",
    "is_non_substantive_text",
    "is_vault_hub_note",
    "normalize_source_type",
]
