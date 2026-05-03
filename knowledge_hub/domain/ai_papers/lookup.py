from __future__ import annotations

import re
from typing import Any

from knowledge_hub.learning.resolver import normalize_term

_LOOKUP_NOISE_RE = re.compile(
    r"\b(paper|papers|arxiv|doi|abstract|summary|summarize|citation|citations|explain|explainer)\b|"
    r"논문의?|논문을?|논문|초록|요약해줘|요약해주세요|요약|설명해줘|설명해주세요|설명|정리해줘|정리해주세요|정리",
    re.IGNORECASE,
)
_TRAILING_PARTICLE_TOKENS = {
    "을",
    "를",
    "은",
    "는",
    "이",
    "가",
    "와",
    "과",
    "의",
    "에",
    "도",
    "만",
    "로",
    "으로",
}
_LOOKUP_ANALYSIS_SUFFIX_RE = re.compile(
    r"(?:핵심\s*)?(?:방법(?:론)?|실험\s*결과|결과|성능|지표|평가|한계|기여|실험)(?:을|를|은|는|이|가|의)?$",
    re.IGNORECASE,
)
_LOOKUP_ANALYSIS_SUFFIX_EN_RE = re.compile(
    r"(?:core\s+)?(?:method(?:ology)?|experimental\s+results?|results?|performance|benchmark|evaluation|limitations?|contributions?|findings)\b$",
    re.IGNORECASE,
)
_COMPARE_NOISE_RE = re.compile(
    r"\b(compare|comparison|versus|vs|difference|tradeoff|trade-off)\b|비교해줘|비교해주세요|비교해|비교|차이|장단점",
    re.IGNORECASE,
)
_COMPARE_SPLIT_RE = re.compile(
    r"\s+(?:vs\.?|versus)\s+|(?:와|과|이랑|랑)(?=\s*[A-Za-z0-9가-힣])",
    re.IGNORECASE,
)
_COMPARE_FILLER_RE = re.compile(
    r"\b(?:paper|papers|perspective|view|core|key|difference|differences|strength|strengths|use\s*case|use\s*cases)\b|"
    r"논문\s*관점(?:에서)?|관점에서|핵심\s*차이(?:와)?|각각|비교해서|잘하(?:는)?\s*상황(?:을)?|잘하(?:는)?|상황(?:을)?",
    re.IGNORECASE,
)


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


def _strip_lookup_analysis_suffixes(value: str) -> str:
    candidate = _clean_text(value)
    while candidate:
        previous = candidate
        candidate = _LOOKUP_ANALYSIS_SUFFIX_RE.sub("", candidate).strip()
        candidate = _LOOKUP_ANALYSIS_SUFFIX_EN_RE.sub("", candidate).strip()
        for suffix in sorted(_TRAILING_PARTICLE_TOKENS, key=len, reverse=True):
            if len(candidate) > len(suffix) + 1 and candidate.endswith(suffix):
                candidate = candidate[: -len(suffix)].strip()
                break
        tokens = candidate.split()
        while tokens and tokens[-1] in _TRAILING_PARTICLE_TOKENS:
            tokens.pop()
        candidate = _clean_text(" ".join(tokens))
        if candidate == previous:
            break
    return candidate


def extract_lookup_title_candidate(query: str) -> str:
    body = _clean_text(query)
    if not body:
        return ""
    stripped = _LOOKUP_NOISE_RE.sub(" ", body)
    stripped = re.sub(r"[?!.;]+", " ", stripped)
    tokens = _clean_text(stripped).split()
    while tokens and tokens[-1] in _TRAILING_PARTICLE_TOKENS:
        tokens.pop()
    return _strip_lookup_analysis_suffixes(" ".join(tokens))


def extract_compare_title_candidates(query: str) -> list[str]:
    body = _clean_text(query)
    if not body:
        return []
    parts = _COMPARE_SPLIT_RE.split(body)
    if len(parts) < 2:
        return []
    candidates: list[str] = []
    for part in parts[:3]:
        stripped = _LOOKUP_NOISE_RE.sub(" ", part)
        stripped = _COMPARE_NOISE_RE.sub(" ", stripped)
        stripped = _COMPARE_FILLER_RE.sub(" ", stripped)
        stripped = re.sub(r"[?!.;]+", " ", stripped)
        tokens = _clean_text(stripped).split()
        while tokens and tokens[-1] in _TRAILING_PARTICLE_TOKENS:
            tokens.pop()
        candidate = _strip_lookup_analysis_suffixes(" ".join(tokens))
        if not candidate:
            continue
        if re.search(r"[가-힣]", candidate) and re.search(r"[A-Za-z]", candidate) and not re.fullmatch(
            r"[A-Z][A-Z0-9.+-]{1,12}",
            candidate,
        ):
            continue
        if (
            len(candidate.split()) >= 3
            or looks_like_explicit_paper_title(candidate)
            or re.fullmatch(r"[A-Z][A-Z0-9.+-]{1,12}", candidate)
        ):
            candidates.append(candidate)
    candidates.extend(re.findall(r"(?:[A-Z]{2,}[A-Za-z0-9.+-]*|[A-Z][a-z]?[A-Z][A-Za-z0-9.+-]*)", body))
    return _dedupe_lines(candidates, limit=3)


def looks_like_explicit_paper_title(candidate: str) -> bool:
    token = _clean_text(candidate)
    if not token:
        return False
    if len(token.split()) >= 4:
        return True
    if re.search(r"\d", token):
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9.+-]{1,12}", token):
        return True
    return False


def lookup_match_strength(form: str, row: dict[str, Any]) -> int:
    normalized_form = normalize_term(form)
    if not normalized_form:
        return 0
    paper_id = normalize_term(str(row.get("paper_id") or ""))
    title = normalize_term(str(row.get("title") or ""))
    search_text = normalize_term(str(row.get("search_text") or ""))
    if paper_id and (paper_id == normalized_form or paper_id.startswith(normalized_form)):
        return 5
    if title and (title == normalized_form or title.startswith(normalized_form)):
        return 4
    if title and re.search(rf"(?<![a-z0-9]){re.escape(normalized_form)}(?![a-z0-9])", title):
        return 3
    if search_text and re.search(rf"(?<![a-z0-9]){re.escape(normalized_form)}(?![a-z0-9])", search_text):
        return 2
    return 0


def local_title_lookup_match_strength(form: str, row: dict[str, Any]) -> int:
    normalized_form = normalize_term(form)
    if not normalized_form:
        return 0
    paper_id = normalize_term(str(row.get("arxiv_id") or row.get("paper_id") or ""))
    title = normalize_term(str(row.get("title") or ""))
    if paper_id and paper_id == normalized_form:
        return 7
    if title and title == normalized_form:
        return 6
    if title and title.startswith(normalized_form):
        return 5
    if title and len(normalized_form) >= 16 and normalized_form in title:
        return 4
    return 0


def resolve_lookup_from_local_titles(
    entities: list[str],
    *,
    sqlite_db: Any | None = None,
) -> tuple[list[str], list[str]]:
    if not sqlite_db or not entities:
        return [], []

    search_papers = getattr(sqlite_db, "search_papers", None)
    if not callable(search_papers):
        return [], []

    resolved_ids: list[str] = []
    resolved_titles: list[str] = []
    for form in _dedupe_lines(entities[:3], limit=3):
        best_score = 0
        best_row: dict[str, Any] | None = None
        for row in list(search_papers(form, limit=5) or []):
            if not isinstance(row, dict):
                continue
            score = local_title_lookup_match_strength(form, row)
            if score > best_score:
                best_score = score
                best_row = row
        if best_row is None or best_score < 4:
            continue
        paper_id = _clean_text(best_row.get("arxiv_id") or best_row.get("paper_id"))
        title = _clean_text(best_row.get("title"))
        if paper_id:
            resolved_ids.append(paper_id)
        if title:
            resolved_titles.append(title)
    return _dedupe_lines(resolved_ids, limit=3), _dedupe_lines(resolved_titles, limit=3)


def resolve_lookup(
    entities: list[str],
    *,
    sqlite_db: Any | None = None,
) -> tuple[list[str], list[str]]:
    if not sqlite_db or not entities:
        return [], []

    search_cards = getattr(sqlite_db, "search_paper_cards_v2", None)
    if not callable(search_cards):
        return [], []

    resolved_ids: list[str] = []
    resolved_titles: list[str] = []
    for form in _dedupe_lines(entities[:3], limit=3):
        best_score = 0
        best_row: dict[str, Any] | None = None
        for row in list(search_cards(form, limit=5) or []):
            if not isinstance(row, dict):
                continue
            score = lookup_match_strength(form, row)
            if score > best_score:
                best_score = score
                best_row = row
        if best_row is None or best_score < 3:
            continue
        paper_id = _clean_text(best_row.get("paper_id"))
        title = _clean_text(best_row.get("title"))
        if paper_id:
            resolved_ids.append(paper_id)
        if title:
            resolved_titles.append(title)
    return _dedupe_lines(resolved_ids, limit=3), _dedupe_lines(resolved_titles, limit=3)


__all__ = [
    "extract_compare_title_candidates",
    "extract_lookup_title_candidate",
    "local_title_lookup_match_strength",
    "looks_like_explicit_paper_title",
    "lookup_match_strength",
    "resolve_lookup",
    "resolve_lookup_from_local_titles",
]
