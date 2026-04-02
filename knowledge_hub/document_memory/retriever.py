"""Retrieve document-memory units without changing default search behavior."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from knowledge_hub.document_memory.models import DocumentMemoryUnit
from knowledge_hub.document_memory.payloads import unit_payload


def _query_terms(query: str) -> list[str]:
    return [part.casefold() for part in re.split(r"[^0-9A-Za-z가-힣._/-]+", str(query or "").strip()) if part.strip()]


def _text_overlap_score(terms: list[str], *parts: str) -> float:
    haystack = " ".join(str(part or "") for part in parts).casefold()
    score = 0.0
    for term in terms:
        if term in haystack:
            score += 1.0
    if terms and " ".join(terms) in haystack:
        score += 2.0
    return score


def _provenance_text(item: dict[str, Any]) -> str:
    provenance = dict(item.get("provenance") or {})
    heading_path = provenance.get("heading_path")
    if isinstance(heading_path, (list, tuple)):
        heading_text = " ".join(str(part or "") for part in heading_path if str(part or "").strip())
    else:
        heading_text = str(heading_path or "")
    return " ".join(
        [
            str(provenance.get("file_path") or ""),
            str(provenance.get("source_ref") or ""),
            heading_text,
            " ".join(str(tag or "") for tag in list(item.get("tags") or []) if str(tag or "").strip()),
            " ".join(str(link or "") for link in list(item.get("links") or []) if str(link or "").strip()),
        ]
    )


def _tokenize(text: str) -> list[str]:
    return [part for part in re.split(r"[^0-9A-Za-z가-힣]+", str(text or "").casefold()) if part]


def _normalized_variants(token: str) -> set[str]:
    value = str(token or "").casefold().strip()
    if not value:
        return set()
    variants = {value}
    compact = value.replace("-", "").replace("_", "")
    if compact:
        variants.add(compact)
    if value.endswith("s") and len(value) >= 4:
        variants.add(value[:-1])
    if compact.endswith("s") and len(compact) >= 4:
        variants.add(compact[:-1])
    return {item for item in variants if item}


def _tokens_overlap(query_terms: list[str], haystack_terms: set[str]) -> list[str]:
    haystack_variants: set[str] = set()
    for token in haystack_terms:
        haystack_variants.update(_normalized_variants(token))
    overlap: list[str] = []
    for term in query_terms:
        term_variants = _normalized_variants(term)
        if term_variants & haystack_variants:
            overlap.append(term)
    return overlap


def _title_match_signal(query: str, document_title: str, unit_title: str = "") -> float:
    query_terms = [token for token in _tokenize(query) if len(token) >= 2]
    if not query_terms:
        return 0.0
    title_terms = set(_tokenize(document_title))
    unit_terms = set(_tokenize(unit_title))
    haystack_terms = title_terms | unit_terms
    if not haystack_terms:
        return 0.0
    overlap = _tokens_overlap(query_terms, haystack_terms)
    if not overlap:
        return 0.0
    ratio = len(overlap) / max(1, len(set(query_terms)))
    signal = 2.5 * ratio
    if ratio >= 0.66:
        signal += 1.25
    if len(overlap) >= 2:
        signal += 0.5
    return round(signal, 3)


def _is_temporal_query(query: str) -> bool:
    token = str(query or "")
    return bool(
        re.search(r"\b(latest|recent|updated|newest|changed since|before|after|since)\b", token, re.IGNORECASE)
        or re.search(r"최근|최신|업데이트|이전|이후", token)
    )


def _is_update_or_version_query(query: str) -> bool:
    token = str(query or "")
    return bool(
        re.search(r"\b(update|updated|latest|recent|version|revision|revised|release|newest|changed)\b", token, re.IGNORECASE)
        or re.search(r"업데이트|최신|최근|버전|개정|릴리즈|변경", token)
    )


def _is_explainer_query(query: str) -> bool:
    token = str(query or "")
    return bool(
        re.search(r"\b(explain|purpose|role|difference|summary|overview|what is)\b", token, re.IGNORECASE)
        or re.search(r"설명|목적|역할|차이|요약|개요|무엇", token)
    )


def _parse_iso(value: Any) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        if token.endswith("Z"):
            return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(timezone.utc)
        parsed = datetime.fromisoformat(token)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        pass
    if re.fullmatch(r"\d{4}", token):
        try:
            return datetime(int(token), 1, 1, tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _temporal_signal(query: str, item: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if not _is_temporal_query(query):
        return 0.0, {"enabled": False}
    event_dt = _parse_iso(item.get("event_date") or "")
    document_dt = _parse_iso(item.get("document_date") or "")
    observed_dt = _parse_iso(item.get("observed_at") or item.get("updated_at") or "")
    active_dt = event_dt or document_dt or observed_dt
    field_name = "event_date" if event_dt else "document_date" if document_dt else "observed_at" if observed_dt else ""
    observed_only = bool(observed_dt and not event_dt and not document_dt)
    source_type = str(item.get("source_type") or "").strip().lower()
    if active_dt is None:
        return 0.0, {"enabled": True, "matchedField": "", "matchedValue": ""}
    current_year = datetime.now(timezone.utc).year
    signal = 0.0
    if re.search(r"\b(before|prior to|earlier than)\b|이전", query, re.IGNORECASE):
        year_match = re.search(r"(20\d{2})", query)
        if year_match:
            signal = 1.5 if active_dt.year <= int(year_match.group(1)) else -0.75
    elif re.search(r"\b(after|since)\b|이후", query, re.IGNORECASE):
        year_match = re.search(r"(20\d{2})", query)
        if year_match:
            signal = 1.5 if active_dt.year >= int(year_match.group(1)) else -0.75
        else:
            signal = max(0.0, 1.8 - ((current_year - active_dt.year) * 0.3))
    else:
        signal = max(0.0, 1.8 - ((current_year - active_dt.year) * 0.3))
    if observed_only and source_type == "web":
        signal = signal * 0.35 if signal > 0.0 else signal * 0.5
    return round(signal, 3), {
        "enabled": True,
        "matchedField": field_name,
        "matchedValue": active_dt.isoformat(),
        "weakObservedAtOnly": bool(observed_only and source_type == "web"),
    }


def _updates_relation_signal(query: str, sqlite_db: Any, document_id: str) -> tuple[float, list[str]]:
    if not _is_update_or_version_query(query) and not _is_temporal_query(query):
        return 0.0, []
    if sqlite_db is None or not document_id or not getattr(sqlite_db, "list_memory_relations", None):
        return 0.0, []
    incoming = sqlite_db.list_memory_relations(dst_form="document_memory", dst_id=document_id, relation_type="updates", limit=5)
    outgoing = sqlite_db.list_memory_relations(src_form="document_memory", src_id=document_id, relation_type="updates", limit=5)
    score = (0.35 * len(incoming)) - (0.15 * len(outgoing))
    relation_ids = [
        str(item.get("relation_id") or "").strip()
        for item in [*incoming, *outgoing]
        if str(item.get("relation_id") or "").strip()
    ]
    return round(score, 3), relation_ids


def _is_named_query(query: str) -> bool:
    tokens = _tokenize(query)
    if any(token.isdigit() and len(token) >= 4 for token in tokens):
        return True
    if any(marker in str(query or "") for marker in (":", '"', "'", "“", "”")):
        return True
    latin_tokens = [token for token in tokens if re.search(r"[a-z]", token)]
    return len(latin_tokens) >= 3


def _has_specific_title_intent(query: str, document_title: str, unit_title: str = "") -> bool:
    if not _is_named_query(query):
        return False
    query_terms = [token for token in _tokenize(query) if len(token) >= 2]
    if not query_terms:
        return False
    haystack_terms = set(_tokenize(document_title)) | set(_tokenize(unit_title))
    if not haystack_terms:
        return False
    overlap = _tokens_overlap(query_terms, haystack_terms)
    if not overlap:
        return False
    ratio = len(set(overlap)) / max(1, len(set(query_terms)))
    return len(overlap) >= 2 and ratio >= 0.5


def _web_temporal_text_bonus(query: str, *, source_type: str, document_title: str = "", unit_title: str = "", source_ref: str = "", section_path: str = "") -> float:
    if str(source_type or "").casefold() != "web" or not _is_temporal_query(query):
        return 0.0
    haystack = " ".join([document_title, unit_title, source_ref, section_path]).casefold()
    marker_hits = sum(
        1
        for marker in ("updated", "latest", "recent", "version", "release", "guide", "guideline", "watchlist", "reference", "최신", "최근", "업데이트", "버전", "가이드", "레퍼런스")
        if marker in haystack
    )
    if marker_hits <= 0:
        return 0.0
    return min(1.1, 0.35 * marker_hits)


def _generic_title_penalty(query: str, summary: dict[str, Any]) -> float:
    source_type = str(summary.get("source_type") or "").casefold()
    document_title = str(summary.get("document_title") or summary.get("title") or "").strip()
    if not _has_specific_title_intent(query, document_title, str(summary.get("title") or "")):
        return 0.0
    title_tokens = _tokenize(document_title)
    if source_type == "paper":
        return 0.0
    penalty = 0.0
    if len(title_tokens) <= 2:
        penalty += 1.5
    generic_terms = {"transformer", "retrieval", "memory", "attention", "rag", "diffusion", "agent", "agents"}
    if title_tokens and all(token in generic_terms for token in title_tokens):
        penalty += 1.5
    return penalty


def _source_type_boost(query: str, source_type: str, document_title: str = "", unit_title: str = "") -> float:
    if not _has_specific_title_intent(query, document_title, unit_title):
        return 0.0
    token = str(source_type or "").casefold()
    if token == "paper":
        return 1.75
    if token == "vault":
        return 0.0
    return 0.25


def _looks_web_archive_surface(item: dict[str, Any]) -> bool:
    source_type = str(item.get("source_type") or "").casefold()
    if source_type != "web":
        return False
    provenance = dict(item.get("provenance") or {})
    title = str(item.get("document_title") or item.get("title") or "")
    source_ref = str(item.get("source_ref") or provenance.get("source_ref") or "")
    excerpt = " ".join(
        [
            str(item.get("contextual_summary") or ""),
            str(item.get("source_excerpt") or ""),
            str(item.get("document_thesis") or ""),
            str(item.get("context_header") or ""),
        ]
    ).casefold()
    if "arxiv.org/abs/" in source_ref:
        return True
    if re.match(r"^\[\d{4}\.\d{4,5}\]", title.strip()):
        return True
    archive_markers = (
        "submitted on",
        "view pdf",
        "references & citations",
        "google scholar",
        "semantic scholar",
        "bibtex",
        "computer science >",
        "arxiv:",
        "submission history",
        "full-text links",
    )
    return sum(1 for marker in archive_markers if marker in excerpt) >= 2


def _web_archive_penalty(query: str, item: dict[str, Any]) -> float:
    if not _looks_web_archive_surface(item):
        return 0.0
    if _has_specific_title_intent(
        query,
        str(item.get("document_title") or item.get("title") or ""),
        str(item.get("title") or ""),
    ):
        return 0.0
    if _is_temporal_query(query) or _is_update_or_version_query(query):
        return 1.2
    return 4.8


def _broad_explainer_source_bias(query: str, item: dict[str, Any]) -> float:
    if not _is_explainer_query(query):
        return 0.0
    if _has_specific_title_intent(
        query,
        str(item.get("document_title") or item.get("title") or ""),
        str(item.get("title") or ""),
    ):
        return 0.0
    source_type = str(item.get("source_type") or "").casefold()
    if source_type == "paper" and not _looks_web_archive_surface(item):
        return 0.7
    if source_type == "web" and _looks_web_archive_surface(item):
        return -0.45
    return 0.0


def _type_signal(query: str, unit_type: str) -> float:
    token = str(query or "").casefold()
    unit = str(unit_type or "").casefold()
    preferred: dict[str, tuple[str, ...]] = {
        "method": ("method", "approach", "방법"),
        "result": ("result", "results", "evaluation", "finding", "결과", "평가"),
        "limitation": ("limitation", "limitations", "한계"),
        "summary": ("summary", "abstract", "overview", "요약", "개요"),
        "background": ("background", "introduction", "배경", "소개"),
    }
    for wanted_type, hints in preferred.items():
        if any(hint in token for hint in hints):
            return 2.5 if unit == wanted_type else -0.2
    return 0.6 if unit in {"summary", "section"} else 0.0


def _placeholder_summary_penalty(query: str, summary: dict[str, Any]) -> float:
    token = str(query or "").casefold()
    if any(hint in token for hint in ("metadata", "메타데이터", "status", "상태")):
        return 0.0
    haystack = " ".join(
        [
            str(summary.get("title") or ""),
            str(summary.get("contextual_summary") or ""),
            str(summary.get("source_excerpt") or ""),
            str(summary.get("document_thesis") or ""),
        ]
    ).casefold()
    penalty = 0.0
    if "pending_summary" in haystack or "요약본/번역본이 아직 등록되지 않았습니다" in haystack:
        penalty += 2.25
    if "번역 완료 후" in haystack:
        penalty += 0.75
    if any(
        marker in haystack
        for marker in (
            "요약을 바로 작성할 수 없습니다",
            "원문이 필요합니다",
            "논문 원문",
            "제공된 정보만으로는",
            "cannot summarize",
            "unable to summarize",
            "need the original paper text",
        )
    ):
        penalty += 2.5
    if "메타데이터" == str(summary.get("title") or "").strip():
        penalty += 1.25
    return penalty


def _metadata_unit_penalty(query: str, unit: dict[str, Any]) -> float:
    token = str(query or "").casefold()
    if any(hint in token for hint in ("metadata", "메타데이터", "status", "상태")):
        return 0.0
    title = str(unit.get("title") or "").strip().casefold()
    section_path = str(unit.get("section_path") or "").casefold()
    excerpt = str(unit.get("source_excerpt") or "").casefold()
    penalty = 0.0
    if title in {"메타데이터", "metadata"} or section_path.endswith("메타데이터"):
        penalty += 2.0
    if "pending_summary" in excerpt or "요약본/번역본이 아직 등록되지 않았습니다" in excerpt:
        penalty += 1.5
    return penalty


def _appendix_unit_penalty(query: str, unit: dict[str, Any]) -> float:
    token = str(query or "").casefold()
    if any(hint in token for hint in ("appendix", "supplement", "reference", "references", "bibliography", "부록", "참고문헌")):
        return 0.0
    if str(unit.get("source_type") or "").strip().casefold() != "paper":
        return 0.0
    provenance = dict(unit.get("provenance") or {})
    quality_signals = dict(provenance.get("quality_signals") or {})
    title = str(unit.get("title") or "").strip().casefold()
    section_path = str(unit.get("section_path") or "").casefold()
    excerpt = str(unit.get("source_excerpt") or "").casefold()
    haystack = " ".join([title, section_path, excerpt[:160]])
    appendix_like = bool(quality_signals.get("appendix_like"))
    if not appendix_like:
        appendix_like = any(
            marker in haystack
            for marker in ("appendix", "supplementary", "supplemental", "references", "reference", "bibliography", "부록", "참고문헌")
        )
    if not appendix_like:
        return 0.0
    penalty = 4.5
    if title.startswith(("appendix", "supplement", "references", "bibliography", "부록", "참고문헌")):
        penalty += 1.5
    return penalty


def _path_depth(section_path: str) -> int:
    token = str(section_path or "").strip()
    if not token:
        return 0
    return len([part for part in token.split(" > ") if part.strip()])


def _section_parent_path(section_path: str) -> str:
    parts = [part.strip() for part in str(section_path or "").split(" > ") if part.strip()]
    if len(parts) <= 1:
        return ""
    return " > ".join(parts[:-1])


def _unit_payload_with_signals(value: dict[str, Any], retrieval_signals: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = unit_payload(value)
    if retrieval_signals:
        payload["retrievalSignals"] = dict(retrieval_signals)
    return payload


def _build_matched_segment(
    ordered_units: list[dict[str, Any]],
    *,
    anchor_unit: dict[str, Any],
    scored_by_unit_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not ordered_units or not anchor_unit:
        return {}
    anchor_id = str(anchor_unit.get("unit_id") or "")
    anchor_index = next(
        (index for index, item in enumerate(ordered_units) if str(item.get("unit_id") or "") == anchor_id),
        -1,
    )
    if anchor_index < 0:
        return {}
    anchor_parent = _section_parent_path(str(anchor_unit.get("section_path") or ""))
    segment_units = [anchor_unit]
    for direction in (-1, 1):
        cursor = anchor_index + direction
        while 0 <= cursor < len(ordered_units):
            candidate = ordered_units[cursor]
            candidate_parent = _section_parent_path(str(candidate.get("section_path") or ""))
            if candidate_parent != anchor_parent:
                break
            segment_units.append(candidate)
            break
    segment_units.sort(key=lambda item: (int(item.get("order_index") or 0), str(item.get("unit_id") or "")))
    units_payload = [
        _unit_payload_with_signals(item, scored_by_unit_id.get(str(item.get("unit_id") or ""), {}))
        for item in segment_units
    ]
    segment_text = "\n\n".join(
        str(item.get("source_excerpt") or item.get("contextual_summary") or "").strip()
        for item in segment_units
        if str(item.get("source_excerpt") or item.get("contextual_summary") or "").strip()
    )
    return {
        "anchorUnitId": anchor_id,
        "units": units_payload,
        "segmentText": segment_text,
        "segmentSignals": {
            "strategy": "adjacent_unit_stitch",
            "unitCount": len(segment_units),
            "anchorOrderIndex": int(anchor_unit.get("order_index") or 0),
            "windowStartOrderIndex": int(segment_units[0].get("order_index") or 0),
            "windowEndOrderIndex": int(segment_units[-1].get("order_index") or 0),
        },
    }


class DocumentMemoryRetriever:
    def __init__(self, sqlite_db):
        self.sqlite_db = sqlite_db

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        token = str(document_id or "").strip()
        if not token:
            return None
        summary = self.sqlite_db.get_document_memory_summary(token)
        units = self.sqlite_db.list_document_memory_units(token, limit=200)
        if not summary and not units:
            return None
        document_title = str((summary or {}).get("document_title") or (units[0].get("document_title") if units else "") or token)
        source_type = str((summary or {}).get("source_type") or (units[0].get("source_type") if units else "") or "")
        return {
            "documentId": token,
            "documentTitle": document_title,
            "sourceType": source_type,
            "summary": unit_payload(summary),
            "units": [unit_payload(item) for item in units],
        }

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        query_token = str(query or "").strip()
        raw_summaries = self.sqlite_db.search_document_memory_units(
            query_token,
            limit=max(1, int(limit)) * 2,
            unit_types=["document_summary"],
        )
        if not raw_summaries:
            raw_summaries = self.sqlite_db.search_document_memory_units(
                query_token,
                limit=max(1, int(limit)) * 2,
            )
        terms = _query_terms(query_token)
        summaries: list[dict[str, Any]] = []
        for summary in raw_summaries:
            metadata_text = _provenance_text(summary)
            summary_query_match = _text_overlap_score(
                terms,
                str(summary.get("document_title") or ""),
                str(summary.get("title") or ""),
                str(summary.get("source_ref") or ""),
                str(summary.get("contextual_summary") or ""),
                str(summary.get("context_header") or ""),
                str(summary.get("document_thesis") or ""),
                str(summary.get("search_text") or ""),
                str(summary.get("document_date") or ""),
                str(summary.get("event_date") or ""),
                str(summary.get("observed_at") or ""),
                metadata_text,
            )
            title_match = _title_match_signal(
                query_token,
                str(summary.get("document_title") or summary.get("title") or ""),
                str(summary.get("title") or ""),
            )
            summary_penalty = _placeholder_summary_penalty(query_token, summary)
            generic_penalty = _generic_title_penalty(query_token, summary)
            source_boost = _source_type_boost(
                query_token,
                str(summary.get("source_type") or ""),
                str(summary.get("document_title") or summary.get("title") or ""),
                str(summary.get("title") or ""),
            )
            temporal_score, temporal_meta = _temporal_signal(query_token, summary)
            relation_boost, relation_ids = _updates_relation_signal(query_token, self.sqlite_db, str(summary.get("document_id") or ""))
            archive_penalty = _web_archive_penalty(query_token, summary)
            explainer_bias = _broad_explainer_source_bias(query_token, summary)
            summary["_document_memory_doc_score"] = (
                summary_query_match
                + title_match
                + source_boost
                + explainer_bias
                + temporal_score
                + _web_temporal_text_bonus(
                    query_token,
                    source_type=str(summary.get("source_type") or ""),
                    document_title=str(summary.get("document_title") or ""),
                    unit_title=str(summary.get("title") or ""),
                    source_ref=str((summary.get("provenance") or {}).get("source_ref") or ""),
                    section_path=str((summary.get("provenance") or {}).get("heading_path") or ""),
                )
                + relation_boost
                + (0.45 * _text_overlap_score(terms, metadata_text))
                + (0.5 * float(summary.get("confidence") or 0.0))
                - summary_penalty
                - archive_penalty
                - generic_penalty
            )
            summary["_document_memory_doc_penalty"] = summary_penalty
            summary["_document_memory_title_match"] = title_match
            summary["_document_memory_generic_penalty"] = generic_penalty
            summary["_document_memory_source_boost"] = source_boost
            summary["_document_memory_explainer_bias"] = explainer_bias
            summary["_document_memory_archive_penalty"] = archive_penalty
            summary["_document_memory_temporal_score"] = temporal_score
            summary["_document_memory_temporal_meta"] = temporal_meta
            summary["_document_memory_update_boost"] = relation_boost
            summary["_document_memory_relation_ids"] = relation_ids
            summaries.append(summary)
        summaries.sort(
            key=lambda item: (
                -float(item.get("_document_memory_doc_score") or 0.0),
                -float(item.get("confidence") or 0.0),
                str(item.get("document_id") or ""),
            )
        )
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for summary in summaries:
            document_id = str(summary.get("document_id") or "")
            if not document_id or document_id in seen:
                continue
            seen.add(document_id)
            document_summary = self.sqlite_db.get_document_memory_summary(document_id) or summary
            related = self.sqlite_db.list_document_memory_units(document_id, limit=40)
            related_non_summary = [
                item
                for item in related
                if str(item.get("unit_type") or "") != "document_summary"
            ]
            ordered_units = sorted(
                related_non_summary,
                key=lambda item: (int(item.get("order_index") or 0), str(item.get("unit_id") or "")),
            )
            summary_signals = {
                "summaryQueryMatch": _text_overlap_score(
                    terms,
                    str(document_summary.get("title") or ""),
                    str(document_summary.get("contextual_summary") or ""),
                    str(document_summary.get("context_header") or ""),
                    str(document_summary.get("document_thesis") or ""),
                    _provenance_text(document_summary),
                ),
                "documentThesis": str(document_summary.get("document_thesis") or ""),
                "titleMatch": round(float(summary.get("_document_memory_title_match") or 0.0), 3),
                "namedTitleBias": round(float(summary.get("_document_memory_title_match") or 0.0), 3),
                "sourceTypeBoost": round(float(summary.get("_document_memory_source_boost") or 0.0), 3),
                "explainerSourceBias": round(float(summary.get("_document_memory_explainer_bias") or 0.0), 3),
                "webArchivePenalty": round(float(summary.get("_document_memory_archive_penalty") or 0.0), 3),
                "genericTitlePenalty": round(float(summary.get("_document_memory_generic_penalty") or 0.0), 3),
                "placeholderPenalty": round(float(summary.get("_document_memory_doc_penalty") or 0.0), 3),
                "timeRelevance": round(float(summary.get("_document_memory_temporal_score") or 0.0), 3),
                "updateRelationBoost": round(float(summary.get("_document_memory_update_boost") or 0.0), 3),
                "temporalSignals": dict(summary.get("_document_memory_temporal_meta") or {}),
                "memoryRelationsUsed": list(summary.get("_document_memory_relation_ids") or []),
                "updatesPreferred": bool(float(summary.get("_document_memory_update_boost") or 0.0) > 0.0),
            }
            scored_units: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
            for unit in related_non_summary:
                unit_metadata_text = _provenance_text(unit)
                query_score = _text_overlap_score(
                    terms,
                    str(unit.get("title") or ""),
                    str(unit.get("section_path") or ""),
                    str(unit.get("contextual_summary") or ""),
                    str(unit.get("source_excerpt") or ""),
                    str(unit.get("context_header") or ""),
                    str(unit.get("search_text") or ""),
                    unit_metadata_text,
                )
                summary_score = _text_overlap_score(
                    terms,
                    str(document_summary.get("contextual_summary") or ""),
                    str(document_summary.get("document_thesis") or ""),
                    _provenance_text(document_summary),
                )
                confidence = float(unit.get("confidence") or 0.0)
                depth_penalty = 0.12 * max(0, _path_depth(str(unit.get("section_path") or "")) - 1)
                type_bonus = _type_signal(query_token, str(unit.get("unit_type") or ""))
                metadata_penalty = _metadata_unit_penalty(query_token, unit)
                appendix_penalty = _appendix_unit_penalty(query_token, unit)
                archive_penalty = _web_archive_penalty(query_token, unit)
                explainer_bias = _broad_explainer_source_bias(query_token, unit)
                title_match = _title_match_signal(
                    query_token,
                    str(unit.get("document_title") or ""),
                    str(unit.get("title") or ""),
                )
                source_boost = _source_type_boost(
                    query_token,
                    str(unit.get("source_type") or ""),
                    str(unit.get("document_title") or ""),
                    str(unit.get("title") or ""),
                )
                web_temporal_bonus = _web_temporal_text_bonus(
                    query_token,
                    source_type=str(unit.get("source_type") or ""),
                    document_title=str(unit.get("document_title") or ""),
                    unit_title=str(unit.get("title") or ""),
                    source_ref=str((unit.get("provenance") or {}).get("source_ref") or ""),
                    section_path=str(unit.get("section_path") or ""),
                )
                total = (
                    query_score
                    + title_match
                    + source_boost
                    + explainer_bias
                    + web_temporal_bonus
                    + (0.35 * summary_score)
                    + (0.35 * _text_overlap_score(terms, unit_metadata_text))
                    + (0.75 * confidence)
                    + type_bonus
                    - depth_penalty
                    - metadata_penalty
                    - appendix_penalty
                    - archive_penalty
                )
                signals = {
                    "queryMatch": round(query_score, 3),
                    "titleMatch": round(title_match, 3),
                    "sourceTypeBoost": round(source_boost, 3),
                    "explainerSourceBias": round(explainer_bias, 3),
                    "webTemporalTextBonus": round(web_temporal_bonus, 3),
                    "summaryMatch": round(summary_score, 3),
                    "confidence": round(confidence, 3),
                    "typeSignal": round(type_bonus, 3),
                    "structuralProximity": round(-depth_penalty, 3),
                    "metadataPenalty": round(metadata_penalty, 3),
                    "appendixPenalty": round(appendix_penalty, 3),
                    "webArchivePenalty": round(archive_penalty, 3),
                    "sectionDepth": _path_depth(str(unit.get("section_path") or "")),
                }
                scored_units.append((total, unit, signals))
            scored_units.sort(
                key=lambda item: (
                    -item[0],
                    -float(item[1].get("confidence") or 0.0),
                    int(item[1].get("order_index") or 0),
                )
            )
            matched_unit_payload = unit_payload(summary)
            matched_segment: dict[str, Any] = {}
            if scored_units:
                best_score, best_unit, best_signals = scored_units[0]
                scored_by_unit_id = {
                    str(unit.get("unit_id") or ""): {"score": round(score, 3), **signals}
                    for score, unit, signals in scored_units
                }
                matched_unit_payload = _unit_payload_with_signals(
                    best_unit,
                    {
                        "score": round(best_score, 3),
                        **best_signals,
                    },
                )
                matched_segment = _build_matched_segment(
                    ordered_units,
                    anchor_unit=best_unit,
                    scored_by_unit_id=scored_by_unit_id,
                )
            items.append(
                {
                    "documentId": document_id,
                    "documentTitle": str(summary.get("document_title") or document_id),
                    "sourceType": str(summary.get("source_type") or ""),
                    "matchedUnit": matched_unit_payload,
                    "documentSummary": _unit_payload_with_signals(document_summary, summary_signals),
                    "documentThesis": str(document_summary.get("document_thesis") or ""),
                    "matchedSegment": matched_segment,
                    "retrievalSignals": {
                        "strategy": "summary_first_hierarchical",
                        "summaryQueryMatch": round(summary_signals["summaryQueryMatch"], 3),
                        "titleMatch": round(summary_signals["titleMatch"], 3),
                    "namedTitleBias": round(summary_signals["namedTitleBias"], 3),
                    "sourceTypeBoost": round(summary_signals["sourceTypeBoost"], 3),
                    "explainerSourceBias": round(summary_signals["explainerSourceBias"], 3),
                    "webArchivePenalty": round(summary_signals["webArchivePenalty"], 3),
                    "genericTitlePenalty": round(summary_signals["genericTitlePenalty"], 3),
                    "placeholderPenalty": round(summary_signals["placeholderPenalty"], 3),
                        "timeRelevance": round(summary_signals["timeRelevance"], 3),
                        "updateRelationBoost": round(summary_signals["updateRelationBoost"], 3),
                        "temporalSignals": dict(summary_signals["temporalSignals"]),
                        "memoryRelationsUsed": list(summary_signals["memoryRelationsUsed"]),
                        "updatesPreferred": bool(summary_signals["updatesPreferred"]),
                        "candidateCount": len(related_non_summary),
                    },
                    "relatedUnits": [
                        _unit_payload_with_signals(
                            unit,
                            {
                                "score": round(score, 3),
                                **signals,
                            },
                        )
                        for score, unit, signals in scored_units[:4]
                    ],
                }
            )
            if len(items) >= max(1, int(limit)):
                break
        return items
