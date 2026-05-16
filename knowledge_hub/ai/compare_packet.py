from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from typing import Any

from knowledge_hub.ai.answer_contracts import NON_EVIDENCE_SOURCE_SCHEMES, NON_EVIDENCE_SOURCE_TYPES


COMPARE_PACKET_SCHEMA = "knowledge-hub.compare-packet.v1"


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _hash_text(*parts: Any, length: int = 24) -> str:
    text = "\n".join(str(part or "") for part in parts if str(part or "").strip())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: max(8, int(length))]


def _source_scheme(source_id: Any) -> str:
    token = _clean_text(source_id).lower()
    if not token or ":" not in token:
        return ""
    scheme = token.split(":", 1)[0]
    return scheme if re.fullmatch(r"[a-z_][a-z0-9_+.-]*", scheme) else ""


def _is_non_evidence_ref(item: dict[str, Any]) -> bool:
    source_type = _clean_text(item.get("source_type") or item.get("sourceType")).lower()
    if source_type in NON_EVIDENCE_SOURCE_TYPES:
        return True
    scheme = _source_scheme(item.get("source_id") or item.get("sourceId") or item.get("spanRef"))
    return bool(scheme and scheme in NON_EVIDENCE_SOURCE_SCHEMES)


def _content_hash(item: dict[str, Any]) -> str:
    return _clean_text(
        item.get("content_hash")
        or item.get("contentHash")
        or item.get("source_content_hash")
        or item.get("sourceContentHash")
    )


def _span_locator(item: dict[str, Any]) -> str:
    locator = _clean_text(
        item.get("span_locator")
        or item.get("spanLocator")
        or item.get("locator")
        or item.get("parent_chunk_span")
        or item.get("parentChunkSpan")
        or item.get("chunk_span")
        or item.get("chunkSpan")
        or item.get("unit_id")
        or item.get("unitId")
    )
    if locator:
        return locator
    char_start = _clean_text(
        item.get("char_start")
        or item.get("charStart")
        or item.get("start_offset")
        or item.get("startOffset")
        or item.get("chunk_start")
        or item.get("chunkStart")
    )
    char_end = _clean_text(
        item.get("char_end")
        or item.get("charEnd")
        or item.get("end_offset")
        or item.get("endOffset")
        or item.get("chunk_end")
        or item.get("chunkEnd")
    )
    return f"chars:{char_start}-{char_end}" if char_start and char_end else ""


def _span_offset_available(item: dict[str, Any]) -> bool:
    explicit = _bool_or_none(
        item.get("spanOffsetAvailable") if "spanOffsetAvailable" in item else item.get("span_offset_available")
    )
    if explicit is not None:
        return explicit
    locator = _span_locator(item)
    if re.search(r"(?:chars?|bytes?)[:=]\d+\s*[-:]\s*\d+", locator, re.IGNORECASE):
        return True
    if re.search(r"\b\d+\s*[-:]\s*\d+\b", locator):
        return True
    return bool(
        _clean_text(
            item.get("char_start") or item.get("charStart") or item.get("start_offset") or item.get("startOffset")
        )
        and _clean_text(
            item.get("char_end") or item.get("charEnd") or item.get("end_offset") or item.get("endOffset")
        )
    )


def _source_document_id(item: dict[str, Any]) -> str:
    return _clean_text(item.get("document_id") or item.get("documentId") or item.get("parent_id") or item.get("parentId"))


def _chunk_id(item: dict[str, Any]) -> str:
    return _clean_text(item.get("chunk_id") or item.get("chunkId") or item.get("unit_id") or item.get("unitId"))


def _is_stale_span(item: dict[str, Any]) -> bool:
    derivative = dict(item.get("derivative_source") or item.get("derivativeSource") or {})
    return bool(item.get("stale")) or bool(derivative.get("stale"))


def _has_strict_span_provenance(item: dict[str, Any]) -> bool:
    explicit = _bool_or_none(item.get("strictSpanBacked") if "strictSpanBacked" in item else item.get("strict_span_backed"))
    if explicit is False:
        return False
    if _bool_or_none(item.get("fallbackSpan") if "fallbackSpan" in item else item.get("fallback_span")) is True:
        return False
    source_id = _clean_text(item.get("source_id") or item.get("sourceId") or item.get("target"))
    return bool(source_id and _content_hash(item) and _span_offset_available(item) and not _is_stale_span(item))


def _span_ref(item: dict[str, Any], *, fallback_index: int) -> dict[str, Any]:
    source_id_for_strict = _clean_text(item.get("source_id") or item.get("sourceId") or item.get("target"))
    source_id = source_id_for_strict or _clean_text(item.get("spanRef"))
    span_ref = _clean_text(item.get("span_ref") or item.get("spanRef")) or f"span:{fallback_index}"
    content_hash = _content_hash(item)
    span_locator = _span_locator(item)
    span_offset_available = _span_offset_available({**item, "spanLocator": span_locator})
    fallback_span = _bool_or_none(item.get("fallbackSpan") if "fallbackSpan" in item else item.get("fallback_span")) is True
    strict_span_backed = False if fallback_span else _has_strict_span_provenance(
        {**item, "sourceId": source_id_for_strict, "contentHash": content_hash, "spanLocator": span_locator}
    )
    span = {
        "spanRef": span_ref,
        "sourceId": source_id,
        "sourceType": _clean_text(item.get("source_type") or item.get("sourceType")),
        "contentHash": content_hash,
        "sourceContentHash": content_hash,
        "spanLocator": span_locator,
        "contentHashAvailable": bool(content_hash),
        "spanLocatorAvailable": bool(span_locator),
        "spanOffsetAvailable": bool(span_offset_available),
        "strictSpanBacked": bool(strict_span_backed),
        "fallbackSpan": bool(fallback_span),
        "quote": str(item.get("quote") or item.get("text") or item.get("excerpt") or "")[:500],
    }
    citation_label = _clean_text(item.get("citation_label") or item.get("citationLabel") or item.get("label"))
    document_id = _source_document_id(item)
    chunk_id = _chunk_id(item)
    if citation_label:
        span["citationLabel"] = citation_label
    if document_id:
        span["documentId"] = document_id
    if chunk_id:
        span["chunkId"] = chunk_id
    return span


def _source_id(item: dict[str, Any]) -> str:
    return _clean_text(
        item.get("source_id")
        or item.get("sourceId")
        or item.get("citation_target")
        or item.get("target")
        or item.get("source_ref")
        or item.get("sourceRef")
    )


def _source_span_ref(item: dict[str, Any], *, fallback_index: int) -> dict[str, Any] | None:
    source_id = _source_id(item)
    if not source_id:
        return None
    source_type = _clean_text(item.get("source_type") or item.get("sourceType"))
    span_ref = _clean_text(
        item.get("span_ref")
        or item.get("spanRef")
        or item.get("span_locator")
        or item.get("spanLocator")
        or item.get("parent_chunk_span")
        or item.get("parentChunkSpan")
        or item.get("parent_id")
        or item.get("parentId")
    )
    quote = str(item.get("quote") or item.get("text") or item.get("excerpt") or item.get("document") or item.get("title") or "")
    return _span_ref(
        {
            **item,
            "spanRef": span_ref or f"source:{_hash_text(source_id, quote, fallback_index, length=16)}",
            "sourceId": source_id,
            "sourceType": source_type,
            "spanLocator": span_ref,
            "quote": quote[:500],
            "citationLabel": _clean_text(item.get("citation_label") or item.get("citationLabel")),
            "evidenceKind": _clean_text(item.get("evidence_kind") or item.get("evidenceKind")),
            "fallbackSpan": True,
            "strictSpanBacked": False,
        },
        fallback_index=fallback_index,
    )


def _span_key(item: dict[str, Any]) -> tuple[str, str]:
    return (
        _clean_text(item.get("sourceId") or item.get("source_id")),
        _clean_text(item.get("spanRef") or item.get("span_ref")),
    )


def _strict_source_coverage_ready(spans: list[dict[str, Any]]) -> bool:
    normalized: list[dict[str, Any]] = []
    for index, span in enumerate(spans, start=1):
        item = dict(span or {})
        if _is_non_evidence_ref(item):
            continue
        normalized.append(_span_ref(item, fallback_index=index))
    if not normalized:
        return False
    source_ids = {_clean_text(item.get("sourceId")) for item in normalized if _clean_text(item.get("sourceId"))}
    strict_source_ids = {
        _clean_text(item.get("sourceId"))
        for item in normalized
        if bool(item.get("strictSpanBacked")) and _clean_text(item.get("sourceId"))
    }
    fallback_source_ids = {
        _clean_text(item.get("sourceId"))
        for item in normalized
        if bool(item.get("fallbackSpan")) and _clean_text(item.get("sourceId"))
    }
    # Fallback spans remain non-evidence. They may coexist with strict spans as
    # diagnostics, but every represented source must still be covered by a
    # strict span before an existing dimension can recover to supported.
    return (
        len(strict_source_ids) >= 2
        and bool(source_ids)
        and source_ids.issubset(strict_source_ids)
        and fallback_source_ids.issubset(strict_source_ids)
    )


def _strict_supporting_spans(
    spans: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    citation_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    citation_by_ref: dict[str, dict[str, Any]] = {}
    for citation in list(citations or []):
        item = dict(citation or {})
        normalized = _span_ref(
            {**item, "strictSpanBacked": True, "fallbackSpan": False},
            fallback_index=len(citation_by_ref) + 1,
        )
        if not normalized["strictSpanBacked"]:
            continue
        citation_by_key.setdefault(_span_key(normalized), normalized)
        citation_by_ref.setdefault(_clean_text(normalized.get("spanRef")), normalized)

    strict_spans: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_span in enumerate(list(spans or []), start=1):
        span = dict(raw_span or {})
        key = _span_key(span)
        citation = citation_by_key.get(key) or citation_by_ref.get(_clean_text(span.get("spanRef") or span.get("span_id"))) or {}
        normalized = _span_ref({**citation, **span, "strictSpanBacked": True, "fallbackSpan": False}, fallback_index=index)
        if not normalized["strictSpanBacked"] or _is_non_evidence_ref(normalized):
            continue
        normalized_key = _span_key(normalized)
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        strict_spans.append(normalized)

    for index, citation in enumerate(list(citations or []), start=len(strict_spans) + 1):
        normalized = _span_ref({**dict(citation or {}), "strictSpanBacked": True, "fallbackSpan": False}, fallback_index=index)
        normalized_key = _span_key(normalized)
        if not normalized["strictSpanBacked"] or normalized_key in seen or _is_non_evidence_ref(normalized):
            continue
        seen.add(normalized_key)
        strict_spans.append(normalized)
    return strict_spans


def _source_supporting_spans(sources: list[dict[str, Any]], citations: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    citation_by_source: dict[str, dict[str, Any]] = {}
    for citation in list(citations or []):
        item = dict(citation or {})
        source_id = _source_id(item)
        if source_id:
            citation_by_source.setdefault(source_id, item)

    spans: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_source in enumerate(sources, start=1):
        source = dict(raw_source or {})
        source_id = _source_id(source)
        if not source_id:
            continue
        citation = citation_by_source.get(source_id) or {}
        merged = {**citation, **source}
        span = _source_span_ref(merged, fallback_index=index)
        if not span or _is_non_evidence_ref(span):
            continue
        key = (_clean_text(span.get("sourceId")), _clean_text(span.get("spanRef")))
        if key in seen:
            continue
        seen.add(key)
        spans.append(span)
    return spans


_LABEL_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}|[가-힣]{2,}")
_LABEL_STOPWORDS = {
    "compare",
    "comparison",
    "source",
    "sources",
    "paper",
    "papers",
    "explain",
    "difference",
    "differences",
    "논문",
    "기준",
    "비교",
    "차이",
    "설명",
    "처리",
    "하는",
    "방식",
}

_SOURCE_MENTION_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]{1,}|\d{4}\.\d{4,5}", re.IGNORECASE)
_SOURCE_MENTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "baseline",
    "benchmark",
    "benchmarks",
    "current",
    "generation",
    "knowledge",
    "large",
    "learning",
    "memory",
    "model",
    "models",
    "paper",
    "papers",
    "retrieval",
    "retrieval-augmented",
    "survey",
    "system",
    "systems",
    "the",
    "with",
}
_ANSWERABILITY_RISK_RE = re.compile(
    r"(?:latest|current corpus|exact numeric|rank(?:ing)?|최신|현재\s*코퍼스|정확한\s*수치|순위|단정)",
    re.IGNORECASE,
)
_COMPARABLE_ROLE_LABELS = {
    "method": "method",
    "result": "result",
    "metric": "metric",
    "limitation": "limitation",
    "problem": "problem",
    "scope": "scope",
    "paper_summary": "summary",
}
_COMPARABLE_SLOT_LABELS = {
    "problem": "problem",
    "method": "method",
    "result": "result",
    "dataset": "dataset",
    "metric": "metric",
    "limitation": "limitation",
}
_EXPLICIT_SOURCE_ALIASES_BY_ID = {
    "alexnet-2012": {"alexnet", "cnn"},
    "1312.5602": {"dqn", "deep q-network"},
    "1406.2661": {"gan"},
    "1409.3215": {"seq2seq", "sequence to sequence"},
    "1502.03167": {"batchnorm", "batch normalization"},
    "1512.03385": {"resnet"},
    "1706.03762": {"transformer", "attention is all you need"},
    "1707.06347": {"ppo", "proximal policy optimization"},
    "1810.04805": {"bert"},
    "2005.11401": {"rag", "retrieval-augmented generation"},
    "2005.14165": {"gpt"},
    "2006.11239": {"diffusion", "denoising diffusion probabilistic models"},
    "2007.01282": {"fid", "fusion-in-decoder", "fusion in decoder"},
    "2010.11929": {"vit", "vision transformer"},
    "2201.11903": {"cot", "chain-of-thought", "chain of thought"},
    "2310.11511": {"self-rag", "self rag"},
    "2312.00752": {"mamba"},
    "2404.16130": {"graphrag", "graph rag"},
    "2410.05779": {"lightrag", "light rag"},
}
_ALIAS_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_LOW_SIGNAL_EVIDENCE_TEXT_RE = re.compile(
    r"^\s*(?:\\(?:newcommand|renewcommand|DeclareMathOperator|def|usepackage|documentclass)\b|%+)",
    re.IGNORECASE,
)
_BLOCK_PREFIX_RE = re.compile(r"^\s*\[Block\s+\d+\]\s*", re.IGNORECASE)


def _query_mentions_alias(query: str, alias: str) -> bool:
    query_text = _clean_text(query).casefold()
    alias_text = _clean_text(alias).casefold()
    if not query_text or not alias_text:
        return False
    parts = _ALIAS_TOKEN_RE.findall(alias_text)
    if not parts:
        return False
    pattern = r"(?<![A-Za-z0-9])" + r"[\s_.:/+-]+".join(re.escape(part) for part in parts) + r"(?![A-Za-z0-9])"
    return bool(re.search(pattern, query_text, re.IGNORECASE))


def _source_payload_explicitly_named(query: str, *, source_id: str, title: str = "") -> bool:
    source_token = _clean_text(source_id)
    if source_token and source_token.casefold() in _clean_text(query).casefold():
        return True
    if title and _query_mentions_alias(query, title):
        return True
    for alias in _EXPLICIT_SOURCE_ALIASES_BY_ID.get(source_token.casefold(), set()):
        if _query_mentions_alias(query, alias):
            return True
    return False


def _query_focus_label(query: str, sources: list[dict[str, Any]], *, max_tokens: int = 8) -> str:
    text_parts = [str(query or "")]
    text_parts.extend(str(item.get("title") or item.get("parent_label") or item.get("parentLabel") or "") for item in sources)
    tokens: list[str] = []
    seen: set[str] = set()
    for token in _LABEL_TOKEN_RE.findall(" ".join(text_parts)):
        normalized = token.casefold()
        if normalized in _LABEL_STOPWORDS or normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    return " ".join(tokens) if tokens else "evidence comparison"


def _label_is_low_signal(label: str) -> bool:
    value = _clean_text(label)
    lowered = value.casefold()
    return (
        not value
        or value.startswith("||||")
        or lowered.startswith("comparison ")
        or lowered in {"dimension", "evidence comparison"}
    )


def _claim_card_id(item: dict[str, Any]) -> str:
    return _clean_text(item.get("claimCardId") or item.get("claim_card_id") or item.get("claimId") or item.get("claim_id"))


def _group_claim_ids(group: dict[str, Any]) -> list[str]:
    return [_clean_text(value) for value in list(group.get("claimCardIds") or group.get("claim_card_ids") or []) if _clean_text(value)]


def _label_from_frame(frame: dict[str, Any]) -> str:
    for key in ("metric", "dataset", "task", "comparator"):
        value = _clean_text(frame.get(key))
        if value:
            return value
    return ""


def _claim_text(card: dict[str, Any]) -> str:
    parts = [
        _clean_text(card.get("summaryText") or card.get("summary_text")),
        _clean_text(card.get("resultValueText") or card.get("result_value_text")),
        _clean_text(card.get("resultDirection") or card.get("result_direction")),
    ]
    return " | ".join(part for part in parts if part)


def _text_signal_ready(value: Any) -> bool:
    text = _clean_text(value)
    body = _BLOCK_PREFIX_RE.sub("", text)
    if len(body) < 16:
        return False
    return not _LOW_SIGNAL_EVIDENCE_TEXT_RE.search(body)


def _text_has_low_signal_prefix(value: Any) -> bool:
    text = _clean_text(value)
    body = _BLOCK_PREFIX_RE.sub("", text)
    return bool(body and _LOW_SIGNAL_EVIDENCE_TEXT_RE.search(body))


def _claim_supporting_spans(card: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_anchors = [dict(item or {}) for item in list(card.get("evidenceAnchors") or card.get("evidence_anchors") or [])]
    if evidence_anchors:
        spans: list[dict[str, Any]] = []
        for index, anchor in enumerate(evidence_anchors, start=1):
            anchor_id = _clean_text(anchor.get("anchorId") or anchor.get("anchor_id"))
            spans.append(
                {
                    "spanRef": anchor_id or f"{_claim_card_id(card)}:anchor:{index}",
                    "sourceId": _clean_text(anchor.get("sourceId") or anchor.get("source_id") or card.get("sourceId") or card.get("source_id")),
                    "sourceType": _clean_text(anchor.get("sourceType") or anchor.get("source_type") or card.get("sourceKind") or card.get("source_kind") or "paper"),
                    "contentHash": _clean_text(anchor.get("contentHash") or anchor.get("content_hash") or anchor.get("sourceContentHash") or anchor.get("source_content_hash")),
                    "sourceContentHash": _clean_text(anchor.get("sourceContentHash") or anchor.get("source_content_hash") or anchor.get("contentHash") or anchor.get("content_hash")),
                    "spanLocator": _clean_text(anchor.get("spanLocator") or anchor.get("span_locator")),
                    "documentId": _clean_text(anchor.get("documentId") or anchor.get("document_id")),
                    "chunkId": _clean_text(anchor.get("chunkId") or anchor.get("chunk_id")),
                    "citationLabel": _clean_text(anchor.get("citationLabel") or anchor.get("citation_label")),
                    "quote": str(anchor.get("quote") or anchor.get("excerpt") or "")[:500],
                }
            )
        return spans
    anchor_ids = [_clean_text(value) for value in list(card.get("evidenceAnchorIds") or card.get("evidence_anchor_ids") or [])]
    excerpts = [_clean_text(value) for value in list(card.get("anchorExcerpts") or card.get("anchor_excerpts") or [])]
    if not anchor_ids and not excerpts:
        return []
    count = max(len(anchor_ids), len(excerpts), 1)
    spans: list[dict[str, Any]] = []
    for index in range(count):
        spans.append(
            {
                "spanRef": anchor_ids[index] if index < len(anchor_ids) and anchor_ids[index] else f"{_claim_card_id(card)}:anchor:{index + 1}",
                "sourceId": _clean_text(card.get("sourceId") or card.get("source_id")),
                "sourceType": _clean_text(card.get("sourceKind") or card.get("source_kind") or "paper"),
                "quote": excerpts[index] if index < len(excerpts) else _clean_text(card.get("summaryText") or card.get("summary_text")),
            }
        )
    return spans


def _ordered_group_cards(*, group: dict[str, Any], claim_cards_by_id: dict[str, dict[str, Any]], resolved_source_ids: list[str]) -> list[dict[str, Any]]:
    cards = [claim_cards_by_id[claim_id] for claim_id in _group_claim_ids(group) if claim_id in claim_cards_by_id]
    if not cards:
        return []
    order = {source_id: index for index, source_id in enumerate(resolved_source_ids)}
    return sorted(
        cards,
        key=lambda item: (
            order.get(_clean_text(item.get("sourceId") or item.get("source_id")), len(order) + 1),
            _clean_text(item.get("sourceId") or item.get("source_id")),
            _claim_card_id(item),
        ),
    )


def _source_id_from_card(card: dict[str, Any]) -> str:
    return _clean_text(card.get("sourceId") or card.get("source_id"))


def _source_title_from_card(card: dict[str, Any]) -> str:
    summary = _clean_text(card.get("summaryText") or card.get("summary_text"))
    title = summary.split("|", 1)[0].strip() if "|" in summary else summary
    return _clean_text(card.get("title") or card.get("sourceTitle") or card.get("source_title") or title)


def _source_mention_tokens(card: dict[str, Any]) -> set[str]:
    values = [
        _source_id_from_card(card),
        _source_title_from_card(card),
        card.get("documentId") or card.get("document_id"),
    ]
    tokens: set[str] = set()
    for value in values:
        for token in _SOURCE_MENTION_TOKEN_RE.findall(_clean_text(value)):
            lowered = token.casefold()
            if lowered in _SOURCE_MENTION_STOPWORDS:
                continue
            if re.fullmatch(r"\d{4}\.\d{4,5}", token) or len(lowered) >= 4:
                tokens.add(lowered)
    return tokens


def _source_explicitly_named(query: str, cards: list[dict[str, Any]]) -> bool:
    source_id = _source_id_from_card(cards[0]) if cards else ""
    for card in list(cards or []):
        if _source_payload_explicitly_named(
            query,
            source_id=source_id or _source_id_from_card(card),
            title=_clean_text(card.get("title") or card.get("parent_label") or card.get("parentLabel")),
        ):
            return True
    return False


def _card_evidence_role(card: dict[str, Any]) -> str:
    for anchor in list(card.get("evidenceAnchors") or card.get("evidence_anchors") or []):
        role = _clean_text(anchor.get("evidenceRole") or anchor.get("evidence_role")).casefold()
        if role:
            return role
    return _clean_text(card.get("claimType") or card.get("claim_type")).casefold()


def _card_strict_spans(card: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for index, span in enumerate(_claim_supporting_spans(card), start=1):
        normalized = _span_ref({**span, "strictSpanBacked": True, "fallbackSpan": False}, fallback_index=index)
        if normalized["strictSpanBacked"] and not _is_non_evidence_ref(normalized):
            spans.append(normalized)
    return spans


def _claim_card_signal_ready(card: dict[str, Any]) -> bool:
    if not _text_signal_ready(_claim_text(card)):
        return False
    spans = _claim_supporting_spans(card)
    return not any(_text_has_low_signal_prefix(span.get("quote")) for span in spans)


def _source_mention_tokens_from_payload(payload: dict[str, Any]) -> set[str]:
    values = [
        payload.get("paperId") or payload.get("paper_id"),
        payload.get("sourceId") or payload.get("source_id"),
        payload.get("title"),
        payload.get("cardId") or payload.get("card_id"),
    ]
    tokens: set[str] = set()
    for value in values:
        for token in _SOURCE_MENTION_TOKEN_RE.findall(_clean_text(value)):
            lowered = token.casefold()
            if lowered in _SOURCE_MENTION_STOPWORDS:
                continue
            if re.fullmatch(r"\d{4}\.\d{4,5}", token) or len(lowered) >= 4:
                tokens.add(lowered)
    return tokens


def _slot_source_id(payload: dict[str, Any]) -> str:
    return _clean_text(payload.get("paperId") or payload.get("paper_id") or payload.get("sourceId") or payload.get("source_id"))


def _source_titles_note(source_titles: list[str]) -> str:
    titles = [_clean_text(title) for title in source_titles if _clean_text(title)]
    if not titles:
        return ""
    return f"Source titles: {'; '.join(titles[:2])}."


def _slot_source_explicitly_named(query: str, payload: dict[str, Any]) -> bool:
    return _source_payload_explicitly_named(
        query,
        source_id=_slot_source_id(payload),
        title=_clean_text(payload.get("title")),
    )


def _slot_strict_spans(slot: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for index, ref in enumerate(list(slot.get("evidenceRefs") or slot.get("evidence_refs") or []), start=1):
        item = dict(ref or {})
        normalized = _span_ref({**item, "strictSpanBacked": True, "fallbackSpan": False}, fallback_index=index)
        if normalized["strictSpanBacked"] and not _is_non_evidence_ref(normalized):
            spans.append(normalized)
    return spans


def _slot_text(slot: dict[str, Any]) -> str:
    return _clean_text(slot.get("text") or slot.get("summaryText") or slot.get("summary_text"))


def _slot_text_signal_ready(slot: dict[str, Any]) -> bool:
    return _text_signal_ready(_slot_text(slot))


def _synthesized_slot_dimensions(
    *,
    query: str,
    query_frame: dict[str, Any],
    paper_knowledge_slots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if _ANSWERABILITY_RISK_RE.search(_clean_text(query)):
        return []
    resolved_source_ids = [
        _clean_text(value)
        for value in list(query_frame.get("resolved_source_ids") or query_frame.get("resolvedSourceIds") or [])
        if _clean_text(value)
    ]
    if len(resolved_source_ids) < 2:
        return []

    payloads_by_source: dict[str, dict[str, Any]] = {}
    for payload in list(paper_knowledge_slots or []):
        item = dict(payload or {})
        source_id = _slot_source_id(item)
        if source_id in resolved_source_ids and source_id not in payloads_by_source:
            payloads_by_source[source_id] = item
    selected_source_ids = [source_id for source_id in resolved_source_ids[:2] if source_id in payloads_by_source]
    if len(selected_source_ids) < 2:
        return []
    if any(not _slot_source_explicitly_named(query, payloads_by_source[source_id]) for source_id in selected_source_ids):
        return []

    slots_by_type: dict[str, dict[str, dict[str, Any]]] = {}
    for source_id in selected_source_ids:
        for slot in list(payloads_by_source[source_id].get("slots") or []):
            slot_item = dict(slot or {})
            slot_type = _clean_text(slot_item.get("slotType") or slot_item.get("slot_type")).casefold()
            if slot_type not in _COMPARABLE_SLOT_LABELS or not _slot_text_signal_ready(slot_item) or not _slot_strict_spans(slot_item):
                continue
            slots_by_type.setdefault(slot_type, {}).setdefault(source_id, slot_item)

    dimensions: list[dict[str, Any]] = []
    for slot_type, source_slots in slots_by_type.items():
        if any(source_id not in source_slots for source_id in selected_source_ids):
            continue
        ordered_slots = [source_slots[source_id] for source_id in selected_source_ids]
        supporting_spans: list[dict[str, Any]] = []
        for slot in ordered_slots:
            supporting_spans.extend(_slot_strict_spans(slot))
        if not _strict_source_coverage_ready(supporting_spans):
            continue
        label = _COMPARABLE_SLOT_LABELS.get(slot_type, slot_type)
        dimensions.append(
            {
                "dimensionId": f"paper-slot:{slot_type}",
                "label": label,
                "leftClaim": _slot_text(ordered_slots[0]),
                "rightClaim": _slot_text(ordered_slots[1]),
                "comparisonStatus": "supported",
                "supportingSpans": supporting_spans,
                "notes": _clean_text(
                    "Generated from Paper Knowledge Slots with strict evidence anchor coverage. "
                    + _source_titles_note([payloads_by_source[source_id].get("title") for source_id in selected_source_ids])
                ),
            }
        )
    return dimensions


def _merge_synthesized_dimensions(*dimension_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for dimensions in dimension_sets:
        for dimension in dimensions:
            label = _clean_text(dimension.get("label")).casefold()
            dimension_id = _clean_text(dimension.get("dimensionId") or dimension.get("dimension_id")).casefold()
            key = label or dimension_id
            if key and key in seen_labels:
                continue
            if key:
                seen_labels.add(key)
            merged.append(dimension)
    return merged


def _dimension_merge_key(dimension: dict[str, Any]) -> str:
    label = _clean_text(dimension.get("label")).casefold()
    if label:
        return label
    return _clean_text(dimension.get("dimensionId") or dimension.get("dimension_id")).casefold()


def _merge_group_dimensions_with_slot_dimensions(
    group_dimensions: list[dict[str, Any]],
    slot_dimensions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not slot_dimensions:
        return group_dimensions
    slot_by_key = {_dimension_merge_key(item): item for item in slot_dimensions if _dimension_merge_key(item)}
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dimension in group_dimensions:
        key = _dimension_merge_key(dimension)
        status = _clean_text(dimension.get("status") or dimension.get("comparisonStatus") or "unknown").casefold()
        if key and key in slot_by_key and status in {"insufficient", "unknown"}:
            merged.append(slot_by_key[key])
            seen.add(key)
            continue
        merged.append(dimension)
        if key:
            seen.add(key)
    for dimension in slot_dimensions:
        key = _dimension_merge_key(dimension)
        if key and key in seen:
            continue
        merged.append(dimension)
        if key:
            seen.add(key)
    return merged


def _synthesized_claim_dimensions(
    *,
    query: str,
    query_frame: dict[str, Any],
    claim_cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if _ANSWERABILITY_RISK_RE.search(_clean_text(query)):
        return []
    resolved_source_ids = [
        _clean_text(value)
        for value in list(query_frame.get("resolved_source_ids") or query_frame.get("resolvedSourceIds") or [])
        if _clean_text(value)
    ]
    if len(resolved_source_ids) < 2:
        return []

    cards_by_source: dict[str, list[dict[str, Any]]] = {}
    for card in claim_cards:
        source_id = _source_id_from_card(card)
        if source_id in resolved_source_ids:
            cards_by_source.setdefault(source_id, []).append(dict(card or {}))
    selected_source_ids = [source_id for source_id in resolved_source_ids[:2] if cards_by_source.get(source_id)]
    if len(selected_source_ids) < 2:
        return []
    if any(not _source_explicitly_named(query, cards_by_source[source_id]) for source_id in selected_source_ids):
        return []

    cards_by_role: dict[str, dict[str, dict[str, Any]]] = {}
    for source_id in selected_source_ids:
        for card in cards_by_source[source_id]:
            role = _card_evidence_role(card)
            if role not in _COMPARABLE_ROLE_LABELS or not _claim_card_signal_ready(card) or not _card_strict_spans(card):
                continue
            cards_by_role.setdefault(role, {}).setdefault(source_id, card)

    dimensions: list[dict[str, Any]] = []
    for role, source_cards in cards_by_role.items():
        if any(source_id not in source_cards for source_id in selected_source_ids):
            continue
        ordered_cards = [source_cards[source_id] for source_id in selected_source_ids]
        supporting_spans: list[dict[str, Any]] = []
        for card in ordered_cards:
            supporting_spans.extend(_card_strict_spans(card))
        if not _strict_source_coverage_ready(supporting_spans):
            continue
        label = _COMPARABLE_ROLE_LABELS.get(role, role)
        dimensions.append(
            {
                "dimensionId": f"claim-card-role:{role}",
                "label": label,
                "leftClaim": _claim_text(ordered_cards[0]),
                "rightClaim": _claim_text(ordered_cards[1]),
                "comparisonStatus": "supported",
                "supportingSpans": supporting_spans,
                "notes": _clean_text(
                    "Generated from ask-v2 claim-card evidence anchors with strict source span coverage. "
                    + _source_titles_note([_source_title_from_card(card) for card in ordered_cards])
                ),
            }
        )
    return dimensions


def build_compare_packet_from_runtime(
    *,
    query: str,
    source_type: str,
    family: str,
    runtime_execution: dict[str, Any],
    query_frame: dict[str, Any],
    claim_cards: list[dict[str, Any]],
    claim_alignment: dict[str, Any],
    paper_knowledge_slots: list[dict[str, Any]] | None = None,
    evidence_policy: dict[str, Any] | None = None,
    comparison_verification: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if _clean_text(source_type).lower() != "paper":
        return None
    if _clean_text(family).lower() != "paper_compare":
        return None
    if _clean_text(runtime_execution.get("used")).lower() != "ask_v2":
        return None
    cards_by_id = {_claim_card_id(item): dict(item or {}) for item in claim_cards if _claim_card_id(dict(item or {}))}
    slot_payloads = [dict(item or {}) for item in list(paper_knowledge_slots or [])]
    groups = [dict(item or {}) for item in list((claim_alignment or {}).get("groups") or [])]
    if not cards_by_id and not slot_payloads:
        return None

    resolved_source_ids = [_clean_text(value) for value in list(query_frame.get("resolved_source_ids") or query_frame.get("resolvedSourceIds") or [])]
    if not groups:
        dimensions = _merge_synthesized_dimensions(
            _synthesized_slot_dimensions(
                query=query,
                query_frame=query_frame,
                paper_knowledge_slots=slot_payloads,
            ),
            _synthesized_claim_dimensions(
                query=query,
                query_frame=query_frame,
                claim_cards=list(cards_by_id.values()),
            ),
        )
        if not dimensions:
            return None
        return build_compare_packet_contract(
            query=query,
            dimensions=dimensions,
            retrieval_signals=[],
            policy=dict(evidence_policy or {}),
        )

    verification_conflicts = {
        _clean_text(item.get("groupKey"))
        for item in list((comparison_verification or {}).get("conflicts") or [])
        if _clean_text(item.get("groupKey"))
    }
    dimensions: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        group_cards = _ordered_group_cards(group=group, claim_cards_by_id=cards_by_id, resolved_source_ids=resolved_source_ids)
        if not group_cards:
            continue
        frame = dict(group.get("canonicalFrame") or group.get("frame") or {})
        supporting_spans: list[dict[str, Any]] = []
        for card in group_cards:
            supporting_spans.extend(_claim_supporting_spans(card))
        distinct_cards: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for card in group_cards:
            source_id = _clean_text(card.get("sourceId") or card.get("source_id"))
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)
            distinct_cards.append(card)
        group_key = _clean_text(group.get("groupKey")) or f"compare-group:{index}"
        if int(group.get("conflictingClaimCount") or 0) > 0 or group_key in verification_conflicts:
            status = "conflict"
        elif len(distinct_cards) >= 2 and supporting_spans:
            status = "supported"
        else:
            status = "insufficient"
        dimensions.append(
            {
                "dimensionId": group_key,
                "label": _label_from_frame(frame) or f"Comparison {index}",
                "leftClaim": _claim_text(distinct_cards[0]) if distinct_cards else "",
                "rightClaim": _claim_text(distinct_cards[1]) if len(distinct_cards) > 1 else "",
                "comparisonStatus": status,
                "supportingSpans": supporting_spans,
                "notes": _clean_text(group.get("conditionText")),
            }
        )

    dimensions = _merge_group_dimensions_with_slot_dimensions(
        dimensions,
        _synthesized_slot_dimensions(
            query=query,
            query_frame=query_frame,
            paper_knowledge_slots=slot_payloads,
        ),
    )
    if not dimensions:
        return None
    return build_compare_packet_contract(
        query=query,
        dimensions=dimensions,
        retrieval_signals=[],
        policy=dict(evidence_policy or {}),
    )


def build_compare_packet_contract(
    *,
    query: str,
    dimensions: list[dict[str, Any]],
    retrieval_signals: list[dict[str, Any]] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_dimensions: list[dict[str, Any]] = []
    excluded_non_evidence = 0
    unknown_count = 0
    conflict_count = 0
    strict_supported_dimension_count = 0
    fallback_only_dimension_count = 0

    for index, raw_dimension in enumerate(dimensions, start=1):
        dimension = dict(raw_dimension or {})
        supporting_spans: list[dict[str, Any]] = []
        for span_index, raw_span in enumerate(list(dimension.get("supporting_spans") or dimension.get("supportingSpans") or []), start=1):
            span_item = dict(raw_span or {})
            if _is_non_evidence_ref(span_item):
                excluded_non_evidence += 1
                continue
            supporting_spans.append(_span_ref(span_item, fallback_index=span_index))

        status = _clean_text(dimension.get("status") or dimension.get("comparisonStatus") or "unknown").lower()
        if status not in {"supported", "conflict", "unknown", "insufficient"}:
            status = "unknown"
        if status == "conflict":
            conflict_count += 1
        if status in {"unknown", "insufficient"} or not supporting_spans:
            unknown_count += 1
        span_source_ids = {_clean_text(item.get("sourceId")) for item in supporting_spans if _clean_text(item.get("sourceId"))}
        strict_source_ids = {
            _clean_text(item.get("sourceId"))
            for item in supporting_spans
            if bool(item.get("strictSpanBacked")) and _clean_text(item.get("sourceId"))
        }
        strict_span_count = sum(1 for item in supporting_spans if bool(item.get("strictSpanBacked")))
        fallback_span_count = sum(1 for item in supporting_spans if bool(item.get("fallbackSpan")))
        if status in {"supported", "conflict"} and span_source_ids and span_source_ids.issubset(strict_source_ids):
            strict_supported_dimension_count += 1
        elif supporting_spans and fallback_span_count == len(supporting_spans):
            fallback_only_dimension_count += 1

        normalized_dimensions.append(
            {
                "dimensionId": _clean_text(dimension.get("dimension_id") or dimension.get("dimensionId")) or f"dim:{index}",
                "label": _clean_text(dimension.get("label") or dimension.get("name") or f"Dimension {index}"),
                "leftClaim": _clean_text(dimension.get("left_claim") or dimension.get("leftClaim")),
                "rightClaim": _clean_text(dimension.get("right_claim") or dimension.get("rightClaim")),
                "comparisonStatus": status,
                "supportingSpans": supporting_spans,
                "notes": _clean_text(dimension.get("notes")),
            }
        )

    answerable = bool(normalized_dimensions) and strict_supported_dimension_count == len(normalized_dimensions)
    packet_id = _hash_text(query, [item["dimensionId"] for item in normalized_dimensions], [item["comparisonStatus"] for item in normalized_dimensions])
    created_at = datetime.now(timezone.utc).isoformat()
    strict_span_total = sum(
        1
        for dimension in normalized_dimensions
        for span in dimension["supportingSpans"]
        if bool(span.get("strictSpanBacked"))
    )
    fallback_span_total = sum(
        1
        for dimension in normalized_dimensions
        for span in dimension["supportingSpans"]
        if bool(span.get("fallbackSpan"))
    )
    return {
        "schema": COMPARE_PACKET_SCHEMA,
        "packetId": packet_id,
        "packet_id": packet_id,
        "query": str(query or ""),
        "createdAt": created_at,
        "created_at": created_at,
        "dimensions": normalized_dimensions,
        "retrievalSignals": [dict(item or {}) for item in list(retrieval_signals or [])],
        "policy": dict(policy or {}),
        "coverage": {
            "dimensionCount": len(normalized_dimensions),
            "supportedDimensionCount": sum(1 for item in normalized_dimensions if item["comparisonStatus"] == "supported"),
            "conflictDimensionCount": conflict_count,
            "unknownDimensionCount": unknown_count,
            "strictSupportedDimensionCount": strict_supported_dimension_count,
            "fallbackOnlyDimensionCount": fallback_only_dimension_count,
            "supportingSpanCount": sum(len(item["supportingSpans"]) for item in normalized_dimensions),
            "strictSpanBackedCount": strict_span_total,
            "fallbackSpanCount": fallback_span_total,
            "excludedNonEvidenceSpanCount": excluded_non_evidence,
            "answerable": bool(answerable),
            "answerabilityRule": "requires_strict_span_backed_supporting_spans",
        },
    }


def build_compare_packet_from_sources(
    *,
    query: str,
    sources: list[dict[str, Any]],
    citations: list[dict[str, Any]] | None = None,
    strict_spans: list[dict[str, Any]] | None = None,
    strict_citations: list[dict[str, Any]] | None = None,
    existing_packet: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    source_items = [dict(item or {}) for item in list(sources or []) if isinstance(item, dict)]
    strict_source_spans = _strict_supporting_spans(strict_spans, strict_citations)
    source_spans = _source_supporting_spans(source_items, citations=citations)
    if not strict_source_spans and not source_spans:
        if not existing_packet:
            return None
        packet = build_compare_packet_contract(
            query=query,
            dimensions=[dict(item or {}) for item in list(existing_packet.get("dimensions") or []) if isinstance(item, dict)],
            retrieval_signals=list(existing_packet.get("retrievalSignals") or []),
            policy={**dict(existing_packet.get("policy") or {}), **dict(policy or {})},
        )
        existing_packet_id = _clean_text(existing_packet.get("packet_id") or existing_packet.get("packetId"))
        if existing_packet_id:
            packet["packet_id"] = existing_packet_id
            packet["packetId"] = existing_packet_id
        return packet

    focus_label = _query_focus_label(query, source_items)
    if existing_packet:
        dimensions = [dict(item or {}) for item in list(existing_packet.get("dimensions") or []) if isinstance(item, dict)]
        if not dimensions:
            dimensions = [
                {
                    "dimensionId": "retrieved-source-coverage",
                    "label": focus_label,
                    "comparisonStatus": "insufficient",
                    "supportingSpans": [],
                }
            ]
        existing_source_ids = {
            _clean_text(span.get("sourceId") or span.get("source_id"))
            for dimension in dimensions
            for span in list(dimension.get("supportingSpans") or dimension.get("supporting_spans") or [])
            if isinstance(span, dict)
        }
        existing_span_keys = {
            _span_key(span)
            for dimension in dimensions
            for span in list(dimension.get("supportingSpans") or dimension.get("supporting_spans") or [])
            if isinstance(span, dict)
        }
        missing_strict_spans = [
            span
            for span in strict_source_spans
            if _span_key(span) not in existing_span_keys
        ]
        strict_source_ids = {_clean_text(span.get("sourceId")) for span in missing_strict_spans}
        missing_source_spans = [
            span
            for span in source_spans
            if _clean_text(span.get("sourceId")) not in existing_source_ids
            and _clean_text(span.get("sourceId")) not in strict_source_ids
        ]
        if missing_strict_spans or missing_source_spans:
            first_dimension = dimensions[0]
            first_dimension["supportingSpans"] = (
                list(first_dimension.get("supportingSpans") or []) + missing_strict_spans + missing_source_spans
            )
        for dimension in dimensions:
            if _label_is_low_signal(str(dimension.get("label") or "")):
                dimension["label"] = focus_label
            notes = _clean_text(dimension.get("notes"))
            if focus_label and focus_label.casefold() not in notes.casefold():
                dimension["notes"] = _clean_text(f"{notes} Query focus: {focus_label}")
            if _clean_text(dimension.get("comparisonStatus") or dimension.get("status")).lower() == "insufficient":
                spans = [dict(item or {}) for item in list(dimension.get("supportingSpans") or dimension.get("supporting_spans") or [])]
                if _strict_source_coverage_ready(spans):
                    dimension["comparisonStatus"] = "supported"
                    dimension["notes"] = _clean_text(
                        f"{dimension.get('notes') or ''} Insufficient claim alignment recovered by strict retrieved-source spans."
                    )
        packet = build_compare_packet_contract(
            query=query,
            dimensions=dimensions,
            retrieval_signals=list(existing_packet.get("retrievalSignals") or []),
            policy={**dict(existing_packet.get("policy") or {}), **dict(policy or {})},
        )
        existing_packet_id = _clean_text(existing_packet.get("packet_id") or existing_packet.get("packetId"))
        if existing_packet_id:
            packet["packet_id"] = existing_packet_id
            packet["packetId"] = existing_packet_id
        return packet

    left = source_items[0] if source_items else {}
    right = source_items[1] if len(source_items) > 1 else {}
    return build_compare_packet_contract(
        query=query,
        dimensions=[
            {
                "dimensionId": "retrieved-source-coverage",
                "label": focus_label,
                "leftClaim": _clean_text(left.get("title") or left.get("excerpt")),
                "rightClaim": _clean_text(right.get("title") or right.get("excerpt")),
                "comparisonStatus": "insufficient",
                "supportingSpans": strict_source_spans + source_spans,
                "notes": "Source coverage fallback from retrieved evidence; no claim-aligned compare packet was available.",
            }
        ],
        retrieval_signals=[],
        policy=dict(policy or {}),
    )


__all__ = [
    "COMPARE_PACKET_SCHEMA",
    "build_compare_packet_contract",
    "build_compare_packet_from_runtime",
    "build_compare_packet_from_sources",
]
