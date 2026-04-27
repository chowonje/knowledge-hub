from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from typing import Any


EVIDENCE_PACKET_SCHEMA = "knowledge-hub.evidence-packet.v1"
ANSWER_CONTRACT_SCHEMA = "knowledge-hub.answer-contract.v1"
VERIFICATION_VERDICT_SCHEMA = "knowledge-hub.verification-verdict.v1"
NON_EVIDENCE_SOURCE_SCHEMES = {
    "belief",
    "decision",
    "outcome",
    "ontology",
    "learning_node",
    "learning_edge",
    "learning_path",
    "learning_resource",
    "memory_relation",
    "entity_merge",
    "entity_split",
}
NON_EVIDENCE_SOURCE_TYPES = {
    "belief",
    "decision",
    "outcome",
    "ontology_claim",
    "ontology_relation",
    "kg_relation",
    "learning_graph",
    "learning_node",
    "learning_edge",
    "learning_path",
    "learning_resource",
    "memory_relation",
    "entity_merge",
    "entity_split",
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _hash_text(*parts: Any, length: int = 16) -> str:
    text = "\n".join(str(part or "") for part in parts if str(part or "").strip())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: max(8, int(length))]


def _normalize_classification(value: Any) -> str:
    token = _clean_text(value).upper()
    return token if token in {"P0", "P1", "P2", "P3"} else "UNKNOWN"


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _source_scheme(value: Any) -> str:
    token = _clean_text(value).lower()
    if not token or ":" not in token:
        return ""
    scheme = token.split(":", 1)[0]
    return scheme if re.fullmatch(r"[a-z_][a-z0-9_+.-]*", scheme) else ""


def parse_span_offsets(*values: Any) -> tuple[int | None, int | None]:
    for value in values:
        token = str(value or "").strip()
        if not token:
            continue
        match = re.search(r"(?:chars?|bytes?)[:=](\d+)\s*[-:]\s*(\d+)", token, re.IGNORECASE)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            if end >= start:
                return start, end
        match = re.search(r"\b(\d+)\s*[-:]\s*(\d+)\b", token)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            if end >= start:
                return start, end
    return None, None


def _evidence_span(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    span_locator = _clean_text(
        item.get("span_locator")
        or item.get("spanLocator")
        or item.get("parent_chunk_span")
        or item.get("chunk_span")
        or item.get("unit_id")
    )
    char_start = _int_or_none(
        _first_present(
            item.get("char_start"),
            item.get("charStart"),
            item.get("chunk_start"),
            item.get("chunkStart"),
            item.get("start_offset"),
            item.get("startOffset"),
        )
    )
    char_end = _int_or_none(
        _first_present(
            item.get("char_end"),
            item.get("charEnd"),
            item.get("chunk_end"),
            item.get("chunkEnd"),
            item.get("end_offset"),
            item.get("endOffset"),
        )
    )
    if char_start is None or char_end is None:
        parsed_start, parsed_end = parse_span_offsets(span_locator)
        char_start = parsed_start if char_start is None else char_start
        char_end = parsed_end if char_end is None else char_end
    quote = str(item.get("excerpt") or item.get("text") or "")[:1000]
    text = quote if quote.strip() else _clean_text(item.get("title") or f"span {index}")
    source_id = _clean_text(item.get("source_id") or item.get("sourceId") or item.get("citation_target") or item.get("title"))
    if not source_id:
        source_id = f"source:{index}"
    source_hash = _clean_text(item.get("source_content_hash") or item.get("sourceContentHash"))
    snippet_hash = _clean_text(item.get("snippet_hash") or item.get("snippetHash")) or _hash_text(text, length=32)
    content_hash = source_hash or None
    classification = _normalize_classification(item.get("policy_class") or item.get("classification"))
    allow_external = item.get("allow_external") if "allow_external" in item else item.get("allowExternal")
    policy_allowed = item.get("policy_allowed") if "policy_allowed" in item else item.get("policyAllowed")
    explicit_allowed = _bool_or_none(policy_allowed)
    explicit_external = _bool_or_none(allow_external)
    allowed = explicit_allowed if explicit_allowed is not None else classification != "P0"
    external_allowed = explicit_external if explicit_external is not None else classification not in {"P0", "UNKNOWN"}
    retrieval_scores = {
        "score": float(item.get("score") or 0.0),
        "semantic": float(item.get("semantic_score") or item.get("semanticScore") or 0.0),
        "lexical": float(item.get("lexical_score") or item.get("lexicalScore") or 0.0),
    }
    return {
        "spanRef": f"span:{index}",
        "span_id": f"span:{index}",
        "citationLabel": _clean_text(item.get("citation_label") or f"S{index}"),
        "citation_label": _clean_text(item.get("citation_label") or f"S{index}"),
        "sourceId": source_id,
        "source_id": source_id,
        "sourceScheme": _source_scheme(source_id),
        "source_scheme": _source_scheme(source_id),
        "source_type": _clean_text(item.get("source_type") or item.get("sourceType")),
        "sourceRef": _clean_text(item.get("source_ref") or item.get("sourceRef") or source_id),
        "sourceContentHash": source_hash,
        "source_content_hash": source_hash,
        "content_hash": content_hash,
        "contentHashAvailable": bool(source_hash),
        "charStart": char_start,
        "char_start": char_start,
        "charEnd": char_end,
        "char_end": char_end,
        "spanOffsetAvailable": char_start is not None and char_end is not None,
        "spanLocator": span_locator,
        "locator": span_locator,
        "text": text,
        "retrievalScores": retrieval_scores,
        "retrieval_scores": retrieval_scores,
        "policy": {
            "classification": classification,
            "allowed": bool(allowed),
            "allowExternal": bool(external_allowed),
            "external_allowed": bool(external_allowed),
        },
        "evidenceKind": _clean_text(item.get("evidence_kind") or item.get("evidenceKind")),
        "derivativeSource": dict(item.get("derivative_source") or item.get("derivativeSource") or {}),
        "snippetHash": snippet_hash,
    }


def _non_evidence_reason(item: dict[str, Any]) -> str:
    source_type = _clean_text(item.get("source_type") or item.get("sourceType")).lower()
    if source_type in NON_EVIDENCE_SOURCE_TYPES:
        return f"non_evidence_source_type:{source_type}"
    source_scheme = _source_scheme(
        item.get("source_id")
        or item.get("sourceId")
        or item.get("source_ref")
        or item.get("sourceRef")
    )
    if source_scheme in NON_EVIDENCE_SOURCE_SCHEMES:
        return f"non_evidence_source_scheme:{source_scheme}"
    return ""


def _retrieval_signal_entry(item: dict[str, Any], span: dict[str, Any], *, reason: str) -> dict[str, Any]:
    source_id = _clean_text(span.get("source_id") or item.get("source_id") or item.get("sourceId"))
    source_type = _clean_text(span.get("source_type") or item.get("source_type") or item.get("sourceType"))
    source_scheme = _source_scheme(source_id)
    evidence_kind = _clean_text(span.get("evidenceKind") or item.get("evidence_kind") or item.get("evidenceKind"))
    signal_id = _hash_text(reason, source_id, source_type, evidence_kind, length=24)
    return {
        "signalId": signal_id,
        "signal_id": signal_id,
        "sourceId": source_id,
        "source_id": source_id,
        "sourceType": source_type,
        "source_type": source_type,
        "sourceScheme": source_scheme,
        "source_scheme": source_scheme,
        "reason": reason,
        "evidenceKind": evidence_kind,
        "derivativeSource": dict(span.get("derivativeSource") or item.get("derivative_source") or item.get("derivativeSource") or {}),
        "citationLabel": _clean_text(span.get("citation_label") or item.get("citation_label") or item.get("citationLabel")),
    }


def _append_unique_signal(bucket: list[dict[str, Any]], signal: dict[str, Any]) -> None:
    key = (
        _clean_text(signal.get("source_id")),
        _clean_text(signal.get("source_type")),
        _clean_text(signal.get("reason")),
        _clean_text(signal.get("evidenceKind")),
    )
    existing = {
        (
            _clean_text(item.get("source_id")),
            _clean_text(item.get("source_type")),
            _clean_text(item.get("reason")),
            _clean_text(item.get("evidenceKind")),
        )
        for item in bucket
    }
    if key not in existing:
        bucket.append(signal)


def _span_is_stale(span: dict[str, Any]) -> bool:
    derivative = dict(span.get("derivativeSource") or span.get("derivative_source") or {})
    return bool(span.get("stale")) or bool(derivative.get("stale"))


def _span_has_strict_provenance(span: dict[str, Any]) -> bool:
    return (
        bool(span.get("contentHashAvailable"))
        and bool(span.get("spanOffsetAvailable"))
        and bool(span.get("source_id") or span.get("sourceId"))
        and not _span_is_stale(span)
    )


def build_evidence_packet_contract(
    *,
    query: str,
    retrieval_mode: str,
    pipeline_result: Any,
    evidence_packet: Any,
    strict: bool = True,
) -> dict[str, Any]:
    try:
        plan_payload = dict(pipeline_result.plan.to_dict() or {})
    except Exception:
        plan_payload = {}
    query_frame = dict(plan_payload.get("queryFrame") or {})
    evidence = [dict(item or {}) for item in list(getattr(evidence_packet, "evidence", []) or [])]
    raw_spans = [_evidence_span(item, index=index) for index, item in enumerate(evidence, start=1)]
    excluded_non_evidence = [span for span in raw_spans if _non_evidence_reason(span)]
    excluded_low_provenance = [
        span
        for span in raw_spans
        if not _non_evidence_reason(span) and not _span_has_strict_provenance(span)
    ]
    spans = [
        span
        for span in raw_spans
        if not _non_evidence_reason(span) and (not strict or _span_has_strict_provenance(span))
    ]
    policy_payload = dict(getattr(evidence_packet, "evidence_policy", {}) or {})
    query_id = _hash_text(query, retrieval_mode, [span.get("sourceId") for span in spans])
    assembled_at = datetime.now(timezone.utc).isoformat()
    answerable = bool(dict(getattr(evidence_packet, "evidence_packet", {}) or {}).get("answerable", bool(spans)))
    policy_class = _normalize_classification(policy_payload.get("classification") or policy_payload.get("policyClass"))
    span_classes = {_normalize_classification((span.get("policy") or {}).get("classification")) for span in raw_spans}
    if "P0" in span_classes:
        policy_class = "P0"
    elif policy_class == "UNKNOWN" and span_classes - {"UNKNOWN"}:
        policy_class = sorted(span_classes - {"UNKNOWN"})[0]
    allow_external = policy_payload.get("allow_external") if "allow_external" in policy_payload else policy_payload.get("allowExternal")
    explicit_external = _bool_or_none(allow_external)
    external_allowed = explicit_external if explicit_external is not None else policy_class not in {"P0", "UNKNOWN"}
    if any(not bool((span.get("policy") or {}).get("external_allowed")) for span in raw_spans):
        external_allowed = False
    explicit_allowed = _bool_or_none(policy_payload.get("allowed"))
    policy_allowed = explicit_allowed if explicit_allowed is not None else policy_class != "P0"
    complete_spans = sum(
        1
        for span in spans
        if bool(span.get("contentHashAvailable")) and bool(span.get("spanOffsetAvailable"))
    )
    coverage_status = "none" if not spans else ("complete" if complete_spans == len(spans) else "partial")
    if strict and raw_spans and not spans:
        answerable = False
        coverage_status = "insufficient"
    if not answerable:
        coverage_status = "insufficient"
    return {
        "schema": EVIDENCE_PACKET_SCHEMA,
        "queryId": query_id,
        "packet_id": query_id,
        "query": str(query or ""),
        "queryFrame": query_frame,
        "retrievalMode": str(retrieval_mode or ""),
        "spans": spans,
        "policy": {
            "classification": policy_class,
            "allowed": bool(policy_allowed),
            "allowExternal": bool(external_allowed),
            "external_allowed": bool(external_allowed),
            "policyKey": _clean_text(policy_payload.get("policyKey") or policy_payload.get("policy_key")),
        },
        "answerable": answerable,
        "coverage": {
            "status": coverage_status,
            "span_count": len(spans),
            "raw_span_count": len(raw_spans),
            "source_count": len({span.get("source_id") for span in spans if span.get("source_id")}),
            "excluded_low_provenance": len(excluded_low_provenance) if strict else 0,
            "excluded_non_evidence": len(excluded_non_evidence),
            "excluded_stale": sum(1 for span in excluded_low_provenance if _span_is_stale(span)) if strict else 0,
        },
        "assembledAt": assembled_at,
        "created_at": assembled_at,
        "assemblerVersion": "evidence-packet-contract-v1",
    }


def _claim_like_sentences(answer: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", str(answer or "")) if part.strip()]
    if not sentences and str(answer or "").strip():
        sentences = [str(answer or "").strip()]
    return [
        item
        for item in sentences
        if len(item) >= 4 and not item.endswith(":")
    ]


_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]{2,}")
_STOP_TOKENS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "하다",
    "있다",
    "된다",
    "입니다",
    "합니다",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(str(text or ""))
        if token.lower() not in _STOP_TOKENS
    }


def _citation_refs_for_claim(sentence: str, citations: list[dict[str, Any]]) -> list[str]:
    sentence_tokens = _tokens(sentence)
    if not sentence_tokens:
        return []
    refs: list[str] = []
    for citation in citations:
        citation_text = " ".join(
            str(citation.get(key) or "")
            for key in ("quote", "text", "target", "source_id", "label", "citationLabel")
        )
        citation_tokens = _tokens(citation_text)
        if not citation_tokens:
            continue
        overlap = sentence_tokens & citation_tokens
        if len(overlap) >= 2 or (len(sentence_tokens) <= 4 and bool(overlap)):
            refs.append(str(citation.get("spanRef") or citation.get("span_id") or ""))
    return [ref for ref in refs if ref]


def build_verification_verdict(verification: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(verification or {})
    status = _clean_text(payload.get("status")).lower()
    unsupported = int(payload.get("unsupportedClaimCount") or payload.get("claimUnsupportedCount") or 0)
    uncertain = int(payload.get("uncertainClaimCount") or 0)
    needs_caution = bool(payload.get("needsCaution"))
    if status in {"verified", "pass"} and unsupported == 0 and not needs_caution:
        verdict = "pass"
    elif status in {"skipped", "abstain"}:
        verdict = "abstain"
    else:
        verdict = "fail" if (unsupported or uncertain or needs_caution or status in {"failed", "caution", "fail"}) else "abstain"
    rewrite_allowed = verdict == "pass"
    checked_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema": VERIFICATION_VERDICT_SCHEMA,
        "verdict": verdict,
        "status": status or "unknown",
        "unsupportedClaimCount": unsupported,
        "uncertainClaimCount": uncertain,
        "supportedClaimCount": int(payload.get("supportedClaimCount") or 0),
        "needsCaution": needs_caution,
        "reason": _clean_text(payload.get("summary") or payload.get("reason")),
        "checked_at": checked_at,
        "rewriteAllowed": bool(rewrite_allowed),
        "rewritePolicy": "citation_alignment_or_framing_only" if rewrite_allowed else "blocked_by_verification_gate",
        "recommended_action": "return" if verdict == "pass" else "abstain",
    }


def build_answer_contract(
    *,
    answer: str,
    evidence_packet: Any,
    verification: dict[str, Any] | None = None,
    rewrite: dict[str, Any] | None = None,
    routing_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = [dict(item or {}) for item in list(getattr(evidence_packet, "evidence", []) or [])]
    citations_payload = list(getattr(evidence_packet, "citations", []) or [])
    citations: list[dict[str, Any]] = []
    retrieval_signals: list[dict[str, Any]] = []
    for index in range(1, max(len(evidence), len(citations_payload)) + 1):
        citation = dict(citations_payload[index - 1] or {}) if index - 1 < len(citations_payload) else {}
        item = evidence[index - 1] if index - 1 < len(evidence) else {}
        span = _evidence_span(item, index=index)
        signal_reason = _non_evidence_reason({**item, **span})
        if signal_reason:
            _append_unique_signal(
                retrieval_signals,
                _retrieval_signal_entry(item, span, reason=signal_reason),
            )
            continue
        if not _span_has_strict_provenance(span):
            continue
        citations.append(
            {
                "spanRef": f"span:{index}",
                "span_id": f"span:{index}",
                "citationLabel": _clean_text(citation.get("label") or item.get("citation_label") or f"S{index}"),
                "label": _clean_text(citation.get("label") or item.get("citation_label") or f"S{index}"),
                "target": _clean_text(citation.get("target") or item.get("citation_target") or item.get("source_id")),
                "source_id": span["source_id"],
                "content_hash": span["content_hash"],
                "char_start": span["char_start"],
                "char_end": span["char_end"],
                "quote": str(item.get("excerpt") or "")[:500],
                "text": str(span.get("text") or "")[:500],
            }
        )
    evidence_payload = dict(getattr(evidence_packet, "evidence_packet", {}) or {})
    claim_sentences = _claim_like_sentences(answer)
    claim_citation_map = [
        {
            "sentence": sentence,
            "citationRefs": _citation_refs_for_claim(sentence, citations),
        }
        for sentence in claim_sentences
    ]
    citation_backed = sum(1 for item in claim_citation_map if item["citationRefs"])
    coverage_ratio = 1.0 if not claim_sentences else round(citation_backed / max(1, len(claim_sentences)), 4)
    unsupported = int((verification or {}).get("unsupportedClaimCount") or (verification or {}).get("claimUnsupportedCount") or 0)
    abstain = evidence_payload.get("answerable") is False or not str(answer or "").strip()
    route = dict(routing_meta or {})
    model_id = _clean_text(route.get("model") or route.get("model_id") or route.get("modelId"))
    provider = _clean_text(route.get("provider"))
    if provider and model_id and "/" not in model_id:
        model_id = f"{provider}/{model_id}"
    if not model_id:
        model_id = "unknown"
    prompt_hash = _clean_text(route.get("prompt_hash") or route.get("promptHash")) or _hash_text("prompt", model_id, len(answer or ""))
    evidence_packet_id = _hash_text([item.get("source_id") for item in evidence], len(citations_payload))
    generated_at = datetime.now(timezone.utc).isoformat()
    coverage_status = "none" if not citations else ("complete" if coverage_ratio >= 1.0 else "partial")
    if abstain:
        coverage_status = "insufficient"
    return {
        "schema": ANSWER_CONTRACT_SCHEMA,
        "answer_id": _hash_text(answer, evidence_packet_id),
        "evidence_packet_id": evidence_packet_id,
        "answerText": str(answer or ""),
        "answer_text": str(answer or ""),
        "citations": citations,
        "abstain": bool(abstain),
        "abstainReason": _clean_text(evidence_payload.get("answerableDecisionReason")) if abstain else "",
        "coverage": {
            "status": coverage_status,
            "citation_count": len(citations),
            "supported_span_count": citation_backed,
            "claim_count": len(claim_sentences),
            "unmapped_claim_count": max(0, len(claim_sentences) - citation_backed),
            "unsupported_claim_count": unsupported,
            "excluded_non_evidence_signal_count": len(retrieval_signals),
        },
        "coverageRatio": coverage_ratio,
        "claimLikeSentenceCount": len(claim_sentences),
        "citationBackedSentenceCount": citation_backed,
        "citationClaimMap": claim_citation_map,
        "retrievalSignals": retrieval_signals,
        "retrieval_signals": retrieval_signals,
        "verificationVerdict": build_verification_verdict(verification),
        "rewrite": {
            "attempted": bool((rewrite or {}).get("attempted")),
            "applied": bool((rewrite or {}).get("applied")),
            "finalAnswerSource": _clean_text((rewrite or {}).get("finalAnswerSource")),
        },
        "modelId": model_id,
        "model_id": model_id,
        "promptHash": prompt_hash,
        "prompt_hash": prompt_hash,
        "generatedAt": generated_at,
        "created_at": generated_at,
    }


__all__ = [
    "ANSWER_CONTRACT_SCHEMA",
    "EVIDENCE_PACKET_SCHEMA",
    "VERIFICATION_VERDICT_SCHEMA",
    "build_answer_contract",
    "build_evidence_packet_contract",
    "build_verification_verdict",
    "parse_span_offsets",
]
