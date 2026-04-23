from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from typing import Any


EVIDENCE_PACKET_SCHEMA = "knowledge-hub.evidence-packet.v1"
ANSWER_CONTRACT_SCHEMA = "knowledge-hub.answer-contract.v1"
VERIFICATION_VERDICT_SCHEMA = "knowledge-hub.verification-verdict.v1"


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _hash_text(*parts: Any, length: int = 16) -> str:
    text = "\n".join(str(part or "") for part in parts if str(part or "").strip())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: max(8, int(length))]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


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
        item.get("char_start")
        or item.get("charStart")
        or item.get("chunk_start")
        or item.get("chunkStart")
        or item.get("start_offset")
        or item.get("startOffset")
    )
    char_end = _int_or_none(
        item.get("char_end")
        or item.get("charEnd")
        or item.get("chunk_end")
        or item.get("chunkEnd")
        or item.get("end_offset")
        or item.get("endOffset")
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
    snippet_hash = _clean_text(item.get("snippet_hash") or item.get("snippetHash"))
    content_hash = source_hash or snippet_hash or _hash_text(source_id, span_locator, text)
    classification = _clean_text(item.get("policy_class") or item.get("classification")) or "P2"
    allow_external = item.get("allow_external") if "allow_external" in item else item.get("allowExternal")
    policy_allowed = item.get("policy_allowed") if "policy_allowed" in item else item.get("policyAllowed", True)
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
            "allowed": bool(policy_allowed),
            "allowExternal": allow_external,
            "external_allowed": bool(allow_external),
        },
        "evidenceKind": _clean_text(item.get("evidence_kind") or item.get("evidenceKind")),
        "derivativeSource": dict(item.get("derivative_source") or item.get("derivativeSource") or {}),
        "snippetHash": snippet_hash,
    }


def build_evidence_packet_contract(
    *,
    query: str,
    retrieval_mode: str,
    pipeline_result: Any,
    evidence_packet: Any,
) -> dict[str, Any]:
    try:
        plan_payload = dict(pipeline_result.plan.to_dict() or {})
    except Exception:
        plan_payload = {}
    query_frame = dict(plan_payload.get("queryFrame") or {})
    evidence = [dict(item or {}) for item in list(getattr(evidence_packet, "evidence", []) or [])]
    spans = [_evidence_span(item, index=index) for index, item in enumerate(evidence, start=1)]
    policy_payload = dict(getattr(evidence_packet, "evidence_policy", {}) or {})
    query_id = _hash_text(query, retrieval_mode, [span.get("sourceId") for span in spans])
    assembled_at = datetime.now(timezone.utc).isoformat()
    answerable = bool(dict(getattr(evidence_packet, "evidence_packet", {}) or {}).get("answerable", bool(spans)))
    policy_class = _clean_text(policy_payload.get("classification") or policy_payload.get("policyClass")) or "P2"
    allow_external = policy_payload.get("allow_external") if "allow_external" in policy_payload else policy_payload.get("allowExternal")
    complete_spans = sum(
        1
        for span in spans
        if bool(span.get("contentHashAvailable")) and bool(span.get("spanOffsetAvailable"))
    )
    coverage_status = "none" if not spans else ("complete" if complete_spans == len(spans) else "partial")
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
            "allowed": bool(policy_payload.get("allowed", True)),
            "allowExternal": allow_external,
            "external_allowed": bool(allow_external),
            "policyKey": _clean_text(policy_payload.get("policyKey") or policy_payload.get("policy_key")),
        },
        "answerable": answerable,
        "coverage": {
            "status": coverage_status,
            "span_count": len(spans),
            "source_count": len({span.get("source_id") for span in spans if span.get("source_id")}),
        },
        "assembledAt": assembled_at,
        "created_at": assembled_at,
        "assemblerVersion": "evidence-packet-contract-v1",
    }


def _claim_like_sentences(answer: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", str(answer or "")) if part.strip()]
    if not sentences and str(answer or "").strip():
        sentences = [str(answer or "").strip()]
    return [
        item
        for item in sentences
        if len(item) >= 12 and not item.endswith(":")
    ]


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
    rewrite_allowed = verdict == "pass" or (unsupported == 0 and status == "caution")
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
        "rewritePolicy": "citation_alignment_or_framing_only" if rewrite_allowed else "blocked_for_unsupported_claims",
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
    for index, raw in enumerate(citations_payload, start=1):
        citation = dict(raw or {})
        item = evidence[index - 1] if index - 1 < len(evidence) else {}
        span = _evidence_span(item, index=index)
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
    citation_backed = min(len(citations), len(claim_sentences))
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
            "unsupported_claim_count": unsupported,
        },
        "coverageRatio": coverage_ratio,
        "claimLikeSentenceCount": len(claim_sentences),
        "citationBackedSentenceCount": citation_backed,
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
