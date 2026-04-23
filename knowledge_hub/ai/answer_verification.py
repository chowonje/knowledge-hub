from __future__ import annotations

from typing import Any

from knowledge_hub.ai.rag_support import (
    clean_text as _clean_text,
    extract_json_payload as _extract_json_payload,
    jaccard as _jaccard,
    tokenize as _tokenize,
    truncate_text as _truncate_text,
)
from knowledge_hub.core.sanitizer import redact_p0
from knowledge_hub.learning.policy import evaluate_policy_for_payload


def heuristic_answer_verification(
    searcher,
    *,
    answer: str,
    evidence: list[dict[str, Any]],
    answer_signals: dict[str, Any],
    contradicting_beliefs: list[dict[str, Any]],
    route_meta: dict[str, Any],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    claim_texts = searcher._split_answer_claims(answer)
    if not claim_texts:
        return {
            "status": "skipped",
            "supportedClaimCount": 0,
            "unsupportedClaimCount": 0,
            "uncertainClaimCount": 0,
            "conflictMentioned": searcher._answer_mentions_conflict(answer),
            "needsCaution": bool(answer_signals.get("caution_required")),
            "summary": "답변에서 검증 가능한 구체 claim을 충분히 추출하지 못했습니다.",
            "warnings": list(dict.fromkeys([*(warnings or []), "answer verification skipped: no concrete claims found"])),
            "claims": [],
            "route": {**dict(route_meta or {}), "mode": "heuristic"},
        }

    evidence_rows: list[tuple[str, set[str], str]] = []
    for item in evidence:
        excerpt = _clean_text(f"{item.get('title', '')} {item.get('excerpt', '')}")
        evidence_rows.append((str(item.get("title") or ""), _tokenize(excerpt), excerpt))

    claims: list[dict[str, Any]] = []
    for claim in claim_texts:
        claim_tokens = _tokenize(claim)
        best_title = ""
        best_score = 0.0
        for title, evidence_tokens, _excerpt in evidence_rows:
            overlap = _jaccard(claim_tokens, evidence_tokens)
            if overlap > best_score:
                best_score = overlap
                best_title = title
        if best_score < 0.08:
            verdict = "unsupported"
            reason = "근거 excerpt/summary와의 어휘 중첩이 매우 낮아 직접 지지를 확인하지 못했습니다."
            titles: list[str] = []
        else:
            verdict = "uncertain"
            reason = "일부 유사 근거는 있으나 직접 지지 여부를 보수적으로 확정하지 않았습니다."
            titles = [best_title] if best_title else []
        claims.append(
            {
                "claim": claim,
                "verdict": verdict,
                "evidenceTitles": titles,
                "reason": reason,
            }
        )

    unsupported_count = sum(1 for item in claims if item["verdict"] == "unsupported")
    uncertain_count = sum(1 for item in claims if item["verdict"] == "uncertain")
    contradiction_present = bool(answer_signals.get("contradictory_source_count")) or bool(contradicting_beliefs)
    conflict_mentioned = not contradiction_present or searcher._answer_mentions_conflict(answer)
    needs_caution = contradiction_present and not conflict_mentioned
    needs_caution = bool(needs_caution or unsupported_count > 0 or uncertain_count > 0)
    summary = (
        f"휴리스틱 검증 결과 unsupported {unsupported_count}건, uncertain {uncertain_count}건으로 근거 충실성에 주의가 필요합니다."
        if needs_caution
        else "휴리스틱 검증에서 뚜렷한 위험 신호는 찾지 못했습니다."
    )
    return {
        "status": "caution" if needs_caution else "verified",
        "supportedClaimCount": 0,
        "unsupportedClaimCount": unsupported_count,
        "uncertainClaimCount": uncertain_count,
        "conflictMentioned": bool(conflict_mentioned),
        "needsCaution": bool(needs_caution),
        "summary": summary,
        "warnings": list(dict.fromkeys([*(warnings or []), "answer verification used heuristic fallback"])),
        "claims": claims,
        "route": {**dict(route_meta or {}), "mode": "heuristic"},
    }


def normalize_answer_verification(
    searcher,
    *,
    raw: dict[str, Any],
    answer: str,
    answer_signals: dict[str, Any],
    contradicting_beliefs: list[dict[str, Any]],
    route_meta: dict[str, Any],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    contradiction_present = bool(answer_signals.get("contradictory_source_count")) or bool(contradicting_beliefs)
    claims_payload = raw.get("claims")
    normalized_claims: list[dict[str, Any]] = []
    if isinstance(claims_payload, list):
        for item in claims_payload[:12]:
            if not isinstance(item, dict):
                continue
            claim = _clean_text(str(item.get("claim") or ""))
            verdict = str(item.get("verdict") or "").strip().lower()
            if not claim or verdict not in {"supported", "uncertain", "unsupported"}:
                continue
            titles = [str(title).strip() for title in list(item.get("evidenceTitles") or []) if str(title).strip()][:5]
            normalized_claims.append(
                {
                    "claim": claim,
                    "verdict": verdict,
                    "evidenceTitles": titles,
                    "reason": _truncate_text(str(item.get("reason") or ""), 240),
                }
            )

    supported_count = sum(1 for item in normalized_claims if item["verdict"] == "supported")
    unsupported_count = sum(1 for item in normalized_claims if item["verdict"] == "unsupported")
    uncertain_count = sum(1 for item in normalized_claims if item["verdict"] == "uncertain")
    conflict_mentioned = bool(raw.get("conflictMentioned")) if "conflictMentioned" in raw else searcher._answer_mentions_conflict(answer)
    needs_caution = bool(raw.get("needsCaution")) if "needsCaution" in raw else False
    if contradiction_present and not conflict_mentioned:
        needs_caution = True
    if unsupported_count > 0 or uncertain_count > 0:
        needs_caution = True

    explicit_status = str(raw.get("status") or "").strip().lower()
    if explicit_status in {"verified", "caution", "failed", "skipped"}:
        status = explicit_status
    elif unsupported_count == 0 and uncertain_count == 0 and not needs_caution:
        status = "verified"
    else:
        status = "caution"

    summary = _truncate_text(str(raw.get("summary") or ""), 240)
    if not summary:
        if status == "verified":
            summary = f"검증 결과 {supported_count}건의 claim이 근거와 직접 연결되었습니다."
        else:
            summary = f"검증 결과 unsupported {unsupported_count}건, uncertain {uncertain_count}건이 있어 답변을 보수적으로 읽어야 합니다."
    return {
        "status": status,
        "supportedClaimCount": supported_count,
        "unsupportedClaimCount": unsupported_count,
        "uncertainClaimCount": uncertain_count,
        "conflictMentioned": bool(conflict_mentioned),
        "needsCaution": bool(needs_caution),
        "summary": summary,
        "warnings": list(dict.fromkeys(warnings or [])),
        "claims": normalized_claims,
        "route": {**dict(route_meta or {}), "mode": "llm"},
    }


def verify_answer(
    searcher,
    *,
    query: str,
    answer: str,
    evidence: list[dict[str, Any]],
    answer_signals: dict[str, Any],
    contradicting_beliefs: list[dict[str, Any]],
    allow_external: bool,
) -> dict[str, Any]:
    evidence_context = searcher._build_answer_verification_context(
        evidence=evidence,
        answer_signals=answer_signals,
        contradicting_beliefs=contradicting_beliefs,
    )
    verifier_llm, route_meta, route_warnings = searcher._resolve_llm_for_verification(
        query=query,
        context=evidence_context,
        source_count=len(evidence),
        allow_external=allow_external,
    )
    warnings = list(route_warnings)
    route_name = str((route_meta or {}).get("route") or "").strip().lower()
    route_reasons = {str(item or "").strip().lower() for item in list((route_meta or {}).get("reasons") or []) if str(item or "").strip()}
    external_route = route_name in {"mini", "strong"}

    if verifier_llm is None:
        if route_name == "fallback-only" and "config_missing" in route_reasons:
            return {
                "status": "skipped",
                "supportedClaimCount": 0,
                "unsupportedClaimCount": 0,
                "uncertainClaimCount": 0,
                "conflictMentioned": searcher._answer_mentions_conflict(answer),
                "needsCaution": False,
                "summary": "검증 라우트를 사용할 수 없어 답변 검증을 건너뛰었습니다.",
                "warnings": list(dict.fromkeys(warnings)),
                "claims": [],
                "route": {**dict(route_meta or {}), "mode": "skipped"},
            }
        return heuristic_answer_verification(
            searcher,
            answer=answer,
            evidence=evidence,
            answer_signals=answer_signals,
            contradicting_beliefs=contradicting_beliefs,
            route_meta=route_meta,
            warnings=warnings,
        )

    external_policy = evaluate_policy_for_payload(
        allow_external=external_route,
        raw_texts=[evidence_context, answer],
        mode="rag-answer-verification-external" if external_route else "rag-answer-verification-local",
    )
    original_classification = external_policy.classification
    safe_context = evidence_context
    if external_route and not external_policy.allowed and original_classification == "P0":
        safe_context = redact_p0(evidence_context)
        redacted_policy = evaluate_policy_for_payload(
            allow_external=True,
            raw_texts=[safe_context, answer],
            mode="rag-answer-verification-redacted",
        )
        if redacted_policy.allowed:
            external_policy = redacted_policy
        else:
            warnings.append("answer verification skipped external verifier due to P0 policy block")
            return {
                "status": "skipped",
                "supportedClaimCount": 0,
                "unsupportedClaimCount": 0,
                "uncertainClaimCount": 0,
                "conflictMentioned": searcher._answer_mentions_conflict(answer),
                "needsCaution": bool(answer_signals.get("caution_required")),
                "summary": "정책상 외부 검증을 수행하지 못해 답변 검증을 건너뛰었습니다.",
                "warnings": list(dict.fromkeys(warnings)),
                "claims": [],
                "route": {**dict(route_meta or {}), "mode": "skipped"},
            }

    try:
        raw_output = verifier_llm.generate(
            searcher._build_answer_verification_prompt(query=query, answer=answer),
            safe_context,
        )
    except Exception as error:
        warnings.append(f"answer verification llm failed: {error}")
        return heuristic_answer_verification(
            searcher,
            answer=answer,
            evidence=evidence,
            answer_signals=answer_signals,
            contradicting_beliefs=contradicting_beliefs,
            route_meta=route_meta,
            warnings=warnings,
        )

    parsed = _extract_json_payload(str(raw_output or ""))
    if not parsed:
        warnings.append("answer verification llm returned non-json output")
        return heuristic_answer_verification(
            searcher,
            answer=answer,
            evidence=evidence,
            answer_signals=answer_signals,
            contradicting_beliefs=contradicting_beliefs,
            route_meta=route_meta,
            warnings=warnings,
        )

    return normalize_answer_verification(
        searcher,
        raw=parsed,
        answer=answer,
        answer_signals=answer_signals,
        contradicting_beliefs=contradicting_beliefs,
        route_meta=route_meta,
        warnings=warnings,
    )
