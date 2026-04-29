from __future__ import annotations

from knowledge_hub.ai.answer_verification import verify_answer
from knowledge_hub.ai.rag_support import (
    build_paper_answer_readiness_p1_conservative_answer,
    clean_text as _clean_text,
    paper_answer_readiness_p1_fallback_enabled,
)
from knowledge_hub.core.sanitizer import redact_p0
from knowledge_hub.learning.policy import evaluate_policy_for_payload


def should_apply_conservative_fallback(verification: dict) -> bool:
    if not bool(verification.get("needsCaution")):
        return False
    if int(verification.get("unsupportedClaimCount") or 0) > 0:
        return True
    if int(verification.get("claimUnsupportedCount") or 0) > 0:
        return True
    if int(verification.get("claimConflictCount") or 0) > 0:
        return True
    if int(verification.get("claimWeakCount") or 0) > 0 and int(verification.get("supportedClaimCount") or 0) == 0:
        return True
    if int(verification.get("supportedClaimCount") or 0) == 0:
        return True
    return False


def _unsupported_claim_count(verification: dict) -> int:
    return max(
        int(verification.get("unsupportedClaimCount") or 0),
        int(verification.get("claimUnsupportedCount") or 0),
    )


def _gate_fallback_warning(verification: dict) -> str:
    if _unsupported_claim_count(verification) > 0:
        return "answer rewrite skipped: unsupported claims require conservative fallback"
    supported_count = int(verification.get("supportedClaimCount") or 0)
    if supported_count > 0:
        return ""
    if int(verification.get("claimWeakCount") or 0) > 0:
        return "answer rewrite skipped: weak claims require conservative fallback"
    if int(verification.get("uncertainClaimCount") or 0) > 0:
        return "answer rewrite skipped: uncertain claims require conservative fallback"
    return ""


def apply_conservative_fallback_if_needed(
    searcher,
    *,
    query: str,
    answer: str,
    rewrite_meta: dict,
    verification: dict,
    evidence: list[dict],
    answer_signals: dict,
    contradicting_beliefs: list[dict],
    allow_external: bool,
    routing_meta: dict | None = None,
):
    rewrite_applied = bool((rewrite_meta or {}).get("applied"))
    fallback_required = bool((rewrite_meta or {}).get("requiresConservativeFallback"))
    if not rewrite_applied and not fallback_required:
        return answer, rewrite_meta, verification
    if not should_apply_conservative_fallback(verification):
        return answer, rewrite_meta, verification

    if paper_answer_readiness_p1_fallback_enabled(
        config=getattr(searcher, "config", None),
        answer_signals=answer_signals,
        evidence=evidence,
        allow_external=allow_external,
        routing_meta=routing_meta,
    ):
        conservative_answer = build_paper_answer_readiness_p1_conservative_answer(evidence=evidence)
    else:
        conservative_answer = searcher._build_conservative_answer(
            verification=verification,
            evidence=evidence,
        )
    if not conservative_answer or conservative_answer == _clean_text(answer):
        return answer, rewrite_meta, verification

    fallback_verification = verify_answer(
        searcher,
        query=query,
        answer=conservative_answer,
        evidence=evidence,
        answer_signals=answer_signals,
        contradicting_beliefs=contradicting_beliefs,
        allow_external=allow_external,
    )
    updated_meta = dict(rewrite_meta or {})
    fallback_warnings = list(dict.fromkeys([*list(updated_meta.get("warnings") or []), "answer conservative fallback applied"]))
    updated_meta.update(
        {
            "attempted": True,
            "applied": True,
            "summary": "검증 후에도 주의 신호가 남아, 답변을 더 짧고 보수적인 형태로 정리했습니다.",
            "finalAnswerSource": "conservative_fallback",
            "warnings": fallback_warnings,
        }
    )
    return conservative_answer, updated_meta, fallback_verification


def rewrite_answer(
    searcher,
    *,
    query: str,
    answer: str,
    evidence: list[dict],
    answer_signals: dict,
    verification: dict,
    contradicting_beliefs: list[dict],
    allow_external: bool,
):
    triggered_by = searcher._should_rewrite_answer(verification)
    rewrite_meta = searcher._default_answer_rewrite(answer=answer)
    gate_warning = _gate_fallback_warning(verification)
    if gate_warning:
        rewrite_meta.update(
            {
                "attempted": False,
                "applied": False,
                "triggeredBy": list(triggered_by or ["verification_gate"]),
                "attemptCount": 0,
                "summary": "근거가 약하거나 불확실한 claim이 감지되어 LLM 재작성 대신 보수적 fallback으로 넘겼습니다.",
                "finalAnswerSource": "original",
                "requiresConservativeFallback": True,
                "warnings": [gate_warning],
            }
        )
        return answer, rewrite_meta
    if not triggered_by:
        return answer, rewrite_meta

    rewrite_context = searcher._build_answer_rewrite_context(
        evidence=evidence,
        answer_signals=answer_signals,
        contradicting_beliefs=contradicting_beliefs,
        verification=verification,
    )
    rewrite_llm, route_meta, route_warnings = searcher._resolve_llm_for_rewrite(
        query=query,
        context=rewrite_context,
        source_count=len(evidence),
        allow_external=allow_external,
    )
    rewrite_meta = {
        "attempted": True,
        "applied": False,
        "triggeredBy": list(triggered_by),
        "attemptCount": 1,
        "summary": "답변 충실성 경고로 인해 1회 재작성을 시도했지만 원본을 유지했습니다.",
        "originalAnswer": answer,
        "finalAnswerSource": "original",
        "route": dict(route_meta or {}),
        "warnings": list(route_warnings or []),
    }

    if rewrite_llm is None:
        rewrite_meta["warnings"] = list(
            dict.fromkeys([*list(rewrite_meta.get("warnings") or []), "answer rewrite skipped: rewrite route unavailable"])
        )
        return answer, rewrite_meta

    external_route = str((route_meta or {}).get("route") or "").strip().lower() in {"mini", "strong"}
    safe_context = rewrite_context
    rewrite_policy = evaluate_policy_for_payload(
        allow_external=external_route,
        raw_texts=[rewrite_context, answer],
        mode="rag-answer-rewrite-external" if external_route else "rag-answer-rewrite-local",
    )
    original_classification = rewrite_policy.classification
    if external_route and not rewrite_policy.allowed and original_classification == "P0":
        redacted_context = redact_p0(rewrite_context)
        if redacted_context != rewrite_context:
            safe_context = redacted_context
            redacted_policy = evaluate_policy_for_payload(
                allow_external=True,
                raw_texts=[safe_context, answer],
                mode="rag-answer-rewrite-redacted",
            )
            if redacted_policy.allowed:
                rewrite_policy = redacted_policy
    if not rewrite_policy.allowed:
        rewrite_meta["warnings"] = list(
            dict.fromkeys(
                [
                    *list(rewrite_meta.get("warnings") or []),
                    "answer rewrite skipped due to policy block",
                    *list(rewrite_policy.warnings or []),
                ]
            )
        )
        return answer, rewrite_meta

    try:
        rewritten = _clean_text(
            str(
                rewrite_llm.generate(
                    searcher._build_answer_rewrite_prompt(
                        query=query,
                        original_answer=answer,
                        verification=verification,
                        triggered_by=triggered_by,
                    ),
                    safe_context,
                )
                or ""
            )
        )
    except Exception as error:
        rewrite_meta["warnings"] = list(
            dict.fromkeys([*list(rewrite_meta.get("warnings") or []), f"answer rewrite failed: {error}"])
        )
        return answer, rewrite_meta

    if not rewritten:
        rewrite_meta["warnings"] = list(
            dict.fromkeys([*list(rewrite_meta.get("warnings") or []), "answer rewrite returned empty output"])
        )
        return answer, rewrite_meta

    rewrite_meta.update(
        {
            "applied": True,
            "summary": "근거 밖 주장 또는 누락된 주의 문구를 줄이기 위해 답변을 1회 재작성했습니다.",
            "finalAnswerSource": "rewritten",
            "warnings": list(dict.fromkeys(rewrite_meta.get("warnings") or [])),
        }
    )
    return rewritten, rewrite_meta
