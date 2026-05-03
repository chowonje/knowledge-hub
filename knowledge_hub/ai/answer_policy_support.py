from __future__ import annotations

from typing import Any

from knowledge_hub.core.sanitizer import redact_p0
from knowledge_hub.learning.policy import evaluate_policy_for_payload


def with_claim_context(context: str, claim_context: str) -> str:
    base = str(context or "").strip()
    extra = str(claim_context or "").strip()
    if not extra:
        return base
    if not base:
        return extra
    return f"{base}\n\n---\n\n{extra}"


def apply_claim_consensus_to_verification(
    verification: dict[str, Any],
    claim_consensus: dict[str, Any],
    *,
    force_weak_caution: bool = True,
    merge_mode: str = "strict",
) -> dict[str, Any]:
    payload = dict(verification or {})
    consensus = dict(claim_consensus or {})
    support_count = int(consensus.get("supportCount") or 0)
    conflict_count = int(consensus.get("conflictCount") or 0)
    weak_count = int(consensus.get("weakClaimCount") or 0)
    unsupported_count = int(consensus.get("unsupportedClaimCount") or 0)

    payload["claimVerificationSummary"] = str(consensus.get("claimVerificationSummary") or "")
    payload["claimConflictCount"] = conflict_count
    payload["claimWeakCount"] = weak_count
    payload["claimUnsupportedCount"] = unsupported_count
    payload["claimConflicts"] = list(consensus.get("conflicts") or [])
    payload["claimConsensusMode"] = str(merge_mode or "strict").strip().lower() or "strict"

    if payload["claimConsensusMode"] != "strict":
        return payload

    payload["supportedClaimCount"] = max(int(payload.get("supportedClaimCount") or 0), support_count)
    payload["uncertainClaimCount"] = max(int(payload.get("uncertainClaimCount") or 0), weak_count)
    payload["unsupportedClaimCount"] = max(int(payload.get("unsupportedClaimCount") or 0), unsupported_count)

    needs_caution = bool(payload.get("needsCaution"))
    if conflict_count > 0 or unsupported_count > 0 or (force_weak_caution and weak_count > 0):
        needs_caution = True
    payload["needsCaution"] = needs_caution

    status = str(payload.get("status") or "").strip().lower()
    if status in {"verified", "ok"} and needs_caution:
        payload["status"] = "caution"

    warnings = list(payload.get("warnings") or [])
    if conflict_count > 0:
        warnings.append(f"claim adjudication conflict: conflicts={conflict_count}")
    elif unsupported_count > 0:
        warnings.append(f"claim adjudication weak grounding: unsupported={unsupported_count}")
    elif weak_count > 0:
        warnings.append(f"claim adjudication caution: weak={weak_count}")
    payload["warnings"] = list(dict.fromkeys(warnings))

    summary = str(payload.get("summary") or "").strip()
    consensus_summary = str(consensus.get("claimVerificationSummary") or "").strip()
    if consensus_summary:
        suffix = (
            f"claim adjudication={consensus_summary}"
            f" (support={support_count}, weak={weak_count}, unsupported={unsupported_count}, conflicts={conflict_count})"
        )
        payload["summary"] = summary if summary.endswith(suffix) else (f"{summary} {suffix}".strip() if summary else suffix)

    return payload


def evaluate_policy(
    *,
    context: str,
    allow_external: bool,
    route_mode: str,
) -> tuple[str, Any, str]:
    external_route = route_mode in {"api", "fixed"}
    safe_context = context
    external_policy = evaluate_policy_for_payload(
        allow_external=external_route,
        raw_texts=[context],
        mode="rag-external" if external_route else "rag-local",
    )
    original_classification = external_policy.classification
    if route_mode == "api" and external_policy.classification == "P0":
        redacted_context = redact_p0(context)
        if redacted_context != context:
            safe_context = redacted_context
            redacted_policy = evaluate_policy_for_payload(
                allow_external=True,
                raw_texts=[safe_context],
                mode="rag-external-redacted",
            )
            if redacted_policy.allowed:
                external_policy = redacted_policy
    return safe_context, external_policy, original_classification


def policy_payload(
    *,
    original_classification: str,
    effective_policy: Any,
    safe_context: str,
    original_context: str,
    allow_external: bool,
) -> dict[str, Any]:
    return {
        "originalClassification": original_classification,
        "effectiveClassification": effective_policy.classification,
        "contextRedacted": safe_context != original_context,
        "externalCall": effective_policy.to_dict(),
        "warnings": effective_policy.warnings,
        "allowExternalRequested": bool(allow_external),
    }


__all__ = [
    "apply_claim_consensus_to_verification",
    "evaluate_policy",
    "policy_payload",
    "with_claim_context",
]
