"""Provider-side outbound policy guard.

Second-line defense for external model calls. RAG/service layers already gate
payloads, but provider adapters must also enforce P0 blocking before outbound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from uuid import uuid4

from knowledge_hub.learning.policy import evaluate_policy_for_payload


POLICY_BLOCKED_OUTBOUND = "POLICY_BLOCKED_OUTBOUND"


@dataclass
class OutboundPolicyDecision:
    allowed: bool
    classification: str
    warnings: list[str]
    rule: str
    trace_id: str
    policy_errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "allowed": self.allowed,
            "warnings": list(self.warnings),
            "rule": self.rule,
            "trace_id": self.trace_id,
            "policyErrors": list(self.policy_errors),
        }


@dataclass
class BatchOutboundPolicyReport:
    checked_count: int
    allowed_count: int
    blocked_count: int
    blocked_indices: list[int]
    warnings: list[str]
    trace_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "checkedCount": self.checked_count,
            "allowedCount": self.allowed_count,
            "blockedCount": self.blocked_count,
            "blockedIndices": list(self.blocked_indices),
            "warnings": list(self.warnings),
            "trace_id": self.trace_id,
        }


class OutboundPolicyError(RuntimeError):
    """Raised when provider outbound call violates local-first policy."""

    def __init__(self, decision: OutboundPolicyDecision):
        self.code = POLICY_BLOCKED_OUTBOUND
        self.decision = decision
        message = f"{POLICY_BLOCKED_OUTBOUND}: classification={decision.classification} trace_id={decision.trace_id}"
        super().__init__(message)


def evaluate_outbound_policy(
    *,
    provider: str,
    model: str,
    prompt: str = "",
    context: str = "",
) -> OutboundPolicyDecision:
    trace_id = f"policy_{uuid4().hex[:12]}"
    status = evaluate_policy_for_payload(
        allow_external=True,
        raw_texts=[str(prompt or ""), str(context or "")],
        mode=f"provider-outbound:{provider}",
    )
    warnings = list(status.warnings or [])
    rule = "deny_p0_raw_sensitive" if status.classification == "P0" else "allow_non_p0_with_warning"
    if status.classification == "P1" and "P1 structured facts detected" not in warnings:
        warnings.append("P1 structured facts detected")
    warnings.append(f"provider={provider}")
    warnings.append(f"model={model}")
    return OutboundPolicyDecision(
        allowed=bool(status.allowed),
        classification=status.classification,
        warnings=warnings,
        rule=rule,
        trace_id=trace_id,
        policy_errors=list(status.policy_errors or []),
    )


def enforce_outbound_policy(
    *,
    provider: str,
    model: str,
    prompt: str = "",
    context: str = "",
) -> OutboundPolicyDecision:
    decision = evaluate_outbound_policy(provider=provider, model=model, prompt=prompt, context=context)
    if not decision.allowed:
        raise OutboundPolicyError(decision)
    return decision


def evaluate_outbound_policy_batch(
    *,
    provider: str,
    model: str,
    texts: list[str],
    chunk_size: int = 128,
) -> BatchOutboundPolicyReport:
    trace_id = f"policy_batch_{uuid4().hex[:12]}"
    blocked_indices: list[int] = []
    warnings: list[str] = [f"provider={provider}", f"model={model}"]
    checked = 0

    def _chunks(items: list[tuple[int, str]]) -> Iterable[list[tuple[int, str]]]:
        size = max(1, int(chunk_size))
        for idx in range(0, len(items), size):
            yield items[idx: idx + size]

    enumerated = [(idx, str(text or "")) for idx, text in enumerate(texts)]
    for batch in _chunks(enumerated):
        for idx, text in batch:
            checked += 1
            status = evaluate_policy_for_payload(
                allow_external=True,
                raw_texts=[text],
                mode=f"provider-outbound:{provider}",
            )
            if not status.allowed:
                blocked_indices.append(idx)
            if status.classification == "P1":
                warnings.append(f"P1 warning at index={idx}")

    allowed_count = checked - len(blocked_indices)
    return BatchOutboundPolicyReport(
        checked_count=checked,
        allowed_count=allowed_count,
        blocked_count=len(blocked_indices),
        blocked_indices=blocked_indices,
        warnings=list(dict.fromkeys(warnings)),
        trace_id=trace_id,
    )
