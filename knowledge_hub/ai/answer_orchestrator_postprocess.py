from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnswerPostprocessDeps:
    verify_answer_fn: Any
    rewrite_answer_fn: Any
    apply_conservative_fallback_if_needed_fn: Any
    apply_claim_consensus_to_verification_fn: Any


@dataclass(frozen=True)
class AnswerPostprocessResult:
    initial_answer_verification: dict[str, Any]
    final_answer: str
    final_answer_rewrite: dict[str, Any]
    final_answer_verification: dict[str, Any]
    verification_warnings: list[str]


class AnswerPostprocess:
    def __init__(self, deps: AnswerPostprocessDeps) -> None:
        self._deps = deps

    def run(
        self,
        *,
        query: str,
        initial_answer: str,
        evidence_packet: Any,
        claim_consensus: dict[str, Any],
        claim_consensus_merge_mode: str,
        allow_external: bool,
    ) -> AnswerPostprocessResult:
        deps = self._deps
        initial_answer_verification = deps.verify_answer_fn(
            query=query,
            answer=initial_answer,
            evidence=evidence_packet.evidence,
            answer_signals=evidence_packet.answer_signals,
            contradicting_beliefs=evidence_packet.contradicting_beliefs,
            allow_external=allow_external,
        )
        initial_answer_verification = deps.apply_claim_consensus_to_verification_fn(
            initial_answer_verification,
            claim_consensus,
            merge_mode=claim_consensus_merge_mode,
        )
        final_answer, answer_rewrite = deps.rewrite_answer_fn(
            query=query,
            answer=initial_answer,
            evidence=evidence_packet.evidence,
            answer_signals=evidence_packet.answer_signals,
            verification=initial_answer_verification,
            contradicting_beliefs=evidence_packet.contradicting_beliefs,
            allow_external=allow_external,
        )
        final_answer_verification = initial_answer_verification
        if bool(answer_rewrite.get("applied")):
            final_answer_verification = deps.verify_answer_fn(
                query=query,
                answer=final_answer,
                evidence=evidence_packet.evidence,
                answer_signals=evidence_packet.answer_signals,
                contradicting_beliefs=evidence_packet.contradicting_beliefs,
                allow_external=allow_external,
            )
            final_answer_verification = deps.apply_claim_consensus_to_verification_fn(
                final_answer_verification,
                claim_consensus,
                merge_mode=claim_consensus_merge_mode,
            )
        final_answer, answer_rewrite, final_answer_verification = deps.apply_conservative_fallback_if_needed_fn(
            query=query,
            answer=final_answer,
            rewrite_meta=answer_rewrite,
            verification=final_answer_verification,
            evidence=evidence_packet.evidence,
            answer_signals=evidence_packet.answer_signals,
            contradicting_beliefs=evidence_packet.contradicting_beliefs,
            allow_external=allow_external,
        )
        final_answer_verification = deps.apply_claim_consensus_to_verification_fn(
            final_answer_verification,
            claim_consensus,
            merge_mode=claim_consensus_merge_mode,
        )
        verification_warnings = list(final_answer_verification.get("warnings") or [])
        if bool(final_answer_verification.get("needsCaution")):
            verification_warnings.append(
                "answer verification caution:"
                f" unsupported={int(final_answer_verification.get('unsupportedClaimCount') or 0)}"
                f" uncertain={int(final_answer_verification.get('uncertainClaimCount') or 0)}"
                f" conflict_mentioned={bool(final_answer_verification.get('conflictMentioned'))}"
            )
        elif str(final_answer_verification.get("status") or "").strip().lower() in {"failed", "skipped"}:
            verification_warnings.append(
                f"answer verification {str(final_answer_verification.get('status') or 'skipped').strip().lower()}"
            )
        verification_warnings = list(
            dict.fromkeys([*verification_warnings, *list(answer_rewrite.get("warnings") or [])])
        )
        return AnswerPostprocessResult(
            initial_answer_verification=initial_answer_verification,
            final_answer=final_answer,
            final_answer_rewrite=answer_rewrite,
            final_answer_verification=final_answer_verification,
            verification_warnings=verification_warnings,
        )


__all__ = [
    "AnswerPostprocess",
    "AnswerPostprocessDeps",
    "AnswerPostprocessResult",
]
