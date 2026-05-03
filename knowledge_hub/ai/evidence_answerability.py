from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerabilityInputs:
    evidence_count: int
    substantive_count: int
    all_evidence_non_substantive: bool
    source_mismatch_count: int
    unique_paper_count: int
    resolved_compare_paper_count: int
    requires_multiple_sources: bool
    is_temporal: bool
    is_abstention: bool
    calibration_hardened: bool
    eval_profile: str
    normalized_requested_source: str
    top1_substantive: bool
    top1_rejected_reason: str
    top1_temporal_grounded: bool
    top1_observed_at_only: bool
    top1_direct_score: float
    top1_direct: bool
    direct_answer_count: int
    temporal_grounded_count: int
    weak_observed_only_count: int
    high_trust_count: int
    memory_provenance_count: int
    preferred_source_count: int
    contradicting_beliefs_present: bool
    top1_is_temporal_web_candidate: bool
    top1_has_web_temporal_textual_marker: bool
    paper_top1_non_substantive_refusal: bool


def evaluate_answerability(inputs: AnswerabilityInputs) -> tuple[bool, str, list[str]]:
    answerable = False
    answerable_reason = "no_evidence"
    insufficient_reasons: list[str] = []

    if inputs.evidence_count <= 0:
        insufficient_reasons.append("no_evidence")
    elif inputs.all_evidence_non_substantive:
        insufficient_reasons.append("non_substantive_evidence")
        answerable_reason = "non_substantive_only"
    elif inputs.substantive_count <= 0:
        insufficient_reasons.append("non_substantive_evidence")
        answerable_reason = "no_substantive_evidence"
    elif inputs.source_mismatch_count >= inputs.evidence_count:
        insufficient_reasons.append("source_mismatch")
        answerable_reason = "source_mismatch_only"
    # Multi-source compare: require at least two distinct papers in retrieved evidence.
    # Do not gate on resolved_compare_paper_count alone — the frame can under-resolve
    # while retrieval still surfaces two paper IDs (false need_multiple_papers / no_result).
    if inputs.requires_multiple_sources and inputs.unique_paper_count < 2:
        insufficient_reasons.append("need_multiple_papers")
        answerable_reason = "need_multiple_papers"

    if not inputs.top1_substantive and inputs.top1_rejected_reason == "vault_hub_noise":
        insufficient_reasons.append("non_substantive_evidence")
        answerable_reason = "vault_hub_top1"
    elif not inputs.top1_substantive and inputs.paper_top1_non_substantive_refusal:
        insufficient_reasons.append("non_substantive_evidence")
        answerable_reason = "paper_refusal_excerpt"

    if inputs.contradicting_beliefs_present:
        insufficient_reasons.append("contradicting_beliefs_present")
        answerable_reason = "contradicting_beliefs_present"

    if inputs.calibration_hardened and inputs.is_temporal and inputs.top1_is_temporal_web_candidate:
        if not inputs.top1_substantive:
            insufficient_reasons.append("non_substantive_evidence")
            answerable_reason = "weak_web_temporal_grounding"
        elif (
            inputs.top1_observed_at_only
            or not inputs.top1_temporal_grounded
            or not inputs.top1_has_web_temporal_textual_marker
        ):
            insufficient_reasons.append("missing_temporal_grounding")
            answerable_reason = "weak_web_temporal_grounding"

    if inputs.eval_profile == "candidate-v6" and inputs.evidence_count > 0 and inputs.top1_substantive and not inputs.top1_direct:
        if inputs.is_temporal:
            insufficient_reasons.append("low_confidence_evidence")
            answerable_reason = "insufficient_for_latest_claim"
        elif inputs.is_abstention:
            insufficient_reasons.append("low_confidence_evidence")
            answerable_reason = "weak_support_only"

    if not insufficient_reasons:
        if inputs.eval_profile == "on-control":
            answerable = inputs.substantive_count > 0 and (
                inputs.preferred_source_count > 0
                or inputs.high_trust_count > 0
                or inputs.memory_provenance_count > 0
            )
            answerable_reason = (
                "strict_eval_threshold_met" if answerable else "strict_quality_threshold_not_met"
            )
        else:
            answerable = inputs.substantive_count > 0
            if inputs.is_temporal and inputs.weak_observed_only_count > 0 and inputs.temporal_grounded_count <= 0:
                answerable = False
                insufficient_reasons.append("missing_temporal_grounding")
                answerable_reason = "weak_observed_at_only"
            elif (
                inputs.calibration_hardened
                and inputs.is_temporal
                and inputs.top1_is_temporal_web_candidate
                and (
                    inputs.top1_observed_at_only
                    or not inputs.top1_temporal_grounded
                    or not inputs.top1_has_web_temporal_textual_marker
                )
            ):
                answerable = False
                insufficient_reasons.append("missing_temporal_grounding")
                answerable_reason = "weak_web_temporal_grounding"
            elif inputs.eval_profile == "candidate-v6":
                if inputs.is_abstention:
                    strict_abstention = (
                        inputs.top1_substantive
                        and inputs.top1_direct
                        and inputs.substantive_count >= 2
                        and inputs.direct_answer_count >= 1
                    )
                    answerable = strict_abstention
                    answerable_reason = (
                        "strict_abstention_threshold_met"
                        if answerable
                        else "strict_abstention_threshold_not_met"
                    )
                elif inputs.is_temporal:
                    direct_temporal = (
                        inputs.top1_substantive
                        and inputs.top1_direct
                        and inputs.temporal_grounded_count >= 1
                    )
                    answerable = direct_temporal
                    answerable_reason = (
                        "substantive_evidence_found"
                        if answerable
                        else "insufficient_for_latest_claim"
                    )
                elif inputs.normalized_requested_source == "vault":
                    answerable = inputs.top1_substantive and (
                        inputs.top1_direct_score >= 0.8
                        or inputs.direct_answer_count >= 1
                        or inputs.memory_provenance_count > 0
                        or inputs.high_trust_count > 0
                    )
                    answerable_reason = (
                        "substantive_evidence_found" if answerable else "weak_support_only"
                    )
                elif inputs.normalized_requested_source in {"", "all"}:
                    # Align with vault: require strong direct overlap or trusted/multi-signal relief — not score == 1.0 only.
                    answerable = inputs.top1_substantive and (
                        inputs.top1_direct_score >= 0.8
                        or inputs.direct_answer_count >= 1
                        or inputs.memory_provenance_count > 0
                        or inputs.high_trust_count > 0
                    )
                    answerable_reason = (
                        "substantive_evidence_found"
                        if answerable
                        else "direct_but_incomplete"
                    )
                else:
                    answerable = inputs.top1_substantive and (
                        inputs.top1_direct or inputs.direct_answer_count >= 1
                    )
                    answerable_reason = (
                        "substantive_evidence_found" if answerable else "weak_support_only"
                    )
            elif inputs.calibration_hardened and inputs.is_abstention:
                strict_abstention = (
                    inputs.top1_substantive
                    and inputs.substantive_count >= 2
                    and not (inputs.is_temporal and inputs.temporal_grounded_count <= 0)
                )
                answerable = strict_abstention
                answerable_reason = (
                    "strict_abstention_threshold_met"
                    if answerable
                    else "strict_abstention_threshold_not_met"
                )
            else:
                answerable_reason = "substantive_evidence_found"

    if (
        inputs.evidence_count > 0
        and not answerable
        and not insufficient_reasons
        and answerable_reason in {
            "no_evidence",
            "substantive_evidence_found",
            "strict_quality_threshold_not_met",
            "direct_but_incomplete",
            "weak_support_only",
            "insufficient_for_latest_claim",
        }
    ):
        insufficient_reasons.append("low_confidence_evidence")
        answerable_reason = "low_confidence_evidence"

    # Independent gates may append the same tag twice (e.g. substantive_count<=0 + paper refusal).
    insufficient_reasons = list(dict.fromkeys(insufficient_reasons))

    return answerable, answerable_reason, insufficient_reasons


__all__ = ["AnswerabilityInputs", "evaluate_answerability"]
