from __future__ import annotations

import re
from typing import Any


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _evidence_text(evidence: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in evidence or []:
        parts.extend(
            [
                _clean_text(item.get("title")),
                _clean_text(item.get("section_path")),
                _clean_text(item.get("excerpt")),
            ]
        )
    return " ".join(part for part in parts if part).casefold()


def _load_claim_normalizations(sqlite_db: Any, claim_ids: list[str]) -> dict[str, dict[str, Any]]:
    list_claim_normalizations = getattr(sqlite_db, "list_claim_normalizations", None)
    if not callable(list_claim_normalizations) or not claim_ids:
        return {}
    try:
        rows = list_claim_normalizations(
            claim_ids=claim_ids,
            status="normalized",
            limit=max(20, len(claim_ids) * 2),
        )
    except Exception:
        return {}
    normalizations: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        claim_id = _clean_text(row.get("claim_id"))
        if claim_id and claim_id not in normalizations:
            normalizations[claim_id] = dict(row)
    return normalizations


def adjudicate_claims(
    sqlite_db: Any,
    *,
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    limit: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not sqlite_db or not claims:
        return [], {
            "supportCount": 0,
            "conflictCount": 0,
            "weakClaimCount": 0,
            "unsupportedClaimCount": 0,
            "claimVerificationSummary": "unavailable" if claims else "empty",
            "conflicts": [],
        }

    selected_claims = [dict(item) for item in claims[: max(1, int(limit))]]
    claim_ids = [_clean_text(item.get("claim_id")) for item in selected_claims if _clean_text(item.get("claim_id"))]
    normalizations = _load_claim_normalizations(sqlite_db, claim_ids)
    full_text = _evidence_text(evidence)
    items: list[dict[str, Any]] = []
    support_count = 0
    weak_count = 0
    unsupported_count = 0
    conflict_count = 0
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for claim in selected_claims:
        claim_id = _clean_text(claim.get("claim_id"))
        claim_text = _clean_text(claim.get("claim_text"))
        normalization = normalizations.get(claim_id, {})
        evidence_strength = _clean_text(normalization.get("evidence_strength") or "weak").lower()
        result_direction = _clean_text(normalization.get("result_direction") or "unknown").lower()
        numeric_value = normalization.get("result_value_numeric")
        numeric_token = ""
        if numeric_value is not None:
            try:
                numeric_float = float(numeric_value)
                numeric_token = str(int(numeric_float)) if numeric_float.is_integer() else str(numeric_float)
            except Exception:
                numeric_token = _clean_text(numeric_value)
        expected_terms = [
            _clean_text(normalization.get("task")),
            _clean_text(normalization.get("dataset")),
            _clean_text(normalization.get("metric")),
            _clean_text(normalization.get("comparator")),
            _clean_text(normalization.get("result_value_text")),
            _clean_text(normalization.get("condition_text")),
            _clean_text(normalization.get("scope_text")),
            _clean_text(normalization.get("limitation_text")),
        ]
        if numeric_token:
            expected_terms.append(numeric_token)
        expected_terms = [term for term in expected_terms if term]
        matched_terms = [term for term in expected_terms if term.casefold() in full_text]
        matched_titles = [
            _clean_text(item.get("title"))
            for item in evidence or []
            if any(
                term.casefold() in " ".join(
                    [
                        _clean_text(item.get("title")),
                        _clean_text(item.get("section_path")),
                        _clean_text(item.get("excerpt")),
                    ]
                ).casefold()
                for term in matched_terms[:3]
            )
        ]
        matched_titles = [title for title in matched_titles if title][:3]
        evidence_numbers = {token.rstrip("%") for token in re.findall(r"\d+(?:\.\d+)?%?", full_text)}
        numeric_conflict = bool(numeric_token and evidence_numbers and numeric_token.rstrip("%") not in evidence_numbers)
        reasons: list[str] = []
        verdict = "supported"

        if not evidence:
            verdict = "unsupported"
            reasons.append("no_evidence")
        elif numeric_conflict:
            verdict = "contradicted"
            reasons.append("numeric_value_conflict")
        elif evidence_strength == "weak":
            verdict = "weakly_supported"
            reasons.append("weak_evidence_strength")
        elif expected_terms and len(matched_terms) < max(1, min(2, len(expected_terms))):
            verdict = "weakly_supported"
            reasons.append("expected_terms_under_grounded")
        elif claim_text and claim_text.casefold() not in full_text and not matched_terms:
            verdict = "weakly_supported"
            reasons.append("claim_text_not_directly_grounded")

        if result_direction == "better" and any(token in full_text for token in ("worse", "lower", "decrease", "감소", "악화")):
            verdict = "contradicted"
            reasons.append("direction_conflict")
        if result_direction == "worse" and any(token in full_text for token in ("improves", "better", "higher", "향상", "개선")):
            verdict = "contradicted"
            reasons.append("direction_conflict")

        if verdict == "supported":
            support_count += 1
        elif verdict == "weakly_supported":
            weak_count += 1
        elif verdict == "contradicted":
            conflict_count += 1
        else:
            unsupported_count += 1

        key = (
            _clean_text(normalization.get("dataset")),
            _clean_text(normalization.get("metric")),
            _clean_text(normalization.get("comparator")),
        )
        if any(key):
            grouped.setdefault(key, []).append({"claim_id": claim_id, "normalization": normalization})
        items.append(
            {
                "claimId": claim_id,
                "claimText": claim_text,
                "status": verdict,
                "verdict": verdict,
                "matchedEvidenceCount": len(matched_titles),
                "matchedTitles": matched_titles,
                "matchedTerms": matched_terms,
                "reasons": reasons,
                "normalized": {
                    "task": _clean_text(normalization.get("task")),
                    "dataset": _clean_text(normalization.get("dataset")),
                    "metric": _clean_text(normalization.get("metric")),
                    "comparator": _clean_text(normalization.get("comparator")),
                    "resultDirection": _clean_text(normalization.get("result_direction")),
                    "resultValueText": _clean_text(normalization.get("result_value_text")),
                    "resultValueNumeric": normalization.get("result_value_numeric"),
                    "conditionText": _clean_text(normalization.get("condition_text")),
                    "scopeText": _clean_text(normalization.get("scope_text")),
                    "limitationText": _clean_text(normalization.get("limitation_text")),
                    "evidenceStrength": evidence_strength,
                },
            }
        )

    consensus_conflicts: list[dict[str, Any]] = []
    for key, group in grouped.items():
        directions = {
            _clean_text(item["normalization"].get("result_direction"))
            for item in group
            if _clean_text(item["normalization"].get("result_direction")) not in {"", "unknown"}
        }
        numeric_values = {
            item["normalization"].get("result_value_numeric")
            for item in group
            if item["normalization"].get("result_value_numeric") is not None
        }
        if len(directions) > 1 or len(numeric_values) > 1:
            conflict_count += 1
            consensus_conflicts.append(
                {
                    "dataset": key[0],
                    "metric": key[1],
                    "comparator": key[2],
                    "claimIds": [_clean_text(item["claim_id"]) for item in group],
                    "reason": "normalized_claim_conflict",
                }
            )

    summary = {
        "supportCount": support_count,
        "conflictCount": conflict_count,
        "weakClaimCount": weak_count,
        "unsupportedClaimCount": unsupported_count,
        "claimVerificationSummary": "conflicted" if conflict_count else ("weak" if weak_count or unsupported_count else "supported"),
        "conflicts": consensus_conflicts,
    }
    return items, summary


def render_claim_adjudication_context(
    claim_verification: list[dict[str, Any]],
    consensus: dict[str, Any],
    *,
    limit: int = 5,
) -> str:
    if not claim_verification:
        return ""
    lines = [
        "Claim adjudication summary:",
        f"- summary={_clean_text(consensus.get('claimVerificationSummary') or 'unknown')}",
        f"- support={int(consensus.get('supportCount') or 0)}",
        f"- weak={int(consensus.get('weakClaimCount') or 0)}",
        f"- unsupported={int(consensus.get('unsupportedClaimCount') or 0)}",
        f"- conflicts={int(consensus.get('conflictCount') or 0)}",
    ]
    for item in claim_verification[: max(1, int(limit))]:
        normalized = dict(item.get("normalized") or {})
        lines.append(
            "- "
            f"{_clean_text(item.get('claimText') or item.get('claimId'))} "
            f"[{_clean_text(item.get('status') or 'unknown')}] "
            f"dataset={_clean_text(normalized.get('dataset')) or '-'} "
            f"metric={_clean_text(normalized.get('metric')) or '-'} "
            f"value={_clean_text(normalized.get('resultValueText')) or normalized.get('resultValueNumeric') or '-'}"
        )
    return "\n".join(lines).strip()


__all__ = ["adjudicate_claims", "render_claim_adjudication_context"]
