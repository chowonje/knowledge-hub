from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.application.research_review_loop_types import JsonValue


PACK_MARKDOWN_NAME = "research_review_loop_pack.md"
PACK_JSON_NAME = "research_review_loop_pack.json"
PACK_SCHEMA = "knowledge-hub.research-review-loop.pack.v1"


def emit_research_review_loop_pack(
    payload: dict[str, JsonValue],
    *,
    out_dir: str,
) -> dict[str, JsonValue]:
    root = Path(out_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    sidecar = _pack_sidecar(payload)
    markdown_path = root / PACK_MARKDOWN_NAME
    json_path = root / PACK_JSON_NAME
    markdown_path.write_text(_pack_markdown(payload, sidecar), encoding="utf-8")
    json_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "markdownPath": str(markdown_path),
        "jsonPath": str(json_path),
    }


def _pack_sidecar(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    accepted = _claims_with_state(payload, "accepted", canonical_only=True)
    rejected = _claims_with_state(payload, "rejected", canonical_only=False)
    unsure = _claims_with_state(payload, "unsure", canonical_only=False)
    return {
        "schema": PACK_SCHEMA,
        "packId": _context_pack_value(payload, "packId"),
        "sourceScope": payload.get("sourceScope") or {},
        "canonicalWriteAllowed": False,
        "authoritativeAssertCount": len(accepted),
        "reviewedButNoAuthoritativeAssertions": _reviewed_but_no_authoritative_assertions(payload, accepted),
        "memoryProjectionPolicy": _context_pack_dict(payload, "memoryProjectionPolicy"),
        "acceptedClaimIds": [str(item.get("claimId") or "") for item in accepted],
        "rejectedClaimIds": [str(item.get("claimId") or "") for item in rejected],
        "unsureClaimIds": [str(item.get("claimId") or "") for item in unsure],
        "weakConceptIds": [str(item.get("conceptId") or "") for item in _review_backed_rows(payload, "weakConcepts")],
        "openQuestionIds": [str(item.get("questionId") or "") for item in _review_backed_rows(payload, "openQuestions")],
        "acceptedClaims": accepted,
        "rejectedClaims": rejected,
        "unsureClaims": unsure,
        "openQuestions": _review_backed_rows(payload, "openQuestions"),
    }


def _pack_markdown(payload: dict[str, JsonValue], sidecar: dict[str, JsonValue]) -> str:
    lines = [
        f"# Context Pack: {', '.join(_source_ids(payload))}",
        f"packId: {sidecar['packId']}",
        "canonicalWriteAllowed: false",
        "",
        "## ACCEPTED CLAIMS",
        *_claim_lines(payload, _claims_with_state(payload, "accepted", canonical_only=True), prefix="ASSERT"),
        "",
        "## REJECTED CLAIMS - DO NOT ASSERT",
        *_claim_lines(payload, _claims_with_state(payload, "rejected", canonical_only=False), prefix="DO NOT ASSERT"),
        "",
        "## UNSURE",
        *_claim_lines(payload, _claims_with_state(payload, "unsure", canonical_only=False), prefix="UNCERTAIN"),
        "",
        "## OPEN QUESTIONS",
        *_open_question_lines(payload),
        "",
    ]
    return "\n".join(lines)


def _claim_lines(
    payload: dict[str, JsonValue],
    claims: list[dict[str, JsonValue]],
    *,
    prefix: str,
) -> list[str]:
    if not claims:
        return ["- none"]
    decisions = _decisions_by_id(payload)
    spans = _spans_by_id(payload)
    lines: list[str] = []
    for claim in claims:
        decision = decisions.get(str(claim.get("reviewDecisionId") or ""), {})
        evidence = _evidence_summary(claim, spans)
        reason = _one_line(str(decision.get("reason") or ""))
        lines.append(f"- [{claim.get('claimId')}] {prefix}: {_one_line(str(claim.get('claimText') or ''))}")
        if evidence:
            lines.append(f"  evidence: {evidence}")
        if reason:
            lines.append(f"  reason: {reason}")
    return lines


def _open_question_lines(payload: dict[str, JsonValue]) -> list[str]:
    questions = _review_backed_rows(payload, "openQuestions")
    if not questions:
        return ["- none"]
    return [
        f"- [{item.get('questionId')}] {_one_line(str(item.get('questionText') or ''))}"
        for item in questions
    ]


def _claims_with_state(
    payload: dict[str, JsonValue],
    state: str,
    *,
    canonical_only: bool,
) -> list[dict[str, JsonValue]]:
    claims = payload.get("proposedClaims")
    if not isinstance(claims, list):
        return []
    result: list[dict[str, JsonValue]] = []
    for item in claims:
        if not isinstance(item, dict):
            continue
        if str(item.get("state") or "") != state:
            continue
        if not str(item.get("reviewDecisionId") or ""):
            continue
        if canonical_only and not bool(item.get("canonicalEligible")):
            continue
        result.append(item)
    return result


def _evidence_summary(claim: dict[str, JsonValue], spans: dict[str, dict[str, JsonValue]]) -> str:
    summaries: list[str] = []
    span_ids = claim.get("evidenceSpanIds")
    if not isinstance(span_ids, list):
        return ""
    for span_id in span_ids:
        span = spans.get(str(span_id), {})
        locator = _one_line(str(span.get("locator") or ""))
        if not locator:
            continue
        snippet_hash = _one_line(str(span.get("snippetHash") or ""))
        preview = _one_line(str(span.get("textPreview") or ""))
        summaries.append(f"{locator} [{snippet_hash}] {preview}".strip())
    return "; ".join(summaries)


def _decisions_by_id(payload: dict[str, JsonValue]) -> dict[str, dict[str, JsonValue]]:
    rows = payload.get("reviewDecisions")
    if not isinstance(rows, list):
        return {}
    return {str(item.get("decisionId") or ""): item for item in rows if isinstance(item, dict)}


def _spans_by_id(payload: dict[str, JsonValue]) -> dict[str, dict[str, JsonValue]]:
    rows = payload.get("evidenceSpans")
    if not isinstance(rows, list):
        return {}
    return {str(item.get("evidenceSpanId") or ""): item for item in rows if isinstance(item, dict)}


def _source_ids(payload: dict[str, JsonValue]) -> list[str]:
    scope = payload.get("sourceScope")
    if not isinstance(scope, dict):
        return ["unknown"]
    values = scope.get("explicitSourceIds")
    if not isinstance(values, list):
        return ["unknown"]
    return [str(item) for item in values if str(item).strip()] or ["unknown"]


def _context_pack_value(payload: dict[str, JsonValue], key: str) -> str:
    context_pack = payload.get("contextPackPreview")
    if not isinstance(context_pack, dict):
        return ""
    return str(context_pack.get(key) or "")


def _context_pack_dict(payload: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    context_pack = payload.get("contextPackPreview")
    if not isinstance(context_pack, dict):
        return {}
    value = context_pack.get(key)
    if not isinstance(value, dict):
        return {}
    return value


def _reviewed_but_no_authoritative_assertions(
    payload: dict[str, JsonValue],
    accepted: list[dict[str, JsonValue]],
) -> bool:
    context_pack = payload.get("contextPackPreview")
    if isinstance(context_pack, dict):
        value = context_pack.get("reviewedButNoAuthoritativeAssertions")
        if isinstance(value, bool):
            return value
    return bool(payload.get("reviewDecisions")) and not accepted


def _review_backed_rows(payload: dict[str, JsonValue], key: str) -> list[dict[str, JsonValue]]:
    values = payload.get(key)
    if not isinstance(values, list):
        return []
    rows: list[dict[str, JsonValue]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        decision_ids = item.get("supportingDecisionIds")
        if not isinstance(decision_ids, list) or not decision_ids:
            continue
        rows.append(item)
    return rows


def _one_line(value: str) -> str:
    return " ".join(value.split())
