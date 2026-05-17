"""Report-only human review pack for SectionSpan contract candidates.

This helper converts the non-strict SectionSpan contract review into
human-reviewable cards.  It is a review artifact only: it does not create
strict evidence, runtime citations, parser routing, answer integration, DB
changes, index changes, or canonical parsed artifact writes.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_CONTRACT_REVIEW_PACK_SCHEMA_ID = "knowledge-hub.paper.sectionspan-contract-review-pack.v1"
SECTIONSPAN_CONTRACT_REVIEW_SCHEMA_ID = "knowledge-hub.paper.sectionspan-contract-review.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _schema_violations(contract_review: dict[str, Any]) -> list[str]:
    if contract_review.get("schema") != SECTIONSPAN_CONTRACT_REVIEW_SCHEMA_ID:
        return ["sectionspan_contract_review_schema_mismatch"]
    return []


def _unsafe_flags(contract_review: dict[str, Any]) -> list[str]:
    unsafe: list[str] = []
    counts = dict(contract_review.get("counts") or {})
    gate = dict(contract_review.get("gate") or {})
    policy = dict(contract_review.get("policy") or {})
    if contract_review.get("status") != "contract_review_ready":
        unsafe.append("sectionspan_contract_review_not_ready")
    for key in ("strictEligibleCandidates", "citationGradeCandidates", "runtimeEvidenceCandidates"):
        if _safe_int(counts.get(key)) > 0:
            unsafe.append(f"{key}_nonzero")
    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(gate.get(key)):
            unsafe.append(f"{key}_true")
    for key in (
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(policy.get(key)):
            unsafe.append(f"{key}_true")
    return list(dict.fromkeys(unsafe))


def _priority(section_type: str) -> str:
    if section_type == "numbered_section":
        return "approve_candidate_contract"
    if section_type == "abstract":
        return "approve_frontmatter_candidate_contract"
    if section_type == "backmatter":
        return "approve_low_priority_backmatter_candidate_contract"
    return "review_named_section_candidate_contract"


def _questions(section_type: str) -> list[str]:
    common = [
        "Does this row identify a real section boundary rather than layout or navigation noise?",
        "Is this candidate safe to keep as non-strict source-aligned structure?",
        "What extra authority would be required before runtime citation use?",
    ]
    if section_type == "abstract":
        return [
            "Should this frontmatter boundary be kept as a SectionSpan candidate?",
            *common,
        ]
    if section_type == "backmatter":
        return [
            "Should this backmatter boundary be retained for navigation and exclusion logic only?",
            *common,
        ]
    return common


def _card(index: int, candidate: dict[str, Any]) -> dict[str, Any]:
    section_type = str(candidate.get("section_type") or "")
    strict_blockers = list(
        dict.fromkeys(
            [
                "sectionspan_contract_review_pack_only",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_explicit_later_tranche",
                "parser_routing_requires_explicit_later_tranche",
                "answer_integration_requires_explicit_later_tranche",
                *[str(item) for item in list(candidate.get("strict_blockers") or [])],
            ]
        )
    )
    return {
        "card_id": f"sectionspan-contract-review-card:{index:04d}",
        "review_status": "needs_human_review",
        "recommended_action": _priority(section_type),
        "source_contract_candidate_id": str(candidate.get("contract_candidate_id") or ""),
        "source_sectionspan_candidate_id": str(candidate.get("source_candidate_id") or ""),
        "candidate_type": "sectionspan_contract_review_card",
        "paper_id": str(candidate.get("paper_id") or ""),
        "source_parser": str(candidate.get("source_parser") or ""),
        "candidate_text": _clean_text(candidate.get("candidate_text")),
        "section_label": _clean_text(candidate.get("section_label")),
        "section_title": _clean_text(candidate.get("section_title")),
        "section_type": section_type,
        "section_level": _safe_int(candidate.get("section_level")),
        "canonical_span": {
            "chars_start": _safe_int(candidate.get("chars_start")),
            "chars_end": _safe_int(candidate.get("chars_end")),
            "page": _safe_int(candidate.get("page")),
            "sourceContentHash": str(candidate.get("sourceContentHash") or ""),
            "alignmentMethod": str(candidate.get("alignment_method") or ""),
            "alignmentStatus": str(candidate.get("canonical_alignment_status") or ""),
            "confidence": float(candidate.get("confidence") or 0.0),
            "locatorKind": "canonical_generated_markdown",
        },
        "review_questions": _questions(section_type),
        "contract_assessment": {
            "contractStatus": str(candidate.get("contract_status") or ""),
            "contractReady": bool(candidate.get("contract_ready")),
            "contractBlockers": list(candidate.get("contract_blockers") or []),
            "contractRequirements": list(candidate.get("contract_requirements") or []),
        },
        "evidence_tier": "sectionspan_contract_review_card_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": [
            "human_review_pack_is_report_only",
            "sectionspan_contract_cards_are_not_runtime_evidence",
            "later_explicit_promotion_tranche_required",
        ],
    }


def _held_out_rows(contract_review: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(contract_review.get("heldOut") or []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "sourceCandidateId": str(item.get("sourceCandidateId") or ""),
                "paperId": str(item.get("paperId") or ""),
                "candidateText": _clean_text(item.get("candidateText")),
                "reviewClass": str(item.get("reviewClass") or ""),
                "reason": str(item.get("reason") or "held_out_upstream"),
                "reviewStatus": "held_out",
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
            }
        )
    return rows


def _counts(cards: list[dict[str, Any]], held_out: list[dict[str, Any]], violations: list[str], input_count: int) -> dict[str, Any]:
    return {
        "inputContractCandidates": input_count,
        "reviewCardCount": len(cards),
        "heldOutCandidates": len(held_out),
        "strictEligibleCards": 0,
        "citationGradeCards": 0,
        "runtimeEvidenceCards": 0,
        "schemaViolationCount": len([item for item in violations if item.endswith("_mismatch")]),
        "unsafeUpstreamFlagCount": len([item for item in violations if not item.endswith("_mismatch")]),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in cards)),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in cards)),
        "byRecommendedAction": dict(Counter(str(item.get("recommended_action") or "") for item in cards)),
        "heldOutByReason": dict(Counter(str(item.get("reason") or "") for item in held_out)),
    }


def build_sectionspan_contract_review_pack(
    *,
    sectionspan_contract_review_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only human review pack from a SectionSpan contract review."""

    report_path = Path(str(sectionspan_contract_review_report)).expanduser()
    contract_review = _read_json(report_path)
    violations = [*_schema_violations(contract_review), *_unsafe_flags(contract_review)]
    raw_candidates = [
        dict(item)
        for item in list(contract_review.get("contractCandidates") or [])
        if isinstance(item, dict)
    ]
    ready_candidates = [item for item in raw_candidates if item.get("contract_status") == "contract_ready_non_strict"]
    ready_candidates.sort(
        key=lambda item: (
            str(item.get("paper_id") or ""),
            _safe_int(item.get("page")),
            _safe_int(item.get("chars_start")),
            str(item.get("contract_candidate_id") or ""),
        )
    )
    cards = [_card(index, candidate) for index, candidate in enumerate(ready_candidates, start=1)]
    held_out = _held_out_rows(contract_review)
    counts = _counts(cards, held_out, violations, len(raw_candidates))
    ready = not violations and counts["reviewCardCount"] > 0
    return {
        "schema": SECTIONSPAN_CONTRACT_REVIEW_PACK_SCHEMA_ID,
        "status": "review_pack_ready" if ready else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "sectionspanContractReviewReport": str(report_path),
            "sectionspanContractReviewSchema": str(contract_review.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "reviewPackReady": ready,
            "candidateFormalizationReady": ready,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "sectionspan_contract_human_review_ready" if ready else "blocked",
            "schemaViolations": [item for item in violations if item.endswith("_mismatch")],
            "unsafeUpstreamFlags": [item for item in violations if not item.endswith("_mismatch")],
            "recommendedNextTranche": "sectionspan_strict_promotion_design_requires_approval",
        },
        "policy": {
            "reportOnly": True,
            "allCardsNonStrict": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "reviewGuidance": {
            "primaryDecision": "review_sectionspan_contract_cards_before_any_strict_promotion_design",
            "recommendedFirstArtifact": "SectionSpan",
            "doNotPromoteYet": [
                "runtime_answer_citations",
                "strict_evidence_spans",
                "parser_routing",
                "canonical_parsed_artifact_writes",
            ],
        },
        "warnings": [
            "review_cards_are_not_strict_evidence",
            "source_hash_page_and_chars_do_not_make_runtime_citations",
            "generated_markdown_offsets_are_not_original_pdf_offsets",
            "strict_promotion_requires_explicit_later_tranche",
        ],
        "reviewCards": cards,
        "heldOut": held_out,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "inputs",
            "counts",
            "gate",
            "policy",
            "reviewGuidance",
            "warnings",
            "heldOut",
        )
        if key in report
    }


def render_sectionspan_contract_review_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan Contract Human Review Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Review cards: `{int(counts.get('reviewCardCount') or 0)}`",
        f"- Held out: `{int(counts.get('heldOutCandidates') or 0)}`",
        f"- Strict eligible cards: `{int(counts.get('strictEligibleCards') or 0)}`",
        f"- Runtime evidence cards: `{int(counts.get('runtimeEvidenceCards') or 0)}`",
        "",
        "## Boundary",
        "",
        "All cards are `sectionspan_contract_review_card_only`. This pack does not create strict evidence, runtime citations, parser routing, answer integration, canonical parsed artifacts, SQLite writes, indexes, or embeddings.",
        "",
        "## Counts",
        "",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By recommended action: `{json.dumps(counts.get('byRecommendedAction') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Held out by reason: `{json.dumps(counts.get('heldOutByReason') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Review Cards",
        "",
    ]
    for item in list(report.get("reviewCards") or []):
        span = dict(item.get("canonical_span") or {})
        lines.extend(
            [
                f"### `{item.get('card_id', '')}`",
                "",
                f"- Paper: `{item.get('paper_id', '')}`",
                f"- Page: `{span.get('page')}`",
                f"- Section: `{item.get('section_label', '')}` {item.get('section_title', '')}",
                f"- Type: `{item.get('section_type', '')}`",
                f"- Recommended action: `{item.get('recommended_action', '')}`",
                f"- Evidence tier: `{item.get('evidence_tier', '')}`",
                "",
            ]
        )
    lines.extend(["", "## Held Out", ""])
    for item in list(report.get("heldOut") or []):
        lines.append(
            f"- `{item.get('paperId')}` `{item.get('reviewClass')}` {item.get('candidateText')} -> `{item.get('reason')}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_sectionspan_contract_review_pack_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    cards_path = root / "sectionspan-contract-review-cards.json"
    summary_path = root / "sectionspan-contract-review-summary.json"
    markdown_path = root / "sectionspan-contract-review-pack.md"
    cards_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_contract_review_pack_markdown(report), encoding="utf-8")
    return {"cards": str(cards_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only SectionSpan contract human review pack.")
    parser.add_argument("--sectionspan-contract-review-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_contract_review_pack(
        sectionspan_contract_review_report=args.sectionspan_contract_review_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_contract_review_pack_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_CONTRACT_REVIEW_PACK_SCHEMA_ID",
    "build_sectionspan_contract_review_pack",
    "render_sectionspan_contract_review_pack_markdown",
    "write_sectionspan_contract_review_pack_reports",
]
