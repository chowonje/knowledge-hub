"""Report-only SectionSpanCandidate contract review.

This helper checks whether the current non-strict SectionSpanCandidate layer is
coherent enough for candidate-contract review.  It deliberately does not create
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


SECTIONSPAN_CONTRACT_REVIEW_SCHEMA_ID = "knowledge-hub.paper.sectionspan-contract-review.v1"
SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID = "knowledge-hub.paper.sectionspan-candidate-report.v1"
CANDIDATE_LAYER_PROMOTION_POLICY_DRAFT_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-promotion-policy-draft.v1"
)

_REQUIRED_CANDIDATE_FIELDS = (
    "candidate_id",
    "candidate_type",
    "paper_id",
    "source_parser",
    "candidate_text",
    "section_label",
    "section_title",
    "section_type",
    "section_level",
    "canonical_alignment_status",
    "alignment_method",
    "chars_start",
    "chars_end",
    "page",
    "sourceContentHash",
    "confidence",
    "source_span_locator",
    "evidence_tier",
    "strict_eligible",
    "citation_grade",
    "strict_blockers",
    "non_strict_reason",
)
_ALLOWED_SECTION_TYPES = {"numbered_section", "abstract", "backmatter", "named_section"}


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


def _safe_bool(value: Any) -> bool:
    return bool(value) is True


def _schema_violations(sectionspan: dict[str, Any], promotion: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if sectionspan.get("schema") != SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID:
        violations.append("sectionspan_candidate_schema_mismatch")
    if promotion.get("schema") != CANDIDATE_LAYER_PROMOTION_POLICY_DRAFT_SCHEMA_ID:
        violations.append("promotion_policy_draft_schema_mismatch")
    return violations


def _unsafe_flags(sectionspan: dict[str, Any], promotion: dict[str, Any]) -> list[str]:
    unsafe: list[str] = []
    for name, payload in (("sectionspan", sectionspan), ("promotionPolicy", promotion)):
        counts = dict(payload.get("counts") or {})
        policy = dict(payload.get("policy") or {})
        gate = dict(payload.get("gate") or {})
        for key in (
            "strictEligibleCandidates",
            "citationGradeCandidates",
            "runtimeEvidenceCandidates",
            "strictPromotionReadyTracks",
            "parserRoutingReadyTracks",
            "answerIntegrationReadyTracks",
            "runtimePromotionAllowedTracks",
        ):
            if _safe_int(counts.get(key)) > 0:
                unsafe.append(f"{name}_{key}_nonzero")
        for key in (
            "strictEvidenceCreated",
            "runtimePromotionAllowed",
            "parserRoutingChanged",
            "canonicalParsedArtifactsWritten",
            "databaseMutation",
            "reindexOrReembed",
            "answerIntegrationChanged",
            "runtimePolicyChanged",
        ):
            if _safe_bool(policy.get(key)):
                unsafe.append(f"{name}_{key}_true")
        for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady"):
            if _safe_bool(gate.get(key)):
                unsafe.append(f"{name}_{key}_true")
    if str(promotion.get("status") or "") != "draft_ready":
        unsafe.append("promotion_policy_draft_not_ready")
    return list(dict.fromkeys(unsafe))


def _section_track(promotion: dict[str, Any]) -> dict[str, Any]:
    for track in list(promotion.get("promotionTracks") or []):
        if isinstance(track, dict) and track.get("layer") == "sectionspan":
            return track
    return {}


def _track_violations(track: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if not track:
        return ["sectionspan_promotion_track_missing"]
    if track.get("promotion_readiness") != "candidate_formalization_ready_non_strict":
        violations.append("sectionspan_track_not_formalization_ready")
    if not bool(track.get("candidate_layer_formalization_ready")):
        violations.append("sectionspan_track_candidate_formalization_not_ready")
    for key in ("strict_promotion_ready", "parser_routing_ready", "answer_integration_ready", "runtime_evidence"):
        if bool(track.get(key)):
            violations.append(f"sectionspan_track_{key}_true")
    return violations


def _candidate_blockers(candidate: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for field in _REQUIRED_CANDIDATE_FIELDS:
        if field not in candidate:
            blockers.append(f"missing_{field}")
    if candidate.get("candidate_type") != "section_span_candidate":
        blockers.append("candidate_type_not_section_span_candidate")
    if candidate.get("source_parser") != "mineru+pymupdf_alignment":
        blockers.append("source_parser_not_mineru_pymupdf_alignment")
    if _clean_text(candidate.get("candidate_text")) == "":
        blockers.append("missing_candidate_text")
    if _clean_text(candidate.get("section_title")) == "":
        blockers.append("missing_section_title")
    if str(candidate.get("section_type") or "") not in _ALLOWED_SECTION_TYPES:
        blockers.append("unsupported_section_type")
    if candidate.get("canonical_alignment_status") != "aligned":
        blockers.append("canonical_alignment_not_aligned")
    if candidate.get("alignment_method") != "exact":
        blockers.append("alignment_method_not_exact")
    if _safe_int(candidate.get("chars_start")) < 0 or _safe_int(candidate.get("chars_end")) <= _safe_int(candidate.get("chars_start")):
        blockers.append("invalid_chars_start_end")
    if _safe_int(candidate.get("page")) < 1:
        blockers.append("invalid_or_missing_page")
    if _clean_text(candidate.get("sourceContentHash")) == "":
        blockers.append("missing_source_content_hash")
    if candidate.get("evidence_tier") != "sectionspan_candidate_only":
        blockers.append("unexpected_evidence_tier")
    if bool(candidate.get("strict_eligible")):
        blockers.append("strict_eligible_true")
    if bool(candidate.get("citation_grade")):
        blockers.append("citation_grade_true")
    if "runtime_promotion_disabled_for_tranche" not in list(candidate.get("strict_blockers") or []):
        blockers.append("runtime_promotion_blocker_missing")
    if "sectionspan_candidate_layer_not_runtime_evidence" not in list(candidate.get("strict_blockers") or []):
        blockers.append("candidate_layer_runtime_blocker_missing")
    return list(dict.fromkeys(blockers))


def _contract_row(index: int, candidate: dict[str, Any], blockers: list[str]) -> dict[str, Any]:
    section_type = str(candidate.get("section_type") or "")
    contract_ready = not blockers and section_type in _ALLOWED_SECTION_TYPES
    strict_blockers = list(
        dict.fromkeys(
            [
                "sectionspan_contract_review_only",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_explicit_later_tranche",
                "parser_routing_requires_explicit_later_tranche",
                "answer_integration_requires_explicit_later_tranche",
                "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
                *[str(item) for item in list(candidate.get("strict_blockers") or [])],
            ]
        )
    )
    return {
        "contract_candidate_id": f"sectionspan-contract:{candidate.get('paper_id')}:{index:04d}",
        "source_candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_type": "sectionspan_contract_candidate",
        "paper_id": str(candidate.get("paper_id") or ""),
        "source_parser": str(candidate.get("source_parser") or ""),
        "candidate_text": _clean_text(candidate.get("candidate_text")),
        "section_label": _clean_text(candidate.get("section_label")),
        "section_title": _clean_text(candidate.get("section_title")),
        "section_type": section_type,
        "section_level": _safe_int(candidate.get("section_level")),
        "canonical_alignment_status": str(candidate.get("canonical_alignment_status") or ""),
        "alignment_method": str(candidate.get("alignment_method") or ""),
        "chars_start": _safe_int(candidate.get("chars_start")),
        "chars_end": _safe_int(candidate.get("chars_end")),
        "page": _safe_int(candidate.get("page")),
        "sourceContentHash": str(candidate.get("sourceContentHash") or ""),
        "confidence": float(candidate.get("confidence") or 0.0),
        "contract_status": "contract_ready_non_strict" if contract_ready else "blocked",
        "contract_ready": contract_ready,
        "contract_blockers": blockers,
        "contract_requirements": [
            "exact_alignment_to_canonical_generated_text",
            "page_recovered_from_canonical_parsed_artifact",
            "source_content_hash_present",
            "candidate_only_until_later_explicit_strict_promotion",
        ],
        "allowed_next_actions": [
            "human_operator_review",
            "report_only_contract_refinement",
            "later_explicit_sectionspan_strict_promotion_design",
        ],
        "disallowed_actions": [
            "strict_evidence_promotion",
            "runtime_answer_citation",
            "parser_routing",
            "canonical_parsed_artifact_write",
            "database_mutation",
            "reindex_or_reembed",
        ],
        "evidence_tier": "sectionspan_contract_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": [
            "contract_review_is_report_only",
            "sectionspan_candidates_are_not_runtime_evidence",
            "later_explicit_promotion_tranche_required",
        ],
    }


def _held_out_rows(sectionspan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(sectionspan.get("heldOut") or []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "sourceCandidateId": str(item.get("sourceCandidateId") or ""),
                "paperId": str(item.get("paperId") or ""),
                "candidateText": _clean_text(item.get("candidateText")),
                "reviewClass": str(item.get("reviewClass") or ""),
                "reason": str(item.get("reason") or "held_out_upstream"),
                "contractStatus": "held_out",
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
            }
        )
    return rows


def _counts(
    rows: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
    violations: list[str],
    *,
    input_candidates: int,
) -> dict[str, Any]:
    return {
        "inputSectionSpanCandidates": input_candidates,
        "contractCandidateCount": len(rows),
        "contractReadyCandidates": sum(1 for item in rows if item.get("contract_ready")),
        "blockedContractCandidates": sum(1 for item in rows if not item.get("contract_ready")),
        "heldOutCandidates": len(held_out),
        "strictEligibleCandidates": 0,
        "citationGradeCandidates": 0,
        "runtimeEvidenceCandidates": 0,
        "schemaViolationCount": len([item for item in violations if item.endswith("_mismatch")]),
        "unsafeUpstreamFlagCount": len([item for item in violations if not item.endswith("_mismatch")]),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in rows)),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in rows)),
        "byContractStatus": dict(Counter(str(item.get("contract_status") or "") for item in rows)),
        "heldOutByReason": dict(Counter(str(item.get("reason") or "") for item in held_out)),
    }


def build_sectionspan_contract_review(
    *,
    sectionspan_candidate_report: str | Path,
    candidate_layer_promotion_policy_draft_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only SectionSpanCandidate contract review payload."""

    paths = {
        "sectionspanCandidateReport": Path(str(sectionspan_candidate_report)).expanduser(),
        "candidateLayerPromotionPolicyDraftReport": Path(
            str(candidate_layer_promotion_policy_draft_report)
        ).expanduser(),
    }
    sectionspan = _read_json(paths["sectionspanCandidateReport"])
    promotion = _read_json(paths["candidateLayerPromotionPolicyDraftReport"])
    violations = [
        *_schema_violations(sectionspan, promotion),
        *_unsafe_flags(sectionspan, promotion),
        *_track_violations(_section_track(promotion)),
    ]
    candidates = [dict(item) for item in list(sectionspan.get("candidates") or []) if isinstance(item, dict)]
    rows = [
        _contract_row(index, candidate, _candidate_blockers(candidate))
        for index, candidate in enumerate(candidates, start=1)
    ]
    held_out = _held_out_rows(sectionspan)
    counts = _counts(rows, held_out, violations, input_candidates=len(candidates))
    ready = not violations and counts["blockedContractCandidates"] == 0 and counts["contractReadyCandidates"] > 0
    return {
        "schema": SECTIONSPAN_CONTRACT_REVIEW_SCHEMA_ID,
        "status": "contract_review_ready" if ready else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "sectionspanCandidateReport": str(paths["sectionspanCandidateReport"]),
            "candidateLayerPromotionPolicyDraftReport": str(paths["candidateLayerPromotionPolicyDraftReport"]),
            "sectionspanCandidateReportSchema": str(sectionspan.get("schema") or ""),
            "candidateLayerPromotionPolicyDraftSchema": str(promotion.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "sectionspanContractReviewReady": ready,
            "candidateFormalizationReady": ready,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "sectionspan_contract_review_ready_non_strict" if ready else "blocked",
            "schemaViolations": [item for item in violations if item.endswith("_mismatch")],
            "unsafeUpstreamFlags": [item for item in violations if not item.endswith("_mismatch")],
            "recommendedNextTranche": "sectionspan_candidate_contract_human_review",
        },
        "policy": {
            "reportOnly": True,
            "allCandidatesNonStrict": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "contractPrinciples": [
            "sectionspan_contract_rows_are_candidate_only",
            "source_hash_page_and_chars_do_not_create_strict_evidence",
            "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
            "runtime_use_requires_later_explicit_strict_promotion_design",
        ],
        "warnings": [
            "sectionspan_contract_review_does_not_modify_runtime_evidence_policy",
            "no_sectionspan_contract_candidate_is_a_runtime_citation",
            "strict_evidence_and_parser_routing_remain_blocked",
        ],
        "contractCandidates": rows,
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
            "contractPrinciples",
            "warnings",
            "heldOut",
        )
        if key in report
    }


def render_sectionspan_contract_review_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpanCandidate Contract Review",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Input candidates: `{int(counts.get('inputSectionSpanCandidates') or 0)}`",
        f"- Contract candidates: `{int(counts.get('contractCandidateCount') or 0)}`",
        f"- Contract-ready candidates: `{int(counts.get('contractReadyCandidates') or 0)}`",
        f"- Blocked contract candidates: `{int(counts.get('blockedContractCandidates') or 0)}`",
        f"- Held out: `{int(counts.get('heldOutCandidates') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleCandidates') or 0)}`",
        f"- Runtime evidence: `{int(counts.get('runtimeEvidenceCandidates') or 0)}`",
        "",
        "## Boundary",
        "",
        "All rows are `sectionspan_contract_candidate_only`. This report does not create strict evidence, runtime citations, parser routing, answer integration, canonical parsed artifacts, SQLite writes, indexes, or embeddings.",
        "",
        "## Counts",
        "",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Held out by reason: `{json.dumps(counts.get('heldOutByReason') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Contract Candidates",
        "",
    ]
    for item in list(report.get("contractCandidates") or []):
        lines.append(
            f"- `{item.get('paper_id')}` page `{item.get('page')}` `{item.get('section_type')}` "
            f"`{item.get('section_label')}` {item.get('section_title')} -> `{item.get('contract_status')}`"
        )
    lines.extend(["", "## Held Out", ""])
    for item in list(report.get("heldOut") or []):
        lines.append(
            f"- `{item.get('paperId')}` `{item.get('reviewClass')}` {item.get('candidateText')} -> `{item.get('reason')}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_sectionspan_contract_review_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    review_path = root / "sectionspan-contract-review.json"
    summary_path = root / "sectionspan-contract-summary.json"
    markdown_path = root / "sectionspan-contract-review.md"
    review_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_contract_review_markdown(report), encoding="utf-8")
    return {"review": str(review_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only SectionSpanCandidate contract review.")
    parser.add_argument("--sectionspan-candidate-report", required=True, help="Path to sectionspan-candidates.json.")
    parser.add_argument(
        "--candidate-layer-promotion-policy-draft-report",
        required=True,
        help="Path to candidate-layer-promotion-policy-draft.json.",
    )
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_contract_review(
        sectionspan_candidate_report=args.sectionspan_candidate_report,
        candidate_layer_promotion_policy_draft_report=args.candidate_layer_promotion_policy_draft_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_contract_review_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_CONTRACT_REVIEW_SCHEMA_ID",
    "build_sectionspan_contract_review",
    "render_sectionspan_contract_review_markdown",
    "write_sectionspan_contract_review_reports",
]
