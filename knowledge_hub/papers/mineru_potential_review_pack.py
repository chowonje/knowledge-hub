"""Report-only review pack for MinerU potential strict candidates.

The review pack consumes a MinerU source-alignment audit report and emits
human-reviewable cards for candidates that look structurally promising.  It is
still an audit artifact: cards remain candidate-only, do not become strict
evidence, and are not wired into parser routing or answer generation.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


MINERU_POTENTIAL_REVIEW_PACK_SCHEMA_ID = "knowledge-hub.paper.mineru-potential-review-pack.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _paper_markdown_paths(source_alignment_report: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for paper in list(source_alignment_report.get("papers") or []):
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("paperId") or "")
        input_payload = dict(paper.get("input") or {})
        path_value = str(input_payload.get("pymupdfDocumentMarkdownPath") or "").strip()
        if paper_id and path_value:
            paths[paper_id] = Path(path_value).expanduser()
    return paths


def _span_context(markdown_text: str, start: int | None, end: int | None, *, radius: int = 220) -> dict[str, Any]:
    if start is None or end is None:
        return {
            "matchedText": "",
            "contextBefore": "",
            "contextAfter": "",
            "contextRadius": radius,
        }
    left = max(0, start - radius)
    right = min(len(markdown_text), end + radius)
    return {
        "matchedText": markdown_text[start:end],
        "contextBefore": markdown_text[left:start],
        "contextAfter": markdown_text[end:right],
        "contextRadius": radius,
    }


def _recommended_action(candidate: dict[str, Any]) -> tuple[str, str]:
    candidate_type = str(candidate.get("candidate_type") or "")
    if candidate_type == "section_candidate":
        return "review_for_section_span_schema", "SectionSpan"
    if candidate_type == "equation_candidate":
        return "hold_for_quote_only_equation_policy", "EquationArtifactQuoteOnly"
    if candidate_type == "figure_caption_candidate":
        return "hold_until_caption_region_link_review", "FigureCaptionArtifact"
    if candidate_type == "table_candidate":
        return "hold_until_table_cell_provenance", "TableArtifact"
    return "hold_unclassified_candidate", "LayoutElement"


def _review_card(
    *,
    index: int,
    candidate: dict[str, Any],
    markdown_text: str,
) -> dict[str, Any]:
    chars_start = candidate.get("chars_start")
    chars_end = candidate.get("chars_end")
    start = _safe_int(chars_start, default=-1) if chars_start is not None else None
    end = _safe_int(chars_end, default=-1) if chars_end is not None else None
    if start is not None and start < 0:
        start = None
    if end is not None and end < 0:
        end = None
    action, artifact = _recommended_action(candidate)
    context = _span_context(markdown_text, start, end)
    strict_blockers = list(candidate.get("strict_blockers") or [])
    return {
        "cardId": f"mineru-review:{index:04d}",
        "reviewStatus": "needs_human_review",
        "recommendedAction": action,
        "formalArtifactCandidate": artifact,
        "candidateId": str(candidate.get("candidate_id") or ""),
        "candidateType": str(candidate.get("candidate_type") or ""),
        "paperId": str(candidate.get("paper_id") or ""),
        "candidateText": _clean_text(candidate.get("candidate_text")),
        "canonicalSpan": {
            "charsStart": start,
            "charsEnd": end,
            "page": candidate.get("page"),
            "sourceContentHash": candidate.get("sourceContentHash"),
            "sourceContentHashSource": candidate.get("sourceContentHashSource"),
            "alignmentMethod": candidate.get("alignment_method"),
            "alignmentStatus": candidate.get("alignment_status"),
            "alignmentConfidence": candidate.get("confidence"),
            "matchedText": context["matchedText"],
            "contextBefore": context["contextBefore"],
            "contextAfter": context["contextAfter"],
        },
        "sourceSpanLocator": dict(candidate.get("source_span_locator") or {}),
        "promotionAssessment": {
            "candidateClassification": candidate.get("classification"),
            "sourceAlignmentRequirementsMet": bool(candidate.get("strict_requirements_met")),
            "runtimeStrictEligible": False,
            "citationGrade": False,
            "strictBlockers": strict_blockers,
            "humanReviewQuestions": _human_review_questions(str(candidate.get("candidate_type") or "")),
        },
        "evidenceTier": "human_review_candidate_only",
        "strict": False,
        "citationGrade": False,
    }


def _human_review_questions(candidate_type: str) -> list[str]:
    if candidate_type == "section_candidate":
        return [
            "Does the matched text identify a real section or heading rather than front matter noise?",
            "Should this become a formal SectionSpan candidate in a later schema tranche?",
            "Is canonical generated-markdown span provenance sufficient, or is original PDF/source offset recovery required first?",
        ]
    if candidate_type == "equation_candidate":
        return [
            "Is this exact equation text recoverable as a quote-only span?",
            "Is equation numbering or surrounding paragraph provenance required before promotion?",
        ]
    if candidate_type == "figure_caption_candidate":
        return [
            "Does the caption span match the source text exactly?",
            "Can the caption be linked to a figure/image region before citation-grade use?",
        ]
    if candidate_type == "table_candidate":
        return [
            "Is this only a caption/table-region candidate?",
            "What row/column/cell bbox provenance is missing before citation-grade table evidence?",
        ]
    return ["What artifact type, if any, should this candidate become?"]


def _candidate_filter(candidate: dict[str, Any], *, include_classifications: set[str]) -> bool:
    classification = str(candidate.get("classification") or "")
    if classification not in include_classifications:
        return False
    return _candidate_is_reviewable(candidate)


def _candidate_is_reviewable(candidate: dict[str, Any]) -> bool:
    blockers = set(str(item) for item in list(candidate.get("strict_blockers") or []))
    return (
        str(candidate.get("alignment_status") or "") == "aligned"
        and str(candidate.get("alignment_method") or "") == "exact"
        and bool(candidate.get("strict_requirements_met")) is True
        and bool(candidate.get("strict_eligible")) is False
        and bool(candidate.get("citation_grade")) is False
        and str(candidate.get("evidence_tier") or "") == "source_alignment_candidate_only"
        and "runtime_promotion_disabled_for_tranche" in blockers
    )


def _counts(cards: list[dict[str, Any]], *, input_candidate_count: int, skipped_count: int) -> dict[str, Any]:
    by_type = Counter(str(card.get("candidateType") or "") for card in cards)
    by_paper = Counter(str(card.get("paperId") or "") for card in cards)
    by_action = Counter(str(card.get("recommendedAction") or "") for card in cards)
    blocker_counts: Counter[str] = Counter()
    for card in cards:
        blocker_counts.update(
            str(item)
            for item in list((card.get("promotionAssessment") or {}).get("strictBlockers") or [])
        )
    return {
        "inputCandidateCount": input_candidate_count,
        "totalReviewCards": len(cards),
        "emittedCardCount": len(cards),
        "skippedCount": skipped_count,
        "byCandidateType": dict(by_type),
        "byPaper": dict(by_paper),
        "byRecommendedAction": dict(by_action),
        "strictEligibleCards": sum(1 for card in cards if card.get("strict")),
        "citationGradeCards": sum(1 for card in cards if card.get("citationGrade")),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_mineru_potential_review_pack(
    source_alignment_report_path: str | Path,
    *,
    include_classifications: tuple[str, ...] = ("potential_strict_candidate",),
    limit: int | None = None,
) -> dict[str, Any]:
    """Build a human-review pack from a MinerU source-alignment audit report."""

    report_path = Path(str(source_alignment_report_path)).expanduser()
    source_report = _read_json(report_path)
    markdown_paths = _paper_markdown_paths(source_report)
    include_set = set(include_classifications)
    raw_candidates = [dict(candidate) for candidate in list(source_report.get("candidates") or []) if isinstance(candidate, dict)]
    selected = [
        candidate
        for candidate in raw_candidates
        if _candidate_filter(candidate, include_classifications=include_set)
    ]
    selected.sort(
        key=lambda item: (
            str(item.get("paper_id") or ""),
            str(item.get("candidate_type") or ""),
            _safe_int(item.get("page"), default=0),
            _safe_int(item.get("chars_start"), default=0),
            str(item.get("candidate_id") or ""),
        )
    )
    if limit is not None and limit >= 0:
        selected = selected[:limit]

    markdown_cache: dict[str, str] = {}
    cards: list[dict[str, Any]] = []
    missing_markdown: list[str] = []
    for candidate in selected:
        paper_id = str(candidate.get("paper_id") or "")
        markdown_path = markdown_paths.get(paper_id)
        markdown_text = ""
        if markdown_path is not None:
            if paper_id not in markdown_cache:
                try:
                    markdown_cache[paper_id] = markdown_path.read_text(encoding="utf-8")
                except Exception:
                    markdown_cache[paper_id] = ""
                    missing_markdown.append(paper_id)
            markdown_text = markdown_cache.get(paper_id, "")
        else:
            missing_markdown.append(paper_id)
        cards.append(_review_card(index=len(cards) + 1, candidate=candidate, markdown_text=markdown_text))

    counts = _counts(cards, input_candidate_count=len(raw_candidates), skipped_count=len(raw_candidates) - len(cards))
    return {
        "schema": MINERU_POTENTIAL_REVIEW_PACK_SCHEMA_ID,
        "status": "ok" if cards else "empty",
        "generatedAt": _now(),
        "input": {
            "sourceAlignmentReportPath": str(report_path),
            "sourceAlignmentSchema": source_report.get("schema"),
            "includeClassifications": list(include_classifications),
            "limit": limit,
        },
        "counts": counts,
        "policy": {
            "allCardsCandidateOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
        },
        "reviewGuidance": {
            "primaryDecision": "decide_which_candidate_types_deserve_formal_artifact_schema_later",
            "recommendedFirstArtifact": "SectionSpan",
            "doNotPromoteYet": [
                "table_cell_evidence",
                "figure_region_evidence",
                "equation_reasoning",
                "runtime_answer_evidence",
            ],
        },
        "warnings": [
            "review_cards_are_not_strict_evidence",
            "source_alignment_candidates_are_inputs_for_human_review_only",
            "generated_markdown_offsets_are_not_original_pdf_byte_offsets",
            "table_cells_and_figure_regions_are_not_citation_grade_in_this_pack",
        ],
        "missingMarkdownPapers": sorted(set(missing_markdown)),
        "cards": cards,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "input",
            "counts",
            "policy",
            "reviewGuidance",
            "warnings",
            "missingMarkdownPapers",
        )
        if key in report
    }


def render_mineru_potential_review_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# MinerU Potential Strict Candidate Review Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Review cards: `{int(counts.get('totalReviewCards') or 0)}`",
        f"- Strict eligible cards: `{int(counts.get('strictEligibleCards') or 0)}`",
        f"- Citation-grade cards: `{int(counts.get('citationGradeCards') or 0)}`",
        "",
        "## Evidence Tier",
        "",
        "All cards are `human_review_candidate_only`. This pack does not create strict evidence or runtime citations.",
        "The purpose is to decide which candidate types deserve a later formal artifact schema.",
        "",
        "## Counts",
        "",
        f"- By candidate type: `{json.dumps(counts.get('byCandidateType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By recommended action: `{json.dumps(counts.get('byRecommendedAction') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Review Cards",
        "",
    ]
    for card in list(report.get("cards") or []):
        span = dict(card.get("canonicalSpan") or {})
        assessment = dict(card.get("promotionAssessment") or {})
        blockers = list(assessment.get("strictBlockers") or [])
        lines.extend(
            [
                f"### `{card.get('cardId', '')}`",
                "",
                f"- Candidate: `{card.get('candidateId', '')}`",
                f"- Type: `{card.get('candidateType', '')}`",
                f"- Paper: `{card.get('paperId', '')}`",
                f"- Page: `{span.get('page')}`",
                f"- Alignment: `{span.get('alignmentStatus')}` / `{span.get('alignmentMethod')}`",
                f"- Recommended action: `{card.get('recommendedAction', '')}`",
                f"- Formal artifact candidate: `{card.get('formalArtifactCandidate', '')}`",
                f"- Candidate text: {card.get('candidateText', '')}",
                f"- Matched text: {span.get('matchedText', '')}",
                f"- Strict blockers: {', '.join(f'`{item}`' for item in blockers)}",
                "",
            ]
        )
    return "\n".join(lines)


def write_mineru_potential_review_pack_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    cards_path = root / "mineru-potential-review-cards.json"
    summary_path = root / "mineru-potential-review-summary.json"
    markdown_path = root / "mineru-potential-review-pack.md"
    cards_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_mineru_potential_review_pack_markdown(report), encoding="utf-8")
    return {
        "cards": str(cards_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only MinerU potential strict candidate review pack.")
    parser.add_argument("--source-alignment-report", required=True, help="Path to mineru-source-alignment-report.json.")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--include-classification", action="append", default=[], help="Candidate classification to include. Repeatable.")
    parser.add_argument("--limit", type=int, default=-1, help="Optional max number of review cards.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    classifications = tuple(args.include_classification or ["potential_strict_candidate"])
    limit = args.limit if args.limit >= 0 else None
    report = build_mineru_potential_review_pack(
        args.source_alignment_report,
        include_classifications=classifications,
        limit=limit,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_mineru_potential_review_pack_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MINERU_POTENTIAL_REVIEW_PACK_SCHEMA_ID",
    "build_mineru_potential_review_pack",
    "render_mineru_potential_review_pack_markdown",
    "write_mineru_potential_review_pack_reports",
]
