"""Report-only complex-paper QA eval design helpers.

This module designs questions for later evaluation of source-aligned structured
paper candidates.  It does not answer questions, run evals, mutate SQLite,
reindex, reembed, change parser routing, or promote candidate artifacts into
strict evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


COMPLEX_QA_EVAL_DESIGN_SCHEMA_ID = "knowledge-hub.paper.complex-qa-eval-design.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _by_paper(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        paper_id = str(item.get("paper_id") or "")
        if paper_id:
            grouped[paper_id].append(item)
    for items in grouped.values():
        items.sort(key=lambda item: (_safe_int(item.get("page")), _safe_int(item.get("chars_start")), str(item.get("candidate_id") or "")))
    return dict(grouped)


def _is_aligned(item: dict[str, Any]) -> bool:
    return (
        str(item.get("canonical_alignment_status") or "aligned") == "aligned"
        and item.get("chars_start") is not None
        and item.get("chars_end") is not None
        and item.get("page") is not None
        and bool(item.get("sourceContentHash"))
    )


def _question(
    *,
    index: int,
    question_type: str,
    paper_ids: list[str],
    prompt: str,
    target_layers: list[str],
    source_candidate_ids: list[str],
    evidence_requirements: list[str],
    tags: list[str],
    blocked_by_current_candidates: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "question_id": f"complex-paper-candidate-v1-{index:03d}",
        "question_type": question_type,
        "paper_ids": paper_ids,
        "prompt": _clean_text(prompt),
        "target_candidate_layers": target_layers,
        "source_candidate_ids": source_candidate_ids,
        "evidence_requirements": evidence_requirements,
        "current_runtime_expectation": "abstain_until_strict_candidate_runtime_exists",
        "execution_status": "designed_not_executed",
        "strict_evidence_required_before_answering": True,
        "strict_eligible_now": False,
        "citation_grade_now": False,
        "candidate_only": True,
        "blocked_by_current_candidates": blocked_by_current_candidates or [
            "candidate_layers_are_report_only",
            "runtime_promotion_disabled_for_tranche",
        ],
        "tags": tags,
    }


def _section_questions(section_candidates: list[dict[str, Any]], start_index: int) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    grouped = _by_paper(section_candidates)
    for paper_id in sorted(grouped):
        abstract = next((item for item in grouped[paper_id] if item.get("section_type") == "abstract"), None)
        numbered = [
            item
            for item in grouped[paper_id]
            if item.get("section_type") == "numbered_section"
            and str(item.get("section_title") or "").casefold() not in {"introduction", "related work"}
        ]
        selected = [item for item in [abstract, *numbered[:1]] if item is not None]
        for item in selected:
            title = str(item.get("section_title") or item.get("candidate_text") or "the selected section")
            questions.append(
                _question(
                    index=start_index + len(questions),
                    question_type="section_evidence_question",
                    paper_ids=[paper_id],
                    prompt=f"For paper {paper_id}, what claim should be supported from the section '{title}'?",
                    target_layers=["sectionspan"],
                    source_candidate_ids=[str(item.get("candidate_id") or "")],
                    evidence_requirements=[
                        "section boundary candidate",
                        "canonical chars:start-end",
                        "page",
                        "sourceContentHash",
                        "strict runtime evidence before answering",
                    ],
                    tags=["sectionspan", "single_paper", "design_only"],
                )
            )
    return questions


def _figure_questions(figure_candidates: list[dict[str, Any]], start_index: int, *, limit: int = 4) -> list[dict[str, Any]]:
    aligned = [item for item in figure_candidates if _is_aligned(item)]
    blocked = [item for item in figure_candidates if not _is_aligned(item)]
    selected = [*aligned[: max(0, limit - 1)], *blocked[:1]][:limit]
    questions = []
    for item in selected:
        label = str(item.get("figure_label") or "the figure")
        questions.append(
            _question(
                index=start_index + len(questions),
                question_type="figure_caption_question",
                paper_ids=[str(item.get("paper_id") or "")],
                prompt=f"For paper {item.get('paper_id')}, what does {label} show according to its caption?",
                target_layers=["figure_caption"],
                source_candidate_ids=[str(item.get("candidate_id") or "")],
                evidence_requirements=[
                    "caption source span",
                    "page",
                    "sourceContentHash",
                    "verified figure/image region before citation-grade use",
                ],
                blocked_by_current_candidates=[
                    "figure_region_link_unverified",
                    "candidate_layers_are_report_only",
                ],
                tags=["figure_caption", "single_paper", "design_only"],
            )
        )
    return questions


def _table_questions(table_candidates: list[dict[str, Any]], start_index: int) -> list[dict[str, Any]]:
    selected = sorted(table_candidates, key=lambda item: (str(item.get("paper_id") or ""), _safe_int(item.get("page")), str(item.get("table_label") or "")))
    questions = []
    for item in selected:
        label = str(item.get("table_label") or "the table")
        questions.append(
            _question(
                index=start_index + len(questions),
                question_type="table_region_question",
                paper_ids=[str(item.get("paper_id") or "")],
                prompt=f"For paper {item.get('paper_id')}, what does {label} compare or measure, and which exact table-cell values remain unavailable?",
                target_layers=["table_region"],
                source_candidate_ids=[str(item.get("candidate_id") or "")],
                evidence_requirements=[
                    "table caption source span",
                    "table region candidate",
                    "row/column/cell values",
                    "cell-level bbox provenance",
                    "strict runtime evidence before numeric answering",
                ],
                blocked_by_current_candidates=[
                    "table_cell_row_column_bbox_provenance_missing",
                    "candidate_layers_are_report_only",
                ],
                tags=["table_region", "numeric_evidence", "design_only"],
            )
        )
    return questions


def _equation_questions(equation_candidates: list[dict[str, Any]], start_index: int, *, limit: int = 3) -> list[dict[str, Any]]:
    selected = sorted(equation_candidates, key=lambda item: (str(item.get("paper_id") or ""), str(item.get("equation_label") or "")))[:limit]
    questions = []
    for item in selected:
        label = str(item.get("equation_label") or "the equation")
        questions.append(
            _question(
                index=start_index + len(questions),
                question_type="equation_quote_question",
                paper_ids=[str(item.get("paper_id") or "")],
                prompt=f"For paper {item.get('paper_id')}, quote {label} with page and source-span provenance without interpreting it.",
                target_layers=["equation_quote"],
                source_candidate_ids=[str(item.get("candidate_id") or "")],
                evidence_requirements=[
                    "equation text alignment to canonical source span",
                    "page",
                    "sourceContentHash",
                    "quote-only policy",
                    "no semantic interpretation",
                ],
                blocked_by_current_candidates=[
                    "equation_quote_alignment_missing",
                    "candidate_layers_are_report_only",
                ],
                tags=["equation_quote", "quote_only", "design_only"],
            )
        )
    return questions


def _cross_questions(section_candidates: list[dict[str, Any]], start_index: int, *, limit: int = 2) -> list[dict[str, Any]]:
    grouped = _by_paper(section_candidates)
    papers = sorted(grouped)
    pairs = list(zip(papers, papers[1:]))[:limit]
    questions = []
    for left, right in pairs:
        left_item = next((item for item in grouped[left] if item.get("section_type") == "numbered_section"), grouped[left][0])
        right_item = next((item for item in grouped[right] if item.get("section_type") == "numbered_section"), grouped[right][0])
        questions.append(
            _question(
                index=start_index + len(questions),
                question_type="method_comparison_question",
                paper_ids=[left, right],
                prompt=f"Compare the method evidence from paper {left} section '{left_item.get('section_title')}' with paper {right} section '{right_item.get('section_title')}'.",
                target_layers=["sectionspan"],
                source_candidate_ids=[str(left_item.get("candidate_id") or ""), str(right_item.get("candidate_id") or "")],
                evidence_requirements=[
                    "strict section spans from both papers",
                    "page and sourceContentHash for both sources",
                    "no fallback spans",
                    "no paraphrase-only evidence",
                ],
                tags=["method_comparison", "multi_paper", "design_only"],
            )
        )
    return questions


def build_complex_qa_eval_design(
    *,
    structured_summary_report: str | Path,
    sectionspan_report: str | Path,
    figure_caption_report: str | Path,
    equation_quote_report: str | Path,
    table_region_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only complex-paper QA eval design from candidate reports."""

    section = _read_json(sectionspan_report)
    figure = _read_json(figure_caption_report)
    equation = _read_json(equation_quote_report)
    table = _read_json(table_region_report)
    summary = _read_json(structured_summary_report)
    questions: list[dict[str, Any]] = []
    for builder, args in (
        (_section_questions, [list(section.get("candidates") or [])]),
        (_figure_questions, [list(figure.get("candidates") or [])]),
        (_table_questions, [list(table.get("candidates") or [])]),
        (_equation_questions, [list(equation.get("candidates") or [])]),
        (_cross_questions, [list(section.get("candidates") or [])]),
    ):
        questions.extend(builder(*args, start_index=len(questions) + 1))

    by_type = Counter(str(item.get("question_type") or "") for item in questions)
    by_layer = Counter(layer for item in questions for layer in list(item.get("target_candidate_layers") or []))
    by_paper = Counter(paper for item in questions for paper in list(item.get("paper_ids") or []))
    inputs = {
        "structuredSummaryReport": str(Path(str(structured_summary_report)).expanduser()),
        "sectionspanReport": str(Path(str(sectionspan_report)).expanduser()),
        "figureCaptionReport": str(Path(str(figure_caption_report)).expanduser()),
        "equationQuoteReport": str(Path(str(equation_quote_report)).expanduser()),
        "tableRegionReport": str(Path(str(table_region_report)).expanduser()),
    }
    return {
        "schema": COMPLEX_QA_EVAL_DESIGN_SCHEMA_ID,
        "status": "ok" if questions else "empty",
        "generatedAt": _now(),
        "inputs": inputs,
        "sourceSummary": {
            "structuredCandidateSummarySchema": str(summary.get("schema") or ""),
            "candidateLayerTotal": _safe_int((summary.get("counts") or {}).get("totalCandidates")),
            "strictEligibleCandidates": _safe_int((summary.get("counts") or {}).get("strictEligibleCandidates")),
            "citationGradeCandidates": _safe_int((summary.get("counts") or {}).get("citationGradeCandidates")),
        },
        "counts": {
            "questionCount": len(questions),
            "byQuestionType": dict(by_type),
            "byTargetLayer": dict(by_layer),
            "byPaper": dict(by_paper),
            "currentRuntimeAnswerableQuestions": 0,
            "executedQuestions": 0,
            "strictEvidenceCreated": 0,
        },
        "policy": {
            "designOnly": True,
            "questionsExecuted": False,
            "answerGenerationRun": False,
            "allQuestionsRequireStrictEvidenceBeforeAnswering": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "eval_questions_are_design_only",
            "do_not_answer_with_candidate_only_artifacts",
            "candidate_layers_must_be_promoted_by_a_later_explicit_tranche_before_runtime_use",
        ],
        "questions": questions,
    }


def render_complex_qa_eval_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Complex Paper QA Eval Design",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Questions: `{int(counts.get('questionCount') or 0)}`",
        f"- Executed questions: `{int(counts.get('executedQuestions') or 0)}`",
        f"- Current runtime answerable: `{int(counts.get('currentRuntimeAnswerableQuestions') or 0)}`",
        f"- Strict evidence created: `{int(counts.get('strictEvidenceCreated') or 0)}`",
        "",
        "## Policy",
        "",
        "This is a design-only eval set. Do not answer these questions with candidate-only artifacts.",
        "",
        "## Counts",
        "",
        f"- By question type: `{json.dumps(counts.get('byQuestionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By target layer: `{json.dumps(counts.get('byTargetLayer') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Questions",
        "",
    ]
    for item in list(report.get("questions") or []):
        lines.extend(
            [
                f"### `{item.get('question_id', '')}`",
                "",
                f"- Type: `{item.get('question_type', '')}`",
                f"- Papers: `{', '.join(list(item.get('paper_ids') or []))}`",
                f"- Target layers: `{', '.join(list(item.get('target_candidate_layers') or []))}`",
                f"- Prompt: {item.get('prompt', '')}",
                f"- Execution status: `{item.get('execution_status', '')}`",
                f"- Current runtime expectation: `{item.get('current_runtime_expectation', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_complex_qa_eval_design_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    design_path = root / "complex-paper-qa-eval-design.json"
    markdown_path = root / "complex-paper-qa-eval-design.md"
    design_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_complex_qa_eval_design_markdown(report), encoding="utf-8")
    return {
        "design": str(design_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only complex-paper QA eval design.")
    parser.add_argument("--structured-summary-report", required=True, help="Path to structured-candidate-summary.json.")
    parser.add_argument("--sectionspan-report", required=True, help="Path to sectionspan-candidates.json.")
    parser.add_argument("--figure-caption-report", required=True, help="Path to figure-caption-candidates.json.")
    parser.add_argument("--equation-quote-report", required=True, help="Path to equation-quote-candidates.json.")
    parser.add_argument("--table-region-report", required=True, help="Path to table-region-candidates.json.")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print design payload as JSON.")
    args = parser.parse_args(argv)

    report = build_complex_qa_eval_design(
        structured_summary_report=args.structured_summary_report,
        sectionspan_report=args.sectionspan_report,
        figure_caption_report=args.figure_caption_report,
        equation_quote_report=args.equation_quote_report,
        table_region_report=args.table_region_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_complex_qa_eval_design_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPLEX_QA_EVAL_DESIGN_SCHEMA_ID",
    "build_complex_qa_eval_design",
    "render_complex_qa_eval_design_markdown",
    "write_complex_qa_eval_design_reports",
]
