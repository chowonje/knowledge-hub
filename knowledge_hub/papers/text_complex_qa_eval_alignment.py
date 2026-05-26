"""Align complex paper QA cases to v0.1 text-evidence capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import normalize_text, utc_now_iso

TEXT_COMPLEX_QA_ALIGNMENT_CASE_SCHEMA_ID = "knowledge-hub.paper.text-complex-qa-eval-alignment-case.v1"
TEXT_COMPLEX_QA_ALIGNMENT_REPORT_SCHEMA_ID = "knowledge-hub.paper.text-complex-qa-eval-alignment-report.v1"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


@dataclass(frozen=True)
class ComplexQaCase:
    case_id: str
    category: str
    paper_id: str
    question: str
    required_evidence_types: tuple[str, ...]
    target: dict[str, Any] = field(default_factory=dict)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _private_path_leak_count(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return 1 if PRIVATE_PATH_RE.search(encoded) else 0


def default_complex_qa_cases() -> list[ComplexQaCase]:
    return [
        ComplexQaCase(
            case_id="figure-caption-alexnet-figure-1-caption",
            category="figure_caption_qa",
            paper_id="alexnet-2012",
            question="What does Figure 1 show in AlexNet?",
            required_evidence_types=("figure_caption_text",),
            target={"figureLabel": "Figure 1"},
        ),
        ComplexQaCase(
            case_id="figure-caption-alexnet-figure-1-visual-detail",
            category="figure_caption_qa",
            paper_id="alexnet-2012",
            question="In AlexNet Figure 1, which bar or curve is visually higher?",
            required_evidence_types=("figure_caption_text", "visual_inspection"),
            target={"figureLabel": "Figure 1", "visualReasoningRequested": True},
        ),
        ComplexQaCase(
            case_id="table-numeric-alexnet-table-1-result",
            category="table_numeric_qa",
            paper_id="alexnet-2012",
            question="What numeric result is reported in AlexNet Table 1?",
            required_evidence_types=("table_caption_text", "table_numeric_cell"),
            target={"tableLabel": "Table 1"},
        ),
        ComplexQaCase(
            case_id="table-numeric-resnet-table-3-error-rate",
            category="table_numeric_qa",
            paper_id="resnet-2015",
            question="What error rate is reported in ResNet Table 3?",
            required_evidence_types=("table_caption_text", "table_numeric_cell"),
            target={"tableLabel": "Table 3"},
        ),
        ComplexQaCase(
            case_id="equation-resnet-equation-1-residual",
            category="equation_citation_qa",
            paper_id="resnet-2015",
            question="What residual equation is labeled Equation 1 in ResNet?",
            required_evidence_types=("equation_locator_context", "equation_latex_or_text"),
            target={"equationLabel": "Equation 1"},
        ),
        ComplexQaCase(
            case_id="equation-mae-equation-1-missing",
            category="equation_citation_qa",
            paper_id="mae-2021",
            question="What equation is labeled Equation 1 in MAE?",
            required_evidence_types=("equation_locator_context",),
            target={"equationLabel": "Equation 1"},
        ),
        ComplexQaCase(
            case_id="method-resnet-residual-learning-text",
            category="method_comparison_qa",
            paper_id="resnet-2015",
            question="How does ResNet describe residual learning?",
            required_evidence_types=("paragraph_span_text",),
            target={"matchTerms": ["residual", "learning"]},
        ),
        ComplexQaCase(
            case_id="method-mae-masked-autoencoder-text",
            category="method_comparison_qa",
            paper_id="mae-2021",
            question="How does MAE describe masked autoencoders?",
            required_evidence_types=("paragraph_span_text",),
            target={"matchTerms": ["masked", "autoencoder"]},
        ),
        ComplexQaCase(
            case_id="limitation-visual-chart-inspection-v0-1",
            category="limitation_qa",
            paper_id="alexnet-2012",
            question="Can v0.1 decide which visual curve in Figure 1 is higher?",
            required_evidence_types=("visual_inspection",),
            target={"figureLabel": "Figure 1", "visualReasoningRequested": True},
        ),
        ComplexQaCase(
            case_id="limitation-missing-table-99-no-answer",
            category="limitation_qa",
            paper_id="clip-2021",
            question="What exact numeric value is reported in CLIP Table 99?",
            required_evidence_types=("table_caption_text", "table_numeric_cell"),
            target={"tableLabel": "Table 99"},
        ),
    ]


def _report_statuses(reports: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {key: str(value.get("status") or "unknown") for key, value in sorted(reports.items())}


def _evidence_ref(row: dict[str, Any], *, evidence_type: str, locator_kind: str) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "evidenceType": evidence_type,
        "locatorKind": locator_kind,
        "artifactId": str(row.get("artifactId") or ""),
        "paperId": str(row.get("paperId") or ""),
        "sourceContentHash": str(row.get("sourceContentHash") or ""),
        "page": _safe_int(row.get("page")),
        "bbox": list(row.get("bbox") or row.get("captionBbox") or row.get("equationBbox") or []),
        "textHash": str(
            row.get("textHash")
            or row.get("captionTextHash")
            or row.get("equationTextHash")
            or row.get("contextTextHash")
            or ""
        ),
        "strictEvidence": False,
        "answerVisible": False,
    }
    label = row.get("figureLabel") or row.get("tableLabel") or row.get("equationLabel") or row.get("spanType")
    if label:
        ref["label"] = str(label)
    return ref


def _find_figure_qa_row(report: dict[str, Any], *, paper_id: str, figure_label: str, visual: bool) -> dict[str, Any] | None:
    for row in report.get("rows", []):
        if str(row.get("paperId")) != paper_id:
            continue
        if str(row.get("requestedFigureLabel") or "") != figure_label:
            continue
        if bool(row.get("visualReasoningRequested")) == visual:
            return row
    return None


def _find_table_candidate(report: dict[str, Any], *, paper_id: str, table_label: str) -> dict[str, Any] | None:
    for row in report.get("candidates", []):
        if str(row.get("paperId")) == paper_id and str(row.get("tableLabel") or "") == table_label:
            return row
    return None


def _find_equation_candidate(report: dict[str, Any], *, paper_id: str, equation_label: str) -> dict[str, Any] | None:
    for row in report.get("candidates", []):
        if str(row.get("paperId")) == paper_id and str(row.get("equationLabel") or "") == equation_label:
            return row
    return None


def _find_text_span(report: dict[str, Any], *, paper_id: str, match_terms: Sequence[str]) -> dict[str, Any] | None:
    terms = [str(term).lower() for term in match_terms if str(term).strip()]
    for row in report.get("candidates", []):
        text = str(row.get("text") or "").lower()
        if str(row.get("paperId")) == paper_id and row.get("spanType") == "paragraph" and all(term in text for term in terms):
            return row
    return None


def _align_figure_caption_case(case: ComplexQaCase, reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    figure_label = str(case.target.get("figureLabel") or "")
    visual = bool(case.target.get("visualReasoningRequested"))
    row = _find_figure_qa_row(
        reports.get("figure_caption_text_qa", {}),
        paper_id=case.paper_id,
        figure_label=figure_label,
        visual=visual,
    )
    if visual:
        return _row(
            case,
            disposition="visual_unsupported",
            answerability="unsupported",
            blocker="visual_reasoning_not_supported_in_text_evidence_v01",
            missing=("visual_inspection_contract",),
            matched=[],
            mode="caption_text_only",
        )
    if row and row.get("answerabilityStatus") == "answerable":
        packet = row.get("candidateAnswerPacket") or {}
        evidence = dict(packet.get("evidence") or {})
        evidence["artifactId"] = evidence.get("artifactId") or ""
        evidence["bbox"] = evidence.get("bbox") or []
        return _row(
            case,
            disposition="text_answerable",
            answerability="answerable",
            blocker="",
            missing=(),
            matched=[_evidence_ref(evidence, evidence_type="figure_caption_text", locator_kind="page_bbox")],
            mode="caption_text_only",
        )
    return _row(
        case,
        disposition="no_answer",
        answerability="no_answer",
        blocker="figure_caption_not_found",
        missing=("figure_caption_text",),
        matched=[],
        mode="caption_text_only",
    )


def _align_table_case(case: ComplexQaCase, reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = _find_table_candidate(
        reports.get("table_caption_candidates", {}),
        paper_id=case.paper_id,
        table_label=str(case.target.get("tableLabel") or ""),
    )
    if not row:
        return _row(
            case,
            disposition="no_answer",
            answerability="no_answer",
            blocker="table_candidate_not_found",
            missing=("table_caption_text", "table_numeric_cell"),
            matched=[],
            mode="table_text_candidate",
        )
    missing = ["table_cell_identity", "row_column_header_identity"]
    if not row.get("numericCandidate"):
        missing.append("numeric_candidate_text")
    return _row(
        case,
        disposition="candidate_only",
        answerability="candidate_only",
        blocker="table_cell_identity_not_available",
        missing=tuple(missing),
        matched=[_evidence_ref(row, evidence_type="table_caption_text", locator_kind="page_bbox_chars")],
        mode=str(row.get("structureGrade") or "table_text_candidate"),
    )


def _align_equation_case(case: ComplexQaCase, reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = _find_equation_candidate(
        reports.get("equation_locator_context", {}),
        paper_id=case.paper_id,
        equation_label=str(case.target.get("equationLabel") or ""),
    )
    if not row:
        return _row(
            case,
            disposition="no_answer",
            answerability="no_answer",
            blocker="equation_locator_not_found",
            missing=("equation_locator_context",),
            matched=[],
            mode="equation_locator_context_candidate",
        )
    return _row(
        case,
        disposition="candidate_only",
        answerability="candidate_only",
        blocker="equation_latex_reconstruction_not_available",
        missing=("equation_latex_or_text_strict_contract",),
        matched=[_evidence_ref(row, evidence_type="equation_locator_context", locator_kind="page_bbox_chars")],
        mode=str(row.get("locatorGrade") or "equation_locator_context_candidate"),
    )


def _align_text_span_case(case: ComplexQaCase, reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = _find_text_span(
        reports.get("section_paragraph_spans", {}),
        paper_id=case.paper_id,
        match_terms=list(case.target.get("matchTerms") or []),
    )
    if not row:
        return _row(
            case,
            disposition="no_answer",
            answerability="no_answer",
            blocker="paragraph_span_not_found",
            missing=("paragraph_span_text",),
            matched=[],
            mode="paragraph_span_candidate",
        )
    return _row(
        case,
        disposition="text_answerable",
        answerability="answerable",
        blocker="",
        missing=(),
        matched=[_evidence_ref(row, evidence_type="paragraph_span_text", locator_kind="page_bbox_chars")],
        mode="paragraph_span_candidate",
    )


def _row(
    case: ComplexQaCase,
    *,
    disposition: str,
    answerability: str,
    blocker: str,
    missing: Sequence[str],
    matched: Sequence[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    return {
        "schema": TEXT_COMPLEX_QA_ALIGNMENT_CASE_SCHEMA_ID,
        "caseId": case.case_id,
        "category": case.category,
        "paperId": case.paper_id,
        "question": case.question,
        "questionHash": _sha256_text(case.question),
        "requiredEvidenceTypes": list(case.required_evidence_types),
        "disposition": disposition,
        "answerabilityExpectation": answerability,
        "textEvidenceMode": mode,
        "matchedEvidenceRows": len(matched),
        "matchedEvidence": list(matched),
        "missingContracts": list(missing),
        "blockerReason": blocker,
        "strictEvidencePromotionRows": 0,
        "runtimeAnswerVisibleExposureRows": 0,
    }


def align_complex_qa_cases(
    *,
    reports: dict[str, dict[str, Any]],
    cases: Sequence[ComplexQaCase] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in list(cases or default_complex_qa_cases()):
        if case.category == "figure_caption_qa":
            rows.append(_align_figure_caption_case(case, reports))
        elif case.category == "table_numeric_qa":
            rows.append(_align_table_case(case, reports))
        elif case.category == "equation_citation_qa":
            rows.append(_align_equation_case(case, reports))
        elif case.category == "method_comparison_qa":
            rows.append(_align_text_span_case(case, reports))
        elif bool(case.target.get("visualReasoningRequested")):
            rows.append(
                _row(
                    case,
                    disposition="visual_unsupported",
                    answerability="unsupported",
                    blocker="visual_reasoning_not_supported_in_text_evidence_v01",
                    missing=("visual_inspection_contract",),
                    matched=[],
                    mode="text_evidence_v0_1_scope_limit",
                )
            )
        elif case.target.get("tableLabel"):
            rows.append(_align_table_case(case, reports))
        else:
            rows.append(
                _row(
                    case,
                    disposition="no_answer",
                    answerability="no_answer",
                    blocker="unsupported_case_category",
                    missing=tuple(case.required_evidence_types),
                    matched=[],
                    mode="unknown",
                )
            )

    report: dict[str, Any] = {
        "schema": TEXT_COMPLEX_QA_ALIGNMENT_REPORT_SCHEMA_ID,
        "status": "ready" if rows else "blocked",
        "generatedAt": generated_at or utc_now_iso(),
        "inputReportStatuses": _report_statuses(reports),
        "scope": {
            "writes": "report_only",
            "alignmentPolicy": "text_evidence_v0_1",
            "visualLayoutBranchDeferred": True,
            "strictEvidencePromotionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
        },
        "caseRows": len(rows),
        "textAnswerableRows": sum(1 for row in rows if row.get("disposition") == "text_answerable"),
        "candidateOnlyRows": sum(1 for row in rows if row.get("disposition") == "candidate_only"),
        "visualUnsupportedRows": sum(1 for row in rows if row.get("disposition") == "visual_unsupported"),
        "noAnswerRows": sum(1 for row in rows if row.get("disposition") == "no_answer"),
        "categoryCounts": {
            "figure_caption_qa": sum(1 for row in rows if row.get("category") == "figure_caption_qa"),
            "table_numeric_qa": sum(1 for row in rows if row.get("category") == "table_numeric_qa"),
            "equation_citation_qa": sum(1 for row in rows if row.get("category") == "equation_citation_qa"),
            "method_comparison_qa": sum(1 for row in rows if row.get("category") == "method_comparison_qa"),
            "limitation_qa": sum(1 for row in rows if row.get("category") == "limitation_qa"),
        },
        "rows": rows,
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "strictEvidencePromotionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "warnings": [
            "alignment rows classify complex QA feasibility against text evidence candidates only; they do not generate runtime answers"
        ],
        "schemaErrors": [],
    }
    report["privatePathLeakRows"] = _private_path_leak_count(report)
    if report["privatePathLeakRows"]:
        report["status"] = "blocked"
    return report


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Text Complex QA Eval Alignment",
        "",
        f"- status: `{report.get('status')}`",
        f"- caseRows: `{report.get('caseRows')}`",
        f"- textAnswerableRows: `{report.get('textAnswerableRows')}`",
        f"- candidateOnlyRows: `{report.get('candidateOnlyRows')}`",
        f"- visualUnsupportedRows: `{report.get('visualUnsupportedRows')}`",
        f"- noAnswerRows: `{report.get('noAnswerRows')}`",
        f"- privatePathLeakRows: `{report.get('privatePathLeakRows')}`",
        "",
        "## Cases",
        "",
        "| caseId | category | disposition | blocker |",
        "|---|---|---|---|",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"| `{row.get('caseId')}` | `{row.get('category')}` | `{row.get('disposition')}` | "
            f"`{row.get('blockerReason') or ''}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


__all__ = [
    "TEXT_COMPLEX_QA_ALIGNMENT_CASE_SCHEMA_ID",
    "TEXT_COMPLEX_QA_ALIGNMENT_REPORT_SCHEMA_ID",
    "ComplexQaCase",
    "align_complex_qa_cases",
    "default_complex_qa_cases",
    "load_report",
    "write_report",
]
