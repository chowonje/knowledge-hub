"""Validate advisory GPT recommendations for visual hint expansion rows.

This helper captures manually supplied web GPT/Pro recommendation output as
advisory telemetry only. It does not convert recommendations into human
decisions, write a candidate store, index vectors, promote evidence, or expose
visual text at answer runtime.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Sequence

from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_SCHEMA_ID,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_REVIEW_PACK_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_human_product_decision_record import (
    HUMAN_DECISIONS,
)


VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_VALIDATION_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-gpt-recommendation-output-validation.v1"
)
VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_VALIDATION_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-gpt-recommendation-validation-row.v1"
)

READY_DECISION = "ready_for_project_side_human_decision_synthesis"
NEXT_RECOMMENDED_TRANCHE = "visual_annotation_expansion_pack_design_003"
DEFAULT_OUTPUT_REF = (
    "eval/knowledgeos/reports/"
    "visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_001.manual.json"
)
DEFAULT_PACK_REF = (
    "eval/knowledgeos/reports/"
    "visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack.v1.json"
)

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"FILL_IN|PLACEHOLDER|TODO", re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitized_report_ref(path: Path, *, project_root: Path | None = None) -> str:
    resolved = path.expanduser()
    if project_root is not None:
        try:
            rel = resolved.resolve().relative_to(project_root.resolve())
            return rel.as_posix()
        except Exception:
            pass
    return f"input_reports/{resolved.name}"


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _contains_placeholder(value: Any) -> bool:
    return bool(PLACEHOLDER_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _output_rows(output: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in list(output.get("rows") or []) if isinstance(row, dict)]


def _pack_rows(gpt_review_pack: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in list(gpt_review_pack.get("batchBundles") or []):
        if not isinstance(batch, dict):
            continue
        for row in list(batch.get("rows") or []):
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _rows_by_hint_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {normalize_text(row.get("hintCandidateId")): row for row in rows}


def _duplicate_hint_ids(rows: Sequence[dict[str, Any]]) -> list[str]:
    counts = Counter(normalize_text(row.get("hintCandidateId")) for row in rows)
    return sorted(hint_id for hint_id, count in counts.items() if hint_id and count > 1)


def _scope(row_count: int) -> dict[str, Any]:
    return {
        "writes": "report_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "manualOperatorWebModelRunCompletedExternally": True,
        "manualGptRecommendationRows": int(row_count),
        "finalHumanDecisionRows": 0,
        "candidateStoreWriteRows": 0,
        "vectorIndexing": False,
        "indexEligibleRows": 0,
        "strictEvidencePromotionRows": 0,
        "runtimeAnswerVisibleExposureRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "answerabilityGateBypassRows": 0,
        "cropWriteRows": 0,
        "pageImageWriteRows": 0,
        "wholeImageWriteRows": 0,
        "wholeImageGptRows": 0,
        "candidateStoreMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
    }


def _validation_policy() -> dict[str, Any]:
    return {
        "allowedUse": "recommendation_only",
        "recommendationsAreFinalHumanDecisions": False,
        "recommendationsCanAuthorizeApply": False,
        "recommendationsCanWriteCandidateStore": False,
        "projectSideGateOwnsFinalDecision": True,
        "gptRole": "advisory_image_layout_recommendation_only",
        "allowedSuggestedDecisions": list(HUMAN_DECISIONS),
        "nextPracticalWork": NEXT_RECOMMENDED_TRANCHE,
    }


def _required_contract_violations(row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if normalize_text(row.get("suggestedDecision")) not in HUMAN_DECISIONS:
        violations.append("suggested_decision_not_allowed")
    if not normalize_text(row.get("sourceDecisionRowId")):
        violations.append("missing_source_decision_row_id")
    if not normalize_text(row.get("hintCandidateId")):
        violations.append("missing_hint_candidate_id")
    if not normalize_text(row.get("sourceCandidateId")):
        violations.append("missing_source_candidate_id")
    if not normalize_text(row.get("recommendationRationale")):
        violations.append("missing_recommendation_rationale")
    if not normalize_text(row.get("risk")):
        violations.append("missing_risk")
    if row.get("needsHumanCheck") is not True:
        violations.append("needs_human_check_not_true")
    for field in (
        "finalHumanDecision",
        "applyAllowed",
        "candidateStoreWrite",
        "strictEvidence",
        "citationGrade",
        "answerableWithoutTextEvidence",
        "runtimeVisible",
        "indexEligible",
    ):
        if row.get(field) is not False:
            violations.append(f"{field}_not_false")
    if _contains_placeholder(row):
        violations.append("placeholder_text_present")
    if _contains_private_path(row):
        violations.append("private_path_leak")
    return violations


def _matched_row(
    *,
    output_row: dict[str, Any],
    pack_row: dict[str, Any],
    row_violations: list[str],
) -> dict[str, Any]:
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_VALIDATION_ROW_SCHEMA_ID,
        "sourceDecisionRowId": normalize_text(output_row.get("sourceDecisionRowId")),
        "sourcePackDecisionRowId": normalize_text(pack_row.get("decisionRowId")),
        "hintCandidateId": normalize_text(output_row.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(output_row.get("sourceCandidateId")),
        "paperId": normalize_text(pack_row.get("paperId")),
        "paperRef": normalize_text(pack_row.get("paperRef")),
        "sourceContentHash": normalize_text(pack_row.get("sourceContentHash")),
        "page": int(pack_row.get("page") or 0),
        "bbox": list(pack_row.get("bbox") or []),
        "candidateType": normalize_text(pack_row.get("candidateType")),
        "suggestedDecision": normalize_text(output_row.get("suggestedDecision")),
        "recommendationRationale": normalize_text(output_row.get("recommendationRationale")),
        "risk": normalize_text(output_row.get("risk")),
        "needsHumanCheck": output_row.get("needsHumanCheck") is True,
        "policy": {
            "allowedUse": "recommendation_only",
            "finalHumanDecision": False,
            "applyAllowed": False,
            "candidateStoreWrite": False,
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
        "validation": {
            "matchedPackRow": True,
            "matchedDecisionRowId": normalize_text(output_row.get("sourceDecisionRowId"))
            == normalize_text(pack_row.get("decisionRowId")),
            "matchedSourceCandidateId": normalize_text(output_row.get("sourceCandidateId"))
            == normalize_text(pack_row.get("sourceCandidateId")),
            "policyCompliant": not row_violations,
            "violationReasons": row_violations,
        },
    }


def _schema_like_errors(output: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(output, dict):
        return ["output_not_object"]
    if output.get("schema") != VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_SCHEMA_ID:
        errors.append("unexpected_output_schema")
    rows = output.get("rows")
    if not isinstance(rows, list):
        errors.append("rows_not_array")
        return errors
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"row_{index}_not_object")
    return errors


def build_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation(
    output: dict[str, Any],
    gpt_review_pack: dict[str, Any],
    *,
    output_ref: str = DEFAULT_OUTPUT_REF,
    source_gpt_review_pack_ref: str = DEFAULT_PACK_REF,
    generated_at: str | None = None,
) -> dict[str, Any]:
    pack_rows = _pack_rows(gpt_review_pack)
    pack_by_hint = _rows_by_hint_id(pack_rows)
    output_rows = _output_rows(output)
    output_by_hint = _rows_by_hint_id(output_rows)
    duplicate_ids = _duplicate_hint_ids(output_rows)
    expected_ids = set(pack_by_hint)
    output_ids = {normalize_text(row.get("hintCandidateId")) for row in output_rows if normalize_text(row.get("hintCandidateId"))}
    missing_ids = sorted(expected_ids - output_ids)
    extra_ids = sorted(output_ids - expected_ids)
    schema_errors = _schema_like_errors(output)

    violations: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    policy_violation_ids: set[str] = set()
    private_path_leak_ids: set[str] = set()
    placeholder_ids: set[str] = set()

    for error in schema_errors:
        violations.append({"hintCandidateId": "", "kind": "output_schema_violation", "message": error})
    for hint_id in duplicate_ids:
        violations.append(
            {
                "hintCandidateId": hint_id,
                "kind": "duplicate_hint_candidate_id",
                "message": "Output contains more than one row for hintCandidateId.",
            }
        )
    for hint_id in missing_ids:
        violations.append(
            {
                "hintCandidateId": hint_id,
                "kind": "missing_hint_candidate_id",
                "message": "Output is missing a required GPT review-pack row.",
            }
        )
    for hint_id in extra_ids:
        violations.append(
            {
                "hintCandidateId": hint_id,
                "kind": "extra_hint_candidate_id",
                "message": "Output contains a row absent from the GPT review pack.",
            }
        )

    for output_row in output_rows:
        hint_id = normalize_text(output_row.get("hintCandidateId"))
        row_violations = _required_contract_violations(output_row)
        pack_row = pack_by_hint.get(hint_id)
        if pack_row:
            if normalize_text(output_row.get("sourceDecisionRowId")) != normalize_text(pack_row.get("decisionRowId")):
                row_violations.append("source_decision_row_id_mismatch")
            if normalize_text(output_row.get("sourceCandidateId")) != normalize_text(pack_row.get("sourceCandidateId")):
                row_violations.append("source_candidate_id_mismatch")
        if row_violations:
            policy_violation_ids.add(hint_id)
            if "private_path_leak" in row_violations:
                private_path_leak_ids.add(hint_id)
            if "placeholder_text_present" in row_violations:
                placeholder_ids.add(hint_id)
            for reason in row_violations:
                violations.append(
                    {
                        "hintCandidateId": hint_id,
                        "kind": reason,
                        "message": "Recommendation row violates advisory-only output contract.",
                    }
                )
        if pack_row and hint_id not in duplicate_ids:
            validation_rows.append(
                _matched_row(output_row=output_row, pack_row=pack_row, row_violations=row_violations)
            )

    matched_rows = [
        row
        for row in validation_rows
        if row.get("validation", {}).get("policyCompliant")
        and row.get("validation", {}).get("matchedDecisionRowId")
        and row.get("validation", {}).get("matchedSourceCandidateId")
    ]
    private_path_leak_rows = len(private_path_leak_ids)
    if _contains_private_path({key: value for key, value in output.items() if key != "rows"}):
        private_path_leak_rows += 1
        violations.append(
            {
                "hintCandidateId": "",
                "kind": "private_path_leak",
                "message": "Top-level output payload contains a private local path token.",
            }
        )

    decisions = Counter(normalize_text(row.get("suggestedDecision")) for row in output_rows)
    counts = {
        "sourcePackRows": len(pack_rows),
        "outputRows": len(output_rows),
        "matchedRows": len(matched_rows),
        "missingRows": len(missing_ids),
        "extraRows": len(extra_ids),
        "duplicateRows": len(duplicate_ids),
        "approvedRecommendationRows": int(decisions.get("approve_store_candidate_only", 0)),
        "holdRecommendationRows": int(decisions.get("hold_pending_more_context", 0)),
        "rejectRecommendationRows": int(decisions.get("reject_visual_hint_candidate", 0)),
        "recropRecommendationRows": int(decisions.get("request_recrop_or_reannotation", 0)),
        "policyViolationRows": len(policy_violation_ids),
        "placeholderRows": len(placeholder_ids),
        "finalHumanDecisionRows": sum(1 for row in output_rows if row.get("finalHumanDecision") is not False),
        "applyAllowedRows": sum(1 for row in output_rows if row.get("applyAllowed") is not False),
        "candidateStoreWriteRows": sum(1 for row in output_rows if row.get("candidateStoreWrite") is not False),
        "strictEvidenceRows": sum(1 for row in output_rows if row.get("strictEvidence") is not False),
        "citationGradeRows": sum(1 for row in output_rows if row.get("citationGrade") is not False),
        "answerableWithoutTextEvidenceRows": sum(
            1 for row in output_rows if row.get("answerableWithoutTextEvidence") is not False
        ),
        "runtimeVisibleRows": sum(1 for row in output_rows if row.get("runtimeVisible") is not False),
        "indexEligibleRows": sum(1 for row in output_rows if row.get("indexEligible") is not False),
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": len(schema_errors),
        "blockedRows": 0,
    }
    counts["blockedRows"] = int(
        counts["missingRows"]
        + counts["extraRows"]
        + counts["duplicateRows"]
        + counts["policyViolationRows"]
        + counts["privatePathLeakRows"]
        + counts["schemaViolationCount"]
    )

    report: dict[str, Any] = {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_VALIDATION_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "sourceRecommendationOutput": {
            "schema": normalize_text(output.get("schema")),
            "reportRef": normalize_text(output_ref),
            "rows": len(output_rows),
        },
        "sourceGptReviewPack": {
            "schema": normalize_text(gpt_review_pack.get("schema")),
            "status": normalize_text(gpt_review_pack.get("status")),
            "reportRef": normalize_text(source_gpt_review_pack_ref),
            "packRows": len(pack_rows),
        },
        "scope": _scope(len(output_rows)),
        "policy": _validation_policy(),
        "counts": counts,
        "recommendationRowsDetail": validation_rows,
        "violations": violations,
        "warnings": [
            "GPT recommendations are advisory only and are not final human/product decisions.",
            "This validation does not approve rows, write the candidate store, index vectors, or expose hints at runtime.",
            "Project-side gates own storage, indexing, evidence, and answerability decisions.",
            "The next practical work should return to image/layout annotation expansion, not another approval architecture gate.",
        ],
    }
    if (
        output.get("schema")
        != VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_SCHEMA_ID
        or gpt_review_pack.get("schema") != VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_REVIEW_PACK_SCHEMA_ID
        or gpt_review_pack.get("status") != "ready"
        or counts["blockedRows"]
        or counts["matchedRows"] != counts["sourcePackRows"]
        or counts["outputRows"] != counts["sourcePackRows"]
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    lines = [
        "# Visual Retrieval Hint GPT Recommendation Output Validation 001",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- sourcePackRows: `{counts.get('sourcePackRows')}`",
        f"- outputRows: `{counts.get('outputRows')}`",
        f"- matchedRows: `{counts.get('matchedRows')}`",
        f"- approvedRecommendationRows: `{counts.get('approvedRecommendationRows')}`",
        f"- holdRecommendationRows: `{counts.get('holdRecommendationRows')}`",
        f"- recropRecommendationRows: `{counts.get('recropRecommendationRows')}`",
        f"- rejectRecommendationRows: `{counts.get('rejectRecommendationRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        "",
        "## Boundary",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- modelCalls: `{scope.get('modelCalls')}`",
        f"- webModelCalls: `{scope.get('webModelCalls')}`",
        f"- finalHumanDecisionRows: `{counts.get('finalHumanDecisionRows')}`",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        f"- indexEligibleRows: `{counts.get('indexEligibleRows')}`",
        f"- runtimeVisibleRows: `{counts.get('runtimeVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        "",
        "## Recommendation Summary",
        "",
        "| suggestedDecision | rows |",
        "|---|---:|",
        f"| approve_store_candidate_only | {counts.get('approvedRecommendationRows')} |",
        f"| hold_pending_more_context | {counts.get('holdRecommendationRows')} |",
        f"| request_recrop_or_reannotation | {counts.get('recropRecommendationRows')} |",
        f"| reject_visual_hint_candidate | {counts.get('rejectRecommendationRows')} |",
    ]
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": str(report_json), "markdown": str(report_md)}


__all__ = [
    "DEFAULT_OUTPUT_REF",
    "DEFAULT_PACK_REF",
    "NEXT_RECOMMENDED_TRANCHE",
    "READY_DECISION",
    "VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_VALIDATION_SCHEMA_ID",
    "build_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation",
    "load_json",
    "render_markdown_report",
    "sanitized_report_ref",
    "write_visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_validation",
]
