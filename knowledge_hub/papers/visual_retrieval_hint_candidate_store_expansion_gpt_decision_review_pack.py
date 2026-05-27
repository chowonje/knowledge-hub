"""Build a GPT-facing recommendation pack for visual hint store decisions.

The pack is for manual web GPT/Pro recommendation assistance only. GPT output
is not a final human decision, does not approve rows, and cannot write the
candidate store, index vectors, promote evidence, or expose hints at runtime.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Sequence

from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_human_product_decision_record import (
    HUMAN_DECISIONS,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_RECORD_SCHEMA_ID,
)
from knowledge_hub.papers.visual_retrieval_hint_candidate_store_expansion_review import (
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_SCHEMA_ID,
)


VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-gpt-decision-review-pack.v1"
)
VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-gpt-decision-recommendation-output.v1"
)
VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_BATCH_TEMPLATE_SCHEMA_ID = (
    "knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-gpt-decision-recommendation-batch-template.v1"
)

READY_DECISION = "ready_for_manual_gpt_decision_recommendation_run"
NEXT_RECOMMENDED_TRANCHE = "visual_retrieval_hint_candidate_store_expansion_gpt_recommendation_output_capture"
DEFAULT_PACK_ID = "visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack_001"
DEFAULT_PACK_DIR_REF = (
    "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack_001"
)

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


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


def _scope(batch_count: int, artifact_count: int) -> dict[str, Any]:
    return {
        "writes": "report_and_operator_prompt_files_only",
        "apiCalls": False,
        "modelCalls": False,
        "webModelCalls": False,
        "manualOperatorWebModelRunRequired": True,
        "operatorPromptRows": int(batch_count),
        "recommendationTemplateRows": int(batch_count),
        "bundleArtifactRows": int(artifact_count),
        "completedGptRecommendationRows": 0,
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
        "canonicalParsedArtifactWriteRows": 0,
    }


def _rows_by_hint_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {normalize_text(row.get("hintCandidateId")): row for row in rows}


def _decision_rows(decision_record: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in list(decision_record.get("decisionRowsDetail") or []) if isinstance(row, dict)]


def _review_rows(review_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in list(review_report.get("reviewRowsDetail") or []) if isinstance(row, dict)]


def _pack_row(index: int, decision_row: dict[str, Any], review_row: dict[str, Any] | None) -> dict[str, Any]:
    review_row = review_row or {}
    return {
        "rowNumber": index,
        "decisionRowId": normalize_text(decision_row.get("decisionRowId")),
        "sourceReviewRowId": normalize_text(decision_row.get("sourceReviewRowId")),
        "hintCandidateId": normalize_text(decision_row.get("hintCandidateId")),
        "sourceCandidateId": normalize_text(decision_row.get("sourceCandidateId")),
        "paperId": normalize_text(decision_row.get("paperId")),
        "paperRef": normalize_text(decision_row.get("paperRef")),
        "sourceContentHash": normalize_text(decision_row.get("sourceContentHash")),
        "page": int(decision_row.get("page") or 0),
        "bbox": list(decision_row.get("bbox") or []),
        "candidateType": normalize_text(decision_row.get("candidateType")),
        "currentDecision": normalize_text(decision_row.get("decision")),
        "derivedTextForRetrievalSnippet": normalize_text(review_row.get("derivedTextForRetrievalSnippet")),
        "visibleTextSnippet": normalize_text(review_row.get("visibleTextSnippet")),
        "retrievalKeywordCount": int(review_row.get("retrievalKeywordCount") or 0),
        "allowedSuggestedDecisions": list(HUMAN_DECISIONS),
        "gptOutputPolicy": {
            "finalHumanDecision": False,
            "applyAllowed": False,
            "candidateStoreWrite": False,
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
    }


def _chunks(rows: Sequence[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [list(rows[index : index + size]) for index in range(0, len(rows), size)]


def _batch_stem(batch_number: int) -> str:
    return f"batch_{int(batch_number):02d}"


def _recommendation_template(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_SCHEMA_ID,
        "rows": [
            {
                "sourceDecisionRowId": row.get("decisionRowId"),
                "hintCandidateId": row.get("hintCandidateId"),
                "sourceCandidateId": row.get("sourceCandidateId"),
                "suggestedDecision": "FILL_IN_ONE_ALLOWED_DECISION",
                "recommendationRationale": "FILL_IN_SHORT_REASON",
                "risk": "FILL_IN_LOW_MEDIUM_HIGH_WITH_REASON",
                "needsHumanCheck": True,
                "finalHumanDecision": False,
                "applyAllowed": False,
                "candidateStoreWrite": False,
                "strictEvidence": False,
                "citationGrade": False,
                "answerableWithoutTextEvidence": False,
                "runtimeVisible": False,
                "indexEligible": False,
            }
            for row in rows
        ],
    }


def build_gpt_recommendation_batch_template(batch_bundle: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in list(batch_bundle.get("rows") or []) if isinstance(row, dict)]
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_BATCH_TEMPLATE_SCHEMA_ID,
        "batchId": normalize_text(batch_bundle.get("batchId")),
        "batchNumber": int(batch_bundle.get("batchNumber") or 0),
        "targetOutputSchema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_SCHEMA_ID,
        "rows": _recommendation_template(rows)["rows"],
        "warnings": [
            "This is a recommendation template, not completed GPT output.",
            "GPT recommendations are not final human decisions.",
            "Keep finalHumanDecision=false, applyAllowed=false, candidateStoreWrite=false, strictEvidence=false, citationGrade=false, runtimeVisible=false, and indexEligible=false.",
        ],
    }


def render_gpt_review_batch_prompt(batch_bundle: dict[str, Any]) -> str:
    rows = [row for row in list(batch_bundle.get("rows") or []) if isinstance(row, dict)]
    skeleton = _recommendation_template(rows)
    lines = [
        f"# Visual Retrieval Hint Decision Recommendation Batch {int(batch_bundle.get('batchNumber') or 0):02d}",
        "",
        "Use only the row text provided in this prompt. Do not use outside sources, web search, PDFs, screenshots, full pages, or images.",
        "Your job is to recommend a decision for a human/product reviewer. You are not making the final decision.",
        f"Return JSON only with top-level schema `{VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_SCHEMA_ID}` and a `rows` array.",
        "For every row, set `finalHumanDecision=false`, `applyAllowed=false`, `candidateStoreWrite=false`, `strictEvidence=false`, `citationGrade=false`, `answerableWithoutTextEvidence=false`, `runtimeVisible=false`, and `indexEligible=false`.",
        "",
        "Allowed `suggestedDecision` values:",
    ]
    for decision in HUMAN_DECISIONS:
        lines.append(f"- `{decision}`")
    lines.extend(
        [
            "",
            "Decision guidance:",
            "- Suggest `approve_store_candidate_only` only when the retrieval hint is coherent, useful for future retrieval, and clearly remains non-evidence.",
            "- Suggest `hold_pending_more_context` when the row may be useful but the text is incomplete, uncertain, or needs product review.",
            "- Suggest `reject_visual_hint_candidate` when the row is off-topic, too noisy, or not useful as a retrieval hint.",
            "- Suggest `request_recrop_or_reannotation` when the row appears affected by crop/context problems or a likely annotation mismatch.",
            "",
            "## Rows To Review",
            "",
            "| # | paperId | type | page | currentDecision | hintCandidateId | sourceCandidateId |",
            "|---:|---|---|---:|---|---|---|",
        ]
    )
    for index, row in enumerate(rows, start=1):
        lines.append(
            "| {index} | {paperId} | {candidateType} | {page} | `{decision}` | `{hint}` | `{source}` |".format(
                index=index,
                paperId=row.get("paperId"),
                candidateType=row.get("candidateType"),
                page=row.get("page"),
                decision=row.get("currentDecision"),
                hint=row.get("hintCandidateId"),
                source=row.get("sourceCandidateId"),
            )
        )
    lines.extend(["", "## Row Context", ""])
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"### Row {index}",
                f"- sourceDecisionRowId: `{row.get('decisionRowId')}`",
                f"- hintCandidateId: `{row.get('hintCandidateId')}`",
                f"- paperId: `{row.get('paperId')}`",
                f"- candidateType: `{row.get('candidateType')}`",
                f"- page: `{row.get('page')}`",
                f"- currentDecision: `{row.get('currentDecision')}`",
                f"- derivedTextForRetrievalSnippet: {row.get('derivedTextForRetrievalSnippet')}",
                f"- visibleTextSnippet: {row.get('visibleTextSnippet')}",
                f"- retrievalKeywordCount: `{row.get('retrievalKeywordCount')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## JSON Shape To Return",
            "",
            "```json",
            json.dumps(skeleton, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack(
    decision_record: dict[str, Any],
    review_report: dict[str, Any],
    *,
    pack_id: str = DEFAULT_PACK_ID,
    source_decision_record_ref: str = (
        "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_human_product_decision_record.v1.json"
    ),
    source_review_report_ref: str = (
        "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_review.v1.json"
    ),
    pack_dir_ref: str = DEFAULT_PACK_DIR_REF,
    batch_size: int = 8,
    generated_at: str | None = None,
) -> dict[str, Any]:
    decision_rows = _decision_rows(decision_record)
    review_by_hint = _rows_by_hint_id(_review_rows(review_report))
    pack_rows = [
        _pack_row(index, row, review_by_hint.get(normalize_text(row.get("hintCandidateId"))))
        for index, row in enumerate(decision_rows, start=1)
    ]
    batch_bundles: list[dict[str, Any]] = []
    for batch_index, rows in enumerate(_chunks(pack_rows, batch_size), start=1):
        stem = _batch_stem(batch_index)
        batch_bundles.append(
            {
                "batchId": f"{pack_id}_{stem}",
                "batchNumber": batch_index,
                "rowCount": len(rows),
                "promptRef": f"{pack_dir_ref}/{stem}_prompt.md",
                "recommendationTemplateRef": f"{pack_dir_ref}/{stem}_recommendation_template.v1.json",
                "rows": rows,
            }
        )
    private_path_leak_rows = 1 if _contains_private_path(batch_bundles) else 0
    artifact_rows = len(batch_bundles) * 2
    counts = {
        "sourceDecisionRows": len(decision_rows),
        "sourceReviewRows": len(_review_rows(review_report)),
        "gptReviewRows": len(pack_rows),
        "batchRows": len(batch_bundles),
        "operatorPromptRows": len(batch_bundles),
        "recommendationTemplateRows": len(batch_bundles),
        "bundleArtifactRows": artifact_rows,
        "completedGptRecommendationRows": 0,
        "finalHumanDecisionRows": 0,
        "applyDesignCandidateRows": 0,
        "candidateStoreWriteRows": 0,
        "privatePathLeakRows": int(private_path_leak_rows),
        "schemaViolationCount": 0,
    }
    report: dict[str, Any] = {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_REVIEW_PACK_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION,
        "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        "packId": pack_id,
        "sourceDecisionRecord": {
            "schema": normalize_text(decision_record.get("schema")),
            "status": normalize_text(decision_record.get("status")),
            "reportRef": normalize_text(source_decision_record_ref),
            "decisionRows": len(decision_rows),
            "humanDecisionRows": int(dict(decision_record.get("counts") or {}).get("humanDecisionRows") or 0),
            "candidateStoreWriteRows": int(dict(decision_record.get("counts") or {}).get("candidateStoreWriteRows") or 0),
        },
        "sourceReviewReport": {
            "schema": normalize_text(review_report.get("schema")),
            "status": normalize_text(review_report.get("status")),
            "reportRef": normalize_text(source_review_report_ref),
            "reviewRows": len(_review_rows(review_report)),
            "blockedRows": int(dict(review_report.get("counts") or {}).get("blockedRows") or 0),
        },
        "targetRecommendationOutput": {
            "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_SCHEMA_ID,
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
        "scope": _scope(batch_count=len(batch_bundles), artifact_count=artifact_rows),
        "counts": counts,
        "batchBundles": batch_bundles,
        "operatorInstructions": [
            "Run one batch at a time in web GPT/Pro if you want recommendation assistance.",
            "Do not attach images or PDFs for this pack; use only the text in each batch prompt.",
            "Treat GPT output as non-final recommendation data, not as the human/product decision record.",
            "A human must copy decisions into a separate decision file and validate them before any apply-design tranche.",
        ],
        "warnings": [
            "This pack is not GPT output and contains recommendation templates.",
            "GPT recommendations must not be promoted to final human decisions automatically.",
            "GPT recommendations are not strict evidence, citation-grade evidence, or runtime answer-visible text.",
            "This pack makes no in-repo model/API/web call.",
        ],
    }
    if (
        decision_record.get("schema")
        != VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DECISION_RECORD_SCHEMA_ID
        or review_report.get("schema") != VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_REVIEW_SCHEMA_ID
        or not pack_rows
        or private_path_leak_rows
        or dict(decision_record.get("counts") or {}).get("candidateStoreWriteRows")
        or dict(review_report.get("counts") or {}).get("blockedRows")
    ):
        report["status"] = "blocked"
        report["decision"] = "blocked"
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    scope = dict(report.get("scope") or {})
    lines = [
        "# Visual Retrieval Hint Candidate Store Expansion GPT Decision Review Pack",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- packId: `{report.get('packId')}`",
        f"- gptReviewRows: `{counts.get('gptReviewRows')}`",
        f"- batchRows: `{counts.get('batchRows')}`",
        f"- completedGptRecommendationRows: `{counts.get('completedGptRecommendationRows')}`",
        f"- finalHumanDecisionRows: `{counts.get('finalHumanDecisionRows')}`",
        f"- candidateStoreWriteRows: `{counts.get('candidateStoreWriteRows')}`",
        "",
        "## Boundary",
        "",
        f"- writes: `{scope.get('writes')}`",
        f"- modelCalls: `{scope.get('modelCalls')}`",
        f"- webModelCalls: `{scope.get('webModelCalls')}`",
        f"- manualOperatorWebModelRunRequired: `{scope.get('manualOperatorWebModelRunRequired')}`",
        f"- vectorIndexing: `{scope.get('vectorIndexing')}`",
        f"- runtimeAnswerVisibleExposureRows: `{scope.get('runtimeAnswerVisibleExposureRows')}`",
        f"- strictEvidencePromotionRows: `{scope.get('strictEvidencePromotionRows')}`",
        "",
        "## Batch Files",
        "",
        "| batch | rows | prompt | recommendation template |",
        "|---:|---:|---|---|",
    ]
    for batch in report.get("batchBundles", []):
        lines.append(
            "| {batch} | {rows} | `{prompt}` | `{template}` |".format(
                batch=batch.get("batchNumber"),
                rows=batch.get("rowCount"),
                prompt=batch.get("promptRef"),
                template=batch.get("recommendationTemplateRef"),
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report.get("warnings", []):
            lines.append(f"- `{warning}`")
    return "\n".join(lines).rstrip() + "\n"


def write_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
    pack_dir: Path,
) -> dict[str, Any]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    pack_dir.mkdir(parents=True, exist_ok=True)
    batch_files: list[dict[str, str]] = []
    for batch in report.get("batchBundles", []):
        stem = _batch_stem(int(batch.get("batchNumber") or 0))
        prompt_path = pack_dir / f"{stem}_prompt.md"
        template_path = pack_dir / f"{stem}_recommendation_template.v1.json"
        prompt_path.write_text(render_gpt_review_batch_prompt(batch), encoding="utf-8")
        template_path.write_text(
            json.dumps(build_gpt_recommendation_batch_template(batch), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        batch_files.append({"prompt": str(prompt_path), "recommendationTemplate": str(template_path)})
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": str(report_json), "markdown": str(report_md), "batchFiles": batch_files}


__all__ = [
    "DEFAULT_PACK_DIR_REF",
    "DEFAULT_PACK_ID",
    "NEXT_RECOMMENDED_TRANCHE",
    "READY_DECISION",
    "VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_BATCH_TEMPLATE_SCHEMA_ID",
    "VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_RECOMMENDATION_OUTPUT_SCHEMA_ID",
    "VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_GPT_REVIEW_PACK_SCHEMA_ID",
    "build_gpt_recommendation_batch_template",
    "build_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack",
    "load_json",
    "render_gpt_review_batch_prompt",
    "render_markdown_report",
    "sanitized_report_ref",
    "write_visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack",
]
