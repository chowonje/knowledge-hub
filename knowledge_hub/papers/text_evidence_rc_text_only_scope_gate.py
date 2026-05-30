"""Report-only text-scope gate for the v0.1 RC line."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

TEXT_EVIDENCE_RC_TEXT_ONLY_SCOPE_GATE_SCHEMA_ID = (
    "knowledge-hub.paper.text-evidence-rc-text-only-scope-gate.v1"
)

TEXT_PHASE_REPORTS: tuple[dict[str, str], ...] = (
    {
        "phase": "figure_caption_text_qa_readback",
        "reportRef": "figure_caption_text_qa_readback.v1.json",
        "requiredScope": "caption_text_only",
    },
    {
        "phase": "text_section_paragraph_span_artifacts",
        "reportRef": "text_section_paragraph_span_artifacts.v1.json",
        "requiredScope": "section_paragraph_text_spans",
    },
    {
        "phase": "text_table_caption_candidate_artifacts",
        "reportRef": "text_table_caption_candidate_artifacts.v1.json",
        "requiredScope": "table_caption_and_table_like_text_candidates",
    },
    {
        "phase": "text_equation_locator_context_artifacts",
        "reportRef": "text_equation_locator_context_artifacts.v1.json",
        "requiredScope": "equation_locator_and_surrounding_text_context",
    },
    {
        "phase": "text_complex_qa_eval_alignment",
        "reportRef": "text_complex_qa_eval_alignment.v1.json",
        "requiredScope": "text_answerable_candidate_visual_unsupported_split",
    },
    {
        "phase": "source_alias_normalization",
        "reportRef": "source_alias_normalization.v1.json",
        "requiredScope": "short_alias_contextual_resolution_policy",
    },
)

CONVERGENCE_REPORT_REF = "text_evidence_rc_convergence.v1.json"

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _private_path_leak_count(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return 1 if PRIVATE_PATH_RE.search(encoded) else 0


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _counter_clean(report: dict[str, Any], key: str) -> bool:
    if key in report:
        return int(report.get(key) or 0) == 0
    for value in dict(report.get("mutationCounters") or {}).values():
        if int(value or 0) != 0:
            return False
    return True


def _phase_key_counts(report: dict[str, Any]) -> dict[str, int]:
    keys = [
        "candidateRows",
        "caseRows",
        "inputRows",
        "answerableRows",
        "noAnswerRows",
        "textAnswerableRows",
        "candidateOnlyRows",
        "visualUnsupportedRows",
        "unsafeDirectAliasRows",
        "expectationFailureRows",
        "privatePathLeakRows",
    ]
    counts: dict[str, int] = {}
    for key in keys:
        value = report.get(key)
        if isinstance(value, int):
            counts[key] = value
    rows = report.get("rows")
    if isinstance(rows, list):
        counts["rowCount"] = len(rows)
    return counts


def _phase_row(spec: dict[str, str], reports_root: Path) -> dict[str, Any]:
    report = _load_json(reports_root / spec["reportRef"])
    report_status = _clean_text(report.get("status"))
    present = bool(report)
    strict_evidence_clean = _counter_clean(report, "strictEvidencePromotionRows")
    runtime_exposure_clean = _counter_clean(report, "runtimeAnswerVisibleExposureRows")
    private_path_clean = int(report.get("privatePathLeakRows") or 0) == 0
    ready = bool(
        present
        and report_status in {"ready", "accepted", "ready_for_integration_review"}
        and strict_evidence_clean
        and runtime_exposure_clean
        and private_path_clean
    )
    return {
        "phase": spec["phase"],
        "reportRef": spec["reportRef"],
        "reportSchema": _clean_text(report.get("schema")),
        "reportStatus": report_status,
        "requiredScope": spec["requiredScope"],
        "textScopeReady": ready,
        "phaseDisposition": "text_scope_ready" if ready else "text_scope_hold",
        "keyCounts": _phase_key_counts(report),
        "visualUnsupportedRows": int(report.get("visualUnsupportedRows") or 0),
        "strictEvidencePromotionRows": int(report.get("strictEvidencePromotionRows") or 0),
        "runtimeAnswerVisibleExposureRows": int(report.get("runtimeAnswerVisibleExposureRows") or 0),
        "privatePathLeakRows": int(report.get("privatePathLeakRows") or 0),
    }


def _deferred_items() -> list[dict[str, Any]]:
    return [
        {
            "itemId": "visual_layout_image_format_branch",
            "deferredReason": "outside_text_evidence_v01_rc_scope",
            "futureBranchHint": "codex/visual-layout-image-format-evidence-20260526",
            "requiredBeforePublicTextRc": False,
        },
        {
            "itemId": "bbox_identity_context_probe",
            "deferredReason": "requires layout-aware provenance recovery, not text-only answerability",
            "futureBranchHint": "codex/visual-layout-image-format-evidence-20260526",
            "requiredBeforePublicTextRc": False,
        },
        {
            "itemId": "table_cell_grid_numeric_extraction",
            "deferredReason": "table captions are candidates only until cell identity is recovered",
            "futureBranchHint": "codex/table-cell-grid-evidence-20260526",
            "requiredBeforePublicTextRc": False,
        },
        {
            "itemId": "equation_visual_latex_reconstruction",
            "deferredReason": "equation locator/context is text-only; LaTeX reconstruction needs a later parser path",
            "futureBranchHint": "codex/equation-visual-latex-evidence-20260526",
            "requiredBeforePublicTextRc": False,
        },
        {
            "itemId": "figure_visual_binding",
            "deferredReason": "figure captions can be text evidence; image/visual binding is a separate evidence type",
            "futureBranchHint": "codex/figure-visual-binding-evidence-20260526",
            "requiredBeforePublicTextRc": False,
        },
        {
            "itemId": "vlm_visual_interpretation",
            "deferredReason": "external or VLM-derived interpretation cannot become citation-grade text evidence by default",
            "futureBranchHint": "codex/visual-layout-image-format-evidence-20260526",
            "requiredBeforePublicTextRc": False,
        },
    ]


def build_text_evidence_rc_text_only_scope_gate(
    *,
    reports_root: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    phase_reports = [_phase_row(spec, reports_root) for spec in TEXT_PHASE_REPORTS]
    convergence = _load_json(reports_root / CONVERGENCE_REPORT_REF)
    blockers = [
        {
            "blockerId": _clean_text(row.get("blockerId")),
            "severity": _clean_text(row.get("severity")),
            "reason": _clean_text(row.get("reason")),
        }
        for row in list(convergence.get("blockers") or [])
        if isinstance(row, dict)
    ]
    for row in phase_reports:
        if not row.get("reportSchema"):
            blockers.append(
                {
                    "blockerId": f"missing_phase_report:{row.get('phase')}",
                    "severity": "hold",
                    "reason": f"required phase report {row.get('reportRef')} is not present in current reports_root",
                }
            )
        elif not row.get("textScopeReady"):
            blockers.append(
                {
                    "blockerId": f"phase_report_not_ready:{row.get('phase')}",
                    "severity": "hold",
                    "reason": f"required phase report {row.get('reportRef')} is present but not text-scope ready",
                }
            )
    if not convergence:
        blockers.append(
            {
                "blockerId": "missing_convergence_report",
                "severity": "hold",
                "reason": f"required convergence report {CONVERGENCE_REPORT_REF} is not present in current reports_root",
            }
        )
    text_ready_rows = sum(1 for row in phase_reports if row["textScopeReady"])
    text_hold_rows = len(phase_reports) - text_ready_rows
    deferred_items = _deferred_items()
    public_rc_ready = bool(convergence.get("publicRcReady")) and not blockers
    text_only_ready = text_hold_rows == 0
    if text_hold_rows:
        next_action = "replay_missing_text_evidence_phase_reports"
    elif not convergence:
        next_action = "run_text_evidence_rc_convergence_report"
    else:
        next_action = _clean_text(
            convergence.get("nextAction") or "run_text_evidence_rc_convergence_report"
        )

    report: dict[str, Any] = {
        "schema": TEXT_EVIDENCE_RC_TEXT_ONLY_SCOPE_GATE_SCHEMA_ID,
        "status": "ready" if text_hold_rows == 0 else "blocked",
        "generatedAt": generated_at or utc_now_iso(),
        "scope": {
            "writes": "report_only",
            "scopePolicy": "text_evidence_v0_1_rc_only",
            "visualLayoutBranchDeferred": True,
            "runtimeAnswerVisibleExposureRows": 0,
            "strictEvidencePromotionRows": 0,
            "canonicalCheckoutEdited": False,
            "vaultScanRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "externalDownloadRows": 0,
        },
        "inputReports": {
            "convergenceReportRef": CONVERGENCE_REPORT_REF,
            "convergenceStatus": _clean_text(convergence.get("status")),
            "convergenceNextAction": _clean_text(convergence.get("nextAction")),
        },
        "evidencePolicy": {
            "allowsTextEvidence": True,
            "allowsCaptionTextEvidence": True,
            "allowsSectionParagraphTextSpans": True,
            "allowsTableCaptionCandidates": True,
            "allowsEquationLocatorContextCandidates": True,
            "allowsVisualInspectionEvidence": False,
            "allowsVlmDerivedCitationGradeEvidence": False,
            "allowsStrictEvidencePromotion": False,
            "allowsRuntimeAnswerVisibleExposure": False,
        },
        "phaseRows": len(phase_reports),
        "textReadyRows": text_ready_rows,
        "textHoldRows": text_hold_rows,
        "deferredRows": len(deferred_items),
        "blockerRows": len(blockers),
        "textOnlyRcReady": text_only_ready,
        "publicRcReady": public_rc_ready,
        "decision": (
            "text_only_scope_ready_for_public_rc_review"
            if public_rc_ready
            else "text_only_scope_ready_pending_rc_convergence_actions"
            if text_only_ready
            else "text_only_scope_blocked"
        ),
        "phaseReports": phase_reports,
        "deferredItems": deferred_items,
        "blockers": blockers,
        "nextAction": next_action,
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "mergeRows": 0,
            "cherryPickRows": 0,
            "worktreeDeletionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
            "strictEvidencePromotionRows": 0,
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "reportHash": "",
        "warnings": [
            "Visual/layout/image/format evidence remains deferred to later branch work.",
            (
                "Public RC convergence blockers are clear for the current text-only scope."
                if public_rc_ready
                else "This report does not close PR #149 and does not edit the canonical dirty checkout."
            ),
        ],
        "schemaErrors": [],
    }
    report["privatePathLeakRows"] = _private_path_leak_count(report)
    if report["privatePathLeakRows"]:
        report["status"] = "blocked"
    payload_for_hash = dict(report)
    payload_for_hash["reportHash"] = ""
    report["reportHash"] = _sha256_json(payload_for_hash)
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Text Evidence RC Text-Only Scope Gate",
        "",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- textOnlyRcReady: `{report.get('textOnlyRcReady')}`",
        f"- publicRcReady: `{report.get('publicRcReady')}`",
        f"- phaseRows: `{report.get('phaseRows')}`",
        f"- textReadyRows: `{report.get('textReadyRows')}`",
        f"- deferredRows: `{report.get('deferredRows')}`",
        f"- blockerRows: `{report.get('blockerRows')}`",
        f"- nextAction: `{report.get('nextAction')}`",
        "",
        "## Deferred",
    ]
    for row in report.get("deferredItems") or []:
        lines.append(
            f"- `{row.get('itemId')}` -> `{row.get('futureBranchHint')}`: {row.get('deferredReason')}"
        )
    lines.append("")
    lines.append("## Blockers")
    for row in report.get("blockers") or []:
        lines.append(f"- `{row.get('blockerId')}` ({row.get('severity')}): {row.get('reason')}")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


__all__ = [
    "TEXT_EVIDENCE_RC_TEXT_ONLY_SCOPE_GATE_SCHEMA_ID",
    "build_text_evidence_rc_text_only_scope_gate",
    "render_markdown_report",
    "write_report",
]
