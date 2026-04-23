"""Deterministic AI canon card-quality audit helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.card_feedback import build_card_remediation_plan
from knowledge_hub.papers.memory_builder import _summary_value_is_unusable
from knowledge_hub.papers.memory_quality import is_generic_limitation
from knowledge_hub.papers.public_surface import build_public_memory_card, build_public_summary_card
from knowledge_hub.papers.text_quality import clean_text, spillover_issues

SCHEMA_ID = "knowledge-hub.paper.canon-quality-audit.result.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "ai_canon" / "ai_canon_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "ai_canon"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "canon_quality_report.json"
DEFAULT_SELECTOR_PATH = DEFAULT_OUTPUT_DIR / "canon_needs_review.txt"

_SUMMARY_FIELDS = (
    ("problem", False),
    ("coreIdea", False),
    ("whatIsNew", False),
    ("whenItMatters", False),
    ("methodSteps", True),
    ("keyResults", True),
    ("limitations", True),
)
_MEMORY_FIELDS = ("problemContext", "methodCore", "evidenceCore", "limitations")
_SUMMARY_REBUILD_ISSUES = {
    "summary_artifact_missing",
    "summary_artifact_unusable",
    "fallback_used",
    "front_matter_spillover",
    "table_caption_spillover",
    "raw_english_spillover",
}
_SOURCE_REPAIR_ISSUES = {
    "likely_semantic_mismatch",
    "front_matter_spillover",
}
_MEMORY_REBUILD_ISSUES = {
    "memory_card_missing",
    "memory_card_unusable",
    "empty_problem_context",
    "empty_method",
    "empty_evidence",
    "generic_limitation",
}
_CONCEPT_REFRESH_ISSUES = {"concept_links_missing"}


def _clean_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        raw = values
    elif isinstance(values, tuple):
        raw = list(values)
    else:
        raw = [values]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = clean_text(item)
        lowered = token.casefold()
        if not token or lowered in seen:
            continue
        seen.add(lowered)
        out.append(token)
    return out


def _append_unique(items: list[str], *values: str) -> list[str]:
    for value in values:
        token = clean_text(value)
        if token and token not in items:
            items.append(token)
    return items


def _usable_text(value: Any) -> str:
    token = clean_text(value)
    if not token or _summary_value_is_unusable(token):
        return ""
    return token


def load_canon_manifest(path: Path | None = None) -> list[dict[str, str]]:
    manifest_path = Path(path or DEFAULT_MANIFEST_PATH).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"canon manifest not found: {manifest_path}")
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {str(key or ""): clean_text(value) for key, value in dict(raw or {}).items()}
            paper_id = row.get("paper_id") or row.get("paperId") or row.get("id") or ""
            if not paper_id:
                continue
            rows.append(
                {
                    "paperId": paper_id,
                    "title": row.get("title") or "",
                    "tranche": row.get("tranche") or "",
                    "targetPrimaryLane": row.get("target_primary_lane") or row.get("targetPrimaryLane") or "",
                    "targetSecondaryTags": row.get("target_secondary_tags") or row.get("targetSecondaryTags") or "",
                    "sourceStatus": row.get("source_status") or row.get("sourceStatus") or "",
                    "cardQuality": row.get("card_quality") or row.get("cardQuality") or "",
                    "notes": row.get("notes") or "",
                }
            )
    return rows


def _scan_summary_spillover(summary: dict[str, Any], *, title: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for field_name, is_list in _SUMMARY_FIELDS:
        raw_value = summary.get(field_name)
        values = list(raw_value or []) if is_list and isinstance(raw_value, list) else [raw_value]
        for index, value in enumerate(values):
            token = clean_text(value)
            if not token:
                continue
            for issue in spillover_issues(token, title=title):
                findings.append(
                    {
                        "scope": "summary",
                        "field": field_name,
                        "index": index if is_list else None,
                        "issue": issue,
                        "excerpt": token[:240],
                    }
                )
    return findings


def _scan_memory_spillover(memory_card: dict[str, Any], *, title: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for field_name in _MEMORY_FIELDS:
        token = clean_text(memory_card.get(field_name))
        if not token:
            continue
        for issue in spillover_issues(token, title=title):
            findings.append(
                {
                    "scope": "memory",
                    "field": field_name,
                    "index": None,
                    "issue": issue,
                    "excerpt": token[:240],
                }
            )
    return findings


def _base_issues(
    *,
    summary_payload: dict[str, Any],
    memory_payload: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, Any], dict[str, Any], list[str]]:
    summary = dict(summary_payload.get("summary") or {})
    memory_card = dict(memory_payload.get("memoryCard") or {})
    warnings = _clean_list([*list(summary_payload.get("warnings") or []), *list(memory_payload.get("warnings") or [])])
    summary_status = clean_text((summary_payload.get("artifactStatus") or {}).get("summary") or summary_payload.get("status"))
    memory_status = clean_text((memory_payload.get("artifactStatus") or {}).get("memory") or memory_payload.get("status"))
    quality = dict(summary_payload.get("quality") or {})
    issues: list[str] = []

    if summary_status == "missing":
        _append_unique(issues, "summary_artifact_missing")
    elif summary_status in {"degraded", "failed", "partial", "stale"}:
        _append_unique(issues, "summary_artifact_unusable")

    if memory_status == "missing":
        _append_unique(issues, "memory_card_missing")
    elif memory_status in {"degraded", "failed", "partial", "stale"}:
        _append_unique(issues, "memory_card_unusable")

    if bool(summary_payload.get("fallbackUsed")):
        _append_unique(issues, "fallback_used")

    memory_quality_flag = clean_text(memory_card.get("qualityFlag") or memory_card.get("quality_flag") or "")
    if not _usable_text(memory_card.get("problemContext")):
        _append_unique(issues, "empty_problem_context")
    if not _usable_text(memory_card.get("methodCore")):
        _append_unique(issues, "empty_method")
    if not _usable_text(memory_card.get("evidenceCore")):
        _append_unique(issues, "empty_evidence")
    if is_generic_limitation(memory_card.get("limitations")):
        _append_unique(issues, "generic_limitation")
    if not list(memory_payload.get("conceptsDetailed") or []):
        _append_unique(issues, "concept_links_missing")

    for warning in warnings:
        if warning in {
            "summary_artifact_missing",
            "summary_artifact_unusable",
            "memory_card_missing",
            "memory_card_unusable",
            "fallback_used",
            "likely_semantic_mismatch",
            "concept_links_missing",
        }:
            _append_unique(issues, warning)

    needs_review = bool((quality.get("displayFlags") or {}).get("needsReview")) or memory_quality_flag == "needs_review"
    if needs_review and not issues:
        for reason in list(quality.get("reasons") or []):
            token = clean_text(reason)
            if token in {"likely_semantic_mismatch", "summary_artifact_unusable", "memory_card_unusable"}:
                _append_unique(issues, token)

    return issues, warnings, summary, memory_card, quality


def _remediation_plan_for_item(
    *,
    issues: list[str],
    warnings: list[str],
    summary: dict[str, Any],
    memory_card: dict[str, Any],
) -> dict[str, Any]:
    return build_card_remediation_plan(
        issues=list(issues),
        artifact_flags={
            "hasSummary": bool(_usable_text(summary.get("oneLine")) or _usable_text(summary.get("coreIdea"))),
            "hasMemory": bool(
                _usable_text(memory_card.get("paperCore"))
                or _usable_text(memory_card.get("methodCore"))
                or _usable_text(memory_card.get("evidenceCore"))
            ),
        },
        summary_snapshot={
            "oneLine": clean_text(summary.get("oneLine")),
            "coreIdea": clean_text(summary.get("coreIdea")),
            "problem": clean_text(summary.get("problem")),
            "methodSteps": list(summary.get("methodSteps") or []),
            "keyResults": list(summary.get("keyResults") or []),
            "limitations": list(summary.get("limitations") or []),
        },
        memory_snapshot={
            "paperCore": clean_text(memory_card.get("paperCore")),
            "problemContext": clean_text(memory_card.get("problemContext")),
            "methodCore": clean_text(memory_card.get("methodCore")),
            "evidenceCore": clean_text(memory_card.get("evidenceCore")),
            "limitations": clean_text(memory_card.get("limitations")),
            "qualityFlag": clean_text(memory_card.get("qualityFlag") or memory_card.get("quality_flag")),
            "conceptLinks": list(memory_card.get("conceptLinks") or []),
        },
        observed_warnings=list(warnings),
    )


def audit_canon_quality(
    khub,
    *,
    manifest_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = load_canon_manifest(manifest_path)
    sqlite_db = khub.sqlite_db()
    resolved_manifest_path = Path(manifest_path or DEFAULT_MANIFEST_PATH).expanduser().resolve()
    resolved_output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    items: list[dict[str, Any]] = []

    for row in manifest:
        paper_id = clean_text(row.get("paperId"))
        paper = sqlite_db.get_paper(paper_id) or {}
        blocked_reasons: list[str] = []
        if not paper:
            blocked_reasons.append("paper_not_found")

        summary_payload = build_public_summary_card(khub, paper_id=paper_id)
        memory_payload = build_public_memory_card(khub, paper_id=paper_id)
        summary_status = clean_text((summary_payload.get("artifactStatus") or {}).get("summary") or summary_payload.get("status"))
        memory_status = clean_text((memory_payload.get("artifactStatus") or {}).get("memory") or memory_payload.get("status"))
        issues, warnings, summary, memory_card, quality = _base_issues(
            summary_payload=summary_payload,
            memory_payload=memory_payload,
        )
        title = clean_text(memory_payload.get("paperTitle") or summary_payload.get("paperTitle") or row.get("title") or paper_id)
        spillover_findings = _scan_summary_spillover(summary, title=title) + _scan_memory_spillover(memory_card, title=title)
        for finding in spillover_findings:
            _append_unique(issues, clean_text(finding.get("issue")))

        plan = _remediation_plan_for_item(
            issues=issues,
            warnings=warnings,
            summary=summary,
            memory_card=memory_card,
        )
        proposed_actions = [clean_text(action.get("code")) for action in list(plan.get("actions") or []) if clean_text(action.get("code"))]
        memory_quality_flag = clean_text(memory_card.get("qualityFlag") or memory_card.get("quality_flag"))
        needs_review = bool(
            issues
            or blocked_reasons
            or bool((quality.get("displayFlags") or {}).get("needsReview"))
            or memory_quality_flag == "needs_review"
            or summary_status != "ok"
            or memory_status != "ok"
        )

        items.append(
            {
                "paperId": paper_id,
                "title": title,
                "tranche": clean_text(row.get("tranche")),
                "summaryStatus": summary_status or "missing",
                "memoryStatus": memory_status or "missing",
                "memoryQualityFlag": memory_quality_flag or "unscored",
                "issues": list(issues),
                "spilloverFindings": spillover_findings,
                "observedWarnings": warnings,
                "proposedActions": proposed_actions,
                "blockedReasons": blocked_reasons,
                "needsReview": bool(needs_review),
                "manifest": {
                    "targetPrimaryLane": clean_text(row.get("targetPrimaryLane")),
                    "targetSecondaryTags": clean_text(row.get("targetSecondaryTags")),
                    "sourceStatus": clean_text(row.get("sourceStatus")),
                    "cardQuality": clean_text(row.get("cardQuality")),
                    "notes": clean_text(row.get("notes")),
                },
            }
        )

    needs_review_ids = [str(item.get("paperId") or "") for item in items if bool(item.get("needsReview"))]
    counts = {
        "ok": len([item for item in items if not bool(item.get("needsReview"))]),
        "needsReview": len(needs_review_ids),
        "blocked": len([item for item in items if list(item.get("blockedReasons") or [])]),
    }
    return {
        "schema": SCHEMA_ID,
        "status": "ok",
        "apply": False,
        "manifestPath": str(resolved_manifest_path),
        "outputDir": str(resolved_output_dir),
        "reportPath": str((resolved_output_dir / DEFAULT_REPORT_PATH.name).resolve()),
        "selectorPath": str((resolved_output_dir / DEFAULT_SELECTOR_PATH.name).resolve()),
        "targetCount": len(items),
        "needsReviewCount": len(needs_review_ids),
        "counts": counts,
        "defaults": {
            "provider": "openai",
            "model": "gpt-5.4",
            "allowExternal": True,
            "llmMode": "strong",
        },
        "items": items,
    }


def write_canon_audit_outputs(payload: dict[str, Any], *, output_dir: Path | None = None) -> dict[str, str]:
    resolved_output_dir = Path(output_dir or payload.get("outputDir") or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    report_path = resolved_output_dir / DEFAULT_REPORT_PATH.name
    selector_path = resolved_output_dir / DEFAULT_SELECTOR_PATH.name
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    selector_path.write_text(
        "".join(f"{item.get('paperId')}\n" for item in list(payload.get("items") or []) if bool(item.get("needsReview"))),
        encoding="utf-8",
    )
    return {
        "reportPath": str(report_path),
        "selectorPath": str(selector_path),
    }


def remediation_needs_source_repair(issues: list[str] | tuple[str, ...]) -> bool:
    return bool(_SOURCE_REPAIR_ISSUES & {clean_text(item) for item in list(issues or []) if clean_text(item)})


def remediation_needs_summary_rebuild(issues: list[str] | tuple[str, ...]) -> bool:
    return bool(_SUMMARY_REBUILD_ISSUES & {clean_text(item) for item in list(issues or []) if clean_text(item)})


def remediation_needs_memory_rebuild(issues: list[str] | tuple[str, ...]) -> bool:
    return bool(_MEMORY_REBUILD_ISSUES & {clean_text(item) for item in list(issues or []) if clean_text(item)})


def remediation_needs_concept_refresh(issues: list[str] | tuple[str, ...]) -> bool:
    return bool(_CONCEPT_REFRESH_ISSUES & {clean_text(item) for item in list(issues or []) if clean_text(item)})


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "SCHEMA_ID",
    "audit_canon_quality",
    "load_canon_manifest",
    "remediation_needs_concept_refresh",
    "remediation_needs_memory_rebuild",
    "remediation_needs_source_repair",
    "remediation_needs_summary_rebuild",
    "write_canon_audit_outputs",
]
