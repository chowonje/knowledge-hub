from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from knowledge_hub.notes.models import KoNoteApproval, KoNoteQuality, KoNoteReview


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def should_apply_concept_item(item: dict[str, Any]) -> tuple[bool, str]:
    if str(item.get("item_type") or "") != "concept":
        return True, ""
    if str(item.get("status") or "") == "approved":
        return True, ""
    payload = dict(item.get("payload_json") or {})
    flag = KoNoteQuality.from_payload(payload).flag
    if flag == "ok":
        return True, ""
    if flag == "needs_review":
        return False, "quality_flag=needs_review"
    if flag == "reject":
        return False, "quality_flag=reject"
    return False, "quality_flag=unscored"


def _quality_counts(items: list[dict[str, Any]], *, item_type: str) -> dict[str, int]:
    counts = {"ok": 0, "needs_review": 0, "reject": 0, "unscored": 0}
    for item in items:
        if str(item.get("item_type") or "") != str(item_type or ""):
            continue
        payload = dict(item.get("payload_json") or {})
        flag = KoNoteQuality.from_payload(payload).flag
        if flag not in counts:
            flag = "unscored"
        counts[flag] += 1
    counts["total"] = sum(counts.values())
    return counts


def concept_quality_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return _quality_counts(items, item_type="concept")


def source_quality_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return _quality_counts(items, item_type="source")


def merge_quality_counts(*groups: dict[str, int]) -> dict[str, int]:
    counts = {"ok": 0, "needs_review": 0, "reject": 0, "unscored": 0, "total": 0}
    for group in groups:
        for key in counts:
            counts[key] += int((group or {}).get(key) or 0)
    return counts


def review_queue_counts(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "source": {"total": 0, "needs_review": 0, "reject": 0, "unscored": 0},
        "concept": {"total": 0, "needs_review": 0, "reject": 0, "unscored": 0},
    }
    for item in items:
        item_type = str(item.get("item_type") or "")
        if item_type not in counts:
            continue
        payload = dict(item.get("payload_json") or {})
        review = KoNoteReview.from_payload(payload)
        if not review.queue:
            continue
        flag = KoNoteQuality.from_payload(payload).flag
        if flag not in {"needs_review", "reject", "unscored"}:
            flag = "unscored"
        counts[item_type]["total"] += 1
        counts[item_type][flag] += 1
    counts["combined"] = {
        key: int(counts["source"].get(key, 0)) + int(counts["concept"].get(key, 0))
        for key in ("total", "needs_review", "reject", "unscored")
    }
    return counts


def approval_summary(items: list[dict[str, Any]]) -> dict[str, dict[str, int] | int]:
    auto_counts = {"source": 0, "concept": 0}
    review_counts = {"source": 0, "concept": 0}
    for item in items:
        item_type = str(item.get("item_type") or "")
        if item_type not in {"source", "concept"}:
            continue
        payload = dict(item.get("payload_json") or {})
        approval = KoNoteApproval.from_payload(payload)
        if approval.is_auto():
            auto_counts[item_type] += 1
        review = KoNoteReview.from_payload(payload)
        if review.has_decision("approved"):
            review_counts[item_type] += 1
    auto_counts["total"] = int(auto_counts["source"]) + int(auto_counts["concept"])
    review_counts["total"] = int(review_counts["source"]) + int(review_counts["concept"])
    return {"autoApproved": auto_counts, "approvedFromReview": review_counts}


def remediation_summary(items: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "attempted": 0,
        "improved": 0,
        "unchanged": 0,
        "failed": 0,
        "regressed": 0,
        "recommendedFull": 0,
    }
    for item in items:
        remediation = KoNoteReview.from_payload(dict(item.get("payload_json") or {})).remediation
        if remediation.attempt_count > 0:
            summary["attempted"] += 1
        status = remediation.last_attempt_status
        if status in summary:
            summary[status] += 1
        if remediation.last_improved:
            summary["improved"] += 1
        if remediation.recommended_strategy == "full":
            summary["recommendedFull"] += 1
    return summary


def report_apply_backlog_count(items: list[dict[str, Any]]) -> int:
    backlog = 0
    for item in items:
        status = str(item.get("status") or "")
        if status == "approved":
            backlog += 1
            continue
        if status != "staged":
            continue
        item_type = str(item.get("item_type") or "")
        payload = dict(item.get("payload_json") or {})
        if item_type == "source":
            if KoNoteQuality.from_payload(payload).flag != "reject":
                backlog += 1
            continue
        if item_type == "concept":
            allowed, _ = should_apply_concept_item(item)
            if allowed:
                backlog += 1
    return backlog


def review_item_view(item: dict[str, Any]) -> dict[str, Any]:
    payload = dict(item.get("payload_json") or {})
    quality = KoNoteQuality.from_payload(payload)
    review = KoNoteReview.from_payload(payload)
    approval = KoNoteApproval.from_payload(payload)
    remediation = review.remediation
    approval_mode = ""
    approval_by = ""
    approval_at = ""
    approval_policy_version = ""
    if approval.is_auto():
        approval_mode = "auto"
        approval_by = approval.approved_by
        approval_at = approval.approved_at
        approval_policy_version = approval.policy_version
    elif review.has_decision("approved") and review.decision is not None:
        decision = review.decision
        approval_mode = "review"
        approval_by = decision.reviewer
        approval_at = decision.reviewed_at
    return {
        "id": item.get("id"),
        "itemType": item.get("item_type"),
        "status": item.get("status"),
        "qualityFlag": quality.flag,
        "reviewQueue": review.queue,
        "reviewReasons": list(review.reasons),
        "reviewPatchHints": list(review.patch_hints),
        "titleKo": item.get("title_ko"),
        "titleEn": item.get("title_en"),
        "stagingPath": item.get("staging_path"),
        "finalPath": item.get("final_path"),
        "approvalMode": approval_mode,
        "approvalBy": approval_by,
        "approvalAt": approval_at,
        "approvalPolicyVersion": approval_policy_version,
        "remediationAttemptCount": remediation.attempt_count,
        "remediationLastStatus": remediation.last_attempt_status,
        "remediationLastImproved": remediation.last_improved,
        "remediationStrategy": remediation.strategy,
        "remediationTargetSectionCount": len(remediation.target_sections),
        "remediationPatchedSectionCount": len(remediation.patched_sections),
        "remediationPatchedLineCount": remediation.last_patched_line_count,
        "remediationPreservedLineCount": remediation.last_preserved_line_count,
        "remediationRecommendedStrategy": remediation.recommended_strategy,
    }
