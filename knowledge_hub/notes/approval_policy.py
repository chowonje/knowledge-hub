from __future__ import annotations

from typing import Any

from knowledge_hub.notes.models import KoNoteApproval, KoNoteQuality, KoNoteReview
from knowledge_hub.notes.workflow_helpers import now_iso

KO_NOTE_AUTO_APPROVE_POLICY_VERSION = "ko-note-auto-approve-v1"


def is_auto_approvable_concept_item(item: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if str(item.get("item_type") or "") != "concept":
        reasons.append("item_type!=concept")
    if str(item.get("status") or "") != "staged":
        reasons.append(f"status={item.get('status')}")
    payload = dict(item.get("payload_json") or {})
    quality = KoNoteQuality.from_payload(payload)
    review = KoNoteReview.from_payload(payload)
    if quality.flag != "ok":
        reasons.append(f"quality_flag={quality.flag}")
    if review.queue:
        reasons.append("review_queue=true")
    if review.has_decision("rejected"):
        reasons.append("review_rejected")
    remediation = review.remediation
    if remediation.recommended_strategy == "full":
        reasons.append("remediation_recommended_strategy=full")
    last_status = remediation.last_attempt_status
    if last_status in {"failed", "regressed"}:
        reasons.append(f"remediation_status={last_status}")
    if remediation.last_attempt_warnings and last_status in {"failed", "regressed"}:
        reasons.append("remediation_warning_active")
    return not reasons, reasons


def build_auto_approval_payload(payload: dict[str, Any], *, signals: list[str]) -> dict[str, Any]:
    updated = dict(payload)
    approval = KoNoteApproval.from_payload(updated)
    approval.mode = "auto"
    approval.policy_version = KO_NOTE_AUTO_APPROVE_POLICY_VERSION
    approval.approved_at = now_iso()
    approval.approved_by = "auto-policy"
    approval.reasons = ["conservative-concept-auto-approve"]
    approval.signals = list(signals)
    updated["approval"] = approval.to_payload()
    return updated


def auto_approve_concept_items_for_apply(
    sqlite_db,
    *,
    run_id: str,
    item_type: str,
    only_approved: bool,
) -> tuple[dict[str, int], list[str]]:
    counts = {"source": 0, "concept": 0}
    warnings: list[str] = []
    if item_type == "source":
        counts["total"] = 0
        return counts, warnings

    staged_items = sqlite_db.list_ko_note_items(
        run_id=run_id,
        item_type="concept",
        status="staged",
        limit=2000,
    )
    for item in staged_items:
        payload = dict(item.get("payload_json") or {})
        eligible, reasons = is_auto_approvable_concept_item(item)
        if not eligible:
            if only_approved and KoNoteQuality.from_payload(payload).flag == "ok":
                warnings.append(
                    "concept item not auto-approved by policy: "
                    f"item={item.get('id')} title={item.get('title_en') or item.get('title_ko') or ''} "
                    f"reasons={','.join(reasons)}"
                )
            continue
        signals = [
            "item_type=concept",
            "status=staged",
            "quality_flag=ok",
            "review_queue=false",
            "review_rejected=false",
            "remediation_recommended_strategy!=full",
        ]
        try:
            updated_payload = build_auto_approval_payload(payload, signals=signals)
            if not sqlite_db.update_ko_note_item_payload(int(item["id"]), payload=updated_payload):
                warnings.append(f"concept item auto-approve skipped: item={item.get('id')} payload_update_failed")
                continue
            if not sqlite_db.update_ko_note_item_status(int(item["id"]), status="approved"):
                sqlite_db.update_ko_note_item_payload(int(item["id"]), payload=payload)
                warnings.append(f"concept item auto-approve skipped: item={item.get('id')} status_update_failed")
                continue
            counts["concept"] += 1
        except Exception as exc:
            try:
                sqlite_db.update_ko_note_item_payload(int(item["id"]), payload=payload)
            except Exception:
                pass
            warnings.append(f"concept item auto-approve skipped: item={item.get('id')} error={type(exc).__name__}")
    counts["total"] = int(counts.get("source", 0)) + int(counts.get("concept", 0))
    return counts, warnings
