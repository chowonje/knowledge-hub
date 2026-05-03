from __future__ import annotations

from typing import Any

from knowledge_hub.notes.approval_policy import auto_approve_concept_items_for_apply
from knowledge_hub.notes.models import KoNoteQuality, KoNoteReview
from knowledge_hub.notes.workflow_helpers import (
    concept_quality_counts,
    merge_quality_counts,
    now_iso,
    should_apply_concept_item,
    source_quality_counts,
)


class KoNoteApplier:
    def __init__(self, materializer):
        self.materializer = materializer
        self.sqlite_db = materializer.sqlite_db

    def apply(
        self,
        *,
        run_id: str,
        item_type: str = "all",
        limit: int = 0,
        only_approved: bool = True,
    ) -> dict[str, Any]:
        ts = now_iso()
        run = self.sqlite_db.get_ko_note_run(run_id)
        if not run:
            return {
                "schema": "knowledge-hub.ko-note.apply.result.v1",
                "status": "failed",
                "runId": str(run_id),
                "applied": 0,
                "skipped": 0,
                "conflicts": 0,
                "warnings": [f"ko note run not found: {run_id}"],
                "ts": ts,
            }
        auto_approved = {"source": 0, "concept": 0, "total": 0}
        auto_approve_warnings: list[str] = []
        if only_approved:
            auto_approved, auto_approve_warnings = auto_approve_concept_items_for_apply(
                self.sqlite_db,
                run_id=run_id,
                item_type=item_type,
                only_approved=only_approved,
            )
        statuses = ["approved"] if only_approved else ["staged", "approved"]
        selected: list[dict[str, Any]] = []
        chosen_type = None if item_type == "all" else item_type
        for status in statuses:
            selected.extend(
                self.sqlite_db.list_ko_note_items(run_id=run_id, item_type=chosen_type, status=status, limit=2000)
            )
        selected.sort(key=lambda item: float(item.get("candidate_score") or 0.0), reverse=True)
        if limit > 0:
            selected = selected[: max(1, int(limit))]
        applied = 0
        skipped = 0
        conflicts = 0
        quality_skipped = 0
        quality_applied = 0
        source_quality_warning_count = 0
        source_quality_applied = 0
        approved_from_review = {"source": 0, "concept": 0}
        warnings: list[str] = list(auto_approve_warnings)
        for item in selected:
            if str(item.get("status")) == "applied":
                skipped += 1
                continue
            payload = dict(item.get("payload_json") or {})
            review = KoNoteReview.from_payload(payload)
            quality = KoNoteQuality.from_payload(payload)
            if str(item.get("item_type") or "") == "concept":
                allowed, reason = should_apply_concept_item(item)
                if not allowed:
                    skipped += 1
                    quality_skipped += 1
                    warnings.append(
                        f"concept item skipped by quality: item={item.get('id')} title={item.get('title_en') or item.get('title_ko') or ''} {reason}"
                    )
                    continue
            if str(item.get("item_type") or "") == "source":
                source_flag = quality.flag
                if source_flag in {"needs_review", "reject"}:
                    source_quality_warning_count += 1
                    warnings.append(
                        f"source item quality warning: item={item.get('id')} title={item.get('title_en') or item.get('title_ko') or ''} quality_flag={source_flag}"
                    )
            if str(item.get("status") or "") == "approved" and review.has_decision("approved"):
                if review.decision is not None:
                    item_key = str(item.get("item_type") or "concept")
                    approved_from_review[item_key] = approved_from_review.get(item_key, 0) + 1
                    quality_flag = quality.flag
                    if quality_flag != "ok":
                        warnings.append(
                            f"approved review override: item={item.get('id')} title={item.get('title_en') or item.get('title_ko') or ''} quality_flag={quality_flag}"
                        )
            if str(item.get("item_type")) == "source":
                result, final_path = self.materializer._apply_source_item(item, run_id)
            else:
                result, final_path = self.materializer._apply_concept_item(item)
            if result in {"applied", "merged"}:
                applied += 1
                if str(item.get("item_type") or "") == "concept":
                    quality_applied += 1
                elif str(item.get("item_type") or "") == "source":
                    source_quality_applied += 1
                self.sqlite_db.update_ko_note_item_status(int(item["id"]), status="applied", final_path=final_path)
            elif result == "conflict-copy":
                conflicts += 1
                if str(item.get("item_type") or "") == "concept":
                    quality_applied += 1
                elif str(item.get("item_type") or "") == "source":
                    source_quality_applied += 1
                self.sqlite_db.update_ko_note_item_status(int(item["id"]), status="applied", final_path=final_path)
            else:
                skipped += 1
                if result == "missing-staging":
                    warnings.append(f"missing staging file for item={item.get('id')}")
        items = self.sqlite_db.list_ko_note_items(run_id=run_id, limit=2000)
        approved_count = sum(1 for item in items if str(item.get("status")) == "approved")
        rejected_count = sum(1 for item in items if str(item.get("status")) == "rejected")
        concept_quality = concept_quality_counts(items)
        source_quality = source_quality_counts(items)
        combined_quality_counts = merge_quality_counts(concept_quality, source_quality)
        self.sqlite_db.update_ko_note_run(
            run_id,
            approved_count=approved_count,
            rejected_count=rejected_count,
            warnings_json=list(dict.fromkeys(warnings)),
        )
        return {
            "schema": "knowledge-hub.ko-note.apply.result.v1",
            "status": "ok",
            "runId": str(run_id),
            "applied": applied,
            "skipped": skipped,
            "conflicts": conflicts,
            "qualitySkipped": quality_skipped,
            "quality": {
                **concept_quality,
                "concept": {
                    **concept_quality,
                    "applied": quality_applied,
                    "skippedByQuality": quality_skipped,
                },
                "source": {
                    **source_quality,
                    "applied": source_quality_applied,
                    "warningCount": source_quality_warning_count,
                },
                "combined": combined_quality_counts,
            },
            "autoApproved": auto_approved,
            "approvedFromReview": {
                **approved_from_review,
                "total": int(approved_from_review.get("source", 0)) + int(approved_from_review.get("concept", 0)),
            },
            "warnings": list(dict.fromkeys(warnings)),
            "ts": ts,
        }

    def reject(
        self,
        *,
        run_id: str,
        item_type: str = "all",
        limit: int = 0,
    ) -> dict[str, Any]:
        items = self.sqlite_db.list_ko_note_items(
            run_id=run_id,
            item_type=None if item_type == "all" else item_type,
            status="staged",
            limit=2000,
        )
        if limit > 0:
            items = items[: max(1, int(limit))]
        rejected = 0
        for item in items:
            self.sqlite_db.update_ko_note_item_status(int(item["id"]), status="rejected")
            rejected += 1
        all_items = self.sqlite_db.list_ko_note_items(run_id=run_id, limit=2000)
        approved_count = sum(1 for item in all_items if str(item.get("status")) == "approved")
        rejected_count = sum(1 for item in all_items if str(item.get("status")) == "rejected")
        self.sqlite_db.update_ko_note_run(
            run_id,
            approved_count=approved_count,
            rejected_count=rejected_count,
        )
        return {
            "status": "ok",
            "runId": str(run_id),
            "rejected": rejected,
            "itemType": item_type,
            "ts": now_iso(),
        }
