from __future__ import annotations

from typing import Any

from knowledge_hub.notes.models import KoNoteQuality, KoNoteReview
from knowledge_hub.notes.workflow_helpers import now_iso, review_item_view
from knowledge_hub.notes.scoring import build_note_review_payload


class KoNoteReviewService:
    def __init__(self, sqlite_db):
        self.sqlite_db = sqlite_db

    def review_list(
        self,
        *,
        run_id: str,
        item_type: str = "all",
        quality_flag: str = "all",
        limit: int = 50,
    ) -> dict[str, Any]:
        ts = now_iso()
        run = self.sqlite_db.get_ko_note_run(run_id)
        if not run:
            return {
                "schema": "knowledge-hub.ko-note.review.list.result.v1",
                "status": "failed",
                "runId": str(run_id),
                "items": [],
                "counts": {},
                "ts": ts,
            }
        items = self.sqlite_db.list_ko_note_items(
            run_id=run_id,
            item_type=None if item_type == "all" else item_type,
            limit=2000,
        )
        filtered: list[dict[str, Any]] = []
        for item in items:
            payload = dict(item.get("payload_json") or {})
            review_meta = KoNoteReview.from_payload(payload)
            quality_meta = KoNoteQuality.from_payload(payload)
            if not review_meta.queue:
                continue
            flag = quality_meta.flag
            if quality_flag != "all" and flag != quality_flag:
                continue
            filtered.append(item)
        filtered.sort(
            key=lambda item: (
                {"reject": 0, "needs_review": 1, "unscored": 2}.get(
                    KoNoteQuality.from_payload(dict(item.get("payload_json") or {})).flag,
                    3,
                ),
                -float(item.get("candidate_score") or 0.0),
                int(item.get("id") or 0),
            )
        )
        limited = filtered[: max(1, int(limit))]
        counts = {
            "total": len(filtered),
            "source": sum(1 for item in filtered if str(item.get("item_type") or "") == "source"),
            "concept": sum(1 for item in filtered if str(item.get("item_type") or "") == "concept"),
            "needs_review": sum(1 for item in filtered if KoNoteQuality.from_payload(dict(item.get("payload_json") or {})).flag == "needs_review"),
            "reject": sum(1 for item in filtered if KoNoteQuality.from_payload(dict(item.get("payload_json") or {})).flag == "reject"),
            "unscored": sum(1 for item in filtered if KoNoteQuality.from_payload(dict(item.get("payload_json") or {})).flag == "unscored"),
        }
        return {
            "schema": "knowledge-hub.ko-note.review.list.result.v1",
            "status": "ok",
            "runId": str(run_id),
            "itemType": str(item_type),
            "qualityFlag": str(quality_flag),
            "counts": counts,
            "items": [review_item_view(item) for item in limited],
            "ts": ts,
        }

    def review_transition(
        self,
        *,
        item_id: int,
        decision_status: str,
        reviewer: str,
        note: str,
    ) -> dict[str, Any]:
        ts = now_iso()
        item = self.sqlite_db.get_ko_note_item(int(item_id))
        if not item:
            return {
                "schema": "knowledge-hub.ko-note.review.result.v1",
                "status": "failed",
                "itemId": int(item_id),
                "warnings": [f"ko note item not found: {item_id}"],
                "ts": ts,
            }
        if str(item.get("status") or "") != "staged":
            return {
                "schema": "knowledge-hub.ko-note.review.result.v1",
                "status": "failed",
                "itemId": int(item_id),
                "warnings": [f"review transition allowed only from staged: item={item_id} status={item.get('status')}"],
                "ts": ts,
            }
        payload = dict(item.get("payload_json") or {})
        review = KoNoteReview.from_payload(
            build_note_review_payload(
                item_type=str(item.get("item_type") or ""),
                quality=KoNoteQuality.from_payload(payload).to_payload(),
                existing_review=KoNoteReview.from_payload(payload).to_payload(),
            )
        )
        review.with_decision(
            status=str(decision_status),
            reviewer=str(reviewer or "cli-user"),
            note=str(note or ""),
            reviewed_at=ts,
            queue=False,
        )
        payload["review"] = review.to_payload()
        self.sqlite_db.update_ko_note_item_payload(int(item_id), payload=payload)
        next_status = "approved" if str(decision_status) == "approved" else "rejected"
        self.sqlite_db.update_ko_note_item_status(int(item_id), status=next_status)
        updated = self.sqlite_db.get_ko_note_item(int(item_id)) or item
        run_id = str(updated.get("run_id") or "")
        all_items = self.sqlite_db.list_ko_note_items(run_id=run_id, limit=2000) if run_id else []
        if run_id:
            self.sqlite_db.update_ko_note_run(
                run_id,
                approved_count=sum(1 for row in all_items if str(row.get("status")) == "approved"),
                rejected_count=sum(1 for row in all_items if str(row.get("status")) == "rejected"),
            )
        return {
            "schema": "knowledge-hub.ko-note.review.result.v1",
            "status": "ok",
            "itemId": int(item_id),
            "runId": run_id,
            "itemType": updated.get("item_type"),
            "decision": str(decision_status),
            "titleKo": updated.get("title_ko"),
            "titleEn": updated.get("title_en"),
            "qualityFlag": KoNoteQuality.from_payload(payload).flag,
            "review": review.to_payload(),
            "ts": ts,
        }

    def review_approve(self, *, item_id: int, reviewer: str = "cli-user", note: str = "") -> dict[str, Any]:
        return self.review_transition(
            item_id=int(item_id),
            decision_status="approved",
            reviewer=str(reviewer or "cli-user"),
            note=str(note or ""),
        )

    def review_reject(self, *, item_id: int, reviewer: str = "cli-user", note: str = "") -> dict[str, Any]:
        return self.review_transition(
            item_id=int(item_id),
            decision_status="rejected",
            reviewer=str(reviewer or "cli-user"),
            note=str(note or ""),
        )
