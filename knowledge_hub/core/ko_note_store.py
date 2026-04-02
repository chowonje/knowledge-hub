"""Korean note materialization store helpers."""

from __future__ import annotations

import json
from typing import Any


class KoNoteStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ko_note_runs (
                run_id TEXT PRIMARY KEY,
                crawl_job_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running'
                    CHECK(status IN ('running','completed','partial','blocked','failed')),
                granularity TEXT NOT NULL DEFAULT 'hybrid',
                korean_depth TEXT NOT NULL DEFAULT 'summary_key_excerpt',
                vault_target TEXT NOT NULL DEFAULT 'documents',
                writeback_mode TEXT NOT NULL DEFAULT 'staging_then_apply',
                source_candidates INTEGER NOT NULL DEFAULT 0,
                source_generated INTEGER NOT NULL DEFAULT 0,
                concept_candidates INTEGER NOT NULL DEFAULT 0,
                concept_generated INTEGER NOT NULL DEFAULT 0,
                approved_count INTEGER NOT NULL DEFAULT 0,
                rejected_count INTEGER NOT NULL DEFAULT 0,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finished_at TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ko_note_runs_job
            ON ko_note_runs(crawl_job_id, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ko_note_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES ko_note_runs(run_id) ON DELETE CASCADE,
                item_type TEXT NOT NULL CHECK(item_type IN ('source','concept')),
                item_key TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'staged'
                    CHECK(status IN ('staged','approved','rejected','applied','blocked')),
                job_id TEXT NOT NULL DEFAULT '',
                record_id TEXT DEFAULT '',
                note_id TEXT DEFAULT '',
                entity_id TEXT DEFAULT '',
                title_en TEXT DEFAULT '',
                title_ko TEXT DEFAULT '',
                candidate_score REAL NOT NULL DEFAULT 0.0,
                translation_level TEXT NOT NULL DEFAULT 'T1'
                    CHECK(translation_level IN ('T1','T2')),
                source_urls_json TEXT NOT NULL DEFAULT '[]',
                evidence_ptrs_json TEXT NOT NULL DEFAULT '[]',
                entity_ids_json TEXT NOT NULL DEFAULT '[]',
                relation_ids_json TEXT NOT NULL DEFAULT '[]',
                payload_json TEXT NOT NULL DEFAULT '{}',
                staging_path TEXT DEFAULT '',
                final_path TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP,
                applied_at TIMESTAMP,
                UNIQUE(run_id, item_type, item_key)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ko_note_items_run_status
            ON ko_note_items(run_id, item_type, status, candidate_score DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ko_note_enrichment_runs (
                run_id TEXT PRIMARY KEY,
                source_run_id TEXT DEFAULT '',
                scope TEXT NOT NULL DEFAULT 'both'
                    CHECK(scope IN ('new','existing-top','both')),
                item_type TEXT NOT NULL DEFAULT 'all'
                    CHECK(item_type IN ('source','concept','all')),
                status TEXT NOT NULL DEFAULT 'running'
                    CHECK(status IN ('running','completed','partial','blocked','failed')),
                source_target_count INTEGER NOT NULL DEFAULT 0,
                source_enriched_count INTEGER NOT NULL DEFAULT 0,
                concept_target_count INTEGER NOT NULL DEFAULT 0,
                concept_enriched_count INTEGER NOT NULL DEFAULT 0,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finished_at TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ko_note_enrichment_runs_source
            ON ko_note_enrichment_runs(source_run_id, updated_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ko_note_enrichment_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES ko_note_enrichment_runs(run_id) ON DELETE CASCADE,
                note_item_id INTEGER DEFAULT 0,
                target_path TEXT DEFAULT '',
                item_type TEXT NOT NULL CHECK(item_type IN ('source','concept')),
                status TEXT NOT NULL DEFAULT 'queued'
                    CHECK(status IN ('queued','enriched','skipped','blocked','failed')),
                route TEXT DEFAULT '',
                provider TEXT DEFAULT '',
                model TEXT DEFAULT '',
                model_fingerprint TEXT DEFAULT '',
                evidence_pack_hash TEXT DEFAULT '',
                warnings_json TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                enriched_at TIMESTAMP,
                UNIQUE(item_type, note_item_id, target_path, evidence_pack_hash, model_fingerprint)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ko_note_enrichment_items_run_status
            ON ko_note_enrichment_items(run_id, item_type, status, id DESC)
            """
        )
        self.conn.commit()

    def create_run(
        self,
        *,
        run_id: str,
        crawl_job_id: str,
        status: str = "running",
        granularity: str = "hybrid",
        korean_depth: str = "summary_key_excerpt",
        vault_target: str = "documents",
        writeback_mode: str = "staging_then_apply",
        source_candidates: int = 0,
        source_generated: int = 0,
        concept_candidates: int = 0,
        concept_generated: int = 0,
        approved_count: int = 0,
        rejected_count: int = 0,
        warnings: list[str] | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO ko_note_runs (
                run_id, crawl_job_id, status, granularity, korean_depth, vault_target,
                writeback_mode, source_candidates, source_generated, concept_candidates,
                concept_generated, approved_count, rejected_count, warnings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                crawl_job_id=excluded.crawl_job_id,
                status=excluded.status,
                granularity=excluded.granularity,
                korean_depth=excluded.korean_depth,
                vault_target=excluded.vault_target,
                writeback_mode=excluded.writeback_mode,
                source_candidates=excluded.source_candidates,
                source_generated=excluded.source_generated,
                concept_candidates=excluded.concept_candidates,
                concept_generated=excluded.concept_generated,
                approved_count=excluded.approved_count,
                rejected_count=excluded.rejected_count,
                warnings_json=excluded.warnings_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(run_id),
                str(crawl_job_id),
                str(status),
                str(granularity),
                str(korean_depth),
                str(vault_target),
                str(writeback_mode),
                int(source_candidates),
                int(source_generated),
                int(concept_candidates),
                int(concept_generated),
                int(approved_count),
                int(rejected_count),
                json.dumps(warnings or [], ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM ko_note_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["warnings_json"] = json.loads(item.get("warnings_json") or "[]")
        except Exception:
            item["warnings_json"] = []
        return item

    def update_run(self, run_id: str, **updates) -> bool:
        if not updates:
            return False
        encoded: dict[str, Any] = dict(updates)
        if "warnings_json" in encoded and not isinstance(encoded["warnings_json"], str):
            encoded["warnings_json"] = json.dumps(encoded["warnings_json"] or [], ensure_ascii=False)
        fields = []
        params = []
        for key, value in encoded.items():
            fields.append(f"{key} = ?")
            params.append(value)
        fields.append("updated_at = CURRENT_TIMESTAMP")
        if "status" in encoded and str(encoded.get("status")) in {"completed", "partial", "blocked", "failed"}:
            fields.append("finished_at = CURRENT_TIMESTAMP")
        params.append(str(run_id))
        cursor = self.conn.execute(
            f"UPDATE ko_note_runs SET {', '.join(fields)} WHERE run_id = ?",
            params,
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def add_item(
        self,
        *,
        run_id: str,
        item_type: str,
        item_key: str,
        status: str = "staged",
        job_id: str = "",
        record_id: str = "",
        note_id: str = "",
        entity_id: str = "",
        title_en: str = "",
        title_ko: str = "",
        candidate_score: float = 0.0,
        translation_level: str = "T1",
        source_urls: list[str] | None = None,
        evidence_ptrs: list[dict[str, Any]] | None = None,
        entity_ids: list[str] | None = None,
        relation_ids: list[str] | None = None,
        payload: dict[str, Any] | None = None,
        staging_path: str = "",
        final_path: str = "",
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO ko_note_items (
                run_id, item_type, item_key, status, job_id, record_id, note_id, entity_id,
                title_en, title_ko, candidate_score, translation_level, source_urls_json,
                evidence_ptrs_json, entity_ids_json, relation_ids_json, payload_json,
                staging_path, final_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                str(item_type),
                str(item_key),
                str(status),
                str(job_id or ""),
                str(record_id or ""),
                str(note_id or ""),
                str(entity_id or ""),
                str(title_en or ""),
                str(title_ko or ""),
                float(candidate_score or 0.0),
                str(translation_level or "T1"),
                json.dumps(source_urls or [], ensure_ascii=False),
                json.dumps(evidence_ptrs or [], ensure_ascii=False),
                json.dumps(entity_ids or [], ensure_ascii=False),
                json.dumps(relation_ids or [], ensure_ascii=False),
                json.dumps(payload or {}, ensure_ascii=False),
                str(staging_path or ""),
                str(final_path or ""),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def get_item(self, item_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM ko_note_items WHERE id = ?",
            (int(item_id),),
        ).fetchone()
        return self._decode_item(row)

    def list_items(
        self,
        *,
        run_id: str,
        item_type: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        query = "SELECT * FROM ko_note_items WHERE run_id = ?"
        params: list[Any] = [str(run_id)]
        if item_type:
            query += " AND item_type = ?"
            params.append(str(item_type))
        if status:
            query += " AND status = ?"
            params.append(str(status))
        query += " ORDER BY candidate_score DESC, id ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [item for item in (self._decode_item(row) for row in rows) if item]

    def list_existing_items(
        self,
        *,
        item_type: str | None = None,
        statuses: tuple[str, ...] = ("staged", "approved", "applied"),
        limit: int = 5000,
    ) -> list[dict]:
        query = "SELECT * FROM ko_note_items WHERE 1=1"
        params: list[Any] = []
        if item_type:
            query += " AND item_type = ?"
            params.append(str(item_type))
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(str(item) for item in statuses)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        return [item for item in (self._decode_item(row) for row in rows) if item]

    def find_item_by_final_path(
        self,
        *,
        final_path: str,
        item_type: str | None = None,
        statuses: tuple[str, ...] = ("approved", "applied"),
    ) -> dict | None:
        query = "SELECT * FROM ko_note_items WHERE final_path = ?"
        params: list[Any] = [str(final_path)]
        if item_type:
            query += " AND item_type = ?"
            params.append(str(item_type))
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(str(item) for item in statuses)
        query += " ORDER BY id DESC LIMIT 1"
        row = self.conn.execute(query, params).fetchone()
        return self._decode_item(row)

    def update_item_status(
        self,
        item_id: int,
        *,
        status: str,
        final_path: str | None = None,
        staging_path: str | None = None,
    ) -> bool:
        fields = ["status = ?"]
        params: list[Any] = [str(status)]
        if final_path is not None:
            fields.append("final_path = ?")
            params.append(str(final_path))
        if staging_path is not None:
            fields.append("staging_path = ?")
            params.append(str(staging_path))
        if str(status) in {"approved", "rejected"}:
            fields.append("reviewed_at = CURRENT_TIMESTAMP")
        if str(status) == "applied":
            fields.append("applied_at = CURRENT_TIMESTAMP")
        params.append(int(item_id))
        cursor = self.conn.execute(
            f"UPDATE ko_note_items SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def update_item_payload(
        self,
        item_id: int,
        *,
        payload: dict,
        title_en: str | None = None,
        title_ko: str | None = None,
        staging_path: str | None = None,
        final_path: str | None = None,
    ) -> bool:
        fields = ["payload_json = ?"]
        params: list[Any] = [json.dumps(payload or {}, ensure_ascii=False)]
        if title_en is not None:
            fields.append("title_en = ?")
            params.append(str(title_en))
        if title_ko is not None:
            fields.append("title_ko = ?")
            params.append(str(title_ko))
        if staging_path is not None:
            fields.append("staging_path = ?")
            params.append(str(staging_path))
        if final_path is not None:
            fields.append("final_path = ?")
            params.append(str(final_path))
        params.append(int(item_id))
        cursor = self.conn.execute(
            f"UPDATE ko_note_items SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_latest_run(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM ko_note_runs ORDER BY updated_at DESC, created_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["warnings_json"] = json.loads(item.get("warnings_json") or "[]")
        except Exception:
            item["warnings_json"] = []
        return item

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM ko_note_runs ORDER BY updated_at DESC, created_at DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["warnings_json"] = json.loads(item.get("warnings_json") or "[]")
            except Exception:
                item["warnings_json"] = []
            items.append(item)
        return items

    def list_stale_runs(
        self,
        *,
        status: str = "running",
        updated_before_seconds: int,
        limit: int = 200,
    ) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT * FROM ko_note_runs
            WHERE status = ?
              AND updated_at < datetime('now', ?)
            ORDER BY updated_at ASC, created_at ASC
            LIMIT ?
            """,
            (
                str(status),
                f"-{max(1, int(updated_before_seconds))} seconds",
                max(1, int(limit)),
            ),
        ).fetchall()
        items: list[dict] = []
        for row in rows:
            item = dict(row)
            try:
                item["warnings_json"] = json.loads(item.get("warnings_json") or "[]")
            except Exception:
                item["warnings_json"] = []
            items.append(item)
        return items

    def create_enrichment_run(
        self,
        *,
        run_id: str,
        source_run_id: str = "",
        scope: str = "both",
        item_type: str = "all",
        status: str = "running",
        source_target_count: int = 0,
        source_enriched_count: int = 0,
        concept_target_count: int = 0,
        concept_enriched_count: int = 0,
        warnings: list[str] | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO ko_note_enrichment_runs (
                run_id, source_run_id, scope, item_type, status,
                source_target_count, source_enriched_count,
                concept_target_count, concept_enriched_count, warnings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                source_run_id=excluded.source_run_id,
                scope=excluded.scope,
                item_type=excluded.item_type,
                status=excluded.status,
                source_target_count=excluded.source_target_count,
                source_enriched_count=excluded.source_enriched_count,
                concept_target_count=excluded.concept_target_count,
                concept_enriched_count=excluded.concept_enriched_count,
                warnings_json=excluded.warnings_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(run_id),
                str(source_run_id or ""),
                str(scope or "both"),
                str(item_type or "all"),
                str(status or "running"),
                int(source_target_count or 0),
                int(source_enriched_count or 0),
                int(concept_target_count or 0),
                int(concept_enriched_count or 0),
                json.dumps(warnings or [], ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def get_enrichment_run(self, run_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM ko_note_enrichment_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["warnings_json"] = json.loads(item.get("warnings_json") or "[]")
        except Exception:
            item["warnings_json"] = []
        return item

    def update_enrichment_run(self, run_id: str, **updates) -> bool:
        if not updates:
            return False
        encoded = dict(updates)
        if "warnings_json" in encoded and not isinstance(encoded["warnings_json"], str):
            encoded["warnings_json"] = json.dumps(encoded["warnings_json"] or [], ensure_ascii=False)
        fields = []
        params = []
        for key, value in encoded.items():
            fields.append(f"{key} = ?")
            params.append(value)
        fields.append("updated_at = CURRENT_TIMESTAMP")
        if "status" in encoded and str(encoded.get("status")) in {"completed", "partial", "blocked", "failed"}:
            fields.append("finished_at = CURRENT_TIMESTAMP")
        params.append(str(run_id))
        cursor = self.conn.execute(
            f"UPDATE ko_note_enrichment_runs SET {', '.join(fields)} WHERE run_id = ?",
            params,
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def add_enrichment_item(
        self,
        *,
        run_id: str,
        note_item_id: int = 0,
        target_path: str = "",
        item_type: str,
        status: str = "queued",
        route: str = "",
        provider: str = "",
        model: str = "",
        model_fingerprint: str = "",
        evidence_pack_hash: str = "",
        warnings: list[str] | None = None,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO ko_note_enrichment_items (
                run_id, note_item_id, target_path, item_type, status,
                route, provider, model, model_fingerprint, evidence_pack_hash, warnings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                int(note_item_id or 0),
                str(target_path or ""),
                str(item_type),
                str(status or "queued"),
                str(route or ""),
                str(provider or ""),
                str(model or ""),
                str(model_fingerprint or ""),
                str(evidence_pack_hash or ""),
                json.dumps(warnings or [], ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def list_enrichment_items(
        self,
        *,
        run_id: str,
        item_type: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        query = "SELECT * FROM ko_note_enrichment_items WHERE run_id = ?"
        params: list[Any] = [str(run_id)]
        if item_type:
            query += " AND item_type = ?"
            params.append(str(item_type))
        if status:
            query += " AND status = ?"
            params.append(str(status))
        query += " ORDER BY id ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params).fetchall()
        items: list[dict] = []
        for row in rows:
            item = dict(row)
            try:
                item["warnings_json"] = json.loads(item.get("warnings_json") or "[]")
            except Exception:
                item["warnings_json"] = []
            items.append(item)
        return items

    def update_enrichment_item(self, item_id: int, **updates) -> bool:
        if not updates:
            return False
        encoded = dict(updates)
        if "warnings_json" in encoded and not isinstance(encoded["warnings_json"], str):
            encoded["warnings_json"] = json.dumps(encoded["warnings_json"] or [], ensure_ascii=False)
        fields = []
        params = []
        for key, value in encoded.items():
            fields.append(f"{key} = ?")
            params.append(value)
        fields.append("updated_at = CURRENT_TIMESTAMP")
        if "status" in encoded and str(encoded.get("status")) in {"enriched", "skipped", "blocked", "failed"}:
            fields.append("enriched_at = CURRENT_TIMESTAMP")
        params.append(int(item_id))
        cursor = self.conn.execute(
            f"UPDATE ko_note_enrichment_items SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def find_matching_enrichment_item(
        self,
        *,
        note_item_id: int = 0,
        target_path: str = "",
        item_type: str,
        evidence_pack_hash: str,
        model_fingerprint: str,
    ) -> dict | None:
        row = self.conn.execute(
            """
            SELECT * FROM ko_note_enrichment_items
            WHERE note_item_id = ? AND target_path = ? AND item_type = ?
              AND evidence_pack_hash = ? AND model_fingerprint = ?
            ORDER BY id DESC LIMIT 1
            """,
            (
                int(note_item_id or 0),
                str(target_path or ""),
                str(item_type or ""),
                str(evidence_pack_hash or ""),
                str(model_fingerprint or ""),
            ),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["warnings_json"] = json.loads(item.get("warnings_json") or "[]")
        except Exception:
            item["warnings_json"] = []
        return item

    @staticmethod
    def _decode_item(row) -> dict | None:
        if not row:
            return None
        item = dict(row)
        for key in (
            "source_urls_json",
            "evidence_ptrs_json",
            "entity_ids_json",
            "relation_ids_json",
            "payload_json",
        ):
            try:
                item[key] = json.loads(item.get(key) or ("[]" if key != "payload_json" else "{}"))
            except Exception:
                item[key] = [] if key != "payload_json" else {}
        return item
