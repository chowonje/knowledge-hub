"""Application service for Learning Coach MVP v2."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from knowledge_hub.core.config import Config
from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.learning.assessor import grade_session_content, parse_edges_from_session, parse_frontmatter
from knowledge_hub.learning.mapper import generate_learning_map
from knowledge_hub.learning.models import RUN_SCHEMA
from knowledge_hub.learning.models import TEMPLATE_SCHEMA
from knowledge_hub.learning.obsidian_writeback import (
    build_paths,
    resolve_vault_write_adapter,
    write_canvas as write_canvas_file,
    write_hub,
    write_next_branches,
    write_session_template,
    write_trunk_map,
)
from knowledge_hub.learning.recommender import recommend_next


class LearningCoachService:
    def __init__(self, config: Config):
        self.config = config

    def _open_db(self) -> SQLiteDatabase:
        return SQLiteDatabase(self.config.sqlite_path)

    def _record_schema_validation(self, payload: dict) -> dict:
        schema_id = payload.get("schema")
        if isinstance(schema_id, str) and schema_id:
            try:
                annotate_schema_errors(payload, schema_id)
            except Exception:
                # schema validation should never break command execution in non-critical path
                pass
        return payload

    def _resolve_dynamic_dir(self) -> Path:
        repo_dynamic = Path(__file__).resolve().parents[2] / "data" / "dynamic"
        if repo_dynamic.exists():
            return repo_dynamic
        return Path(self.config.sqlite_path).expanduser().resolve().parent / "dynamic"

    def _resolve_vault_adapter(self):
        backend = str(self.config.get_nested("obsidian", "write_backend", default="filesystem") or "filesystem")
        cli_binary = str(self.config.get_nested("obsidian", "cli_binary", default="obsidian") or "obsidian")
        vault_name = str(self.config.get_nested("obsidian", "vault_name", default="") or "")
        return resolve_vault_write_adapter(
            vault_path=self.config.vault_path,
            backend=backend,
            cli_binary=cli_binary,
            vault_name=vault_name,
        )

    def map(
        self,
        topic: str,
        source: str = "all",
        days: int = 180,
        top_k: int = 12,
        writeback: bool = False,
        write_canvas: bool = False,
        allow_external: bool = False,
        run_id: str | None = None,
    ) -> dict:
        run_id = str(run_id or f"learn_map_{uuid4().hex[:12]}")
        db = self._open_db()
        try:
            result = generate_learning_map(
                db=db,
                topic=topic,
                source=source,
                days=days,
                top_k=top_k,
                allow_external=allow_external,
                run_id=run_id,
            )

            db.append_learning_event(
                event_id=f"evt_{uuid4().hex}",
                event_type="learning.map.generated",
                logical_step="map",
                session_id=None,
                payload={
                    "topic": topic,
                    "status": result.get("status"),
                    "trunkCount": len(result.get("trunks", [])),
                    "branchCount": len(result.get("branches", [])),
                },
                policy_class=str(result.get("policy", {}).get("classification", "P2")),
                run_id=run_id,
                request_id=run_id,
                source="learning",
            )

            if writeback:
                if not self.config.vault_path:
                    result.setdefault("warnings", []).append("vault_path not configured; writeback skipped")
                else:
                    paths = build_paths(self.config.vault_path, topic)
                    adapter = self._resolve_vault_adapter()
                    write_hub(paths, topic=topic, session_id="(not-set)", status_line="map-generated", adapter=adapter)
                    write_trunk_map(paths, topic=topic, map_result=result, adapter=adapter)
                    if write_canvas:
                        write_canvas_file(paths=paths, topic=topic, map_result=result, adapter=adapter)
                    result["writeback"] = {
                        "hub": str(paths.hub_file),
                        "trunkMap": str(paths.trunk_map_file),
                    }
                    if write_canvas:
                        result["writeback"]["canvas"] = str(paths.canvas_file)

            return self._record_schema_validation(result)
        finally:
            db.close()

    def assess_template(
        self,
        topic: str,
        session_id: str,
        concept_count: int = 6,
        source: str = "all",
        days: int = 180,
        top_k: int = 12,
        writeback: bool = False,
        allow_external: bool = False,
        run_id: str | None = None,
    ) -> dict:
        run_id = str(run_id or f"learn_template_{uuid4().hex[:12]}")
        db = self._open_db()
        try:
            map_result = generate_learning_map(
                db=db,
                topic=topic,
                source=source,
                days=days,
                top_k=top_k,
                allow_external=allow_external,
                run_id=run_id,
            )
            trunks = map_result.get("trunks") if isinstance(map_result.get("trunks"), list) else []
            selected = [
                str(item.get("canonical_id"))
                for item in trunks[: max(1, min(int(concept_count), len(trunks)))]
                if item.get("canonical_id")
            ]

            topic_slug = str(map_result.get("topicSlug", "")) or topic
            db.upsert_learning_session(
                session_id=session_id,
                topic_slug=topic_slug,
                target_trunk_ids=selected,
                status="draft",
            )
            db.append_learning_event(
                event_id=f"evt_{uuid4().hex}",
                event_type="learning.template.generated",
                logical_step="assess-template",
                session_id=session_id,
                payload={
                    "topic": topic,
                    "targetTrunkCount": len(selected),
                },
                policy_class=str(map_result.get("policy", {}).get("classification", "P2")),
                run_id=run_id,
                request_id=run_id,
                source="learning",
            )

            response = {
                "schema": TEMPLATE_SCHEMA,
                "runId": run_id,
                "topic": topic,
                "status": "ok",
                "session": {
                    "sessionId": session_id,
                    "targetTrunkIds": selected,
                },
                "policy": map_result.get("policy"),
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }

            if writeback:
                if not self.config.vault_path:
                    response["status"] = "error"
                    response["error"] = "vault_path not configured"
                    return response
                paths = build_paths(self.config.vault_path, topic)
                adapter = self._resolve_vault_adapter()
                session_path = write_session_template(
                    paths=paths,
                    topic=topic,
                    session_id=session_id,
                    target_trunk_ids=selected,
                    policy_mode="external-allowed" if allow_external else "local-only",
                    adapter=adapter,
                )
                write_hub(paths, topic=topic, session_id=session_id, status_line="template-ready", adapter=adapter)
                response["session"]["path"] = str(session_path)
                response["writeback"] = {
                    "hub": str(paths.hub_file),
                    "session": str(session_path),
                }
            return self._record_schema_validation(response)
        finally:
            db.close()

    def grade(
        self,
        topic: str,
        session_id: str,
        writeback: bool = False,
        allow_external: bool = False,
        run_id: str | None = None,
    ) -> dict:
        run_id = str(run_id or f"learn_grade_{uuid4().hex[:12]}")
        db = self._open_db()
        try:
            if not self.config.vault_path:
                session = db.get_learning_session(session_id)
                target_trunk_ids = session.get("target_trunk_ids_json") if session else []
                if not isinstance(target_trunk_ids, list):
                    target_trunk_ids = []

                reasons = ["vault_path not configured"]
                payload = {
                    "schema": "knowledge-hub.learning.grade.result.v1",
                    "runId": run_id,
                    "topic": topic,
                    "status": "blocked",
                    "policy": {
                        "mode": "local-only",
                        "allowed": True,
                        "classification": "P2",
                        "blockedReason": None,
                        "policyErrors": [],
                    },
                    "session": {
                        "sessionId": session_id,
                        "path": "",
                        "targetTrunkIds": target_trunk_ids,
                    },
                    "targetTrunkIds": target_trunk_ids,
                    "scores": {
                        "coverage": 0.0,
                        "edgeAccuracy": 0.0,
                        "explanationQuality": 0.0,
                        "final": 0.0,
                        "totalEdges": 0,
                        "validEdges": 0,
                        "minEdges": max(5, max(0, len(target_trunk_ids) - 1)),
                    },
                    "gateDecision": {
                        "passed": False,
                        "status": "blocked",
                        "reasons": reasons,
                    },
                    "weaknesses": [{"reason": "vault_path_not_configured", "severity": "high"}],
                    "policyErrors": [],
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                }
                if not target_trunk_ids and session is None:
                    reasons.append("session not found")
                    payload["weaknesses"][0]["reason"] = "session_not_found_or_no_targets"
                db.append_learning_event(
                    event_id=f"evt_{uuid4().hex}",
                    event_type="learning.grade.blocked",
                    logical_step="grade",
                    session_id=session_id,
                    payload={
                        "topic": topic,
                        "status": payload["status"],
                        "reasons": reasons,
                    },
                    policy_class="P2",
                    run_id=run_id,
                    request_id=run_id,
                    source="learning",
                )
                return payload

            paths = build_paths(self.config.vault_path, topic)
            session_path = paths.session_file(session_id)
            if not session_path.exists():
                payload = {
                    "schema": "knowledge-hub.learning.grade.result.v1",
                    "runId": run_id,
                    "topic": topic,
                    "status": "blocked",
                    "policy": {
                        "mode": "local-only",
                        "allowed": True,
                        "classification": "P2",
                        "blockedReason": None,
                        "policyErrors": [],
                    },
                    "session": {"sessionId": session_id, "path": str(session_path), "targetTrunkIds": []},
                    "targetTrunkIds": [],
                    "scores": {
                        "coverage": 0.0,
                        "edgeAccuracy": 0.0,
                        "explanationQuality": 0.0,
                        "final": 0.0,
                        "totalEdges": 0,
                        "validEdges": 0,
                        "minEdges": 0,
                    },
                    "gateDecision": {"passed": False, "status": "blocked", "reasons": ["session note not found"]},
                    "weaknesses": [{"reason": "session_note_missing", "severity": "high"}],
                    "policyErrors": [],
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                }
                db.append_learning_event(
                    event_id=f"evt_{uuid4().hex}",
                    event_type="learning.grade.blocked",
                    logical_step="grade",
                    session_id=session_id,
                    payload={
                        "topic": topic,
                        "status": payload["status"],
                        "reasons": payload["gateDecision"].get("reasons", []),
                    },
                    policy_class="P2",
                    run_id=run_id,
                    request_id=run_id,
                    source="learning",
                )
                return payload

            content = session_path.read_text(encoding="utf-8")
            result = grade_session_content(
                db=db,
                topic=topic,
                session_id=session_id,
                content=content,
                session_note_path=str(session_path),
                allow_external=allow_external,
                run_id=run_id,
            )
            result["runId"] = run_id

            if writeback:
                status_line = f"grade:{result.get('status')} final={result.get('scores', {}).get('final')}"
                adapter = self._resolve_vault_adapter()
                write_hub(paths, topic=topic, session_id=session_id, status_line=status_line, adapter=adapter)
                result["writeback"] = {
                    "hub": str(paths.hub_file),
                    "session": str(session_path),
                }

            return self._record_schema_validation(result)
        finally:
            db.close()

    def next(
        self,
        topic: str,
        session_id: str,
        source: str = "all",
        days: int = 180,
        top_k: int = 12,
        writeback: bool = False,
        allow_external: bool = False,
        run_id: str | None = None,
    ) -> dict:
        run_id = str(run_id or f"learn_next_{uuid4().hex[:12]}")
        db = self._open_db()
        try:
            map_result = generate_learning_map(
                db=db,
                topic=topic,
                source=source,
                days=days,
                top_k=top_k,
                allow_external=allow_external,
                run_id=run_id,
            )
            dynamic_dir = self._resolve_dynamic_dir()
            result = recommend_next(
                db=db,
                topic=topic,
                session_id=session_id,
                map_result=map_result,
                dynamic_dir=dynamic_dir,
                allow_external=allow_external,
                run_id=run_id,
            )

            if writeback:
                if not self.config.vault_path:
                    result.setdefault("reasoning", []).append("vault_path not configured; writeback skipped")
                else:
                    paths = build_paths(self.config.vault_path, topic)
                    adapter = self._resolve_vault_adapter()
                    write_next_branches(paths, topic=topic, next_result=result, adapter=adapter)
                    write_hub(
                        paths,
                        topic=topic,
                        session_id=session_id,
                        status_line=f"next:{result.get('status')}",
                        adapter=adapter,
                    )
                    result["writeback"] = {
                        "hub": str(paths.hub_file),
                        "nextBranches": str(paths.next_branches_file),
                    }

            return self._record_schema_validation(result)
        finally:
            db.close()

    def run(
        self,
        topic: str,
        session_id: str,
        source: str = "all",
        days: int = 180,
        top_k: int = 12,
        concept_count: int = 6,
        auto_next: bool = False,
        writeback: bool = False,
        canvas: bool = False,
        allow_external: bool = False,
        run_id: str | None = None,
    ) -> dict:
        run_id = str(run_id or f"learn_run_{uuid4().hex[:12]}")
        now = datetime.now(timezone.utc).isoformat()

        map_result = self.map(
            topic=topic,
            source=source,
            days=days,
            top_k=top_k,
            writeback=writeback,
            write_canvas=bool(writeback and canvas),
            allow_external=allow_external,
            run_id=run_id,
        )
        template_result = self.assess_template(
            topic=topic,
            session_id=session_id,
            concept_count=concept_count,
            source=source,
            days=days,
            top_k=top_k,
            writeback=writeback,
            allow_external=allow_external,
            run_id=run_id,
        )

        grade_result: dict | None = None
        next_result: dict | None = None

        if self.config.vault_path:
            session_path = build_paths(self.config.vault_path, topic).session_file(session_id)
            if session_path.exists():
                content = session_path.read_text(encoding="utf-8")
                _, body = parse_frontmatter(content)
                edges, _, _ = parse_edges_from_session(body, session_note_path=str(session_path))
                has_user_edge = len(edges) > 0
                if has_user_edge:
                    grade_result = self.grade(
                        topic=topic,
                        session_id=session_id,
                        writeback=writeback,
                        allow_external=allow_external,
                        run_id=run_id,
                    )
                    if auto_next and grade_result.get("status") not in {"blocked", "error"}:
                        next_result = self.next(
                            topic=topic,
                            session_id=session_id,
                            source=source,
                            days=days,
                            top_k=top_k,
                            writeback=writeback,
                            allow_external=allow_external,
                            run_id=run_id,
                        )

        return self._record_schema_validation({
            "schema": RUN_SCHEMA,
            "runId": run_id,
            "topic": topic,
            "status": "ok",
            "steps": {
                "map": map_result,
                "assessTemplate": template_result,
                "grade": grade_result,
                "next": next_result,
            },
            "createdAt": now,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        })
