"""Application service for Learning Coach MVP v2."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.core.models import LearningGraphCandidate
from knowledge_hub.core.sanitizer import redact_payload
from knowledge_hub.learning.contracts import LearningServiceRepository
from knowledge_hub.knowledge.quality_mode import (
    estimate_quality_mode_cost,
    resolve_quality_mode_route,
)
from knowledge_hub.learning.assessor import grade_session_content, parse_edges_from_session, parse_frontmatter
from knowledge_hub.learning.gap_analyzer import analyze_gaps
from knowledge_hub.learning.graph_builder import LearningGraphBuilder
from knowledge_hub.learning.graph_models import (
    LEARNING_GRAPH_BUILD_SCHEMA,
    LEARNING_GRAPH_PENDING_SCHEMA,
    LEARNING_PATH_SCHEMA,
    LEARNING_REVIEW_SCHEMA,
    LearningEdge,
)
from knowledge_hub.learning.graph_writeback import (
    write_concept_learning_section,
    write_learning_review_notes,
    write_paper_learning_context,
    write_topic_learning_path,
)
from knowledge_hub.learning.mapper import generate_learning_map, slugify_topic
from knowledge_hub.learning.model_router import get_llm_with_routing
from knowledge_hub.learning.models import (
    CHECKPOINT_SCHEMA,
    EXPLAIN_SCHEMA,
    GAP_SCHEMA,
    PATCH_SUGGEST_SCHEMA,
    QUIZ_GENERATE_SCHEMA,
    QUIZ_GRADE_SCHEMA,
    RUN_SCHEMA,
    SESSION_STATE_SCHEMA,
    START_RESUME_SCHEMA,
    TEMPLATE_SCHEMA,
)
from knowledge_hub.learning.obsidian_writeback import (
    build_paths,
    resolve_vault_write_adapter,
    write_canvas as write_canvas_file,
    write_gap_report,
    write_hub,
    write_next_branches,
    write_patch_suggestions,
    write_quiz_report,
    write_session_template,
    write_trunk_map,
)
from knowledge_hub.learning.patch_suggester import suggest_patch
from knowledge_hub.learning.path_generator import LearningPathGenerator
from knowledge_hub.learning.policy import evaluate_policy_for_payload
from knowledge_hub.learning.prerequisites import transitive_prerequisite_map
from knowledge_hub.learning.quiz_engine import generate_quiz, grade_quiz
from knowledge_hub.learning.recommender import recommend_next
from knowledge_hub.learning.resolver import EntityResolver, normalize_term
from knowledge_hub.learning.task_router import get_llm_for_task
from knowledge_hub.learning import service_support as _service_support


SqliteDbFactory = Callable[[], LearningServiceRepository]
RuntimeFactory = Callable[[], Any]


class LearningCoachService:
    def __init__(
        self,
        config: Config,
        *,
        sqlite_db: LearningServiceRepository | None = None,
        sqlite_db_factory: SqliteDbFactory | None = None,
        vector_db: Any | None = None,
        vector_db_factory: RuntimeFactory | None = None,
        embedder: Any | None = None,
        embedder_factory: RuntimeFactory | None = None,
        app_context: Any | None = None,
    ):
        self.config = config
        self._sqlite_db = sqlite_db
        self._sqlite_db_factory = sqlite_db_factory
        self._vector_db = vector_db
        self._vector_db_factory = vector_db_factory
        self._embedder = embedder
        self._embedder_factory = embedder_factory
        self._app_context = app_context
        self._resolved_sqlite_db: LearningServiceRepository | None = None
        self._resolved_vector_db: Any | None = None
        self._resolved_embedder: Any | None = None
        self._sqlite_db_resolved = False
        self._vector_db_resolved = False
        self._embedder_resolved = False

    def _runtime_provider(self) -> Any | None:
        return _service_support.runtime_provider(self)

    def _get_injected_sqlite_db(self) -> LearningServiceRepository | None:
        return _service_support.get_injected_sqlite_db(self)

    def _get_injected_vector_db(self) -> Any | None:
        return _service_support.get_injected_vector_db(self)

    def _get_injected_embedder(self) -> Any | None:
        return _service_support.get_injected_embedder(self)

    def _create_sqlite_db(self) -> LearningServiceRepository:
        return _service_support.create_sqlite_db(self)

    def _create_vector_db(self):
        return _service_support.create_vector_db(self)

    def _create_embedder(self):
        return _service_support.create_embedder(self)

    def _open_db(self) -> LearningServiceRepository:
        return _service_support.open_db(self)

    def _close_db(self, db: LearningServiceRepository | None) -> None:
        _service_support.close_db(self, db)

    def _get_vector_db(self):
        return _service_support.get_vector_db(self)

    def _get_embedder(self):
        return _service_support.get_embedder(self)

    def _build_rag_searcher(self, *, sqlite_db: LearningServiceRepository):
        return _service_support.build_rag_searcher(self, sqlite_db=sqlite_db)

    def _record_schema_validation(self, payload: dict) -> dict:
        return _service_support.record_schema_validation(self, payload)

    def _resolve_dynamic_dir(self) -> Path:
        return _service_support.resolve_dynamic_dir(self)

    def _resolve_vault_adapter(self):
        return _service_support.resolve_vault_adapter(self)

    def _obsidian_backend_kwargs(self) -> dict:
        return _service_support.obsidian_backend_kwargs(self)

    def _resolve_session_context(self, topic: str, session_id: str) -> tuple[str, str, list[str]]:
        return _service_support.resolve_session_context(self, topic, session_id)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _coerce_text_list(items: list[Any] | None) -> list[str]:
        values: list[str] = []
        for item in items or []:
            value = str(item).strip()
            if value and value not in values:
                values.append(value)
        return values

    def _coerce_concept_snapshot(
        self,
        *,
        db: LearningServiceRepository,
        item: Any,
        status: str,
        updated_at: str,
    ) -> dict[str, Any] | None:
        if isinstance(item, dict):
            raw_name = str(item.get("concept_id") or item.get("conceptId") or item.get("name") or item.get("display_name") or "").strip()
            evidence_refs = self._coerce_text_list(item.get("evidence_refs") or item.get("evidenceRefs") or [])
            summary = str(item.get("user_answer_summary") or item.get("userAnswerSummary") or "").strip()
        else:
            raw_name = str(item).strip()
            evidence_refs = []
            summary = ""
        if not raw_name:
            return None

        resolver = EntityResolver(db)
        resolved = resolver.resolve(raw_name, entity_type="concept")
        concept_id = resolved.canonical_id if resolved is not None else normalize_term(raw_name).replace(" ", "_") or raw_name
        display_name = resolved.display_name if resolved is not None else raw_name
        return {
            "conceptId": concept_id,
            "displayName": display_name,
            "status": status,
            "evidenceRefs": evidence_refs,
            "userAnswerSummary": summary,
            "updatedAt": updated_at,
        }

    @staticmethod
    def _merge_concept_states(existing: list[dict[str, Any]], updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {
            str(item.get("conceptId") or ""): dict(item)
            for item in existing
            if isinstance(item, dict) and str(item.get("conceptId") or "").strip()
        }
        for item in updates:
            concept_id = str(item.get("conceptId") or "").strip()
            if not concept_id:
                continue
            merged[concept_id] = dict(item)
        return sorted(merged.values(), key=lambda item: (str(item.get("status") or ""), str(item.get("displayName") or item.get("conceptId") or "")))

    @staticmethod
    def _collect_weak_areas(concept_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "conceptId": str(item.get("conceptId") or ""),
                "displayName": str(item.get("displayName") or item.get("conceptId") or ""),
                "status": str(item.get("status") or ""),
            }
            for item in concept_states
            if str(item.get("status") or "") in {"shaky", "unknown", "misconception"}
        ]

    @staticmethod
    def _collect_next_review_targets(concept_states: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
        return [
            str(item.get("displayName") or item.get("conceptId") or "")
            for item in concept_states
            if str(item.get("status") or "") in {"shaky", "unknown", "misconception"}
        ][: max(1, int(limit))]

    @staticmethod
    def _search_terms(topic: str, question: str, *, limit: int = 8) -> list[str]:
        values: list[str] = []
        for candidate in (
            str(topic or "").strip(),
            slugify_topic(topic).replace("_", " ").strip(),
            str(topic or "").strip().lower(),
            str(question or "").strip(),
        ):
            if candidate and candidate not in values:
                values.append(candidate)
        for token in re.findall(r"[A-Za-z0-9_+-]{2,}|[가-힣]{2,}", f"{topic} {question}"):
            value = str(token).strip()
            if value and value not in values:
                values.append(value)
            if len(values) >= max(2, int(limit)):
                break
        return values[: max(2, int(limit))]

    @staticmethod
    def _summarize_text_snippet(raw: Any, *, limit: int = 220) -> str:
        text = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not text:
            return ""
        return text[:limit].rstrip()

    def _search_local_learning_evidence(
        self,
        *,
        db: LearningServiceRepository,
        topic: str,
        question: str,
        source: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        terms = self._search_terms(topic, question, limit=max(4, int(top_k) * 2))
        include_notes = source in {"all", "note", "web"}
        include_papers = source in {"all", "paper"}
        search_notes = getattr(db, "search_notes", None)
        search_papers = getattr(db, "search_papers", None)
        candidates: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()

        for term in terms:
            if include_notes and callable(search_notes):
                for row in search_notes(term, limit=max(2, int(top_k))):
                    item = dict(row or {})
                    source_type = str(item.get("source_type") or "note")
                    if source == "note" and source_type not in {"note", "vault"}:
                        continue
                    if source == "web" and source_type != "web":
                        continue
                    title = str(item.get("title") or item.get("id") or "").strip()
                    if not title:
                        continue
                    key = ("note", title)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    candidates.append(
                        {
                            "title": title,
                            "sourceType": source_type,
                            "excerpt": self._summarize_text_snippet(item.get("content")),
                        }
                    )
            if include_papers and callable(search_papers):
                for row in search_papers(term, limit=max(2, int(top_k))):
                    item = dict(row or {})
                    title = str(item.get("title") or item.get("arxiv_id") or "").strip()
                    if not title:
                        continue
                    key = ("paper", title)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    excerpt_parts = [
                        self._summarize_text_snippet(item.get("field"), limit=60),
                        self._summarize_text_snippet(item.get("authors"), limit=80),
                        self._summarize_text_snippet(item.get("notes")),
                    ]
                    excerpt = " | ".join(part for part in excerpt_parts if part).strip()
                    candidates.append(
                        {
                            "title": title,
                            "sourceType": "paper",
                            "excerpt": excerpt,
                        }
                    )
            if len(candidates) >= max(1, int(top_k)):
                break
        return candidates[: max(1, int(top_k))]

    @staticmethod
    def _render_local_evidence_answer(topic: str, question: str, evidence_items: list[dict[str, Any]]) -> str:
        highlights: list[str] = []
        for item in evidence_items[:3]:
            title = str(item.get("title") or "").strip()
            excerpt = str(item.get("excerpt") or "").strip()
            source_type = str(item.get("sourceType") or "").strip()
            label = f"{source_type}:{title}" if source_type else title
            highlights.append(f"{label} -> {excerpt or '관련 항목이 확인되었습니다.'}")
        if not highlights:
            return ""
        joined = " / ".join(highlights)
        return (
            f"프로젝트 근거 기준으로 `{topic}`와 질문 `{question}`에 직접 닿는 자료를 먼저 묶으면: "
            f"{joined}"
        )

    def _build_session_gap_payload(
        self,
        *,
        db: LearningServiceRepository,
        topic: str,
        session_id: str,
        run_id: str,
    ) -> dict[str, Any] | None:
        session = db.get_learning_session(session_id) or {}
        target_trunk_ids = session.get("target_trunk_ids_json") if isinstance(session.get("target_trunk_ids_json"), list) else []
        if not target_trunk_ids:
            return None

        state = self._build_session_state_payload(
            db=db,
            topic=topic,
            session=session,
            run_id=run_id,
        )
        session_content, session_path, _ = self._resolve_session_context(topic, session_id)
        resolver = EntityResolver(db)
        used_targets: set[str] = set()
        parse_errors: list[dict[str, Any]] = []
        if session_content:
            _, body = parse_frontmatter(session_content)
            parsed_edges, parse_errors, _ = parse_edges_from_session(
                body or session_content,
                session_note_path=session_path,
            )
            for edge in parsed_edges:
                for raw_term in (edge.source_canonical_id, edge.target_canonical_id):
                    resolved = resolver.resolve(raw_term, entity_type="concept")
                    if resolved is not None and str(resolved.canonical_id or "").strip():
                        used_targets.add(str(resolved.canonical_id))

        concept_states = state.get("conceptStates", []) if isinstance(state.get("conceptStates"), list) else []
        weak_areas = state.get("weakAreas", []) if isinstance(state.get("weakAreas"), list) else []
        display_name_by_id: dict[str, str] = {}
        for collection in (concept_states, weak_areas):
            for item in collection:
                if not isinstance(item, dict):
                    continue
                concept_id = str(item.get("conceptId") or "").strip()
                display_name = str(item.get("displayName") or "").strip()
                if concept_id and display_name:
                    display_name_by_id[concept_id] = display_name

        target_trunks: list[dict[str, Any]] = []
        for raw_trunk_id in target_trunk_ids:
            trunk_id = str(raw_trunk_id).strip()
            if not trunk_id:
                continue
            resolved = resolver.resolve(trunk_id, entity_type="concept")
            display_name = (
                display_name_by_id.get(trunk_id)
                or (str(resolved.display_name).strip() if resolved is not None else "")
                or trunk_id
            )
            target_trunks.append(
                {
                    "canonical_id": trunk_id,
                    "display_name": display_name,
                }
            )

        missing_trunks: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        priority_by_status = {
            "unknown": 1.0,
            "misconception": 0.95,
            "shaky": 0.85,
        }
        for item in weak_areas:
            concept_id = str(item.get("conceptId") or "").strip()
            if not concept_id or concept_id in seen_ids:
                continue
            seen_ids.add(concept_id)
            status = str(item.get("status") or "").strip()
            missing_trunks.append(
                {
                    "canonical_id": concept_id,
                    "display_name": str(item.get("displayName") or concept_id),
                    "priority": priority_by_status.get(status, 0.8),
                    "reason": f"checkpoint state: {status or 'weak'}",
                }
            )
        if used_targets:
            for trunk_id in target_trunk_ids:
                if trunk_id in used_targets or trunk_id in seen_ids:
                    continue
                missing_trunks.append(
                    {
                        "canonical_id": str(trunk_id),
                        "display_name": str(trunk_id),
                        "priority": 0.7,
                        "reason": "target trunk not yet covered in session state",
                    }
                )
        missing_trunks.sort(key=lambda item: (float(item.get("priority") or 0.0), str(item.get("display_name") or "")), reverse=True)
        return {
            "status": "ok",
            "summary": {
                "targetTrunkCount": len(target_trunk_ids),
                "coveredTrunkCount": len([item for item in target_trunk_ids if item in used_targets]),
                "missingTrunkCount": len(missing_trunks),
                "weakEdgeCount": 0,
                "evidenceGapCount": 0,
                "avgGapConfidence": 0.0,
                "normalizationFailureRate": 0.0,
                "parseErrorCount": len(parse_errors),
            },
            "targetTrunkIds": target_trunk_ids,
            "targetTrunks": target_trunks,
            "missingTrunks": missing_trunks[:50],
            "weakEdges": [],
            "evidenceGaps": [],
            "parseErrors": parse_errors[:50],
            "sourceRefs": [session_path] if session_path else [],
            "mapStatus": "session-state",
            "policy": {
                "mode": "local-only",
                "allowed": True,
                "classification": "P2",
                "blockedReason": None,
                "policyErrors": [],
            },
        }

    def _ensure_learning_progress_snapshot(
        self,
        *,
        db: LearningServiceRepository,
        session_id: str,
        topic_slug: str,
        details_update: dict[str, Any],
        gate_status: str | None = None,
    ) -> dict[str, Any]:
        existing = db.get_learning_progress(session_id) or {}
        existing_details = existing.get("details_json") if isinstance(existing.get("details_json"), dict) else {}
        merged_details = dict(existing_details)
        merged_details.update(details_update)
        db.upsert_learning_progress(
            session_id=session_id,
            topic_slug=topic_slug,
            score_final=float(existing.get("score_final", 0.0) or 0.0),
            score_edge_accuracy=float(existing.get("score_edge_accuracy", 0.0) or 0.0),
            score_coverage=float(existing.get("score_coverage", 0.0) or 0.0),
            score_explanation_quality=float(existing.get("score_explanation_quality", 0.0) or 0.0),
            gate_passed=bool(existing.get("gate_passed", 0)),
            gate_status=str(gate_status or existing.get("gate_status") or "draft"),
            weaknesses=existing.get("weaknesses_json") if isinstance(existing.get("weaknesses_json"), list) else [],
            details=merged_details,
        )
        return db.get_learning_progress(session_id) or {}

    def _build_session_state_payload(
        self,
        *,
        db: LearningServiceRepository,
        topic: str,
        session: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any]:
        session_id = str(session.get("session_id") or "")
        topic_slug = str(session.get("topic_slug") or slugify_topic(topic))
        progress = db.get_learning_progress(session_id) or {}
        details = progress.get("details_json") if isinstance(progress.get("details_json"), dict) else {}
        coach_state = details.get("coachState") if isinstance(details.get("coachState"), dict) else {}
        concept_states = coach_state.get("conceptStates") if isinstance(coach_state.get("conceptStates"), list) else []
        quiz_history = coach_state.get("quizHistory") if isinstance(coach_state.get("quizHistory"), list) else []
        weak_areas = coach_state.get("weakAreas") if isinstance(coach_state.get("weakAreas"), list) else self._collect_weak_areas(concept_states)
        next_review_targets = coach_state.get("nextReviewTargets") if isinstance(coach_state.get("nextReviewTargets"), list) else self._collect_next_review_targets(concept_states)
        summary = coach_state.get("summary") if isinstance(coach_state.get("summary"), dict) else {}
        if not summary:
            summary = {
                "gateStatus": str(progress.get("gate_status") or session.get("status") or "draft"),
                "scoreFinal": float(progress.get("score_final", 0.0) or 0.0),
                "weakConceptCount": len(weak_areas),
            }

        return self._record_schema_validation(
            {
                "schema": SESSION_STATE_SCHEMA,
                "runId": run_id,
                "topic": topic,
                "topicSlug": topic_slug,
                "status": "ok",
                "session": {
                    "sessionId": session_id,
                    "topicSlug": topic_slug,
                    "status": str(session.get("status") or "draft"),
                    "targetTrunkIds": session.get("target_trunk_ids_json") if isinstance(session.get("target_trunk_ids_json"), list) else [],
                    "createdAt": session.get("created_at"),
                    "updatedAt": session.get("updated_at"),
                },
                "progress": {
                    "gateStatus": str(progress.get("gate_status") or session.get("status") or "draft"),
                    "gatePassed": bool(progress.get("gate_passed", 0)),
                    "scoreFinal": float(progress.get("score_final", 0.0) or 0.0),
                    "scoreEdgeAccuracy": float(progress.get("score_edge_accuracy", 0.0) or 0.0),
                    "scoreCoverage": float(progress.get("score_coverage", 0.0) or 0.0),
                    "scoreExplanationQuality": float(progress.get("score_explanation_quality", 0.0) or 0.0),
                },
                "summary": summary,
                "conceptStates": concept_states,
                "quizHistory": quiz_history,
                "weakAreas": weak_areas,
                "nextReviewTargets": next_review_targets,
                "recentEvents": db.list_learning_events(session_id=session_id, limit=10),
                "createdAt": self._now_iso(),
                "updatedAt": self._now_iso(),
            }
        )

    def _persist_coach_state(
        self,
        *,
        db: LearningServiceRepository,
        topic: str,
        session_id: str,
        concept_states: list[dict[str, Any]],
        summary: dict[str, Any],
        quiz_entry: dict[str, Any] | None,
        event_type: str,
        logical_step: str,
        event_payload: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any]:
        session = db.get_learning_session(session_id)
        topic_slug = str((session or {}).get("topic_slug") or slugify_topic(topic))
        weak_areas = self._collect_weak_areas(concept_states)
        next_review_targets = self._collect_next_review_targets(concept_states)
        existing = db.get_learning_progress(session_id) or {}
        details = existing.get("details_json") if isinstance(existing.get("details_json"), dict) else {}
        coach_state = details.get("coachState") if isinstance(details.get("coachState"), dict) else {}
        quiz_history = coach_state.get("quizHistory") if isinstance(coach_state.get("quizHistory"), list) else []
        if quiz_entry:
            quiz_history = [dict(quiz_entry), *[item for item in quiz_history if isinstance(item, dict)]][:10]
        snapshot = {
            "summary": summary,
            "conceptStates": concept_states,
            "weakAreas": weak_areas,
            "nextReviewTargets": next_review_targets,
            "quizHistory": quiz_history,
            "updatedAt": self._now_iso(),
        }
        progress = self._ensure_learning_progress_snapshot(
            db=db,
            session_id=session_id,
            topic_slug=topic_slug,
            details_update={"coachState": snapshot},
            gate_status=str(existing.get("gate_status") or (session or {}).get("status") or "draft"),
        )
        db.append_learning_event(
            event_id=f"evt_{uuid4().hex}",
            event_type=event_type,
            logical_step=logical_step,
            session_id=session_id,
            payload=event_payload,
            policy_class="P2",
            run_id=run_id,
            request_id=run_id,
            source="learning",
        )
        refreshed_session = db.get_learning_session(session_id) or {
            "session_id": session_id,
            "topic_slug": topic_slug,
            "target_trunk_ids_json": [],
            "status": progress.get("gate_status", "draft"),
        }
        return self._build_session_state_payload(
            db=db,
            topic=topic,
            session=refreshed_session,
            run_id=run_id,
        )

    @staticmethod
    def _extract_json_object(raw: str) -> dict:
        return _service_support.extract_json_object(raw)

    def _llm_json_generate(self, llm, prompt: str, context_payload: dict, max_tokens: int = 1400) -> tuple[dict, str | None]:
        return _service_support.llm_json_generate(self, llm, prompt, context_payload, max_tokens=max_tokens)

    def _refine_learning_edges(
        self,
        *,
        topic: str,
        edge_candidates: list[LearningEdge],
        candidate_data: dict,
        allow_external: bool,
        sqlite_db: LearningServiceRepository,
    ) -> tuple[list[LearningEdge], dict, list[str]]:
        if not edge_candidates:
            return edge_candidates, {"attempted": False, "applied": 0, "router": None}, []
        monthly_spend_usd = float(sqlite_db.get_quality_mode_monthly_spend() or 0.0)

        node_names = {
            node.node_id: node.canonical_name
            for node in candidate_data.get("nodes", [])
            if getattr(node, "node_id", None)
        }
        review_candidates = []
        for edge in edge_candidates[:24]:
            evidence = edge.evidence if isinstance(edge.evidence, dict) else {}
            provenance = edge.provenance if isinstance(edge.provenance, dict) else {}
            evidence_summary = ""
            if provenance.get("derivedFrom") == "ontology_relation":
                evidence_summary = f"ontology relation {provenance.get('predicateId') or evidence.get('predicateId') or ''}".strip()
            elif provenance.get("derivedFrom") == "ontology_claim":
                evidence_summary = str(evidence.get("claimText") or "")[:220]
            review_candidates.append(
                {
                    "edgeId": edge.edge_id,
                    "source": node_names.get(edge.source_node_id, edge.source_node_id),
                    "target": node_names.get(edge.target_node_id, edge.target_node_id),
                    "edgeType": edge.edge_type,
                    "confidence": round(float(edge.confidence), 3),
                    "derivedFrom": provenance.get("derivedFrom"),
                    "evidenceSummary": evidence_summary,
                }
            )

        quality_decision = resolve_quality_mode_route(
            self.config,
            item_kind="learning",
            requested_allow_external=allow_external,
            requested_mode="auto",
            topic=topic,
            monthly_spend_usd=monthly_spend_usd,
        )
        llm, routing, warnings = get_llm_for_task(
            self.config,
            task_type="learning_graph_refinement",
            allow_external=quality_decision.allow_external,
            query=f"refine learning graph edges for {topic}",
            context=json.dumps(review_candidates, ensure_ascii=False),
            source_count=len(review_candidates),
            force_route=quality_decision.llm_mode,
        )
        warnings = list(warnings or [])
        warnings.extend(quality_decision.warnings)
        if llm is None:
            return edge_candidates, {"attempted": False, "applied": 0, "router": routing.to_dict()}, warnings

        prompt = (
            "You are reviewing proposed learning-graph edges.\n"
            "Return ONLY a JSON object.\n"
            "Do not invent new edges, nodes, or ids.\n"
            "You may keep the current edgeType or change it only among: prerequisite, recommended_before, builds_on, example_of.\n"
            "Schema:\n"
            "{\n"
            '  "reviews": [\n'
            '    {"edgeId":"string","edgeType":"string","confidence":0.0,"reason":"string"}\n'
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Review only the provided candidates\n"
            "- confidence must be between 0 and 1\n"
            "- keep reasons short\n"
            "- prefer conservative downgrades over aggressive upgrades\n"
        )
        llm_context = redact_payload(
            {
                "topic": topic,
                "topicSlug": candidate_data.get("topicSlug"),
                "candidates": review_candidates,
            }
        )
        llm_payload, llm_error = self._llm_json_generate(
            llm,
            prompt,
            llm_context,
            max_tokens=1400,
        )
        if llm_error and quality_decision.allow_external and routing.route == "strong":
            warnings.append(llm_error)
            mini_llm, mini_routing, extra_warnings = get_llm_for_task(
                self.config,
                task_type="learning_graph_refinement",
                allow_external=True,
                query=f"refine learning graph edges for {topic}",
                context=json.dumps(review_candidates, ensure_ascii=False),
                source_count=len(review_candidates),
                force_route="mini",
            )
            warnings.extend(extra_warnings)
            if mini_llm is not None:
                llm_payload, llm_error = self._llm_json_generate(
                    mini_llm,
                    prompt,
                    llm_context,
                    max_tokens=1400,
                )
                if not llm_error:
                    routing = mini_routing
                    warnings.append("fallback_from_strong_call_error")
        if llm_error:
            warnings.append(llm_error)
            return edge_candidates, {"attempted": True, "applied": 0, "router": routing.to_dict()}, warnings

        allowed_edge_types = {"prerequisite", "recommended_before", "builds_on", "example_of"}
        reviews = llm_payload.get("reviews") if isinstance(llm_payload.get("reviews"), list) else []
        review_by_id = {
            str(item.get("edgeId")): item
            for item in reviews
            if isinstance(item, dict) and str(item.get("edgeId") or "").strip()
        }

        refined: list[LearningEdge] = []
        applied = 0
        for edge in edge_candidates:
            review = review_by_id.get(edge.edge_id)
            if not review:
                refined.append(edge)
                continue
            edge_type = str(review.get("edgeType") or edge.edge_type).strip()
            if edge_type not in allowed_edge_types:
                edge_type = edge.edge_type
            try:
                confidence = float(review.get("confidence", edge.confidence) or edge.confidence)
            except Exception:
                confidence = edge.confidence
            confidence = max(0.0, min(1.0, confidence))
            reason = str(review.get("reason") or "").strip()
            provenance = dict(edge.provenance or {})
            provenance["llmReview"] = {
                "provider": routing.provider,
                "model": routing.model,
                "route": routing.route,
                "reason": reason,
            }
            refined.append(
                LearningEdge(
                    edge_id=edge.edge_id,
                    source_node_id=edge.source_node_id,
                    edge_type=edge_type,
                    target_node_id=edge.target_node_id,
                    confidence=confidence,
                    status=edge.status,
                    provenance=provenance,
                    evidence=edge.evidence,
                )
            )
            applied += 1
        if (
            quality_decision.allow_external
            and applied > 0
            and quality_decision.llm_mode in {"mini", "strong"}
        ):
            sqlite_db.record_quality_mode_usage(
                "learning",
                quality_decision.llm_mode,
                estimate_quality_mode_cost(
                    self.config,
                    "learning",
                    quality_decision.llm_mode,
                ),
                topic_slug=slugify_topic(topic),
            )
        return refined, {"attempted": True, "applied": applied, "router": routing.to_dict()}, warnings

    def start_or_resume_topic(
        self,
        topic: str,
        *,
        force_new_session: bool = False,
        source: str = "all",
        days: int = 180,
        top_k: int = 12,
        concept_count: int = 6,
        run_id: str | None = None,
    ) -> dict:
        run_id = str(run_id or f"learn_start_{uuid4().hex[:12]}")
        db = self._open_db()
        try:
            topic_slug = slugify_topic(topic)
            resumed_session = None if force_new_session else next(
                (item for item in db.list_learning_sessions(topic_slug=topic_slug, limit=10) if str(item.get("status") or "") != "completed"),
                None,
            )
            created = False
            if resumed_session is None:
                session_id = f"{topic_slug}-{uuid4().hex[:8]}"
                template = self.assess_template(
                    topic=topic,
                    session_id=session_id,
                    concept_count=concept_count,
                    source=source,
                    days=days,
                    top_k=top_k,
                    writeback=False,
                    allow_external=False,
                    run_id=run_id,
                )
                resumed_session = db.get_learning_session(session_id) or {
                    "session_id": session_id,
                    "topic_slug": topic_slug,
                    "target_trunk_ids_json": template.get("session", {}).get("targetTrunkIds", []),
                    "status": "draft",
                    "created_at": self._now_iso(),
                    "updated_at": self._now_iso(),
                }
                created = True

            state = self._build_session_state_payload(
                db=db,
                topic=topic,
                session=resumed_session,
                run_id=run_id,
            )
            return self._record_schema_validation(
                {
                    "schema": START_RESUME_SCHEMA,
                    "runId": run_id,
                    "topic": topic,
                    "topicSlug": topic_slug,
                    "status": "ok",
                    "created": created,
                    "resumed": not created,
                    "session_id": resumed_session.get("session_id"),
                    "sessionId": resumed_session.get("session_id"),
                    "recent_state_summary": state.get("summary", {}),
                    "weak_areas": state.get("weakAreas", []),
                    "next_review_targets": state.get("nextReviewTargets", []),
                    "recommended_next_step": "take-quiz" if state.get("weakAreas") else "ask-question",
                    "createdAt": self._now_iso(),
                    "updatedAt": self._now_iso(),
                }
            )
        finally:
            self._close_db(db)

    def get_session_state(
        self,
        *,
        topic: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        run_id = str(run_id or f"learn_state_{uuid4().hex[:12]}")
        db = self._open_db()
        try:
            session = None
            resolved_topic = str(topic or "").strip()
            if session_id:
                session = db.get_learning_session(str(session_id))
                if session:
                    resolved_topic = resolved_topic or str(session.get("topic_slug") or session.get("topic") or session.get("topic_slug") or "")
            elif topic:
                topic_slug = slugify_topic(str(topic))
                sessions = db.list_learning_sessions(topic_slug=topic_slug, limit=1)
                session = sessions[0] if sessions else None
                resolved_topic = str(topic)

            if not session:
                return self._record_schema_validation(
                    {
                        "schema": SESSION_STATE_SCHEMA,
                        "runId": run_id,
                        "topic": resolved_topic,
                        "topicSlug": slugify_topic(resolved_topic) if resolved_topic else "",
                        "status": "not_found",
                        "session": {},
                        "progress": {},
                        "summary": {},
                        "conceptStates": [],
                        "quizHistory": [],
                        "weakAreas": [],
                        "nextReviewTargets": [],
                        "recentEvents": [],
                        "createdAt": self._now_iso(),
                        "updatedAt": self._now_iso(),
                    }
                )
            topic_value = resolved_topic or str(session.get("topic_slug") or "")
            return self._build_session_state_payload(
                db=db,
                topic=topic_value,
                session=session,
                run_id=run_id,
            )
        finally:
            self._close_db(db)

    def explain_topic(
        self,
        *,
        topic: str,
        question: str,
        session_id: str | None = None,
        source: str = "all",
        top_k: int = 5,
        min_score: float = 0.3,
        run_id: str | None = None,
    ) -> dict:
        run_id = str(run_id or f"learn_explain_{uuid4().hex[:12]}")
        db = self._open_db()
        try:
            active_session_id = str(session_id or "").strip()
            if not active_session_id:
                start_payload = self.start_or_resume_topic(topic, run_id=run_id)
                active_session_id = str(start_payload.get("session_id") or start_payload.get("sessionId") or "").strip()
            state = self.get_session_state(topic=topic, session_id=active_session_id, run_id=run_id)
            local_evidence = self._search_local_learning_evidence(
                db=db,
                topic=topic,
                question=question,
                source=source,
                top_k=max(1, int(top_k)),
            )
            evidence_refs = [str(item.get("title") or "").strip() for item in local_evidence if str(item.get("title") or "").strip()]
            if evidence_refs:
                answer = self._render_local_evidence_answer(topic, question, local_evidence)
                supplemental = []
            else:
                searcher = self._build_rag_searcher(sqlite_db=db)
                results = searcher.search(
                    question,
                    top_k=max(1, int(top_k)),
                    source_type=source,
                    retrieval_mode="hybrid",
                    alpha=0.7,
                )
                evidence_refs = [
                    str(getattr(item, "metadata", {}).get("title") or "")
                    for item in results
                    if str(getattr(item, "metadata", {}).get("title") or "").strip()
                ]
                if evidence_refs:
                    answer_payload = searcher.generate_answer(
                        question,
                        top_k=max(1, int(top_k)),
                        min_score=float(min_score),
                        source_type=source,
                        retrieval_mode="hybrid",
                        alpha=0.7,
                        allow_external=False,
                        paper_memory_mode="off",
                    )
                    answer = str((answer_payload or {}).get("answer") or "").strip()
                    supplemental = []
                else:
                    llm = getattr(searcher, "llm", None)
                    supplement = ""
                    if llm is not None and hasattr(llm, "generate"):
                        supplement = str(
                            llm.generate(f"{topic}: {question}", context="Provide a concise conceptual explanation.")
                        )[:1200].strip()
                    supplemental = [supplement] if supplement else []
                    answer = "프로젝트 근거를 충분히 찾지 못했습니다. 아래는 모델 기반 보완 설명입니다."
            weak_areas = state.get("weakAreas", []) if isinstance(state.get("weakAreas"), list) else []
            followup_checks = []
            for item in weak_areas[:2]:
                display_name = str(item.get("displayName") or item.get("conceptId") or "").strip()
                if display_name:
                    followup_checks.append(f"`{display_name}`를 자신의 말로 2문장 안에 설명해보세요.")
            if not followup_checks:
                followup_checks = [
                    f"`{topic}`의 핵심 아이디어를 한 문장으로 요약해보세요.",
                    f"`{topic}`에서 가장 헷갈리는 개념 하나를 골라 이유를 말해보세요.",
                ]
            payload = {
                "schema": EXPLAIN_SCHEMA,
                "runId": run_id,
                "topic": topic,
                "status": "ok",
                "session_id": active_session_id,
                "sessionId": active_session_id,
                "question": question,
                "answer": answer,
                "evidence_refs": evidence_refs,
                "evidenceRefs": evidence_refs,
                "supplemental_model_knowledge": supplemental,
                "supplementalModelKnowledge": supplemental,
                "followup_checks": followup_checks,
                "followupChecks": followup_checks,
                "stateSummary": state.get("summary", {}),
                "createdAt": self._now_iso(),
                "updatedAt": self._now_iso(),
            }
            db.append_learning_event(
                event_id=f"evt_{uuid4().hex}",
                event_type="learning.explain.generated",
                logical_step="explain",
                session_id=active_session_id or None,
                payload={
                    "topic": topic,
                    "question": question,
                    "evidenceRefCount": len(evidence_refs),
                    "usedModelSupplement": bool(supplemental),
                },
                policy_class="P2",
                run_id=run_id,
                request_id=run_id,
                source="learning",
            )
            return self._record_schema_validation(payload)
        finally:
            self._close_db(db)

    def checkpoint(
        self,
        *,
        topic: str,
        session_id: str,
        summary: str,
        known_items: list[Any] | None = None,
        shaky_items: list[Any] | None = None,
        unknown_items: list[Any] | None = None,
        misconceptions: list[Any] | None = None,
        writeback: bool = False,
        run_id: str | None = None,
    ) -> dict:
        run_id = str(run_id or f"learn_checkpoint_{uuid4().hex[:12]}")
        db = self._open_db()
        try:
            updated_at = self._now_iso()
            session = db.get_learning_session(session_id)
            if session is None:
                db.upsert_learning_session(
                    session_id=session_id,
                    topic_slug=slugify_topic(topic),
                    target_trunk_ids=[],
                    status="draft",
                )
                session = db.get_learning_session(session_id) or {
                    "session_id": session_id,
                    "topic_slug": slugify_topic(topic),
                    "target_trunk_ids_json": [],
                    "status": "draft",
                }

            existing_state = self.get_session_state(topic=topic, session_id=session_id, run_id=run_id)
            existing_concepts = existing_state.get("conceptStates", []) if isinstance(existing_state.get("conceptStates"), list) else []
            updates = []
            for items, status in (
                (known_items, "known"),
                (shaky_items, "shaky"),
                (unknown_items, "unknown"),
                (misconceptions, "misconception"),
            ):
                for item in items or []:
                    snapshot = self._coerce_concept_snapshot(db=db, item=item, status=status, updated_at=updated_at)
                    if snapshot:
                        updates.append(snapshot)
            concept_states = self._merge_concept_states(existing_concepts, updates)
            weak_areas = self._collect_weak_areas(concept_states)
            next_review_targets = self._collect_next_review_targets(concept_states)
            state_payload = self._persist_coach_state(
                db=db,
                topic=topic,
                session_id=session_id,
                concept_states=concept_states,
                summary={
                    "checkpointSummary": summary,
                    "weakConceptCount": len(weak_areas),
                    "updatedConceptCount": len(updates),
                },
                quiz_entry=None,
                event_type="learning.checkpoint.saved",
                logical_step="checkpoint",
                event_payload={
                    "topic": topic,
                    "summary": summary,
                    "updatedConceptCount": len(updates),
                },
                run_id=run_id,
            )
            payload = {
                "schema": CHECKPOINT_SCHEMA,
                "runId": run_id,
                "topic": topic,
                "status": "ok",
                "saved": True,
                "session_id": session_id,
                "sessionId": session_id,
                "updated_concepts": updates,
                "updatedConcepts": updates,
                "weak_areas": weak_areas,
                "weakAreas": weak_areas,
                "next_review_targets": next_review_targets,
                "nextReviewTargets": next_review_targets,
                "summary": summary,
                "stateSummary": state_payload.get("summary", {}),
                "createdAt": self._now_iso(),
                "updatedAt": self._now_iso(),
            }
            if writeback and self.config.vault_path:
                paths = build_paths(self.config.vault_path, topic)
                adapter = self._resolve_vault_adapter()
                write_hub(paths, topic=topic, session_id=session_id, status_line=f"checkpoint:{len(updates)} updated", adapter=adapter)
                payload["writeback"] = {"hub": str(paths.hub_file)}
            return self._record_schema_validation(payload)
        finally:
            self._close_db(db)

    def analyze_gaps(
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
        run_id = str(run_id or f"learn_gap_{uuid4().hex[:12]}")
        db = self._open_db()
        try:
            content, session_path, raw_evidence_texts = self._resolve_session_context(topic, session_id)
            policy = evaluate_policy_for_payload(
                allow_external=allow_external,
                raw_texts=raw_evidence_texts,
                mode="external-allowed" if allow_external else "local-only",
            )
            if not policy.allowed:
                payload = {
                    "schema": GAP_SCHEMA,
                    "runId": run_id,
                    "topic": topic,
                    "status": "blocked",
                    "policy": policy.to_dict(),
                    "session": {"sessionId": session_id, "path": session_path},
                    "summary": {
                        "targetTrunkCount": 0,
                        "coveredTrunkCount": 0,
                        "missingTrunkCount": 0,
                        "weakEdgeCount": 0,
                        "evidenceGapCount": 0,
                        "avgGapConfidence": 0.0,
                        "normalizationFailureRate": 0.0,
                        "parseErrorCount": 0,
                    },
                    "missingTrunks": [],
                    "weakEdges": [],
                    "evidenceGaps": [],
                    "policyErrors": policy.policy_errors,
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                }
                return self._record_schema_validation(payload)

            result = analyze_gaps(
                db=db,
                topic=topic,
                session_content=content,
                session_note_path=session_path,
                session_id=session_id,
                source=source,
                days=days,
                top_k=top_k,
                min_confidence=float(self.config.get_nested("learning", "gap", "min_confidence", default=0.7)),
                allow_external=allow_external,
                run_id=run_id,
            )

            llm, routing, warnings = get_llm_with_routing(
                self.config,
                allow_external,
                avg_gap_confidence=float(result.get("summary", {}).get("avgGapConfidence", 1.0)),
                normalization_failure_rate=float(result.get("summary", {}).get("normalizationFailureRate", 0.0)),
            )
            # Keep this for observability; actual gap scoring remains ontology-first deterministic.
            _ = llm

            payload = {
                "schema": GAP_SCHEMA,
                "runId": run_id,
                "topic": topic,
                "status": result.get("status", "ok"),
                "policy": result.get("policy", policy.to_dict()),
                "session": {"sessionId": session_id, "path": session_path},
                "summary": result.get("summary", {}),
                "targetTrunkIds": result.get("targetTrunkIds", []),
                "missingTrunks": result.get("missingTrunks", []),
                "weakEdges": result.get("weakEdges", []),
                "evidenceGaps": result.get("evidenceGaps", []),
                "parseErrors": result.get("parseErrors", []),
                "router": routing.to_dict(),
                "sourceRefs": result.get("sourceRefs", []),
                "warnings": warnings,
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }

            if writeback and self.config.vault_path:
                paths = build_paths(self.config.vault_path, topic)
                adapter = self._resolve_vault_adapter()
                write_gap_report(paths, topic=topic, gap_result=payload, adapter=adapter)
                payload["writeback"] = {"gapReport": str(paths.gap_report_file)}

            db.append_learning_event(
                event_id=f"evt_{uuid4().hex}",
                event_type="learning.gap.generated",
                logical_step="gap",
                session_id=session_id,
                payload={
                    "topic": topic,
                    "status": payload.get("status"),
                    "missingTrunkCount": payload.get("summary", {}).get("missingTrunkCount", 0),
                },
                policy_class=str(payload.get("policy", {}).get("classification", "P2")),
                run_id=run_id,
                request_id=run_id,
                source="learning",
            )
            return self._record_schema_validation(payload)
        finally:
            self._close_db(db)

    def generate_quiz(
        self,
        topic: str,
        session_id: str,
        source: str = "all",
        days: int = 180,
        top_k: int = 12,
        mix: str | None = None,
        question_count: int = 6,
        writeback: bool = False,
        allow_external: bool = False,
        run_id: str | None = None,
    ) -> dict:
        run_id = str(run_id or f"learn_quiz_gen_{uuid4().hex[:12]}")
        db = self._open_db()
        try:
            gap_payload = self._build_session_gap_payload(
                db=db,
                topic=topic,
                session_id=session_id,
                run_id=run_id,
            )
        finally:
            self._close_db(db)
        if gap_payload is None:
            gap_payload = self.analyze_gaps(
                topic=topic,
                session_id=session_id,
                source=source,
                days=days,
                top_k=top_k,
                writeback=False,
                allow_external=allow_external,
                run_id=run_id,
            )
        quiz_mix = str(mix or self.config.get_nested("learning", "quiz", "mix", default="mixed"))
        quiz = generate_quiz(
            topic,
            target_trunk_ids=gap_payload.get("targetTrunkIds", []) if isinstance(gap_payload.get("targetTrunkIds"), list) else [],
            target_trunks=gap_payload.get("targetTrunks", []) if isinstance(gap_payload.get("targetTrunks"), list) else [],
            missing_trunks=gap_payload.get("missingTrunks", []) if isinstance(gap_payload.get("missingTrunks"), list) else [],
            mix=quiz_mix,
            question_count=question_count,
        )
        payload = {
            "schema": QUIZ_GENERATE_SCHEMA,
            "runId": run_id,
            "topic": topic,
            "status": "ok",
            "policy": gap_payload.get("policy"),
            "session": {"sessionId": session_id, "path": gap_payload.get("session", {}).get("path", "")},
            "quiz": quiz,
            "gapSummary": gap_payload.get("summary", {}),
            "sourceRefs": gap_payload.get("sourceRefs", []),
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
        if writeback and self.config.vault_path:
            paths = build_paths(self.config.vault_path, topic)
            adapter = self._resolve_vault_adapter()
            write_quiz_report(paths, topic=topic, quiz_payload=payload, adapter=adapter)
            payload["writeback"] = {"quiz": str(paths.quiz_file)}
        return self._record_schema_validation(payload)

    def grade_quiz(
        self,
        topic: str,
        session_id: str,
        answers: list[dict],
        source: str = "all",
        days: int = 180,
        top_k: int = 12,
        writeback: bool = False,
        allow_external: bool = False,
        run_id: str | None = None,
    ) -> dict:
        run_id = str(run_id or f"learn_quiz_grade_{uuid4().hex[:12]}")
        db = self._open_db()
        try:
            quiz_payload = self.generate_quiz(
                topic=topic,
                session_id=session_id,
                source=source,
                days=days,
                top_k=top_k,
                writeback=False,
                allow_external=allow_external,
                run_id=run_id,
            )
            quiz = quiz_payload.get("quiz", {})
            contains_essay = any(str(item.get("type", "")) == "essay" for item in (quiz.get("questions", []) if isinstance(quiz, dict) else []))
            grading = grade_quiz(quiz_payload, answers)
            llm, routing, warnings = get_llm_with_routing(
                self.config,
                allow_external,
                avg_gap_confidence=float(quiz_payload.get("gapSummary", {}).get("avgGapConfidence", 1.0)),
                normalization_failure_rate=float(quiz_payload.get("gapSummary", {}).get("normalizationFailureRate", 0.0)),
                contains_essay=contains_essay,
            )

            answer_texts = [str(item.get("answer", "")) for item in answers if isinstance(item, dict)]
            ext_policy = evaluate_policy_for_payload(
                allow_external=allow_external,
                raw_texts=answer_texts,
                mode="external-allowed" if allow_external else "local-only",
            )
            if allow_external and not ext_policy.allowed:
                payload = {
                    "schema": QUIZ_GRADE_SCHEMA,
                    "runId": run_id,
                    "topic": topic,
                    "status": "blocked",
                    "policy": ext_policy.to_dict(),
                    "session": quiz_payload.get("session"),
                    "quiz": quiz,
                    "grading": redact_payload(grading),
                    "router": routing.to_dict(),
                    "warnings": warnings,
                    "policyErrors": ext_policy.policy_errors,
                    "sourceRefs": quiz_payload.get("sourceRefs", []),
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                }
                return self._record_schema_validation(payload)

            if llm is not None and allow_external:
                prompt = (
                    "You are a strict learning coach. Return ONLY JSON object.\n"
                    "Goal: improve quiz grading feedback quality without changing factual correctness.\n"
                    "Schema:\n"
                    "{\n"
                    '  "overallFeedback": "string",\n'
                    '  "detailOverrides": [{"id":"q1","feedback":"string"}],\n'
                    '  "studyFocus": ["string"],\n'
                    '  "confidence": 0.0\n'
                    "}\n"
                    "Rules: keep concise, Korean language, do not include raw secrets."
                )
                llm_context = redact_payload(
                    {
                        "topic": topic,
                        "quiz": quiz,
                        "answers": answers,
                        "grading": grading,
                        "gapSummary": quiz_payload.get("gapSummary", {}),
                    }
                )
                llm_payload, llm_error = self._llm_json_generate(llm, prompt, llm_context, max_tokens=1200)
                if llm_error:
                    warnings.append(llm_error)
                else:
                    overrides = llm_payload.get("detailOverrides")
                    if isinstance(overrides, list):
                        by_id = {
                            str(item.get("id")): str(item.get("feedback"))
                            for item in overrides
                            if isinstance(item, dict) and item.get("id") and item.get("feedback")
                        }
                        for item in grading.get("details", []):
                            qid = str(item.get("id", ""))
                            if qid in by_id:
                                item["feedback"] = by_id[qid]
                    overall = str(llm_payload.get("overallFeedback", "")).strip()
                    study_focus = llm_payload.get("studyFocus") if isinstance(llm_payload.get("studyFocus"), list) else []
                    extra_feedback: list[str] = []
                    if overall:
                        extra_feedback.append(overall)
                    for item in study_focus:
                        value = str(item).strip()
                        if value:
                            extra_feedback.append(value)
                    if extra_feedback:
                        grading["feedback"] = extra_feedback
                    grading["llmReview"] = {
                        "provider": routing.provider,
                        "model": routing.model,
                        "confidence": float(llm_payload.get("confidence", 0.0) or 0.0),
                    }

            concept_states = []
            submitted_by_id = {
                str(item.get("id")): str(item.get("answer", "")).strip()
                for item in answers
                if isinstance(item, dict)
            }
            updated_at = self._now_iso()
            for detail in grading.get("details", []):
                if not isinstance(detail, dict):
                    continue
                target_concept = str(detail.get("targetConceptId") or "").strip()
                if not target_concept:
                    continue
                submitted_answer = submitted_by_id.get(str(detail.get("id") or ""), "")
                if detail.get("isCorrect"):
                    status = "known"
                elif not submitted_answer:
                    status = "unknown"
                elif str(detail.get("type") or "") in {"mcq", "short"}:
                    status = "misconception"
                else:
                    status = "shaky"
                snapshot = self._coerce_concept_snapshot(
                    db=db,
                    item={
                        "concept_id": target_concept,
                        "user_answer_summary": submitted_answer[:240],
                        "evidence_refs": quiz_payload.get("sourceRefs", []),
                    },
                    status=status,
                    updated_at=updated_at,
                )
                if snapshot:
                    concept_states.append(snapshot)

            existing_state = self.get_session_state(topic=topic, session_id=session_id, run_id=run_id)
            merged_concepts = self._merge_concept_states(
                existing_state.get("conceptStates", []) if isinstance(existing_state.get("conceptStates"), list) else [],
                concept_states,
            )
            state_payload = self._persist_coach_state(
                db=db,
                topic=topic,
                session_id=session_id,
                concept_states=merged_concepts,
                summary={
                    "quizScore": grading.get("score"),
                    "quizPassed": grading.get("passed"),
                    "correct": grading.get("correct"),
                    "total": grading.get("total"),
                },
                quiz_entry={
                    "runId": run_id,
                    "score": grading.get("score"),
                    "total": grading.get("total"),
                    "correct": grading.get("correct"),
                    "passed": grading.get("passed"),
                    "createdAt": updated_at,
                },
                event_type="learning.quiz.graded",
                logical_step="quiz-grade",
                event_payload={
                    "topic": topic,
                    "score": grading.get("score"),
                    "passed": grading.get("passed"),
                    "correct": grading.get("correct"),
                    "total": grading.get("total"),
                },
                run_id=run_id,
            )

            payload = {
                "schema": QUIZ_GRADE_SCHEMA,
                "runId": run_id,
                "topic": topic,
                "status": "ok",
                "policy": quiz_payload.get("policy"),
                "session": quiz_payload.get("session"),
                "quiz": quiz,
                "grading": redact_payload(grading),
                "router": routing.to_dict(),
                "warnings": warnings,
                "sourceRefs": quiz_payload.get("sourceRefs", []),
                "stateSummary": state_payload.get("summary", {}),
                "weakAreas": state_payload.get("weakAreas", []),
                "nextReviewTargets": state_payload.get("nextReviewTargets", []),
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }
            if writeback and self.config.vault_path:
                paths = build_paths(self.config.vault_path, topic)
                adapter = self._resolve_vault_adapter()
                write_quiz_report(paths, topic=topic, quiz_payload=payload, adapter=adapter)
                payload["writeback"] = {"quiz": str(paths.quiz_file)}
            return self._record_schema_validation(payload)
        finally:
            self._close_db(db)

    def suggest_patch(
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
        run_id = str(run_id or f"learn_patch_{uuid4().hex[:12]}")
        gap_payload = self.analyze_gaps(
            topic=topic,
            session_id=session_id,
            source=source,
            days=days,
            top_k=top_k,
            writeback=False,
            allow_external=allow_external,
            run_id=run_id,
        )
        session_path = str(gap_payload.get("session", {}).get("path", ""))
        patch = suggest_patch(
            topic,
            gap_payload=gap_payload,
            session_note_path=session_path,
        )
        llm, routing, warnings = get_llm_with_routing(
            self.config,
            allow_external,
            avg_gap_confidence=float(gap_payload.get("summary", {}).get("avgGapConfidence", 1.0)),
            normalization_failure_rate=float(gap_payload.get("summary", {}).get("normalizationFailureRate", 0.0)),
        )
        if llm is not None and allow_external:
            prompt = (
                "You are a senior technical writing assistant for knowledge graph notes.\n"
                "Return ONLY JSON object to refine patch suggestions.\n"
                "Schema:\n"
                "{\n"
                '  "suggestions":[{"id":"s1","targetPath":"...","sectionTitle":"...","reason":"...","confidence":0.0,"mode":"proposal-only","patchText":"..."}],\n'
                '  "globalNote":"string"\n'
                "}\n"
                "Rules: keep proposal-only mode, preserve ids if possible, Korean language."
            )
            llm_context = redact_payload(
                {
                    "topic": topic,
                    "gapSummary": gap_payload.get("summary", {}),
                    "missingTrunks": gap_payload.get("missingTrunks", [])[:10],
                    "weakEdges": gap_payload.get("weakEdges", [])[:10],
                    "suggestions": patch.get("suggestions", []),
                }
            )
            llm_payload, llm_error = self._llm_json_generate(llm, prompt, llm_context, max_tokens=1800)
            if llm_error:
                warnings.append(llm_error)
            else:
                refined = llm_payload.get("suggestions")
                if isinstance(refined, list):
                    normalized: list[dict] = []
                    for item in refined[:10]:
                        if not isinstance(item, dict):
                            continue
                        normalized.append(
                            {
                                "id": str(item.get("id", f"s_{len(normalized)+1}")),
                                "targetPath": str(item.get("targetPath", session_path)),
                                "sectionTitle": str(item.get("sectionTitle", "## Concept Map Edges")),
                                "reason": str(item.get("reason", "gap_refinement")),
                                "confidence": float(item.get("confidence", 0.5) or 0.5),
                                "mode": "proposal-only",
                                "patchText": str(item.get("patchText", "")).strip(),
                            }
                        )
                    if normalized:
                        patch["suggestions"] = normalized
                        patch["suggestionCount"] = len(normalized)
                note = str(llm_payload.get("globalNote", "")).strip()
                if note:
                    patch["globalNote"] = note
        payload = {
            "schema": PATCH_SUGGEST_SCHEMA,
            "runId": run_id,
            "topic": topic,
            "status": "ok",
            "policy": gap_payload.get("policy"),
            "session": gap_payload.get("session"),
            "mode": "proposal-only",
            "suggestions": patch.get("suggestions", []),
            "router": routing.to_dict(),
            "warnings": warnings,
            "summary": {
                "suggestionCount": patch.get("suggestionCount", 0),
                "missingTrunkCount": gap_payload.get("summary", {}).get("missingTrunkCount", 0),
                "weakEdgeCount": gap_payload.get("summary", {}).get("weakEdgeCount", 0),
            },
            "sourceRefs": patch.get("sourceRefs", []),
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
        if writeback and self.config.vault_path:
            paths = build_paths(self.config.vault_path, topic)
            adapter = self._resolve_vault_adapter()
            write_patch_suggestions(paths, topic=topic, patch_payload=payload, adapter=adapter)
            payload["writeback"] = {"patchSuggestions": str(paths.patch_suggestions_file)}
        return self._record_schema_validation(payload)

    def reinforce(
        self,
        topic: str,
        session_id: str,
        source: str = "all",
        days: int = 180,
        top_k: int = 12,
        top_k_per_gap: int = 3,
        writeback: bool = False,
        allow_external: bool = False,
        run_id: str | None = None,
    ) -> dict:
        """
        지식 보강 추천 (gap 기반 소스 매칭)

        Args:
            topic: 학습 주제
            session_id: 세션 ID
            source: 소스 필터
            days: 최근 일수
            top_k: trunk map top-k
            top_k_per_gap: 각 gap당 추천 소스 수
            writeback: Obsidian writeback 여부
            allow_external: 외부 호출 허용
            run_id: 실행 ID
        """
        run_id = str(run_id or f"learn_reinforce_{uuid4().hex[:12]}")

        # 1) gap 분석
        gap_payload = self.analyze_gaps(
            topic=topic,
            session_id=session_id,
            source=source,
            days=days,
            top_k=top_k,
            writeback=False,
            allow_external=allow_external,
            run_id=run_id,
        )

        # 2) RAG searcher 초기화
        db = self._open_db()
        try:
            searcher = self._build_rag_searcher(sqlite_db=db)

            # 3) 보강 추천 생성
            from knowledge_hub.learning.knowledge_reinforcer import recommend_reinforcements

            result = recommend_reinforcements(
                db=db,
                searcher=searcher,
                topic=topic,
                session_id=session_id,
                gap_result=gap_payload,
                top_k_per_gap=top_k_per_gap,
                allow_external=allow_external,
                run_id=run_id,
            )

            # 4) writeback
            if writeback and self.config.vault_path:
                paths = build_paths(self.config.vault_path, topic)
                adapter = self._resolve_vault_adapter()
                from knowledge_hub.learning.obsidian_writeback import write_reinforcement_plan
                write_reinforcement_plan(paths, topic=topic, result=result, adapter=adapter)
                result["writeback"] = {"reinforcementPlan": str(paths.reinforcement_plan_file)}

            return self._record_schema_validation(result)

        finally:
            self._close_db(db)

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
            self._close_db(db)

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
            self._close_db(db)

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
            self._close_db(db)

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
            self._close_db(db)

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

    def build_learning_graph(
        self,
        topic: str,
        top_k: int = 12,
        allow_external: bool = False,
        run_id: str | None = None,
    ) -> dict:
        run_id = str(run_id or f"learn_graph_{uuid4().hex[:12]}")
        db = self._open_db()
        try:
            builder = LearningGraphBuilder(db)
            candidate_data = builder.build_topic_candidates(topic=topic, top_k=max(1, top_k))
            nodes = candidate_data["nodes"]
            edge_candidates = builder.generate_edge_candidates(
                topic=topic,
                top_k=max(1, top_k),
                candidate_data=candidate_data,
            )
            edge_candidates, refinement, refinement_warnings = self._refine_learning_edges(
                topic=topic,
                edge_candidates=edge_candidates,
                candidate_data=candidate_data,
                allow_external=allow_external,
                sqlite_db=db,
            )
            resource_links = builder.generate_resource_links(
                topic=topic,
                top_k=max(1, top_k),
                candidate_data=candidate_data,
            )

            pending_items: list[LearningGraphCandidate] = []
            for node in nodes:
                db.upsert_learning_graph_node(
                    node_id=node.node_id,
                    entity_id=node.entity_id,
                    node_type=node.node_type,
                    canonical_name=node.canonical_name,
                    difficulty_level=node.difficulty_level,
                    difficulty_score=node.difficulty_score,
                    stage=node.stage,
                    confidence=node.confidence,
                    provenance=node.provenance,
                )
                pending_items.append(
                    LearningGraphCandidate(
                        topic_slug=candidate_data["topicSlug"],
                        item_type="difficulty",
                        payload=node.to_dict(),
                        confidence=node.confidence,
                        provenance=node.provenance,
                    )
                )

            for edge in edge_candidates:
                pending_items.append(
                    LearningGraphCandidate(
                        topic_slug=candidate_data["topicSlug"],
                        item_type="edge",
                        payload=edge.to_dict(),
                        confidence=edge.confidence,
                        provenance=edge.provenance,
                        evidence=edge.evidence,
                    )
                )

            for item in resource_links:
                pending_items.append(
                    LearningGraphCandidate(
                        topic_slug=candidate_data["topicSlug"],
                        item_type="resource_link",
                        payload=item,
                        confidence=float(item.get("confidence", 0.5)),
                        provenance=item.get("provenance") or {},
                    )
                )

            path_payload = LearningPathGenerator().generate_path(
                topic_slug=candidate_data["topicSlug"],
                nodes=[db.learning_graph_store.get_node(node.node_id) or {
                    "node_id": node.node_id,
                    "canonical_name": node.canonical_name,
                    "difficulty_level": node.difficulty_level,
                    "stage": node.stage,
                    "node_type": node.node_type,
                } for node in nodes],
                edges=edge_candidates,
                resource_links=[{
                    "concept_node_id": item.get("conceptNodeId"),
                    "resource_node_id": item.get("resourceNodeId"),
                    "link_type": item.get("linkType"),
                } for item in resource_links],
                approved_only=False,
            )
            pending_items.append(
                LearningGraphCandidate(
                    topic_slug=candidate_data["topicSlug"],
                    item_type="path",
                    payload=path_payload.to_dict(),
                    confidence=0.7,
                    provenance=path_payload.provenance,
                )
            )

            queue_result = builder.queue_pending(topic=topic, items=pending_items)
            db.append_learning_graph_event(
                event_type="learning.graph.generated",
                topic_slug=candidate_data["topicSlug"],
                payload={
                    "runId": run_id,
                    "topic": topic,
                    "nodeCount": len(nodes),
                    "edgeCount": len(edge_candidates),
                    "resourceLinkCount": len(resource_links),
                    "pendingCount": len(queue_result.get("queuedIds", [])),
                },
                actor="learning",
            )
            payload = {
                "schema": LEARNING_GRAPH_BUILD_SCHEMA,
                "runId": run_id,
                "topic": topic,
                "topicSlug": candidate_data["topicSlug"],
                "status": "ok",
                "policy": evaluate_policy_for_payload(
                    allow_external=allow_external,
                    raw_texts=[topic],
                    mode="external-allowed" if allow_external else "local-only",
                ).to_dict(),
                "counts": {
                    "nodes": len(nodes),
                    "edgeCandidates": len(edge_candidates),
                    "resourceLinks": len(resource_links),
                    "pendingQueued": len(queue_result.get("queuedIds", [])),
                },
                "edgeRefinement": refinement,
                "warnings": refinement_warnings,
                "queued": queue_result,
                "nodes": [node.to_dict() for node in nodes],
                "edgeCandidates": [edge.to_dict() for edge in edge_candidates[:50]],
                "resourceLinks": resource_links[:50],
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }
            return self._record_schema_validation(payload)
        finally:
            self._close_db(db)

    def list_learning_graph_pending(
        self,
        topic: str | None = None,
        item_type: str = "all",
        limit: int = 200,
    ) -> dict:
        db = self._open_db()
        try:
            topic_slug = slugify_topic(topic) if topic else None
            items = db.list_learning_graph_pending(
                topic_slug=topic_slug,
                item_type=item_type,
                limit=max(1, limit),
            )
            counts: dict[str, int] = {}
            for item in items:
                counts[str(item.get("item_type"))] = counts.get(str(item.get("item_type")), 0) + 1
            payload = {
                "schema": LEARNING_GRAPH_PENDING_SCHEMA,
                "status": "ok",
                "topic": topic,
                "topicSlug": topic_slug,
                "counts": counts,
                "items": items,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }
            return self._record_schema_validation(payload)
        finally:
            self._close_db(db)

    def apply_learning_graph_pending(self, pending_id: int) -> dict:
        db = self._open_db()
        try:
            pending = db.get_learning_graph_pending(int(pending_id))
            if not pending:
                return self._record_schema_validation({
                    "schema": LEARNING_GRAPH_PENDING_SCHEMA,
                    "status": "error",
                    "error": "pending item not found",
                    "pendingId": int(pending_id),
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                })

            item_type = str(pending.get("item_type") or "")
            payload = pending.get("payload_json") if isinstance(pending.get("payload_json"), dict) else {}
            topic_slug = str(pending.get("topic_slug") or "")
            if item_type == "difficulty":
                db.upsert_learning_graph_node(
                    node_id=str(payload.get("nodeId")),
                    entity_id=payload.get("entityId"),
                    node_type=str(payload.get("nodeType")),
                    canonical_name=str(payload.get("canonicalName")),
                    difficulty_level=str(payload.get("difficultyLevel")),
                    difficulty_score=float(payload.get("difficultyScore", 0.5)),
                    stage=str(payload.get("stage") or payload.get("difficultyLevel") or "intermediate"),
                    confidence=float(payload.get("confidence", pending.get("confidence", 0.5))),
                    provenance=payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {},
                )
            elif item_type == "edge":
                db.upsert_learning_graph_edge(
                    edge_id=str(payload.get("edgeId")),
                    source_node_id=str(payload.get("sourceNodeId")),
                    edge_type=str(payload.get("edgeType")),
                    target_node_id=str(payload.get("targetNodeId")),
                    confidence=float(payload.get("confidence", pending.get("confidence", 0.5))),
                    status="approved",
                    provenance=payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {},
                    evidence=payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {},
                )
            elif item_type == "resource_link":
                db.upsert_learning_graph_resource_link(
                    link_id=str(payload.get("linkId")),
                    concept_node_id=str(payload.get("conceptNodeId")),
                    resource_node_id=str(payload.get("resourceNodeId")),
                    link_type=str(payload.get("linkType")),
                    reading_stage=str(payload.get("readingStage") or "intermediate"),
                    confidence=float(payload.get("confidence", pending.get("confidence", 0.5))),
                    status="approved",
                    provenance=payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {},
                )
            elif item_type == "path":
                latest = db.get_latest_learning_graph_path(topic_slug=topic_slug, status="approved")
                next_version = int(latest.get("version", 0)) + 1 if latest else 1
                db.upsert_learning_graph_path(
                    path_id=str(payload.get("pathId")),
                    topic_slug=str(payload.get("topicSlug") or topic_slug),
                    status="approved",
                    version=next_version,
                    path_payload=payload,
                    score_payload=payload.get("score") if isinstance(payload.get("score"), dict) else {},
                    provenance=payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {},
                )
            db.set_learning_graph_pending_status(int(pending_id), "approved")
            db.append_learning_graph_event(
                event_type="learning.graph.pending.applied",
                topic_slug=topic_slug,
                payload={"pendingId": int(pending_id), "itemType": item_type},
                actor="learning",
            )
            result = {
                "schema": LEARNING_GRAPH_PENDING_SCHEMA,
                "status": "ok",
                "pendingId": int(pending_id),
                "itemType": item_type,
                "topicSlug": topic_slug,
                "applied": True,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }
            return self._record_schema_validation(result)
        finally:
            self._close_db(db)

    def reject_learning_graph_pending(self, pending_id: int) -> dict:
        db = self._open_db()
        try:
            pending = db.get_learning_graph_pending(int(pending_id))
            if not pending:
                return self._record_schema_validation({
                    "schema": LEARNING_GRAPH_PENDING_SCHEMA,
                    "status": "error",
                    "error": "pending item not found",
                    "pendingId": int(pending_id),
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                })
            db.set_learning_graph_pending_status(int(pending_id), "rejected")
            db.append_learning_graph_event(
                event_type="learning.graph.pending.rejected",
                topic_slug=str(pending.get("topic_slug") or ""),
                payload={"pendingId": int(pending_id), "itemType": pending.get("item_type")},
                actor="learning",
            )
            return self._record_schema_validation({
                "schema": LEARNING_GRAPH_PENDING_SCHEMA,
                "status": "ok",
                "pendingId": int(pending_id),
                "itemType": pending.get("item_type"),
                "topicSlug": pending.get("topic_slug"),
                "rejected": True,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            })
        finally:
            self._close_db(db)

    def generate_learning_path(
        self,
        topic: str,
        approved_only: bool = True,
        writeback: bool = False,
        run_id: str | None = None,
    ) -> dict:
        run_id = str(run_id or f"learn_path_{uuid4().hex[:12]}")
        db = self._open_db()
        try:
            builder = LearningGraphBuilder(db)
            scoped = builder.build_topic_candidates(topic=topic, top_k=12)
            scoped_topic_slug = slugify_topic(str(scoped.get("topicSlug") or topic))
            scoped_node_ids = {node.node_id for node in scoped["nodes"]}
            nodes = [
                node for node in db.list_learning_graph_nodes(limit=5000)
                if str(node.get("node_id")) in scoped_node_ids
            ]
            edges = []
            for row in db.list_learning_graph_edges(status="approved" if approved_only else None, limit=5000):
                prov = row.get("provenance_json") if isinstance(row.get("provenance_json"), dict) else {}
                if slugify_topic(str(prov.get("topicSlug") or "")) != scoped_topic_slug:
                    continue
                edges.append(
                    LearningEdge(
                        edge_id=str(row.get("edge_id")),
                        source_node_id=str(row.get("source_node_id")),
                        edge_type=str(row.get("edge_type")),
                        target_node_id=str(row.get("target_node_id")),
                        confidence=float(row.get("confidence", 0.5)),
                        status=str(row.get("status", "approved")),
                        provenance=prov,
                        evidence=row.get("evidence_json") if isinstance(row.get("evidence_json"), dict) else {},
                    )
                )
            if not approved_only:
                pending_items = db.list_learning_graph_pending(
                    topic_slug=scoped_topic_slug,
                    item_type="all",
                    limit=5000,
                )
                seen_edges = {
                    (edge.source_node_id, edge.edge_type, edge.target_node_id)
                    for edge in edges
                }
                for item in pending_items:
                    item_type = str(item.get("item_type") or "")
                    payload = item.get("payload_json") if isinstance(item.get("payload_json"), dict) else {}
                    payload_prov = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
                    payload_topic_slug = slugify_topic(
                        str(
                            payload.get("topicSlug")
                            or payload_prov.get("topicSlug")
                            or item.get("topic_slug")
                            or ""
                        )
                    )
                    if payload_topic_slug and payload_topic_slug != scoped_topic_slug:
                        continue
                    if item_type == "edge":
                        source_node_id = str(payload.get("sourceNodeId") or "")
                        edge_type = str(payload.get("edgeType") or "")
                        target_node_id = str(payload.get("targetNodeId") or "")
                        if not source_node_id or not edge_type or not target_node_id:
                            continue
                        edge_key = (source_node_id, edge_type, target_node_id)
                        if edge_key in seen_edges:
                            continue
                        edges.append(
                            LearningEdge(
                                edge_id=str(payload.get("edgeId") or f"pending_edge_{item.get('pending_id')}"),
                                source_node_id=source_node_id,
                                edge_type=edge_type,
                                target_node_id=target_node_id,
                                confidence=float(payload.get("confidence", item.get("confidence", 0.5))),
                                status="pending",
                                provenance=payload_prov,
                                evidence=payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {},
                            )
                        )
                        seen_edges.add(edge_key)
            resource_links = []
            for row in db.list_learning_graph_resource_links(status="approved" if approved_only else None, limit=5000):
                prov = row.get("provenance_json") if isinstance(row.get("provenance_json"), dict) else {}
                if slugify_topic(str(prov.get("topicSlug") or "")) != scoped_topic_slug:
                    continue
                resource_links.append(row)
            if not approved_only:
                seen_links = {
                    (
                        str(item.get("concept_node_id") or ""),
                        str(item.get("resource_node_id") or ""),
                        str(item.get("link_type") or ""),
                    )
                    for item in resource_links
                }
                pending_items = db.list_learning_graph_pending(
                    topic_slug=scoped_topic_slug,
                    item_type="resource_link",
                    limit=5000,
                )
                for item in pending_items:
                    payload = item.get("payload_json") if isinstance(item.get("payload_json"), dict) else {}
                    payload_prov = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
                    payload_topic_slug = slugify_topic(
                        str(
                            payload.get("topicSlug")
                            or payload_prov.get("topicSlug")
                            or item.get("topic_slug")
                            or ""
                        )
                    )
                    if payload_topic_slug and payload_topic_slug != scoped_topic_slug:
                        continue
                    link_key = (
                        str(payload.get("conceptNodeId") or ""),
                        str(payload.get("resourceNodeId") or ""),
                        str(payload.get("linkType") or ""),
                    )
                    if not all(link_key) or link_key in seen_links:
                        continue
                    resource_links.append(
                        {
                            "link_id": str(payload.get("linkId") or f"pending_link_{item.get('pending_id')}"),
                            "concept_node_id": link_key[0],
                            "resource_node_id": link_key[1],
                            "link_type": link_key[2],
                            "reading_stage": str(payload.get("readingStage") or "intermediate"),
                            "confidence": float(payload.get("confidence", item.get("confidence", 0.5))),
                            "status": "pending",
                            "provenance_json": payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {},
                        }
                    )
                    seen_links.add(link_key)

            path = LearningPathGenerator().generate_path(
                topic_slug=scoped["topicSlug"],
                nodes=nodes,
                edges=edges,
                resource_links=resource_links,
                approved_only=approved_only,
            )

            if approved_only:
                latest = db.get_latest_learning_graph_path(topic_slug=scoped["topicSlug"], status="approved")
                next_version = int(latest.get("version", 0)) + 1 if latest else 1
                db.upsert_learning_graph_path(
                    path_id=path.path_id,
                    topic_slug=path.topic_slug,
                    status="approved",
                    version=next_version,
                    path_payload=path.to_dict(),
                    score_payload=path.score,
                    provenance=path.provenance,
                )

            writeback_result = {}
            if writeback and self.config.vault_path:
                topic_index = write_topic_learning_path(
                    self.config.vault_path,
                    topic=topic,
                    path_payload=path.to_dict(),
                    **self._obsidian_backend_kwargs(),
                )
                node_by_id = {str(node.get("node_id")): node for node in nodes}
                path_node_set = set(path.nodes)
                outgoing: dict[str, list[str]] = {}
                for edge in edges:
                    if (
                        edge.edge_type in {"prerequisite", "recommended_before"}
                        and edge.source_node_id in path_node_set
                        and edge.target_node_id in path_node_set
                    ):
                        outgoing.setdefault(edge.source_node_id, []).append(edge.target_node_id)
                prerequisite_map = transitive_prerequisite_map(
                    path.nodes,
                    edges,
                    include_recommended=True,
                )

                updated_concepts: list[str] = []
                for node_id in path.nodes:
                    node = node_by_id.get(node_id)
                    if not node or str(node.get("node_type")) not in {"concept", "technique"}:
                        continue
                    payload = {
                        "difficultyLevel": node.get("difficulty_level"),
                        "stage": node.get("stage"),
                        "prerequisites": [
                            node_by_id.get(item, {}).get("canonical_name")
                            for item in prerequisite_map.get(node_id, [])
                            if node_by_id.get(item)
                        ],
                        "studyNext": [node_by_id.get(item, {}).get("canonical_name") for item in outgoing.get(node_id, []) if node_by_id.get(item)],
                        "representativePapers": [],
                    }
                    for link in resource_links:
                        if str(link.get("concept_node_id")) == node_id:
                            resource_id = str(link.get("resource_node_id") or "")
                            if resource_id.startswith("lg_node_paper:"):
                                arxiv_id = resource_id.replace("lg_node_paper:", "", 1)
                                paper = db.get_paper(arxiv_id)
                                if paper and paper.get("title"):
                                    payload["representativePapers"].append(str(paper.get("title")))
                    path_written = write_concept_learning_section(
                        self.config.vault_path,
                        str(node.get("canonical_name")),
                        payload,
                        **self._obsidian_backend_kwargs(),
                    )
                    if path_written:
                        updated_concepts.append(str(path_written))

                updated_papers: list[str] = []
                for link in resource_links:
                    resource_id = str(link.get("resource_node_id") or "")
                    if not resource_id.startswith("lg_node_paper:"):
                        continue
                    arxiv_id = resource_id.replace("lg_node_paper:", "", 1)
                    paper = db.get_paper(arxiv_id)
                    concept_node = node_by_id.get(str(link.get("concept_node_id")))
                    if not paper or not paper.get("title") or not concept_node:
                        continue
                    paper_written = write_paper_learning_context(
                        self.config.vault_path,
                        str(paper.get("title")),
                        {
                            "underConcept": concept_node.get("canonical_name"),
                            "readingStage": link.get("reading_stage"),
                            "prerequisites": [
                                node_by_id.get(item, {}).get("canonical_name")
                                for item in prerequisite_map.get(str(link.get("concept_node_id")), [])
                                if node_by_id.get(item)
                            ],
                            "whyReadNow": f"{concept_node.get('canonical_name')} 학습 단계에서 읽을 대표 자료",
                        },
                        **self._obsidian_backend_kwargs(),
                    )
                    if paper_written:
                        updated_papers.append(str(paper_written))
                writeback_result = {
                    "topicIndex": str(topic_index),
                    "conceptNotes": updated_concepts,
                    "paperNotes": updated_papers,
                }

            result = {
                "schema": LEARNING_PATH_SCHEMA,
                "runId": run_id,
                "topic": topic,
                "topicSlug": scoped["topicSlug"],
                "status": "ok",
                "path": path.to_dict(),
                "writeback": writeback_result,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }
            return self._record_schema_validation(result)
        finally:
            self._close_db(db)

    def write_learning_review_surfaces(self, topic: str, *, limit: int = 20) -> dict:
        run_id = f"learn_review_{uuid4().hex[:12]}"
        db = self._open_db()
        try:
            topic_slug = slugify_topic(topic)
            top_concepts = db.list_top_feature_snapshots(
                topic_slug=topic_slug,
                feature_kind="concept",
                limit=max(1, limit),
            )
            pending_claims = db.list_ontology_pending(
                pending_type="claim",
                topic_slug=topic_slug,
                status="pending",
                limit=max(1, limit),
            )
            merge_proposals = db.list_entity_merge_proposals(
                topic_slug=topic_slug,
                status="pending",
                limit=max(1, limit),
            )
            predicate_validation = db.list_predicate_validation_issues(limit=max(1, limit))
            contradiction_hotspots = [
                item
                for item in top_concepts
                if float(item.get("contradiction_score") or 0.0) > 0.0
            ][: max(1, limit)]
            learning_pending = db.list_learning_graph_pending(
                topic_slug=topic_slug,
                item_type="all",
                limit=max(1, limit),
            )
            latest_path = db.get_latest_learning_graph_path(topic_slug=topic_slug, status="approved") or {}
            path_payload = latest_path.get("path_json") if isinstance(latest_path.get("path_json"), dict) else {}
            reading_queue: dict[str, list[dict]] = {"beginner": [], "intermediate": [], "advanced": []}
            study_next_items: list[dict[str, Any]] = []
            path_node_names: dict[str, str] = {}
            for stage_items in (path_payload.get("stages", {}) or {}).values():
                for stage_item in stage_items or []:
                    node_id = str(stage_item.get("nodeId") or "")
                    canonical_name = str(stage_item.get("canonicalName") or "").strip()
                    if node_id and canonical_name:
                        path_node_names[node_id] = canonical_name
            for stage_name in ("beginner", "intermediate", "advanced"):
                for item in (path_payload.get("stages", {}).get(stage_name) or [])[: max(1, limit)]:
                    reading_queue[stage_name].append(
                        {
                            "concept": item.get("canonicalName"),
                            "prerequisites": [
                                path_node_names.get(str(prereq_id))
                                for prereq_id in (item.get("prerequisiteNodeIds") or [])
                                if path_node_names.get(str(prereq_id))
                            ],
                            "papers": [
                                paper.get("resource_node_id")
                                for paper in (item.get("papers") or [])[:5]
                                if paper.get("resource_node_id")
                            ],
                        }
                    )
            for stage_name in ("beginner", "intermediate", "advanced"):
                for item in (path_payload.get("stages", {}).get(stage_name) or [])[:3]:
                    study_next_items.append(
                        {
                            "stage": stage_name,
                            "concept": item.get("canonicalName"),
                            "difficulty": item.get("difficultyLevel"),
                            "prerequisites": [
                                path_node_names.get(str(prereq_id))
                                for prereq_id in (item.get("prerequisiteNodeIds") or [])
                                if path_node_names.get(str(prereq_id))
                            ],
                        }
                    )

            writeback = {}
            if self.config.vault_path:
                writeback = write_learning_review_notes(
                    self.config.vault_path,
                    topic=topic,
                    review_payload={
                        "claims": pending_claims,
                        "topConcepts": top_concepts,
                        "contradictionHotspots": contradiction_hotspots,
                        "mergeProposals": merge_proposals,
                        "predicateValidation": predicate_validation,
                        "learningPending": learning_pending,
                        "studyNextItems": study_next_items,
                        "readingQueue": reading_queue,
                    },
                    **self._obsidian_backend_kwargs(),
                )
            payload = {
                "schema": LEARNING_REVIEW_SCHEMA,
                "runId": run_id,
                "topic": topic,
                "topicSlug": topic_slug,
                "status": "ok",
                "counts": {
                    "claims": len(pending_claims),
                    "topConcepts": len(top_concepts),
                    "contradictionHotspots": len(contradiction_hotspots),
                    "mergeProposals": len(merge_proposals),
                    "predicateValidation": len(predicate_validation),
                    "learningPending": len(learning_pending),
                    "studyNextItems": len(study_next_items),
                },
                "writeback": writeback,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }
            return self._record_schema_validation(payload)
        finally:
            self._close_db(db)
