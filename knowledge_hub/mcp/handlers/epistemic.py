from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

from knowledge_hub.knowledge.ontology_profiles import OntologyProfileManager


async def handle_tool(name: str, arguments: dict[str, Any], ctx: dict[str, Any]):
    emit = ctx["emit"]
    sqlite_db = ctx["sqlite_db"]
    to_int = ctx["to_int"]
    to_float = ctx["to_float"]

    status_ok = ctx["MCP_TOOL_STATUS_OK"]
    status_failed = ctx["MCP_TOOL_STATUS_FAILED"]

    if name == "ontology_profile_list":
        manager = OntologyProfileManager(sqlite_db)
        items = manager.list_profiles()
        return emit(status_ok, {"count": len(items), "items": items})

    if name == "ontology_profile_show":
        manager = OntologyProfileManager(sqlite_db)
        compiled = bool(arguments.get("compiled", False))
        profile_id = str(arguments.get("profile_id", "")).strip()
        if compiled:
            payload = manager.compile_active_profile()
        else:
            if not profile_id:
                return emit(status_failed, {"error": "profile_id가 필요합니다."}, status_message="profile_id required")
            payload = manager.get_profile(profile_id)
        if not payload:
            return emit(status_failed, {"error": "profile not found"}, status_message="profile not found")
        return emit(status_ok, payload)

    if name == "ontology_profile_activate":
        manager = OntologyProfileManager(sqlite_db)
        profile_id = str(arguments.get("profile_id", "")).strip()
        kind = str(arguments.get("kind", "")).strip().lower()
        if not profile_id or not kind:
            return emit(status_failed, {"error": "profile_id와 kind가 필요합니다."}, status_message="profile_id/kind required")
        try:
            payload = manager.activate_profile(kind=kind, profile_id=profile_id)
        except Exception as error:
            return emit(status_failed, {"error": str(error)}, status_message="activate failed")
        return emit(status_ok, payload)

    if name == "ontology_profile_import":
        manager = OntologyProfileManager(sqlite_db)
        source_path = str(arguments.get("source_path", "")).strip()
        if not source_path:
            return emit(status_failed, {"error": "source_path가 필요합니다."}, status_message="source_path required")
        try:
            payload = manager.import_profile(
                source_path,
                profile_id=str(arguments.get("profile_id", "")).strip() or None,
                kind=str(arguments.get("kind", "personal") or "personal"),
            )
        except Exception as error:
            return emit(status_failed, {"error": str(error)}, status_message="import failed")
        return emit(status_ok, payload)

    if name == "ontology_profile_export":
        manager = OntologyProfileManager(sqlite_db)
        profile_id = str(arguments.get("profile_id", "")).strip()
        destination = str(arguments.get("destination", "")).strip()
        if not profile_id or not destination:
            return emit(status_failed, {"error": "profile_id와 destination이 필요합니다."}, status_message="profile_id/destination required")
        try:
            payload = manager.export_profile(profile_id, destination, compiled=bool(arguments.get("compiled", False)))
        except Exception as error:
            return emit(status_failed, {"error": str(error)}, status_message="export failed")
        return emit(status_ok, payload)

    if name == "ontology_proposal_submit":
        manager = OntologyProfileManager(sqlite_db)
        proposal_type = str(arguments.get("proposal_type", "")).strip()
        payload = arguments.get("payload")
        if not proposal_type or not isinstance(payload, dict):
            return emit(status_failed, {"error": "proposal_type과 payload(object)가 필요합니다."}, status_message="invalid proposal")
        item = manager.submit_proposal(
            proposal_type=proposal_type,
            target_profile=str(arguments.get("target_profile", "personal") or "personal"),
            payload=payload,
            source=str(arguments.get("source", "user") or "user"),
        )
        return emit(status_ok, {"item": item})

    if name == "ontology_proposal_list":
        manager = OntologyProfileManager(sqlite_db)
        items = manager.list_proposals(
            status=str(arguments.get("status", "")).strip() or None,
            proposal_type=str(arguments.get("proposal_type", "")).strip() or None,
            target_profile=str(arguments.get("target_profile", "")).strip() or None,
            limit=to_int(arguments.get("limit"), 100, minimum=1, maximum=500),
        )
        return emit(status_ok, {"count": len(items), "items": items})

    if name == "ontology_proposal_apply":
        manager = OntologyProfileManager(sqlite_db)
        proposal_id = to_int(arguments.get("proposal_id"), 0, minimum=1, maximum=1_000_000)
        try:
            item = manager.apply_proposal(proposal_id)
        except Exception as error:
            return emit(status_failed, {"error": str(error)}, status_message="proposal apply failed")
        return emit(status_ok, {"item": item})

    if name == "ontology_proposal_reject":
        manager = OntologyProfileManager(sqlite_db)
        proposal_id = to_int(arguments.get("proposal_id"), 0, minimum=1, maximum=1_000_000)
        try:
            item = manager.reject_proposal(proposal_id, {"reason": str(arguments.get("reason", "") or "")})
        except Exception as error:
            return emit(status_failed, {"error": str(error)}, status_message="proposal reject failed")
        return emit(status_ok, {"item": item})

    if name == "belief_list":
        items = sqlite_db.list_beliefs(
            status=str(arguments.get("status", "")).strip() or None,
            scope=str(arguments.get("scope", "")).strip() or None,
            limit=to_int(arguments.get("limit"), 100, minimum=1, maximum=500),
        )
        return emit(status_ok, {"count": len(items), "items": items})

    if name == "belief_show":
        belief_id = str(arguments.get("belief_id", "")).strip()
        if not belief_id:
            return emit(status_failed, {"error": "belief_id가 필요합니다."}, status_message="belief_id required")
        item = sqlite_db.get_belief(belief_id)
        if not item:
            return emit(status_failed, {"error": "belief not found"}, status_message="belief not found")
        return emit(status_ok, item)

    if name == "belief_upsert":
        statement = str(arguments.get("statement", "")).strip()
        if not statement:
            return emit(status_failed, {"error": "statement가 필요합니다."}, status_message="statement required")
        belief_id = str(arguments.get("belief_id", "")).strip() or f"belief_{uuid4().hex[:12]}"
        sqlite_db.upsert_belief(
            belief_id=belief_id,
            statement=statement,
            scope=str(arguments.get("scope", "global") or "global"),
            status=str(arguments.get("status", "proposed") or "proposed"),
            confidence=to_float(arguments.get("confidence"), 0.5, minimum=0.0, maximum=1.0),
            derived_from_claim_ids=[str(item) for item in arguments.get("derived_from_claim_ids", []) or []],
            support_ids=[str(item) for item in arguments.get("support_ids", []) or []],
            contradiction_ids=[str(item) for item in arguments.get("contradiction_ids", []) or []],
            last_validated_at=str(arguments.get("last_validated_at", "")).strip() or None,
            review_due_at=str(arguments.get("review_due_at", "")).strip() or None,
        )
        return emit(status_ok, {"item": sqlite_db.get_belief(belief_id)})

    if name == "belief_review":
        belief_id = str(arguments.get("belief_id", "")).strip()
        status = str(arguments.get("status", "")).strip()
        if not belief_id or not status:
            return emit(status_failed, {"error": "belief_id와 status가 필요합니다."}, status_message="belief_id/status required")
        item = sqlite_db.review_belief(
            belief_id,
            status=status,
            last_validated_at=str(arguments.get("last_validated_at", "")).strip() or datetime.now(timezone.utc).isoformat(),
            review_due_at=str(arguments.get("review_due_at", "")).strip() or None,
        )
        if not item:
            return emit(status_failed, {"error": "belief not found"}, status_message="belief not found")
        return emit(status_ok, {"item": item})

    if name == "decision_create":
        title = str(arguments.get("title", "")).strip()
        if not title:
            return emit(status_failed, {"error": "title이 필요합니다."}, status_message="title required")
        decision_id = str(arguments.get("decision_id", "")).strip() or f"decision_{uuid4().hex[:12]}"
        sqlite_db.upsert_decision(
            decision_id=decision_id,
            title=title,
            summary=str(arguments.get("summary", "") or ""),
            related_belief_ids=[str(item) for item in arguments.get("related_belief_ids", []) or []],
            chosen_option=str(arguments.get("chosen_option", "") or ""),
            status=str(arguments.get("status", "open") or "open"),
            review_due_at=str(arguments.get("review_due_at", "")).strip() or None,
        )
        return emit(status_ok, {"item": sqlite_db.get_decision(decision_id)})

    if name == "decision_list":
        items = sqlite_db.list_decisions(
            status=str(arguments.get("status", "")).strip() or None,
            limit=to_int(arguments.get("limit"), 100, minimum=1, maximum=500),
        )
        return emit(status_ok, {"count": len(items), "items": items})

    if name == "decision_review":
        decision_id = str(arguments.get("decision_id", "")).strip()
        status = str(arguments.get("status", "")).strip()
        if not decision_id or not status:
            return emit(status_failed, {"error": "decision_id와 status가 필요합니다."}, status_message="decision_id/status required")
        item = sqlite_db.review_decision(
            decision_id,
            status=status,
            review_due_at=str(arguments.get("review_due_at", "")).strip() or None,
        )
        if not item:
            return emit(status_failed, {"error": "decision not found"}, status_message="decision not found")
        return emit(status_ok, {"item": item})

    if name == "outcome_record":
        decision_id = str(arguments.get("decision_id", "")).strip()
        summary = str(arguments.get("summary", "")).strip()
        if not decision_id or not summary:
            return emit(status_failed, {"error": "decision_id와 summary가 필요합니다."}, status_message="decision_id/summary required")
        outcome_id = str(arguments.get("outcome_id", "")).strip() or f"outcome_{uuid4().hex[:12]}"
        sqlite_db.record_outcome(
            outcome_id=outcome_id,
            decision_id=decision_id,
            status=str(arguments.get("status", "observed") or "observed"),
            summary=summary,
            recorded_at=str(arguments.get("recorded_at", "")).strip() or datetime.now(timezone.utc).isoformat(),
        )
        return emit(status_ok, {"item": sqlite_db.get_outcome(outcome_id)})

    if name == "outcome_show":
        decision_id = str(arguments.get("decision_id", "")).strip()
        if not decision_id:
            return emit(status_failed, {"error": "decision_id가 필요합니다."}, status_message="decision_id required")
        items = sqlite_db.list_outcomes(
            decision_id=decision_id,
            limit=to_int(arguments.get("limit"), 50, minimum=1, maximum=500),
        )
        return emit(status_ok, {"count": len(items), "items": items})

    return None
