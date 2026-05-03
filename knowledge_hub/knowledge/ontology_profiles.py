"""Canonical ontology profile manager with DB runtime overlays."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from knowledge_hub.infrastructure.config import DEFAULT_CONFIG_DIR
from knowledge_hub.knowledge.ai_taxonomy import validate_ai_knowledge_profile_schema

PROFILE_KINDS = {"core", "domain", "personal"}
PROPOSAL_TYPES = {"entity_type", "predicate", "profile_patch"}
DEFAULT_ACTIVE_STACK = {
    "core": "core",
    "domain": "research",
    "personal": "ai_knowledge",
}


def _deep_merge(left: Any, right: Any) -> Any:
    if isinstance(left, dict) and isinstance(right, dict):
        merged = {str(k): deepcopy(v) for k, v in left.items()}
        for key, value in right.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged
    if isinstance(left, list) and isinstance(right, list):
        merged = []
        seen: set[str] = set()
        for item in left + right:
            token = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if token in seen:
                continue
            seen.add(token)
            merged.append(deepcopy(item))
        return merged
    return deepcopy(right)


def _builtin_profile_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "ontology_profiles"


def _user_profile_dir() -> Path:
    return DEFAULT_CONFIG_DIR / "ontology_profiles"


class OntologyProfileManager:
    def __init__(self, sqlite_db, profile_dirs: list[Path] | None = None):
        self.sqlite_db = sqlite_db
        self.store = sqlite_db.ontology_profile_store
        self.profile_dirs = profile_dirs or [_builtin_profile_dir(), _user_profile_dir()]

    def list_profiles(self) -> list[dict[str, Any]]:
        active = {item.get("kind"): item.get("profile_id") for item in self.store.list_active_profiles()}
        profiles: list[dict[str, Any]] = []
        seen: set[str] = set()
        for root in self.profile_dirs:
            if not root.exists():
                continue
            for path in sorted(root.glob("*.yaml")):
                data = self._load_profile_path(path)
                profile_id = str(data.get("profile_id", "")).strip()
                if not profile_id or profile_id in seen:
                    continue
                seen.add(profile_id)
                kind = str(data.get("kind", "")).strip() or "domain"
                profiles.append(
                    {
                        "profile_id": profile_id,
                        "kind": kind,
                        "title": str(data.get("title", profile_id)),
                        "source_path": str(path),
                        "active": active.get(kind) == profile_id or DEFAULT_ACTIVE_STACK.get(kind) == profile_id,
                    }
                )
        return profiles

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        target = self._resolve_profile_path(profile_id)
        if target is None:
            return None
        data = self._load_profile_path(target)
        data["source_path"] = str(target)
        return data

    def import_profile(self, source_path: str, *, profile_id: str | None = None, kind: str = "personal") -> dict[str, Any]:
        target_kind = str(kind or "personal").strip().lower()
        if target_kind not in PROFILE_KINDS:
            target_kind = "personal"
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"profile not found: {source}")
        data = self._load_profile_path(source)
        normalized_id = str(profile_id or data.get("profile_id") or source.stem).strip()
        data["profile_id"] = normalized_id
        data["kind"] = target_kind
        self.validate_profile(data)
        destination_dir = _user_profile_dir()
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{normalized_id}.yaml"
        destination.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        data["source_path"] = str(destination)
        return data

    def export_profile(self, profile_id: str, destination: str, *, compiled: bool = False) -> dict[str, Any]:
        payload = self.compile_active_profile() if compiled else self.get_profile(profile_id)
        if payload is None:
            raise FileNotFoundError(f"profile not found: {profile_id}")
        target = Path(destination).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return {"destination": str(target), "compiled": bool(compiled)}

    def activate_profile(self, kind: str, profile_id: str) -> dict[str, Any]:
        target_kind = str(kind or "").strip().lower()
        if target_kind not in PROFILE_KINDS:
            raise ValueError(f"unsupported profile kind: {kind}")
        profile = self.get_profile(profile_id)
        if profile is None:
            raise FileNotFoundError(f"profile not found: {profile_id}")
        profile_kind = str(profile.get("kind", target_kind)).strip().lower() or target_kind
        if profile_kind != target_kind:
            raise ValueError(f"profile {profile_id} is kind={profile_kind}, not kind={target_kind}")
        self.store.set_active_profile(target_kind, profile_id, str(profile.get("source_path", "")))
        compiled = self.compile_active_profile(refresh_runtime=True)
        return {
            "kind": target_kind,
            "profile_id": profile_id,
            "compiled": compiled,
        }

    def compile_active_profile(self, *, refresh_runtime: bool = False) -> dict[str, Any]:
        states = {item.get("kind"): item for item in self.store.list_active_profiles()}
        selected: list[dict[str, Any]] = []
        for kind in ("core", "domain", "personal"):
            profile_id = str((states.get(kind) or {}).get("profile_id") or DEFAULT_ACTIVE_STACK.get(kind) or "").strip()
            if not profile_id:
                continue
            profile = self.get_profile(profile_id)
            if profile:
                selected.append(profile)

        compiled: dict[str, Any] = {
            "schema": "knowledge-hub.ontology.profile.compiled.v1",
            "profiles": [{"profile_id": item.get("profile_id"), "kind": item.get("kind")} for item in selected],
            "entity_types": [],
            "predicates": [],
            "aliases": {},
            "extraction_hints": {},
            "cluster_label_hints": {},
            "retrieval_facets": {},
            "knowledge_classification": {},
        }
        predicates_by_id: dict[str, dict[str, Any]] = {}
        entity_types_seen: set[str] = set()

        for profile in selected:
            for entity_type in profile.get("entity_types", []) if isinstance(profile.get("entity_types"), list) else []:
                token = str(entity_type).strip()
                if token and token not in entity_types_seen:
                    entity_types_seen.add(token)
                    compiled["entity_types"].append(token)
            for predicate in profile.get("predicates", []) if isinstance(profile.get("predicates"), list) else []:
                if not isinstance(predicate, dict):
                    continue
                predicate_id = str(predicate.get("id", "")).strip()
                if not predicate_id:
                    continue
                predicates_by_id[predicate_id] = _deep_merge(predicates_by_id.get(predicate_id, {}), predicate)
            compiled["aliases"] = _deep_merge(compiled["aliases"], profile.get("aliases", {}) if isinstance(profile.get("aliases"), dict) else {})
            compiled["extraction_hints"] = _deep_merge(
                compiled["extraction_hints"],
                profile.get("extraction_hints", {}) if isinstance(profile.get("extraction_hints"), dict) else {},
            )
            compiled["cluster_label_hints"] = _deep_merge(
                compiled["cluster_label_hints"],
                profile.get("cluster_label_hints", {}) if isinstance(profile.get("cluster_label_hints"), dict) else {},
            )
            compiled["retrieval_facets"] = _deep_merge(
                compiled["retrieval_facets"],
                profile.get("retrieval_facets", {}) if isinstance(profile.get("retrieval_facets"), dict) else {},
            )
            compiled["knowledge_classification"] = _deep_merge(
                compiled["knowledge_classification"],
                profile.get("knowledge_classification", {}) if isinstance(profile.get("knowledge_classification"), dict) else {},
            )

        overlays = self.store.list_profile_overlays(status="approved", limit=1_000)
        for overlay in overlays:
            payload = overlay.get("payload") if isinstance(overlay.get("payload"), dict) else {}
            overlay_type = str(overlay.get("overlay_type", "")).strip()
            if overlay_type == "entity_type":
                token = str(payload.get("id", "")).strip()
                if token and token not in entity_types_seen:
                    entity_types_seen.add(token)
                    compiled["entity_types"].append(token)
            elif overlay_type == "predicate":
                predicate_id = str(payload.get("id", "")).strip()
                if predicate_id:
                    predicates_by_id[predicate_id] = _deep_merge(predicates_by_id.get(predicate_id, {}), payload)
            elif overlay_type == "profile_patch":
                compiled = _deep_merge(compiled, payload)

        compiled["predicates"] = [predicates_by_id[key] for key in sorted(predicates_by_id)]
        if refresh_runtime:
            self.store.set_runtime_json("compiled_active_profile", compiled)
        return compiled

    def validate_profile(self, payload: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        if not isinstance(payload, dict):
            return ["profile must be an object"]
        profile_id = str(payload.get("profile_id", "")).strip()
        kind = str(payload.get("kind", "")).strip().lower()
        if not profile_id:
            errors.append("profile_id is required")
        if kind not in PROFILE_KINDS:
            errors.append("kind must be one of core/domain/personal")
        if payload.get("entity_types") is not None and not isinstance(payload.get("entity_types"), list):
            errors.append("entity_types must be a list")
        predicates = payload.get("predicates")
        if predicates is not None and not isinstance(predicates, list):
            errors.append("predicates must be a list")
        seen_predicates: set[str] = set()
        for item in predicates or []:
            if not isinstance(item, dict):
                errors.append("predicate items must be objects")
                continue
            predicate_id = str(item.get("id", "")).strip()
            if not predicate_id:
                errors.append("predicate id is required")
                continue
            if predicate_id in seen_predicates:
                errors.append(f"duplicate predicate id: {predicate_id}")
            seen_predicates.add(predicate_id)
        for key in ("aliases", "extraction_hints", "cluster_label_hints", "retrieval_facets"):
            value = payload.get(key)
            if value is not None and not isinstance(value, dict):
                errors.append(f"{key} must be an object")
        classification = payload.get("knowledge_classification")
        if classification is not None:
            errors.extend(validate_ai_knowledge_profile_schema(classification))
        return errors

    def validate_proposal(self, proposal_type: str, payload: dict[str, Any]) -> list[str]:
        token = str(proposal_type or "").strip()
        errors: list[str] = []
        if token not in PROPOSAL_TYPES:
            return [f"unsupported proposal_type: {proposal_type}"]

        compiled = self.compile_active_profile()
        entity_types = {str(item).strip() for item in compiled.get("entity_types", []) if str(item).strip()}
        predicates = {
            str(item.get("id", "")).strip(): item
            for item in compiled.get("predicates", [])
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }

        if token == "entity_type":
            entity_id = str(payload.get("id", "")).strip()
            if not entity_id:
                errors.append("entity_type proposal requires id")
            if entity_id and entity_id in entity_types:
                errors.append(f"entity_type already exists: {entity_id}")
        elif token == "predicate":
            predicate_id = str(payload.get("id", "")).strip()
            if not predicate_id:
                errors.append("predicate proposal requires id")
            if predicate_id and predicate_id in predicates:
                errors.append(f"predicate already exists: {predicate_id}")
            parent = str(payload.get("parent", "")).strip()
            if parent and parent not in predicates:
                errors.append(f"parent predicate not found: {parent}")
            domain = str(payload.get("domain", "")).strip()
            range_type = str(payload.get("range", "")).strip()
            if domain and domain not in entity_types:
                errors.append(f"domain entity type not found: {domain}")
            if range_type and range_type not in entity_types:
                errors.append(f"range entity type not found: {range_type}")
        elif token == "profile_patch":
            if not isinstance(payload, dict) or not payload:
                errors.append("profile_patch proposal requires non-empty payload")
        return errors

    def submit_proposal(
        self,
        *,
        proposal_type: str,
        target_profile: str,
        payload: dict[str, Any],
        source: str = "user",
    ) -> dict[str, Any]:
        errors = self.validate_proposal(proposal_type, payload)
        proposal_id = self.store.add_profile_proposal(
            proposal_type=proposal_type,
            target_profile=target_profile,
            payload=payload,
            source=source,
            status="pending",
            reason={"validationErrors": errors},
        )
        return self.store.get_profile_proposal(proposal_id) or {"proposal_id": proposal_id}

    def list_proposals(
        self,
        *,
        status: str | None = None,
        proposal_type: str | None = None,
        target_profile: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.store.list_profile_proposals(
            status=status,
            proposal_type=proposal_type,
            target_profile=target_profile,
            limit=limit,
        )

    def apply_proposal(self, proposal_id: int) -> dict[str, Any]:
        proposal = self.store.get_profile_proposal(proposal_id)
        if not proposal:
            raise ValueError(f"proposal not found: {proposal_id}")
        if str(proposal.get("status", "")) != "pending":
            raise ValueError(f"proposal is not pending: {proposal_id}")
        payload = proposal.get("payload") if isinstance(proposal.get("payload"), dict) else {}
        errors = self.validate_proposal(str(proposal.get("proposal_type", "")), payload)
        if errors:
            self.store.update_profile_proposal_status(proposal_id, "rejected", {"validationErrors": errors})
            raise ValueError("; ".join(errors))

        proposal_type = str(proposal.get("proposal_type", "")).strip()
        target_profile = str(proposal.get("target_profile", "")).strip()
        self.store.add_profile_overlay(proposal_type, target_profile, payload, source="proposal_apply", status="approved")
        if proposal_type == "predicate":
            self.sqlite_db.upsert_predicate(
                predicate_id=str(payload.get("id", "")).strip(),
                parent_predicate_id=str(payload.get("parent", "")).strip() or None,
                status="approved_ext",
                description=str(payload.get("description", "")),
                source="ontology_profile_proposal",
                domain_source_type=str(payload.get("domain", "")),
                range_target_type=str(payload.get("range", "")),
            )
        self.store.update_profile_proposal_status(proposal_id, "approved", {"applied": True})
        self.compile_active_profile(refresh_runtime=True)
        return self.store.get_profile_proposal(proposal_id) or {"proposal_id": proposal_id}

    def reject_proposal(self, proposal_id: int, reason: dict[str, Any] | None = None) -> dict[str, Any]:
        ok = self.store.update_profile_proposal_status(proposal_id, "rejected", reason or {"rejected": True})
        if not ok:
            raise ValueError(f"proposal not found: {proposal_id}")
        return self.store.get_profile_proposal(proposal_id) or {"proposal_id": proposal_id}

    def _resolve_profile_path(self, profile_id: str) -> Path | None:
        token = str(profile_id or "").strip()
        if not token:
            return None
        for root in self.profile_dirs:
            candidate = root / f"{token}.yaml"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _load_profile_path(path: Path) -> dict[str, Any]:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"invalid ontology profile: {path}")
        return data
