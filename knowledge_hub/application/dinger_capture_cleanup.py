"""Cleanup policy helpers for Dinger capture runtime artifacts.

This helper only deletes runtime-side artifacts under the capture runtime
directory. Queue packets, Dinger pages, OS items, and any other canonical
surfaces are never mutated here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_hub.application.dinger_capture_processor import (
    _claim_is_stale,
    _read_claim_payload,
    _resolve_claim_stale_after_sec,
    resolve_capture_queue_dir,
    resolve_capture_runtime_dir,
)
from knowledge_hub.application.dinger_os_bridge import is_capture_linked_to_os

_CAPTURE_RUNTIME_SUFFIXES = (
    ("state", ".state.json"),
    ("normalized", ".normalized.json"),
    ("filedResult", ".file-result.json"),
    ("osResult", ".os-capture-result.json"),
)
_CLAIM_SUFFIX = ".claim.json"
_CLEANUP_RESULT_SCHEMA_ID = "knowledge-hub.dinger.capture-cleanup.result.v1"


def _now_iso(now: datetime | None = None) -> str:
    resolved = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    return resolved.isoformat()


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        import json

        data = json.loads(payload)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _runtime_paths_for_capture(runtime_dir: Path, capture_id: str) -> dict[str, Path]:
    return {label: runtime_dir / f"{capture_id}{suffix}" for label, suffix in _CAPTURE_RUNTIME_SUFFIXES}


def _runtime_capture_ids(runtime_dir: Path) -> list[str]:
    capture_ids: set[str] = set()
    if not runtime_dir.exists():
        return []
    for _label, suffix in _CAPTURE_RUNTIME_SUFFIXES:
        for candidate in runtime_dir.glob(f"*{suffix}"):
            if not candidate.is_file():
                continue
            capture_id = candidate.name[: -len(suffix)].strip()
            if capture_id:
                capture_ids.add(capture_id)
    return sorted(capture_ids)


def _claim_paths(runtime_dir: Path) -> list[Path]:
    if not runtime_dir.exists():
        return []
    return sorted(path for path in runtime_dir.glob(f"*{_CLAIM_SUFFIX}") if path.is_file())


def _existing_paths(paths: dict[str, Path]) -> list[Path]:
    return [path for path in paths.values() if path.exists()]


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _packet_path_for_capture(
    *,
    capture_id: str,
    queue_dir: Path,
    state: dict[str, Any],
    normalized: dict[str, Any],
    filed_result: dict[str, Any],
    os_result: dict[str, Any],
) -> Path:
    trace = dict(os_result.get("captureTrace") or {}) if isinstance(os_result.get("captureTrace"), dict) else {}
    packet_path_text = _first_text(
        state.get("packetPath"),
        normalized.get("packetPath"),
        filed_result.get("packetPath"),
        trace.get("packetPath"),
    )
    if packet_path_text:
        return Path(packet_path_text).expanduser().resolve()
    return (queue_dir / f"{capture_id}.json").resolve()


def _packet_snapshot(normalized: dict[str, Any], filed_result: dict[str, Any], os_result: dict[str, Any]) -> dict[str, Any]:
    trace = dict(os_result.get("captureTrace") or {}) if isinstance(os_result.get("captureTrace"), dict) else {}
    normalized_source_refs = list(normalized.get("sourceRefs") or [])
    filed_source_refs = list(filed_result.get("sourceRefs") or [])
    snapshot_candidates = [
        (
            "normalized",
            normalized,
            bool(
                _first_text(normalized.get("captureUrl"), normalized.get("title"))
                or normalized_source_refs
            ),
        ),
        (
            "filed_result",
            filed_result,
            bool(
                _first_text(
                    filed_result.get("captureUrl"),
                    filed_result.get("title"),
                    filed_result.get("relativePath"),
                )
                or filed_source_refs
            ),
        ),
        (
            "os_capture_trace",
            trace,
            bool(
                _first_text(
                    trace.get("captureUrl"),
                    trace.get("title"),
                    trace.get("relativePath"),
                    trace.get("filingOutputPointer"),
                )
            ),
        ),
    ]
    for source, payload, is_available in snapshot_candidates:
        if not is_available:
            continue
        return {
            "present": True,
            "source": source,
            "title": _first_text(payload.get("title")),
            "captureUrl": _first_text(payload.get("captureUrl")),
            "relativePath": _first_text(payload.get("relativePath"), payload.get("filingOutputPointer")),
            "sourceRefCount": len(list(payload.get("sourceRefs") or [])),
        }
    return {
        "present": False,
        "source": "",
        "title": "",
        "captureUrl": "",
        "relativePath": "",
        "sourceRefCount": 0,
    }


def _canonical_refs(
    *,
    state: dict[str, Any],
    filed_result: dict[str, Any],
    os_result: dict[str, Any],
) -> dict[str, Any]:
    item = dict(os_result.get("item") or {}) if isinstance(os_result.get("item"), dict) else {}
    dinger_relative_path = _first_text(
        filed_result.get("relativePath"),
        filed_result.get("filePath"),
        ((state.get("steps") or {}).get("filed") or {}).get("relativePath"),
    )
    os_item_id = _first_text(
        item.get("id"),
        ((state.get("steps") or {}).get("linked_to_os") or {}).get("itemId"),
    )
    return {
        "dingerProjectionPresent": bool(dinger_relative_path),
        "dingerProjectionRef": dinger_relative_path,
        "osItemPresent": bool(os_item_id) or is_capture_linked_to_os(os_result),
        "osItemId": os_item_id,
    }


def _capture_cleanup_entry(*, config, queue_dir: Path, runtime_dir: Path, capture_id: str) -> dict[str, Any]:
    runtime_paths = _runtime_paths_for_capture(runtime_dir, capture_id)
    existing_runtime_paths = _existing_paths(runtime_paths)
    state = _load_optional_json(runtime_paths["state"])
    normalized = _load_optional_json(runtime_paths["normalized"])
    filed_result = _load_optional_json(runtime_paths["filedResult"])
    os_result = _load_optional_json(runtime_paths["osResult"])
    packet_path = _packet_path_for_capture(
        capture_id=capture_id,
        queue_dir=queue_dir,
        state=state,
        normalized=normalized,
        filed_result=filed_result,
        os_result=os_result,
    )
    packet_present = packet_path.exists()
    parsed_artifact_count = sum(1 for payload in (state, normalized, filed_result, os_result) if payload)
    snapshot = _packet_snapshot(normalized, filed_result, os_result)
    canonical_refs = _canonical_refs(state=state, filed_result=filed_result, os_result=os_result)
    if packet_present:
        return {
            "entryType": "capture_runtime",
            "captureId": capture_id,
            "cleanupKind": "queue_packet_present_capture",
            "disposition": "keep",
            "action": "skipped",
            "packetPath": str(packet_path),
            "packetPresent": True,
            "packetSnapshotPresent": snapshot["present"],
            "recoverability": "protected",
            "runtimeArtifactPaths": [str(path) for path in existing_runtime_paths],
            "canonicalRefs": canonical_refs,
            "recommendedAction": "leave_runtime_artifacts_attached_to_live_queue_packet",
            "reasonCodes": ["queue_packet_present"],
        }
    if snapshot["present"]:
        return {
            "entryType": "capture_runtime",
            "captureId": capture_id,
            "cleanupKind": "recoverable_orphan",
            "disposition": "keep",
            "action": "skipped",
            "packetPath": str(packet_path),
            "packetPresent": False,
            "packetSnapshotPresent": True,
            "packetSnapshot": snapshot,
            "recoverability": "recoverable",
            "runtimeArtifactPaths": [str(path) for path in existing_runtime_paths],
            "canonicalRefs": canonical_refs,
            "recommendedAction": "requeue_capture_from_runtime_snapshot_before_cleanup",
            "reasonCodes": ["missing_queue_packet", "recoverable_from_runtime_snapshot"],
        }
    cleanup_kind = "orphan_runtime_artifact" if parsed_artifact_count > 0 else "incomplete_runtime_junk"
    return {
        "entryType": "capture_runtime",
        "captureId": capture_id,
        "cleanupKind": cleanup_kind,
        "disposition": "delete",
        "action": "planned_delete",
        "packetPath": str(packet_path),
        "packetPresent": False,
        "packetSnapshotPresent": False,
        "recoverability": "unrecoverable",
        "runtimeArtifactPaths": [str(path) for path in existing_runtime_paths],
        "canonicalRefs": canonical_refs,
        "recommendedAction": "cleanup_runtime_artifacts",
        "reasonCodes": ["missing_queue_packet", cleanup_kind],
    }


def _claim_cleanup_entry(*, config, queue_dir: Path, claim_path: Path) -> dict[str, Any]:
    claim_payload = _read_claim_payload(claim_path)
    capture_id = _first_text(claim_payload.get("captureId"), claim_path.stem[: -len(".claim")] if claim_path.name.endswith(_CLAIM_SUFFIX) else claim_path.stem)
    if not claim_payload:
        packet_path = (queue_dir / f"{capture_id}.json").resolve()
        return {
            "entryType": "claim_file",
            "captureId": capture_id,
            "cleanupKind": "incomplete_runtime_junk",
            "disposition": "delete",
            "action": "planned_delete",
            "claimPath": str(claim_path),
            "packetPath": str(packet_path),
            "packetPresent": packet_path.exists(),
            "stale": False,
            "staleReason": "claim_payload_invalid",
            "recommendedAction": "remove_invalid_claim_file",
            "reasonCodes": ["invalid_claim_payload"],
        }
    stale_after_sec = _resolve_claim_stale_after_sec(config)
    stale, stale_reason = _claim_is_stale(claim_payload, stale_after_sec=stale_after_sec)
    packet_path = Path(
        _first_text(claim_payload.get("packetPath"), queue_dir / f"{capture_id}.json")
    ).expanduser().resolve()
    if stale:
        return {
            "entryType": "claim_file",
            "captureId": capture_id,
            "cleanupKind": "stale_claim_file",
            "disposition": "delete",
            "action": "planned_delete",
            "claimPath": str(claim_path),
            "packetPath": str(packet_path),
            "packetPresent": packet_path.exists(),
            "stale": True,
            "staleReason": stale_reason,
            "recommendedAction": "remove_stale_claim_file",
            "reasonCodes": ["stale_claim_file", stale_reason],
        }
    return {
        "entryType": "claim_file",
        "captureId": capture_id,
        "cleanupKind": "active_claim_file",
        "disposition": "keep",
        "action": "skipped",
        "claimPath": str(claim_path),
        "packetPath": str(packet_path),
        "packetPresent": packet_path.exists(),
        "stale": False,
        "staleReason": stale_reason,
        "recommendedAction": "leave_claim_file",
        "reasonCodes": ["claim_not_stale", stale_reason],
    }


def _delete_paths(paths: list[Path]) -> tuple[list[str], list[str]]:
    deleted: list[str] = []
    errors: list[str] = []
    for path in paths:
        try:
            path.unlink()
            deleted.append(str(path))
        except FileNotFoundError:
            deleted.append(str(path))
        except OSError as error:
            errors.append(f"{path}: {error}")
    return deleted, errors


def cleanup_dinger_capture_runtime(
    *,
    config,
    dry_run: bool = True,
    confirm: bool = False,
    queue_dir: Path | None = None,
    runtime_dir: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not dry_run and not confirm:
        raise ValueError("explicit confirmation is required when dry_run is False")
    resolved_queue_dir = Path(queue_dir or resolve_capture_queue_dir(config)).expanduser().resolve()
    resolved_runtime_dir = Path(runtime_dir or resolve_capture_runtime_dir(config)).expanduser().resolve()
    capture_ids = _runtime_capture_ids(resolved_runtime_dir)
    claim_paths = _claim_paths(resolved_runtime_dir)
    delete_entries: list[dict[str, Any]] = []
    keep_entries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for capture_id in capture_ids:
        entry = _capture_cleanup_entry(
            config=config,
            queue_dir=resolved_queue_dir,
            runtime_dir=resolved_runtime_dir,
            capture_id=capture_id,
        )
        if entry["disposition"] == "delete":
            target_paths = [Path(path) for path in list(entry.get("runtimeArtifactPaths") or [])]
            entry["targetPathCount"] = len(target_paths)
            if dry_run:
                entry["deletedPaths"] = []
            else:
                deleted_paths, path_errors = _delete_paths(target_paths)
                entry["deletedPaths"] = deleted_paths
                if path_errors:
                    entry["action"] = "delete_failed"
                    entry["errors"] = path_errors
                    errors.extend(
                        {
                            "entryType": "capture_runtime",
                            "captureId": capture_id,
                            "message": message,
                        }
                        for message in path_errors
                    )
                else:
                    entry["action"] = "deleted"
            delete_entries.append(entry)
        else:
            keep_entries.append(entry)

    for claim_path in claim_paths:
        entry = _claim_cleanup_entry(config=config, queue_dir=resolved_queue_dir, claim_path=claim_path)
        if entry["disposition"] == "delete":
            if dry_run:
                entry["deletedPaths"] = []
            else:
                deleted_paths, path_errors = _delete_paths([claim_path])
                entry["deletedPaths"] = deleted_paths
                if path_errors:
                    entry["action"] = "delete_failed"
                    entry["errors"] = path_errors
                    errors.extend(
                        {
                            "entryType": "claim_file",
                            "captureId": str(entry.get("captureId") or ""),
                            "message": message,
                        }
                        for message in path_errors
                    )
                else:
                    entry["action"] = "deleted"
            delete_entries.append(entry)
        else:
            keep_entries.append(entry)

    delete_eligible_paths = 0
    for entry in delete_entries:
        if entry["entryType"] == "capture_runtime":
            delete_eligible_paths += len(list(entry.get("runtimeArtifactPaths") or []))
        else:
            delete_eligible_paths += 1

    deleted_entries = sum(1 for entry in delete_entries if entry.get("action") == "deleted")
    deleted_paths = sum(len(list(entry.get("deletedPaths") or [])) for entry in delete_entries)
    failed_entries = sum(1 for entry in delete_entries if entry.get("action") == "delete_failed")
    status = "ok" if not errors else "failed"
    return {
        "schema": _CLEANUP_RESULT_SCHEMA_ID,
        "status": status,
        "dryRun": bool(dry_run),
        "confirmed": bool(confirm),
        "queueDir": str(resolved_queue_dir),
        "runtimeDir": str(resolved_runtime_dir),
        "policy": {
            "dryRunDefault": True,
            "requiresExplicitConfirmation": True,
            "deletableKinds": [
                "orphan_runtime_artifact",
                "stale_claim_file",
                "incomplete_runtime_junk",
            ],
            "protectedKinds": [
                "queue_packet_present_capture",
                "recoverable_orphan",
                "active_claim_file",
            ],
            "neverDeleteCanonicalSurfaces": [
                "queue_packet",
                "dinger_projection_page",
                "os_item",
                "canonical_store_content",
            ],
        },
        "counts": {
            "scannedCaptureGroups": len(capture_ids),
            "scannedClaimFiles": len(claim_paths),
            "deleteEligibleEntries": len(delete_entries),
            "deleteEligiblePaths": delete_eligible_paths,
            "deletedEntries": deleted_entries,
            "deletedPaths": deleted_paths,
            "keptEntries": len(keep_entries),
            "failedEntries": failed_entries,
        },
        "delete": delete_entries,
        "keep": keep_entries,
        "errors": errors,
        "createdAt": _now_iso(now),
    }


__all__ = ["cleanup_dinger_capture_runtime"]
