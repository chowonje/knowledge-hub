"""Snapshot-backed recovery helpers for orphaned Dinger capture packets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.application.dinger_capture_processor import (
    resolve_capture_packet_snapshot_path,
    resolve_capture_queue_dir,
    resolve_capture_runtime_dir,
)

CAPTURE_PACKET_SCHEMA = "knowledge-hub.dinger.capture.result.v1"
RECOVERY_KIND_EXACT_PACKET_SNAPSHOT = "exact_packet_snapshot"
RECOVERY_REASON_SNAPSHOT_AVAILABLE = "packet_snapshot_available"
RECOVERY_REASON_LEGACY_ORPHAN_MISSING_SNAPSHOT = "legacy_orphan_missing_packet_snapshot"
RECOVERY_REASON_PACKET_SNAPSHOT_MISSING = "packet_snapshot_missing"
RECOVERY_REASON_PACKET_SNAPSHOT_INVALID = "packet_snapshot_invalid"
RECOVERY_REASON_PACKET_SNAPSHOT_CAPTURE_ID_MISMATCH = "packet_snapshot_capture_id_mismatch"


def _read_optional_json_object(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # pragma: no cover - defensive runtime helper
        return {}, str(error) or "invalid_json"
    if not isinstance(payload, dict):
        return {}, "expected_json_object"
    return payload, ""


def _runtime_artifact_kinds(*, runtime_dir: Path, capture_id: str) -> list[str]:
    artifact_map = (
        ("runtime_state", runtime_dir / f"{capture_id}.state.json"),
        ("packet_snapshot", runtime_dir / f"{capture_id}.packet-snapshot.json"),
        ("normalized", runtime_dir / f"{capture_id}.normalized.json"),
        ("file_result", runtime_dir / f"{capture_id}.file-result.json"),
        ("os_result", runtime_dir / f"{capture_id}.os-capture-result.json"),
    )
    return [label for label, path in artifact_map if path.exists()]


def assess_capture_requeue_recovery(*, config, capture_id: str) -> dict[str, Any]:
    resolved_capture_id = str(capture_id or "").strip()
    if not resolved_capture_id:
        raise ValueError("capture_id is required")

    runtime_dir = resolve_capture_runtime_dir(config)
    queue_packet_path = (resolve_capture_queue_dir(config) / f"{resolved_capture_id}.json").expanduser().resolve()
    state_path = runtime_dir / f"{resolved_capture_id}.state.json"
    packet_snapshot_path = resolve_capture_packet_snapshot_path(config, resolved_capture_id)

    state_payload, state_error = _read_optional_json_object(state_path)
    snapshot_payload, snapshot_error = _read_optional_json_object(packet_snapshot_path)
    artifact_kinds = _runtime_artifact_kinds(runtime_dir=runtime_dir, capture_id=resolved_capture_id)

    packet_path_from_runtime = str(
        snapshot_payload.get("packetPath") or state_payload.get("packetPath") or queue_packet_path
    ).strip()
    queue_packet_present = Path(packet_path_from_runtime).expanduser().exists() if packet_path_from_runtime else False

    result = {
        "captureId": resolved_capture_id,
        "recoverable": False,
        "recoveryKind": "",
        "reasonCode": "",
        "reason": "",
        "queuePacketPath": packet_path_from_runtime,
        "queuePacketPresent": queue_packet_present,
        "runtimeStatePath": str(state_path),
        "runtimeStatePresent": bool(state_payload),
        "packetSnapshotPath": str(packet_snapshot_path),
        "packetSnapshotPresent": bool(snapshot_payload),
        "artifactKinds": artifact_kinds,
        "packet": {},
    }

    if snapshot_error:
        result["reasonCode"] = RECOVERY_REASON_PACKET_SNAPSHOT_INVALID
        result["reason"] = f"exact packet snapshot is unreadable: {snapshot_error}"
        return result

    if not snapshot_payload:
        result["reasonCode"] = (
            RECOVERY_REASON_LEGACY_ORPHAN_MISSING_SNAPSHOT
            if artifact_kinds
            else RECOVERY_REASON_PACKET_SNAPSHOT_MISSING
        )
        if artifact_kinds:
            result["reason"] = (
                "runtime artifacts exist but the exact packet snapshot is missing; "
                "legacy orphan captures cannot be requeued safely"
            )
        else:
            result["reason"] = "exact packet snapshot is missing"
        if state_error:
            result["reason"] = f"{result['reason']} (runtime state unreadable: {state_error})"
        return result

    if str(snapshot_payload.get("schema") or "").strip() != CAPTURE_PACKET_SCHEMA:
        result["reasonCode"] = RECOVERY_REASON_PACKET_SNAPSHOT_INVALID
        result["reason"] = (
            "exact packet snapshot does not contain a dinger capture queue packet"
        )
        return result

    snapshot_capture_id = str(snapshot_payload.get("captureId") or "").strip()
    if snapshot_capture_id and snapshot_capture_id != resolved_capture_id:
        result["reasonCode"] = RECOVERY_REASON_PACKET_SNAPSHOT_CAPTURE_ID_MISMATCH
        result["reason"] = (
            f"exact packet snapshot captureId={snapshot_capture_id} does not match requested captureId={resolved_capture_id}"
        )
        return result

    result["recoverable"] = True
    result["recoveryKind"] = RECOVERY_KIND_EXACT_PACKET_SNAPSHOT
    result["reasonCode"] = RECOVERY_REASON_SNAPSHOT_AVAILABLE
    result["reason"] = "exact packet snapshot is available for requeue recovery"
    result["packet"] = snapshot_payload
    return result


__all__ = [
    "RECOVERY_KIND_EXACT_PACKET_SNAPSHOT",
    "RECOVERY_REASON_LEGACY_ORPHAN_MISSING_SNAPSHOT",
    "RECOVERY_REASON_PACKET_SNAPSHOT_CAPTURE_ID_MISMATCH",
    "RECOVERY_REASON_PACKET_SNAPSHOT_INVALID",
    "RECOVERY_REASON_PACKET_SNAPSHOT_MISSING",
    "RECOVERY_REASON_SNAPSHOT_AVAILABLE",
    "assess_capture_requeue_recovery",
]
