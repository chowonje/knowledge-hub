"""CLI-first runtime processor for queued Dinger capture packets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
from typing import Any, Callable
from uuid import uuid4

from knowledge_hub.application.dinger_os_bridge import (
    build_capture_os_link_record,
    is_capture_linked_to_os,
)
from knowledge_hub.infrastructure.config import DEFAULT_CONFIG_DIR

LIFECYCLE_STATUSES = ("captured", "normalized", "filed", "linked_to_os", "failed")
INSPECTABLE_CAPTURE_STATUSES = ("queued", "processing", "filed", "linked_to_os", "failed")
CLAIM_SCHEMA = "knowledge-hub.dinger.capture.claim.v1"
DEFAULT_CLAIM_STALE_AFTER_SEC = 900


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_json_object_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json_object(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _resolve_capture_base_dir(config) -> Path:
    sqlite_path = str(config.get_nested("storage", "sqlite", default=str(DEFAULT_CONFIG_DIR / "knowledge.db")) or "").strip()
    if not sqlite_path:
        return DEFAULT_CONFIG_DIR / "dinger_capture_intake"
    return Path(sqlite_path).expanduser().resolve().parent / "dinger_capture_intake"


def resolve_capture_queue_dir(config) -> Path:
    return _resolve_capture_base_dir(config) / "queue"


def resolve_capture_runtime_dir(config) -> Path:
    return _resolve_capture_base_dir(config) / "runtime"


def resolve_capture_packet_snapshot_path(config, capture_id: str) -> Path:
    return resolve_capture_runtime_dir(config) / f"{capture_id}.packet-snapshot.json"


@dataclass(frozen=True)
class CaptureRuntimePaths:
    capture_id: str
    packet_path: Path
    packet_snapshot_path: Path
    state_path: Path
    normalized_path: Path
    filed_result_path: Path
    os_result_path: Path
    claim_path: Path


def _runtime_paths(*, config, packet_path: Path, capture_id: str, claim_id: str | None = None) -> CaptureRuntimePaths:
    runtime_dir = resolve_capture_runtime_dir(config)
    claim_key = str(claim_id or packet_path.stem or capture_id).strip() or "capture"
    return CaptureRuntimePaths(
        capture_id=capture_id,
        packet_path=packet_path,
        packet_snapshot_path=resolve_capture_packet_snapshot_path(config, capture_id),
        state_path=runtime_dir / f"{capture_id}.state.json",
        normalized_path=runtime_dir / f"{capture_id}.normalized.json",
        filed_result_path=runtime_dir / f"{capture_id}.file-result.json",
        os_result_path=runtime_dir / f"{capture_id}.os-capture-result.json",
        claim_path=runtime_dir / f"{claim_key}.claim.json",
    )


def _source_ref_summary(refs: list[dict[str, Any]]) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        source_type = str(ref.get("sourceType") or "").strip()
        primary = ""
        for key in ("paperId", "url", "noteId", "stableScopeId", "documentScopeId"):
            candidate = str(ref.get(key) or "").strip()
            if candidate:
                primary = candidate
                break
        if source_type and primary:
            summary.append({"sourceType": source_type, "primary": primary})
    return summary


def _load_or_init_state(paths: CaptureRuntimePaths) -> dict[str, Any]:
    now = _now_iso()
    if paths.state_path.exists():
        payload = _read_json_object(paths.state_path)
        payload.setdefault("schema", "knowledge-hub.dinger.capture.runtime.state.v1")
        payload.setdefault("captureId", paths.capture_id)
        payload.setdefault("packetPath", str(paths.packet_path))
        payload.setdefault("idempotencyKey", paths.capture_id or str(paths.packet_path))
        payload.setdefault("currentStatus", "captured")
        payload.setdefault("attemptCount", 0)
        payload.setdefault("createdAt", now)
        payload.setdefault("updatedAt", now)
        payload.setdefault("steps", {})
        payload.setdefault("artifacts", {})
        return payload
    return {
        "schema": "knowledge-hub.dinger.capture.runtime.state.v1",
        "captureId": paths.capture_id,
        "packetPath": str(paths.packet_path),
        "idempotencyKey": paths.capture_id or str(paths.packet_path),
        "currentStatus": "captured",
        "attemptCount": 0,
        "createdAt": now,
        "updatedAt": now,
        "steps": {},
        "artifacts": {},
    }


def _load_existing_state(paths: CaptureRuntimePaths) -> dict[str, Any] | None:
    if not paths.state_path.exists():
        return None
    return _load_or_init_state(paths)


def _persist_state(paths: CaptureRuntimePaths, state: dict[str, Any]) -> None:
    state["updatedAt"] = _now_iso()
    _write_json(paths.state_path, state)


def _ensure_packet_snapshot(paths: CaptureRuntimePaths, *, state: dict[str, Any], packet: dict[str, Any]) -> None:
    if not paths.packet_snapshot_path.exists():
        _write_json(paths.packet_snapshot_path, packet)
    state.setdefault("artifacts", {})["packetSnapshotPath"] = str(paths.packet_snapshot_path)


def _mark_step(
    state: dict[str, Any],
    step: str,
    *,
    step_status: str,
    current_status: str | None = None,
    error: str = "",
    **extra: Any,
) -> None:
    steps = state.setdefault("steps", {})
    entry = dict(steps.get(step) or {})
    entry["status"] = step_status
    entry["updatedAt"] = _now_iso()
    if step_status == "ok":
        entry.setdefault("completedAt", entry["updatedAt"])
    if error:
        entry["error"] = error
    for key, value in extra.items():
        if value not in (None, "", [], {}):
            entry[key] = value
    steps[step] = entry
    if current_status:
        state["currentStatus"] = current_status
    if error:
        state["lastError"] = error
    elif step_status == "ok" and step == "failed":
        state.pop("lastError", None)


def _normalize_capture_packet(packet: dict[str, Any], *, packet_path: Path, idempotency_key: str) -> dict[str, Any]:
    schema_id = str(packet.get("schema") or "").strip()
    if schema_id != "knowledge-hub.dinger.capture.result.v1":
        raise ValueError(f"unsupported packet schema: {schema_id or '(missing)'}")
    status = str(packet.get("status") or "").strip()
    if status == "failed":
        raise ValueError(str(packet.get("error") or "capture packet is already failed"))
    capture_id = str(packet.get("captureId") or "").strip()
    if not capture_id:
        raise ValueError("capture packet is missing captureId")
    if status not in LIFECYCLE_STATUSES:
        raise ValueError(f"unsupported capture status: {status or '(missing)'}")
    source_refs = [dict(item) for item in list(packet.get("sourceRefs") or []) if isinstance(item, dict)]
    return {
        "captureId": capture_id,
        "idempotencyKey": idempotency_key,
        "packetPath": str(packet_path),
        "packetStatus": status,
        "title": str(packet.get("title") or packet.get("pageTitle") or "").strip(),
        "captureUrl": str(packet.get("captureUrl") or packet.get("sourceUrl") or "").strip(),
        "capturedAt": str(packet.get("capturedAt") or "").strip(),
        "client": str(packet.get("client") or "").strip(),
        "tags": [str(item).strip() for item in list(packet.get("tags") or []) if str(item).strip()],
        "sourceRefs": source_refs,
        "sourceRefSummary": _source_ref_summary(source_refs),
        "normalizedAt": _now_iso(),
    }


def _augment_filed_payload(filed_payload: dict[str, Any], *, normalized: dict[str, Any]) -> dict[str, Any]:
    result = dict(filed_payload)
    capture_url = str(result.get("captureUrl") or "").strip()
    if not capture_url:
        normalized_capture_url = str(normalized.get("captureUrl") or "").strip()
        if normalized_capture_url:
            result["captureUrl"] = normalized_capture_url
    if not str(result.get("title") or "").strip() and str(normalized.get("title") or "").strip():
        result["title"] = str(normalized.get("title") or "").strip()
    return result


def _resolve_claim_stale_after_sec(config) -> int:
    value = config.get_nested("dinger", "capture_process", "claim_stale_after_sec", default=DEFAULT_CLAIM_STALE_AFTER_SEC)
    try:
        seconds = int(value or DEFAULT_CLAIM_STALE_AFTER_SEC)
    except (TypeError, ValueError):
        seconds = DEFAULT_CLAIM_STALE_AFTER_SEC
    return max(30, seconds)


def _build_claim_payload(*, paths: CaptureRuntimePaths, stale_after_sec: int) -> dict[str, Any]:
    now = _now_iso()
    return {
        "schema": CLAIM_SCHEMA,
        "claimantId": str(uuid4()),
        "captureId": paths.capture_id,
        "packetPath": str(paths.packet_path),
        "claimPath": str(paths.claim_path),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "claimedAt": now,
        "updatedAt": now,
        "staleAfterSec": stale_after_sec,
    }


def _write_claim_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(data)


def _read_claim_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = _read_json_object(path)
    except Exception:
        return {}
    if str(payload.get("schema") or "").strip() not in {"", CLAIM_SCHEMA}:
        return {}
    return payload


def _claim_is_stale(claim_payload: dict[str, Any], *, stale_after_sec: int) -> tuple[bool, str]:
    pid_value = claim_payload.get("pid")
    try:
        pid = int(pid_value)
    except (TypeError, ValueError):
        pid = 0
    host = str(claim_payload.get("host") or "").strip()
    local_host = socket.gethostname()
    if pid > 0 and (not host or host == local_host):
        if not _pid_is_alive(pid):
            return True, "owner_pid_missing"
        return False, "owner_pid_alive"
    updated_at = _parse_iso_datetime(claim_payload.get("updatedAt") or claim_payload.get("claimedAt"))
    if updated_at is None:
        return False, "claim_timestamp_missing"
    age_sec = (datetime.now(timezone.utc) - updated_at).total_seconds()
    if age_sec > float(stale_after_sec):
        return True, "claim_timed_out"
    return False, "claim_fresh"


def _acquire_packet_claim(paths: CaptureRuntimePaths, *, config) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    stale_after_sec = _resolve_claim_stale_after_sec(config)
    while True:
        claim_payload = _build_claim_payload(paths=paths, stale_after_sec=stale_after_sec)
        try:
            _write_claim_file(paths.claim_path, claim_payload)
            return claim_payload, {}
        except FileExistsError:
            existing = _read_claim_payload(paths.claim_path)
            stale, reason = _claim_is_stale(existing, stale_after_sec=stale_after_sec)
            if stale:
                try:
                    paths.claim_path.unlink()
                except FileNotFoundError:
                    continue
                except OSError:
                    return None, {
                        "claimPath": str(paths.claim_path),
                        "claimStatus": "stale-unremovable",
                        "claimRecovery": reason,
                    }
                continue
            return None, {
                "claimPath": str(paths.claim_path),
                "claimStatus": "active",
                "claimRecovery": reason,
                "claimOwnerPid": str(existing.get("pid") or ""),
                "claimOwnerHost": str(existing.get("host") or ""),
                "claimClaimedAt": str(existing.get("claimedAt") or ""),
                "claimUpdatedAt": str(existing.get("updatedAt") or ""),
            }


def _release_packet_claim(paths: CaptureRuntimePaths, claim_payload: dict[str, Any] | None) -> None:
    if not claim_payload or not paths.claim_path.exists():
        return
    existing = _read_claim_payload(paths.claim_path)
    if str(existing.get("claimantId") or "") != str(claim_payload.get("claimantId") or ""):
        return
    try:
        paths.claim_path.unlink()
    except FileNotFoundError:
        return


def _claimed_item(paths: CaptureRuntimePaths, *, state: dict[str, Any] | None, claim_info: dict[str, Any]) -> dict[str, Any]:
    payload = state or {
        "captureId": paths.capture_id,
        "currentStatus": "captured",
        "idempotencyKey": paths.capture_id or str(paths.packet_path),
        "updatedAt": _now_iso(),
    }
    artifacts = dict(payload.get("artifacts") or {})
    filed_step = dict((payload.get("steps") or {}).get("filed") or {})
    os_step = dict((payload.get("steps") or {}).get("linked_to_os") or {})
    os_bridge = dict(payload.get("osBridge") or {})
    return {
        "captureId": str(payload.get("captureId") or paths.capture_id),
        "status": str(payload.get("currentStatus") or "captured"),
        "idempotent": False,
        "idempotencyKey": str(payload.get("idempotencyKey") or paths.capture_id or paths.packet_path),
        "packetPath": str(paths.packet_path),
        "statePath": str(paths.state_path),
        "normalizedPath": str(artifacts.get("normalizedPath") or paths.normalized_path),
        "filedResultPath": str(artifacts.get("filedResultPath") or paths.filed_result_path),
        "osResultPath": str(artifacts.get("osResultPath") or paths.os_result_path),
        "filedRelativePath": str(filed_step.get("relativePath") or ""),
        "osItemId": str(os_step.get("itemId") or ""),
        "projectId": str(os_step.get("projectId") or ""),
        "linkAction": str(os_step.get("linkAction") or os_bridge.get("linkAction") or ""),
        "linkExplanation": str(os_bridge.get("explanation") or ""),
        "linkReason": dict(os_bridge.get("reason") or {}) if isinstance(os_bridge.get("reason"), dict) else {},
        "osBridgeRelativePath": str(os_bridge.get("relativePath") or filed_step.get("relativePath") or ""),
        "error": str(payload.get("lastError") or ""),
        "updatedAt": str(payload.get("updatedAt") or _now_iso()),
        "claimedByOtherRunner": True,
        **claim_info,
    }


class DingerCaptureProcessor:
    """Orchestrates queued capture packets without introducing new canonical state."""

    def __init__(
        self,
        *,
        khub,
        file_capture: Callable[[Path], dict[str, Any]],
        link_to_os: Callable[[dict[str, Any], dict[str, Any], str | None, str | None], dict[str, Any]],
        project_id: str | None = None,
        slug: str | None = None,
    ) -> None:
        self.khub = khub
        self.file_capture = file_capture
        self.link_to_os = link_to_os
        self.project_id = str(project_id or "").strip() or None
        self.slug = str(slug or "").strip() or None

    def process_queue(self, *, queue_dir: Path | None = None, limit: int | None = None) -> dict[str, Any]:
        resolved_queue_dir = Path(queue_dir or resolve_capture_queue_dir(self.khub.config)).expanduser().resolve()
        packet_paths = sorted(path for path in resolved_queue_dir.glob("*.json") if path.is_file())
        if limit is not None and int(limit) >= 0:
            packet_paths = packet_paths[: int(limit)]
        return self.process_packets(packet_paths=packet_paths, queue_dir=resolved_queue_dir)

    def process_packets(self, *, packet_paths: list[Path], queue_dir: Path | None = None) -> dict[str, Any]:
        items = [self.process_packet(packet_path=Path(path).expanduser().resolve()) for path in packet_paths]
        succeeded = sum(1 for item in items if str(item.get("status") or "") == "linked_to_os")
        failed = sum(1 for item in items if str(item.get("status") or "") == "failed")
        idempotent = sum(1 for item in items if bool(item.get("idempotent")))
        payload = {
            "schema": "knowledge-hub.dinger.capture-process.result.v1",
            "status": "ok" if failed == 0 else "failed",
            "queueDir": str(queue_dir or ""),
            "projectId": self.project_id or "",
            "slug": self.slug or "",
            "counts": {
                "scanned": len(packet_paths),
                "processed": len(items),
                "succeeded": succeeded,
                "failed": failed,
                "idempotent": idempotent,
            },
            "items": items,
            "createdAt": _now_iso(),
        }
        if failed:
            payload["error"] = f"{failed} capture packet(s) failed"
        return payload

    def process_packet(self, *, packet_path: Path) -> dict[str, Any]:
        initial_capture_id = packet_path.stem or "capture"
        claim_key = initial_capture_id
        paths = _runtime_paths(config=self.khub.config, packet_path=packet_path, capture_id=initial_capture_id, claim_id=claim_key)
        claim_paths = paths
        claim_payload, claim_info = _acquire_packet_claim(paths, config=self.khub.config)
        if claim_payload is None:
            state = _load_existing_state(paths)
            return _claimed_item(paths, state=state, claim_info=claim_info)
        state = _load_or_init_state(paths)
        state["attemptCount"] = int(state.get("attemptCount") or 0) + 1
        _persist_state(paths, state)
        try:
            packet = _read_json_object(packet_path)
            capture_id = str(packet.get("captureId") or "").strip() or initial_capture_id
            if capture_id != paths.capture_id:
                paths = _runtime_paths(config=self.khub.config, packet_path=packet_path, capture_id=capture_id, claim_id=claim_key)
                state = _load_or_init_state(paths)
                state["attemptCount"] = int(state.get("attemptCount") or 0) + 1
            idempotency_key = capture_id or str(packet_path)
            state["captureId"] = capture_id
            state["packetPath"] = str(packet_path)
            state["idempotencyKey"] = idempotency_key
            _mark_step(
                state,
                "captured",
                step_status="ok",
                current_status=str(state.get("currentStatus") or "captured"),
                packetStatus=str(packet.get("status") or "").strip(),
                packetSchema=str(packet.get("schema") or "").strip(),
            )
            _ensure_packet_snapshot(paths, state=state, packet=packet)
            if (
                str(state.get("currentStatus") or "") == "linked_to_os"
                and str(((state.get("steps") or {}).get("linked_to_os") or {}).get("status") or "") == "ok"
            ):
                _persist_state(paths, state)
                return self._item_from_state(paths=paths, state=state, idempotent=True)

            normalized = _normalize_capture_packet(packet, packet_path=packet_path, idempotency_key=idempotency_key)
            artifacts = state.setdefault("artifacts", {})
            if str(state.get("currentStatus") or "") not in {"normalized", "filed", "linked_to_os"}:
                _write_json(paths.normalized_path, normalized)
                artifacts["normalizedPath"] = str(paths.normalized_path)
                _mark_step(
                    state,
                    "normalized",
                    step_status="ok",
                    current_status="normalized",
                    artifactPath=str(paths.normalized_path),
                    title=str(normalized.get("title") or ""),
                    captureUrl=str(normalized.get("captureUrl") or ""),
                    sourceRefCount=len(list(normalized.get("sourceRefs") or [])),
                )
                _persist_state(paths, state)

            filed_payload: dict[str, Any]
            filed_step = dict((state.get("steps") or {}).get("filed") or {})
            if str(state.get("currentStatus") or "") in {"filed", "linked_to_os"} and str(filed_step.get("status") or "") == "ok":
                if paths.filed_result_path.exists():
                    filed_payload = _read_json_object(paths.filed_result_path)
                else:
                    filed_payload = self.file_capture(packet_path)
                    filed_payload = _augment_filed_payload(filed_payload, normalized=normalized)
                    _write_json(paths.filed_result_path, filed_payload)
            else:
                filed_payload = self.file_capture(packet_path)
                filed_payload = _augment_filed_payload(filed_payload, normalized=normalized)
                _write_json(paths.filed_result_path, filed_payload)
                if str(filed_payload.get("status") or "").strip() != "ok":
                    raise ValueError(str(filed_payload.get("error") or "dinger file returned non-ok status"))
                artifacts["filedResultPath"] = str(paths.filed_result_path)
                _mark_step(
                    state,
                    "filed",
                    step_status="ok",
                    current_status="filed",
                    artifactPath=str(paths.filed_result_path),
                    relativePath=str(filed_payload.get("relativePath") or ""),
                    filePath=str(filed_payload.get("filePath") or ""),
                    sourceSchema=str(filed_payload.get("sourceSchema") or ""),
                )
                _persist_state(paths, state)
            if not paths.filed_result_path.exists() or _read_json_object(paths.filed_result_path).get("captureUrl") != filed_payload.get("captureUrl"):
                _write_json(paths.filed_result_path, filed_payload)

            os_payload: dict[str, Any]
            os_step = dict((state.get("steps") or {}).get("linked_to_os") or {})
            if str(state.get("currentStatus") or "") == "linked_to_os" and str(os_step.get("status") or "") == "ok":
                if paths.os_result_path.exists():
                    os_payload = _read_json_object(paths.os_result_path)
                    if not state.get("osBridge") and is_capture_linked_to_os(os_payload):
                        state["osBridge"] = build_capture_os_link_record(
                            capture_payload=packet,
                            dinger_payload=filed_payload,
                            os_payload=os_payload,
                            project_id=self.project_id,
                            slug=self.slug,
                        )
                else:
                    os_payload = {}
            else:
                os_payload = self.link_to_os(filed_payload, packet, self.project_id, self.slug)
                _write_json(paths.os_result_path, os_payload)
                if str(os_payload.get("status") or "").strip() != "ok":
                    raise ValueError(str(os_payload.get("error") or "os capture returned non-ok status"))
                if not is_capture_linked_to_os(os_payload):
                    raise ValueError("os capture bridge did not return linked_to_os traceability")
                item = dict(os_payload.get("item") or {}) if isinstance(os_payload.get("item"), dict) else {}
                link_record = build_capture_os_link_record(
                    capture_payload=packet,
                    dinger_payload=filed_payload,
                    os_payload=os_payload,
                    project_id=self.project_id,
                    slug=self.slug,
                )
                artifacts["osResultPath"] = str(paths.os_result_path)
                state["osBridge"] = link_record
                _mark_step(
                    state,
                    "linked_to_os",
                    step_status="ok",
                    current_status="linked_to_os",
                    artifactPath=str(paths.os_result_path),
                    itemId=str(item.get("id") or ""),
                    projectId=str(item.get("projectId") or ""),
                    inboxKind=str(item.get("kind") or ""),
                    summary=str(item.get("summary") or ""),
                    linkAction=str(link_record.get("linkAction") or ""),
                    relativePath=str(link_record.get("relativePath") or ""),
                    packetPath=str(link_record.get("packetPath") or ""),
                )
                _persist_state(paths, state)
            return self._item_from_state(paths=paths, state=state, idempotent=False)
        except Exception as error:
            message = str(error) or "capture processing failed"
            _mark_step(state, "failed", step_status="ok", current_status="failed", error=message)
            _persist_state(paths, state)
            return self._item_from_state(paths=paths, state=state, idempotent=False)
        finally:
            _release_packet_claim(claim_paths, claim_payload)

    def _item_from_state(self, *, paths: CaptureRuntimePaths, state: dict[str, Any], idempotent: bool) -> dict[str, Any]:
        artifacts = dict(state.get("artifacts") or {})
        steps = dict(state.get("steps") or {})
        filed_step = dict(steps.get("filed") or {})
        os_step = dict(steps.get("linked_to_os") or {})
        os_bridge = dict(state.get("osBridge") or {})
        return {
            "captureId": str(state.get("captureId") or paths.capture_id),
            "status": str(state.get("currentStatus") or "captured"),
            "idempotent": bool(idempotent),
            "idempotencyKey": str(state.get("idempotencyKey") or paths.capture_id or paths.packet_path),
            "packetPath": str(paths.packet_path),
            "statePath": str(paths.state_path),
            "normalizedPath": str(artifacts.get("normalizedPath") or paths.normalized_path),
            "filedResultPath": str(artifacts.get("filedResultPath") or paths.filed_result_path),
            "osResultPath": str(artifacts.get("osResultPath") or paths.os_result_path),
            "filedRelativePath": str(filed_step.get("relativePath") or ""),
            "osItemId": str(os_step.get("itemId") or ""),
        "projectId": str(os_step.get("projectId") or self.project_id or ""),
        "linkAction": str(os_step.get("linkAction") or os_bridge.get("linkAction") or ""),
        "linkExplanation": str(os_bridge.get("explanation") or ""),
        "linkReason": dict(os_bridge.get("reason") or {}) if isinstance(os_bridge.get("reason"), dict) else {},
        "osBridgeRelativePath": str(os_bridge.get("relativePath") or filed_step.get("relativePath") or ""),
        "error": str(state.get("lastError") or ""),
        "updatedAt": str(state.get("updatedAt") or _now_iso()),
    }


def _derived_inspect_status(
    *,
    packet: dict[str, Any],
    state: dict[str, Any],
    filed_payload: dict[str, Any],
    os_payload: dict[str, Any],
) -> tuple[str, str]:
    lifecycle_status = str(state.get("currentStatus") or "").strip()
    if not lifecycle_status:
        lifecycle_status = str(packet.get("status") or "").strip() or "captured"

    if lifecycle_status == "linked_to_os" or is_capture_linked_to_os(os_payload):
        return "linked_to_os", "linked_to_os"
    if lifecycle_status == "failed" or str(state.get("lastError") or "").strip():
        return "failed", lifecycle_status or "failed"
    if lifecycle_status == "filed" or filed_payload:
        return "filed", lifecycle_status or "filed"
    if state:
        return "processing", lifecycle_status or "captured"
    return "queued", lifecycle_status or "captured"


def inspect_capture_packet(*, config, packet_path: Path) -> dict[str, Any]:
    resolved_packet_path = Path(packet_path).expanduser().resolve()
    packet = _read_json_object(resolved_packet_path)
    capture_id = str(packet.get("captureId") or "").strip() or resolved_packet_path.stem
    paths = _runtime_paths(config=config, packet_path=resolved_packet_path, capture_id=capture_id)
    state = _read_json_object_if_exists(paths.state_path)
    normalized = _read_json_object_if_exists(paths.normalized_path)
    filed_payload = _read_json_object_if_exists(paths.filed_result_path)
    os_payload = _read_json_object_if_exists(paths.os_result_path)

    inspect_status, lifecycle_status = _derived_inspect_status(
        packet=packet,
        state=state,
        filed_payload=filed_payload,
        os_payload=os_payload,
    )
    steps = dict(state.get("steps") or {})
    filed_step = dict(steps.get("filed") or {})
    linked_step = dict(steps.get("linked_to_os") or {})
    os_bridge = dict(state.get("osBridge") or {})
    source_refs = [
        dict(ref)
        for ref in list(
            normalized.get("sourceRefs")
            or packet.get("sourceRefs")
            or filed_payload.get("sourceRefs")
            or os_bridge.get("sourceRefs")
            or []
        )
        if isinstance(ref, dict)
    ]
    updated_at = (
        str(state.get("updatedAt") or "").strip()
        or str((os_payload.get("item") or {}).get("updatedAt") or "").strip()
        or str(filed_payload.get("createdAt") or "").strip()
        or str(packet.get("createdAt") or packet.get("capturedAt") or "").strip()
        or _now_iso()
    )
    return {
        "captureId": capture_id,
        "status": inspect_status,
        "lifecycleStatus": lifecycle_status,
        "packetStatus": str(packet.get("status") or "").strip() or "captured",
        "queueStatus": str(packet.get("queueStatus") or "").strip(),
        "title": str(packet.get("title") or packet.get("pageTitle") or normalized.get("title") or "").strip(),
        "captureUrl": str(packet.get("captureUrl") or packet.get("sourceUrl") or normalized.get("captureUrl") or "").strip(),
        "capturedAt": str(packet.get("capturedAt") or "").strip(),
        "client": str(packet.get("client") or "").strip(),
        "tags": [str(tag).strip() for tag in list(packet.get("tags") or []) if str(tag).strip()],
        "sourceRefCount": len(source_refs),
        "attemptCount": int(state.get("attemptCount") or 0),
        "packetPath": str(paths.packet_path),
        "statePath": str(paths.state_path),
        "normalizedPath": str(paths.normalized_path),
        "filedResultPath": str(paths.filed_result_path),
        "osResultPath": str(paths.os_result_path),
        "stateExists": bool(state),
        "normalizedExists": bool(normalized),
        "filedExists": bool(filed_payload),
        "osExists": bool(os_payload),
        "filedRelativePath": str(filed_step.get("relativePath") or filed_payload.get("relativePath") or "").strip(),
        "osItemId": str(linked_step.get("itemId") or ((os_payload.get("item") or {}).get("id") or "")).strip(),
        "projectId": str(linked_step.get("projectId") or os_bridge.get("projectId") or "").strip(),
        "linkAction": str(linked_step.get("linkAction") or os_bridge.get("linkAction") or os_payload.get("linkAction") or "").strip(),
        "linkExplanation": str(os_bridge.get("explanation") or os_payload.get("explanation") or "").strip(),
        "linkReason": dict(os_bridge.get("reason") or os_payload.get("reason") or {}) if isinstance(os_bridge.get("reason") or os_payload.get("reason"), dict) else {},
        "error": str(state.get("lastError") or "").strip(),
        "updatedAt": updated_at,
    }


def inspect_capture_queue(*, config, queue_dir: Path | None = None) -> dict[str, Any]:
    resolved_queue_dir = Path(queue_dir or resolve_capture_queue_dir(config)).expanduser().resolve()
    items = [
        inspect_capture_packet(config=config, packet_path=packet_path)
        for packet_path in sorted(path for path in resolved_queue_dir.glob("*.json") if path.is_file())
    ]
    items.sort(key=lambda item: (str(item.get("updatedAt") or ""), str(item.get("captureId") or "")), reverse=True)
    return {
        "queueDir": str(resolved_queue_dir),
        "items": items,
    }


def show_capture_packet(*, config, packet_path: Path) -> dict[str, Any]:
    resolved_packet_path = Path(packet_path).expanduser().resolve()
    packet = _read_json_object(resolved_packet_path)
    capture_id = str(packet.get("captureId") or "").strip() or resolved_packet_path.stem
    paths = _runtime_paths(config=config, packet_path=resolved_packet_path, capture_id=capture_id)
    state = _read_json_object_if_exists(paths.state_path)
    normalized = _read_json_object_if_exists(paths.normalized_path)
    filed_payload = _read_json_object_if_exists(paths.filed_result_path)
    os_payload = _read_json_object_if_exists(paths.os_result_path)
    return {
        "capture": inspect_capture_packet(config=config, packet_path=resolved_packet_path),
        "packet": packet,
        "runtimeState": state,
        "normalized": normalized,
        "filedResult": filed_payload,
        "osResult": os_payload,
    }
