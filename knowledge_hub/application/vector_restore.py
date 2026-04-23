from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any

from knowledge_hub.infrastructure.persistence.vector import compare_vector_stores, inspect_vector_store, list_vector_backups, probe_vector_store_openability


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _timestamped_sibling(root: Path, suffix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root.with_name(f"{root.name}.{suffix}.{stamp}")
    idx = 0
    while candidate.exists():
        idx += 1
        candidate = root.with_name(f"{root.name}.{suffix}.{stamp}.{idx}")
    return candidate


def _copy_inspection(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload or {})
    data.pop("recovery_backup", None)
    return data


def _attach_open_probe(payload: dict[str, Any], root: Path, collection_name: str) -> dict[str, Any]:
    data = dict(payload or {})
    if not data:
        return data
    probe = dict(probe_vector_store_openability(root, collection_name))
    data["openable"] = bool(probe.get("openable"))
    data["openProbeError"] = _clean_text(probe.get("error"))
    data["restorable"] = bool(data.get("available")) and bool(data.get("openable"))
    return data


def assess_vector_restore(
    *,
    config: Any,
    backup_path: str | Path | None = None,
    use_latest_backup: bool = False,
) -> dict[str, Any]:
    root = Path(getattr(config, "vector_db_path", "")).expanduser()
    collection_name = _clean_text(getattr(config, "collection_name", ""))
    active = inspect_vector_store(root, collection_name)
    available_backups = list_vector_backups(root, collection_name)
    selected_path = Path(backup_path).expanduser() if backup_path is not None else None
    selection_mode = "explicit" if selected_path is not None else "latest_backup"
    warnings: list[str] = []
    errors: list[str] = []
    backup_inspection: dict[str, Any] = {}

    if selected_path is None and use_latest_backup and available_backups:
        backup_inspection = _copy_inspection(available_backups[0])
        selected_path = Path(str(backup_inspection.get("path") or "")).expanduser()
    elif selected_path is not None:
        if not selected_path.exists():
            errors.append(f"backup path does not exist: {selected_path}")
        else:
            backup_inspection = _copy_inspection(inspect_vector_store(selected_path, collection_name))
            backup_inspection["path"] = str(selected_path)
    elif not available_backups:
        errors.append(f"no vector backups found next to {root}")

    if selected_path is not None and root.resolve() == selected_path.resolve():
        errors.append("backup path must be different from the active vector path")

    if backup_inspection:
        backup_inspection = _attach_open_probe(backup_inspection, selected_path or root, collection_name)
        if int(backup_inspection.get("chroma_embeddings", 0) or 0) <= 0:
            errors.append("selected backup has no Chroma embeddings; restore would not recover semantic retrieval")
        if bool(backup_inspection.get("available")) and not bool(backup_inspection.get("restorable")):
            error = _clean_text(backup_inspection.get("openProbeError"))
            if error:
                errors.append(f"selected backup cannot open Chroma without repair: {error}")
            else:
                errors.append("selected backup cannot open Chroma without repair")
        if bool(active.get("available")):
            warnings.append(
                f"active vector corpus is currently readable ({int(active.get('total_documents', 0) or 0)} docs); restore is an operator choice"
            )
        if int(backup_inspection.get("total_documents", 0) or 0) <= int(active.get("total_documents", 0) or 0):
            warnings.append("selected backup is not larger than the active vector corpus")

    recommended_command = "khub vector-restore --latest-backup"
    if selected_path is not None and selection_mode == "explicit":
        recommended_command = f"khub vector-restore --backup-path {selected_path}"
    recommended_apply_command = f"{recommended_command} --apply --confirm"
    status = "ok" if backup_inspection and not errors else "blocked"
    return {
        "schema": "knowledge-hub.vector-restore.result.v1",
        "status": status,
        "dryRun": True,
        "confirmed": False,
        "applied": False,
        "selection": {
            "mode": selection_mode,
            "selectedPath": str(selected_path) if selected_path is not None else "",
            "availableBackupCount": len(available_backups),
        },
        "activeVector": _copy_inspection(active),
        "backupVector": backup_inspection,
        "warnings": warnings,
        "errors": errors,
        "action": {
            "canRestore": bool(backup_inspection) and not errors,
            "recommendedPreviewCommand": recommended_command,
            "recommendedApplyCommand": recommended_apply_command,
            "willReplaceActive": root.exists(),
        },
    }


def compare_vector_backup(
    *,
    config: Any,
    backup_path: str | Path | None = None,
    use_latest_backup: bool = False,
    sample_limit: int = 10,
    document_limit: int = 10000,
) -> dict[str, Any]:
    restore_assessment = assess_vector_restore(
        config=config,
        backup_path=backup_path,
        use_latest_backup=use_latest_backup,
    )
    payload = {
        "schema": "knowledge-hub.vector-compare.result.v1",
        "status": "ok" if restore_assessment.get("status") == "ok" else "blocked",
        "selection": dict(restore_assessment.get("selection") or {}),
        "activeVector": dict(restore_assessment.get("activeVector") or {}),
        "backupVector": dict(restore_assessment.get("backupVector") or {}),
        "warnings": list(restore_assessment.get("warnings") or []),
        "errors": list(restore_assessment.get("errors") or []),
        "action": dict(restore_assessment.get("action") or {}),
        "diff": {},
    }
    if payload["status"] != "ok":
        return payload
    active_path = Path(getattr(config, "vector_db_path", "")).expanduser()
    selected_path = Path(str(((payload.get("selection") or {}).get("selectedPath") or ""))).expanduser()
    payload["diff"] = compare_vector_stores(
        active_path,
        selected_path,
        collection_name=_clean_text(getattr(config, "collection_name", "")),
        sample_limit=max(1, int(sample_limit)),
        document_limit=max(1, int(document_limit)),
    )
    payload["action"]["recommendedRestoreCommand"] = str(payload["action"].get("recommendedApplyCommand") or "")
    return payload


def restore_vector_backup(
    *,
    config: Any,
    backup_path: str | Path | None = None,
    use_latest_backup: bool = False,
    apply: bool = False,
    confirm: bool = False,
) -> dict[str, Any]:
    payload = assess_vector_restore(
        config=config,
        backup_path=backup_path,
        use_latest_backup=use_latest_backup,
    )
    payload["dryRun"] = not bool(apply)
    payload["confirmed"] = bool(confirm)
    if not apply:
        return payload
    if not confirm:
        payload["status"] = "failed"
        payload["errors"] = [*list(payload.get("errors") or []), "--confirm is required with --apply"]
        return payload
    if str(payload.get("status") or "").strip() != "ok":
        payload["status"] = "failed"
        return payload

    root = Path(getattr(config, "vector_db_path", "")).expanduser()
    collection_name = _clean_text(getattr(config, "collection_name", ""))
    selected_path = Path(str(((payload.get("selection") or {}).get("selectedPath") or ""))).expanduser()
    preserved_active_path: Path | None = None
    active_existed = root.exists()
    try:
        if active_existed:
            preserved_active_path = _timestamped_sibling(root, "pre_restore")
            shutil.move(str(root), str(preserved_active_path))
        shutil.copytree(str(selected_path), str(root))
        restored = inspect_vector_store(root, collection_name)
        restored = _attach_open_probe(restored, root, collection_name)
        if not bool(restored.get("available")):
            raise RuntimeError("restored vector store is not retrieval-ready")
        if not bool(restored.get("restorable")):
            error = _clean_text(restored.get("openProbeError"))
            if error:
                raise RuntimeError(f"restored vector store cannot open Chroma without repair: {error}")
            raise RuntimeError("restored vector store cannot open Chroma without repair")
        payload["applied"] = True
        payload["activeBackupPath"] = str(preserved_active_path) if preserved_active_path is not None else ""
        payload["restoredVector"] = _copy_inspection(restored)
        return payload
    except Exception as error:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        if preserved_active_path is not None and preserved_active_path.exists():
            shutil.move(str(preserved_active_path), str(root))
        payload["status"] = "failed"
        payload["errors"] = [*list(payload.get("errors") or []), str(error)]
        payload["applied"] = False
        return payload
