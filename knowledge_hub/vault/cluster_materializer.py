from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from knowledge_hub.core.schema_validator import annotate_schema_errors


CLUSTER_MANIFEST_SCHEMA_ID = "knowledge-hub.vault.cluster.materialize.manifest.v1"
CLUSTER_BLOCK_START = "<!-- KHUB_CLUSTER_START -->"
CLUSTER_BLOCK_END = "<!-- KHUB_CLUSTER_END -->"
_CLUSTER_BLOCK_PATTERN = re.compile(
    rf"\n*{re.escape(CLUSTER_BLOCK_START)}.*?{re.escape(CLUSTER_BLOCK_END)}\n*",
    flags=re.DOTALL,
)


class ClusterMaterializationError(RuntimeError):
    """Raised when cluster materialization or rollback cannot proceed."""


@dataclass(slots=True)
class ClusterMaterializationOptions:
    snapshot_path: str | None = None
    apply: bool = False
    output_dir: str = "LearningHub/Cluster_Views"
    manifest_dir: str = ".obsidian/khub/clusters/manifests"
    max_cluster_links: int = 2
    max_bridge_links: int = 1
    include_singletons: bool = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _slugify(value: str, fallback: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "").strip().lower())
    slug = re.sub(r"[^a-z0-9가-힣]+", "-", normalized).strip("-")
    return slug or fallback


def _normalize_rel_path(value: str) -> str:
    return Path(str(value or "")).as_posix().strip()


def _obsidian_target(rel_path: str) -> str:
    path = Path(_normalize_rel_path(rel_path))
    return path.with_suffix("").as_posix()


def _obsidian_link(rel_path: str, alias: str | None = None) -> str:
    target = _obsidian_target(rel_path)
    label = str(alias or "").strip()
    return f"[[{target}|{label}]]" if label else f"[[{target}]]"


def _text_hash(value: str) -> str:
    return sha256(str(value or "").encode("utf-8")).hexdigest()


class VaultClusterMaterializer:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path).expanduser().resolve()
        if not self.vault_path.exists():
            raise ClusterMaterializationError(f"vault path does not exist: {self.vault_path}")

    def materialize(self, options: ClusterMaterializationOptions | None = None) -> dict[str, Any]:
        opts = options or ClusterMaterializationOptions()
        snapshot_path = self._resolve_snapshot_path(opts.snapshot_path)
        snapshot = self._load_snapshot(snapshot_path)
        run_id = f"cluster_{uuid4().hex[:12]}"
        output_root = Path(str(opts.output_dir or "LearningHub/Cluster_Views")).expanduser()
        manifest_root = Path(str(opts.manifest_dir or ".obsidian/khub/clusters/manifests")).expanduser()

        plan = self._build_plan(snapshot, snapshot_path, output_root, opts)
        payload = {
            "schema": CLUSTER_MANIFEST_SCHEMA_ID,
            "runId": run_id,
            "mode": "apply" if opts.apply else "dry-run",
            "generatedAt": _now_iso(),
            "vaultPath": str(self.vault_path),
            "sourceSnapshotPath": str(snapshot_path),
            "outputDir": output_root.as_posix(),
            "manifestDir": manifest_root.as_posix(),
            "managedBlock": {
                "startMarker": CLUSTER_BLOCK_START,
                "endMarker": CLUSTER_BLOCK_END,
            },
            "options": {
                "maxClusterLinks": int(opts.max_cluster_links),
                "maxBridgeLinks": int(opts.max_bridge_links),
                "includeSingletons": bool(opts.include_singletons),
            },
            "counts": {
                "clustersGenerated": len(plan["clusterNotes"]),
                "notesTouched": len(plan["touchedNotes"]),
                "singletonsSkipped": int(plan["singletonsSkipped"]),
            },
            "clusterNotes": plan["clusterNotes"],
            "touchedNotes": plan["touchedNotes"],
            "generatedNotePaths": [item["path"] for item in plan["clusterNotes"]],
            "warnings": list(plan["warnings"]),
        }
        annotate_schema_errors(payload, CLUSTER_MANIFEST_SCHEMA_ID, strict=False)

        for touched in payload["touchedNotes"]:
            note_path = self.vault_path / touched["path"]
            if not note_path.exists():
                touched["missing"] = True
                payload["warnings"].append(f"note missing: {touched['path']}")
                continue
            original = note_path.read_text(encoding="utf-8")
            updated = self._upsert_managed_block(original, touched["managedBlock"])
            touched["beforeHash"] = _text_hash(original)
            touched["afterHash"] = _text_hash(updated)
            touched["changed"] = updated != original
            touched["missing"] = False

        if not opts.apply:
            return payload

        for cluster_note in plan["clusterNotes"]:
            target_path = self.vault_path / cluster_note["path"]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(cluster_note["content"], encoding="utf-8")

        for touched in plan["touchedNotes"]:
            note_path = self.vault_path / touched["path"]
            if not note_path.exists():
                continue
            original = note_path.read_text(encoding="utf-8")
            updated = self._upsert_managed_block(original, touched["managedBlock"])
            if updated != original:
                note_path.write_text(updated, encoding="utf-8")

        manifest_path = self._manifest_path(manifest_root, run_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload["manifestPath"] = str(manifest_path.relative_to(self.vault_path).as_posix())
        annotate_schema_errors(payload, CLUSTER_MANIFEST_SCHEMA_ID, strict=False)
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def revert(self, manifest_path: str | None = None, *, latest: bool = True) -> dict[str, Any]:
        manifest = self._load_manifest(manifest_path=manifest_path, latest=latest)
        generated_deleted = 0
        generated_missing = 0
        touched_cleaned = 0
        touched_missing = 0
        warnings: list[str] = []

        for cluster_note in manifest.get("clusterNotes", []):
            rel_path = _normalize_rel_path(cluster_note.get("path", ""))
            if not rel_path:
                continue
            target = self.vault_path / rel_path
            if target.exists():
                target.unlink()
                self._remove_empty_parents(target.parent)
                generated_deleted += 1
            else:
                generated_missing += 1

        for touched in manifest.get("touchedNotes", []):
            rel_path = _normalize_rel_path(touched.get("path", ""))
            if not rel_path:
                continue
            note_path = self.vault_path / rel_path
            if not note_path.exists():
                touched_missing += 1
                continue
            original = note_path.read_text(encoding="utf-8")
            updated, removed = self._remove_managed_block(original)
            if removed:
                note_path.write_text(updated, encoding="utf-8")
                touched_cleaned += 1
            else:
                warnings.append(f"managed block not found: {rel_path}")

        return {
            "schema": "knowledge-hub.vault.cluster.revert.result.v1",
            "status": "ok",
            "revertedAt": _now_iso(),
            "vaultPath": str(self.vault_path),
            "manifestPath": _normalize_rel_path(manifest.get("manifestPath", "")),
            "counts": {
                "generatedDeleted": generated_deleted,
                "generatedMissing": generated_missing,
                "touchedCleaned": touched_cleaned,
                "touchedMissing": touched_missing,
            },
            "warnings": warnings,
        }

    def _build_plan(
        self,
        snapshot: dict[str, Any],
        snapshot_path: Path,
        output_root: Path,
        options: ClusterMaterializationOptions,
    ) -> dict[str, Any]:
        nodes = {
            str(node.get("id", "")): dict(node)
            for node in snapshot.get("nodes", [])
            if str(node.get("id", "")).strip()
        }
        clusters = {
            str(cluster.get("id", "")): dict(cluster)
            for cluster in snapshot.get("clusters", [])
            if str(cluster.get("id", "")).strip()
        }
        bridge_targets: dict[str, list[str]] = {}
        for edge in snapshot.get("edges", []):
            if str(edge.get("type", "")) != "bridge_hint":
                continue
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if not source or not target:
                continue
            bridge_targets.setdefault(source, [])
            if target not in bridge_targets[source]:
                bridge_targets[source].append(target)

        members_by_cluster: dict[str, list[dict[str, Any]]] = {}
        for node in nodes.values():
            cluster_id = str(node.get("clusterId", "")).strip()
            if not cluster_id:
                continue
            members_by_cluster.setdefault(cluster_id, []).append(node)

        cluster_notes: list[dict[str, Any]] = []
        touched_notes: list[dict[str, Any]] = []
        warnings: list[str] = []
        singletons_skipped = 0

        for cluster_id, cluster in clusters.items():
            members = sorted(
                members_by_cluster.get(cluster_id, []),
                key=lambda item: (-float(item.get("importance", 0.0)), str(item.get("title", ""))),
            )
            if not members:
                continue
            has_bridge = any(bool(member.get("bridge")) for member in members)
            if int(cluster.get("size", len(members))) <= 1 and not options.include_singletons and not has_bridge:
                singletons_skipped += 1
                continue

            label = str(cluster.get("label", "") or cluster_id).strip()
            representative_id = str(cluster.get("representativeNoteId", "")).strip() or str(members[0]["id"])
            representative = nodes.get(representative_id, members[0])
            cluster_rel_path = self._cluster_note_relpath(output_root, cluster_id, label)
            cluster_note = {
                "clusterId": cluster_id,
                "label": label,
                "path": cluster_rel_path,
                "representativeNoteId": representative_id,
                "memberCount": len(members),
                "bridgeCount": sum(1 for member in members if bool(member.get("bridge"))),
                "content": self._render_cluster_note(
                    cluster_id=cluster_id,
                    label=label,
                    representative=representative,
                    members=members,
                    bridge_targets=bridge_targets,
                    snapshot_path=str(snapshot_path),
                    rel_path=cluster_rel_path,
                ),
            }
            cluster_notes.append(cluster_note)

            for member in members:
                note_rel_path = _normalize_rel_path(member.get("path", ""))
                if not note_rel_path:
                    warnings.append(f"node missing path: {member.get('id', '')}")
                    continue
                same_cluster_links = self._select_same_cluster_links(
                    note_id=str(member["id"]),
                    members=members,
                    max_links=max(0, int(options.max_cluster_links)),
                )
                bridge_links = self._select_bridge_links(
                    note_id=str(member["id"]),
                    bridge_targets=bridge_targets,
                    nodes=nodes,
                    max_links=max(0, int(options.max_bridge_links)),
                )
                touched_notes.append(
                    {
                        "path": note_rel_path,
                        "clusterId": cluster_id,
                        "beforeHash": "",
                        "afterHash": "",
                        "changed": False,
                        "missing": False,
                        "managedBlock": self._render_managed_block(
                            cluster_label=label,
                            cluster_note_path=cluster_rel_path,
                            representative=representative if str(member["id"]) != representative_id else None,
                            same_cluster_links=same_cluster_links,
                            bridge_links=bridge_links,
                        ),
                    }
                )

        return {
            "clusterNotes": cluster_notes,
            "touchedNotes": touched_notes,
            "warnings": warnings,
            "singletonsSkipped": singletons_skipped,
        }

    def _render_cluster_note(
        self,
        *,
        cluster_id: str,
        label: str,
        representative: dict[str, Any],
        members: list[dict[str, Any]],
        bridge_targets: dict[str, list[str]],
        snapshot_path: str,
        rel_path: str,
    ) -> str:
        representative_link = _obsidian_link(str(representative.get("path", "")), str(representative.get("title", "")))
        top_members = members[: min(20, len(members))]
        bridge_members = [member for member in members if bridge_targets.get(str(member.get("id", "")))]

        lines = [
            "---",
            "schema: knowledge-hub.vault.cluster.note.v1",
            f"generated_at: {json.dumps(_now_iso())}",
            f"cluster_id: {json.dumps(cluster_id)}",
            f"label: {json.dumps(label)}",
            f"representative_note: {json.dumps(str(representative.get('path', '')))}",
            f"source_snapshot: {json.dumps(snapshot_path)}",
            f"materialized_path: {json.dumps(rel_path)}",
            f"member_count: {len(members)}",
            f"bridge_count: {len(bridge_members)}",
            "---",
            "",
            f"# Cluster: {label}",
            "",
            f"- Generated: {_now_iso()}",
            f"- Representative: {representative_link}",
            f"- Members: {len(members)}",
            f"- Bridge Notes: {len(bridge_members)}",
            "",
            "## Top Members",
        ]
        for member in top_members:
            member_link = _obsidian_link(str(member.get("path", "")), str(member.get("title", "")))
            badge = " bridge" if bool(member.get("bridge")) else ""
            lines.append(f"- {member_link}{badge}")
        if len(members) > len(top_members):
            lines.append(f"- ... and {len(members) - len(top_members)} more")
        lines.append("")
        lines.append("## Bridge Notes")
        if bridge_members:
            for member in bridge_members[:20]:
                lines.append(
                    f"- {_obsidian_link(str(member.get('path', '')), str(member.get('title', '')))}"
                )
        else:
            lines.append("- None")
        lines.append("")
        return "\n".join(lines)

    def _select_same_cluster_links(self, *, note_id: str, members: list[dict[str, Any]], max_links: int) -> list[dict[str, str]]:
        if max_links <= 0:
            return []
        selected: list[dict[str, str]] = []
        for member in members:
            candidate_id = str(member.get("id", ""))
            if candidate_id == note_id:
                continue
            selected.append(
                {
                    "path": str(member.get("path", "")),
                    "title": str(member.get("title", "")),
                }
            )
            if len(selected) >= max_links:
                break
        return selected

    def _select_bridge_links(
        self,
        *,
        note_id: str,
        bridge_targets: dict[str, list[str]],
        nodes: dict[str, dict[str, Any]],
        max_links: int,
    ) -> list[dict[str, str]]:
        if max_links <= 0:
            return []
        selected: list[dict[str, str]] = []
        for target_id in bridge_targets.get(note_id, [])[:max_links]:
            target = nodes.get(target_id)
            if not target:
                continue
            selected.append(
                {
                    "path": str(target.get("path", "")),
                    "title": str(target.get("title", "")),
                }
            )
        return selected

    def _render_managed_block(
        self,
        *,
        cluster_label: str,
        cluster_note_path: str,
        representative: dict[str, Any] | None,
        same_cluster_links: list[dict[str, str]],
        bridge_links: list[dict[str, str]],
    ) -> str:
        lines = [
            CLUSTER_BLOCK_START,
            f"**KHUB Cluster:** {_obsidian_link(cluster_note_path, cluster_label)}",
        ]
        if representative:
            lines.append(
                f"**Representative:** {_obsidian_link(str(representative.get('path', '')), str(representative.get('title', '')))}"
            )
        if same_cluster_links:
            rendered = ", ".join(_obsidian_link(item["path"], item["title"]) for item in same_cluster_links)
            lines.append(f"**Related:** {rendered}")
        if bridge_links:
            rendered = ", ".join(_obsidian_link(item["path"], item["title"]) for item in bridge_links)
            lines.append(f"**Bridge:** {rendered}")
        lines.append(CLUSTER_BLOCK_END)
        return "\n".join(lines)

    def _upsert_managed_block(self, original: str, block: str) -> str:
        body = str(original or "")
        replacement = f"\n\n{block}\n"
        if _CLUSTER_BLOCK_PATTERN.search(body):
            updated = _CLUSTER_BLOCK_PATTERN.sub(replacement, body).rstrip()
            return updated + "\n"
        stripped = body.rstrip()
        if stripped:
            return f"{stripped}{replacement}"
        return block + "\n"

    def _remove_managed_block(self, original: str) -> tuple[str, bool]:
        body = str(original or "")
        updated, count = _CLUSTER_BLOCK_PATTERN.subn("\n\n", body)
        if count == 0:
            return body, False
        updated = re.sub(r"\n{3,}", "\n\n", updated).rstrip()
        return (updated + "\n") if updated else "", True

    def _resolve_snapshot_path(self, explicit: str | None) -> Path:
        if explicit:
            target = Path(explicit).expanduser()
        else:
            target = self.vault_path / ".obsidian" / "khub" / "topology" / "latest.json"
        resolved = target.resolve()
        if not resolved.exists():
            raise ClusterMaterializationError(f"topology snapshot not found: {resolved}")
        return resolved

    def _load_snapshot(self, snapshot_path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise ClusterMaterializationError(f"failed to load topology snapshot: {snapshot_path}: {error}") from error
        if not isinstance(payload, dict):
            raise ClusterMaterializationError(f"invalid topology snapshot payload: {snapshot_path}")
        if str(payload.get("schema", "")).strip() != "knowledge-hub.vault.topology.snapshot.v1":
            raise ClusterMaterializationError(f"unexpected topology schema: {payload.get('schema')}")
        return payload

    def _cluster_note_relpath(self, output_root: Path, cluster_id: str, label: str) -> str:
        slug = _slugify(label, fallback=cluster_id.lower())
        return (output_root / f"{cluster_id}-{slug}.md").as_posix()

    def _manifest_path(self, manifest_root: Path, run_id: str) -> Path:
        return (self.vault_path / manifest_root / f"{run_id}.json").resolve()

    def _remove_empty_parents(self, start: Path) -> None:
        current = start
        while current != self.vault_path and current.is_dir():
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def _load_manifest(self, manifest_path: str | None, *, latest: bool) -> dict[str, Any]:
        if manifest_path:
            target = (self.vault_path / manifest_path).resolve() if not Path(manifest_path).is_absolute() else Path(manifest_path).resolve()
        elif latest:
            candidates = sorted(
                (self.vault_path / ".obsidian" / "khub" / "clusters" / "manifests").glob("*.json"),
                key=lambda item: item.stat().st_mtime,
            )
            if not candidates:
                raise ClusterMaterializationError("no cluster materialization manifest found")
            target = candidates[-1]
        else:
            raise ClusterMaterializationError("manifest path is required when --latest is disabled")

        if not target.exists():
            raise ClusterMaterializationError(f"manifest not found: {target}")
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except Exception as error:
            raise ClusterMaterializationError(f"failed to load manifest: {target}: {error}") from error
        if not isinstance(payload, dict):
            raise ClusterMaterializationError(f"invalid manifest payload: {target}")
        payload["manifestPath"] = str(target.relative_to(self.vault_path).as_posix())
        return payload
