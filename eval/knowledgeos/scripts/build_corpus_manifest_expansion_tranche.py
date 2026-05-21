#!/usr/bin/env python3
"""Build and optionally apply a verified corpus manifest expansion tranche."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.application.corpus_artifacts import (  # noqa: E402
    DEFAULT_CORPUS_MANIFEST_PATH,
    corpus_manifest_entries,
    inspect_corpus_artifact,
    load_corpus_manifest,
)
from knowledge_hub.core.schema_validator import validate_payload  # noqa: E402
from knowledge_hub.infrastructure.config import Config  # noqa: E402


SCHEMA_ID = "knowledge-hub.corpus-manifest-expansion-tranche-plan.v1"
DEFAULT_ALLOWLIST = (
    PROJECT_ROOT / "eval/knowledgeos/fixtures/priority_corpus_manifest_expansion_allowlist.v1.json"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "eval/knowledgeos/fixtures/corpus_manifest.json"
ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}$")


class _PapersDirConfig:
    def __init__(self, base: Any, papers_dir: Path):
        self._base = base
        self.papers_dir = str(papers_dir)

    def get_nested(self, *args: Any, default: Any = None) -> Any:
        if tuple(args) == ("storage", "papers_dir"):
            return self.papers_dir
        if hasattr(self._base, "get_nested"):
            return self._base.get_nested(*args, default=default)
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any], *, ensure_ascii: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=ensure_ascii) + "\n", encoding="utf-8")


def _project_ref(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _default_papers_dir(config: Any) -> Path:
    raw = ""
    if hasattr(config, "get_nested"):
        raw = _clean(config.get_nested("storage", "papers_dir", default=""))
    if not raw:
        raw = _clean(getattr(config, "papers_dir", ""))
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ("." + "khub") / "papers"


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}", size


def _source_path_from_ref(corpus_location_ref: str, papers_dir: Path) -> Path | None:
    ref = _clean(corpus_location_ref)
    prefix = "papers_dir/"
    if not ref.startswith(prefix):
        return None
    rel = Path(ref[len(prefix) :])
    if rel.is_absolute() or ".." in rel.parts:
        return None
    return papers_dir / rel


def _artifact_id(source_id: str) -> str:
    normalized = source_id.replace(".", "_").replace("-", "_")
    if normalized.startswith("paper_"):
        return normalized
    return f"paper_{normalized}"


def _provenance_url(source_id: str, row: dict[str, Any]) -> str:
    if ARXIV_RE.fullmatch(source_id):
        return f"https://arxiv.org/pdf/{source_id}"
    return _clean(row.get("provenance_url"))


def _entry_from_row(row: dict[str, Any], *, before: int, after: int, observed_hash: str, byte_length: int) -> dict[str, Any]:
    source_id = _clean(row.get("source_id"))
    filename = Path(_clean(row.get("canonical_filename"))).name
    return {
        "artifactId": _artifact_id(source_id),
        "sourceIds": [source_id],
        "expectedFilename": filename,
        "expectedSourceContentHash": observed_hash,
        "byteLength": byte_length,
        "provenanceUrl": _provenance_url(source_id, row),
        "license": "external_public_paper_not_redistributed",
        "corpusTier": "local_corpus",
        "notes": (
            f"Priority corpus manifest expansion tranche {before}->{after}; "
            "source bytes reverified locally before registration."
        ),
    }


def _manifest_ids(manifest: dict[str, Any]) -> tuple[set[str], set[str]]:
    source_ids: set[str] = set()
    artifact_ids: set[str] = set()
    for entry in corpus_manifest_entries(manifest):
        artifact_id = _clean(entry.get("artifactId"))
        if artifact_id:
            artifact_ids.add(artifact_id)
        for source_id in entry.get("sourceIds") or []:
            if _clean(source_id):
                source_ids.add(_clean(source_id))
    return source_ids, artifact_ids


def _select_rows(
    allowlist_rows: list[dict[str, Any]],
    *,
    manifest_source_ids: set[str],
    additions_needed: int,
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    skipped_already_manifest = 0
    for row in allowlist_rows:
        source_id = _clean(row.get("source_id"))
        if not source_id:
            continue
        if source_id in manifest_source_ids:
            skipped_already_manifest += 1
            continue
        selected.append(row)
        if len(selected) >= additions_needed:
            break
    return selected, skipped_already_manifest


def build_expansion_tranche_plan(
    *,
    allowlist_path: Path = DEFAULT_ALLOWLIST,
    manifest_path: Path = DEFAULT_MANIFEST,
    papers_dir: Path | None = None,
    target_count: int | None = None,
    batch_size: int = 50,
    apply: bool = False,
    config: Any | None = None,
) -> dict[str, Any]:
    config = config or Config()
    resolved_papers_dir = papers_dir or _default_papers_dir(config)
    allowlist_payload = _load_json(allowlist_path)
    manifest = load_corpus_manifest(manifest_path)
    manifest_entries = corpus_manifest_entries(manifest)
    manifest_source_ids, manifest_artifact_ids = _manifest_ids(manifest)
    before = len(manifest_entries)
    resolved_target = target_count or (before + batch_size)
    additions_needed = max(0, min(batch_size, resolved_target - before))
    selected_rows, skipped_already_manifest = _select_rows(
        list(allowlist_payload.get("allowlist") or []),
        manifest_source_ids=manifest_source_ids,
        additions_needed=additions_needed,
    )
    after = before + len(selected_rows)

    rows: list[dict[str, Any]] = []
    proposed_entries: list[dict[str, Any]] = []
    blocker_rows: list[dict[str, Any]] = []

    for order, row in enumerate(selected_rows, start=1):
        source_id = _clean(row.get("source_id"))
        status = "ready"
        blockers: list[str] = []
        location_ref = _clean(row.get("corpus_location_ref"))
        path = _source_path_from_ref(location_ref, resolved_papers_dir)
        observed_hash = _clean(row.get("observed_sha256"))
        observed_size = int(row.get("byte_length") or 0)
        if not observed_hash.startswith("sha256:"):
            blockers.append("hash_missing")
        if observed_size <= 0:
            blockers.append("byte_length_missing")
        if path is None:
            blockers.append("invalid_corpus_location_ref")
        elif not path.is_file():
            blockers.append("source_missing")
        else:
            actual_hash, actual_size = _sha256_file(path)
            if actual_hash != observed_hash:
                blockers.append("hash_mismatch")
            if actual_size != observed_size:
                blockers.append("byte_length_mismatch")

        proposed_entry = _entry_from_row(
            row,
            before=before,
            after=before + additions_needed,
            observed_hash=observed_hash,
            byte_length=observed_size,
        )
        if proposed_entry["artifactId"] in manifest_artifact_ids:
            blockers.append("duplicate_artifact_id")
        if source_id in manifest_source_ids:
            blockers.append("already_in_manifest")

        resolver_result = inspect_corpus_artifact(
            proposed_entry,
            config=_PapersDirConfig(config, resolved_papers_dir),
        )
        if resolver_result.get("status") != "ok":
            blockers.append(f"resolver_{resolver_result.get('status') or 'blocked'}")

        if blockers:
            status = "blocked"
            blocker_rows.append({"source_id": source_id, "blockers": sorted(set(blockers))})
        else:
            proposed_entries.append(proposed_entry)

        rows.append(
            {
                "order": order,
                "source_id": source_id,
                "title": row.get("title"),
                "candidate_tier": row.get("candidate_tier"),
                "status": status,
                "corpus_location_ref": location_ref,
                "observed_sha256": observed_hash,
                "byte_length": observed_size,
                "resolver_status": resolver_result.get("status"),
                "proposed_manifest_entry": proposed_entry,
                "blockers": sorted(set(blockers)),
            }
        )

    status_counts = Counter(row["status"] for row in rows)
    all_ready = len(rows) == additions_needed and not blocker_rows
    applied = False
    if apply:
        if not all_ready:
            raise ValueError("refusing to apply blocked or incomplete expansion tranche")
        manifest_payload = _load_json(manifest_path)
        manifest_payload.setdefault("schema", "knowledge-hub.corpus-manifest.v1")
        manifest_payload.setdefault("artifacts", [])
        manifest_payload["artifacts"].extend(proposed_entries)
        _write_json(manifest_path, manifest_payload, ensure_ascii=True)
        applied = True

    checks = {
        "manifest_mutation": bool(applied),
        "network_used": False,
        "vault_scan": False,
        "source_bytes_reverified_locally": True,
        "source_missing_excluded": True,
        "ambiguous_excluded": True,
        "papers_dir_ref_only": True,
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "generated_at": _now_iso(),
        "scope_note": (
            "Verified corpus manifest expansion tranche. Selects only current allowlist rows, "
            "recomputes local source hashes, and never promotes source_missing or ambiguous rows."
        ),
        "tranche": {
            "name": f"corpus_manifest_{before}_to_{before + additions_needed}",
            "manifest_rows_before": before,
            "manifest_rows_after_if_applied": before + additions_needed,
            "planned_additions": additions_needed,
            "selection_policy": "first verified allowlist rows not already in corpus_manifest.json",
        },
        "inputs": {
            "expansion_allowlist": _project_ref(allowlist_path),
            "corpus_manifest": _project_ref(manifest_path),
        },
        "checks": checks,
        "counts": {
            "selected_rows": len(rows),
            "ready_rows": int(status_counts.get("ready") or 0),
            "blocked_rows": int(status_counts.get("blocked") or 0),
            "already_in_manifest_skipped_rows": skipped_already_manifest,
            "proposed_manifest_entries": len(proposed_entries),
        },
        "status": "applied" if applied else ("ready" if all_ready else "blocked"),
        "blockers": blocker_rows,
        "proposed_manifest_entries": proposed_entries,
        "rows": rows,
    }
    validation = validate_payload(payload, SCHEMA_ID, strict=True)
    if validation.errors:
        payload["status"] = "blocked"
        payload["schema_violations"] = list(validation.errors)
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    tranche = payload.get("tranche") or {}
    counts = payload.get("counts") or {}
    lines = [
        "# Corpus Manifest Expansion Tranche Plan",
        "",
        f"- status: `{payload.get('status')}`",
        f"- tranche: `{tranche.get('name')}`",
        f"- before: **{tranche.get('manifest_rows_before', 0)}**",
        f"- after if applied: **{tranche.get('manifest_rows_after_if_applied', 0)}**",
        f"- selected rows: **{counts.get('selected_rows', 0)}**",
        f"- ready rows: **{counts.get('ready_rows', 0)}**",
        f"- blocked rows: **{counts.get('blocked_rows', 0)}**",
        "",
        "## Selected Rows",
        "",
    ]
    for row in payload.get("rows") or []:
        lines.append(
            f"- `{row.get('source_id')}` status=`{row.get('status')}` "
            f"tier=`{row.get('candidate_tier')}` ref=`{row.get('corpus_location_ref')}`"
        )
    if not payload.get("rows"):
        lines.append("- none")
    if payload.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        for row in payload.get("blockers") or []:
            lines.append(f"- `{row.get('source_id')}`: {', '.join(row.get('blockers') or [])}")
    lines.append("")
    return "\n".join(lines)


def _default_report_paths(payload: dict[str, Any]) -> tuple[Path, Path]:
    tranche = payload.get("tranche") or {}
    name = _clean(tranche.get("name")) or "corpus_manifest_expansion_tranche"
    report_dir = PROJECT_ROOT / "eval/knowledgeos/reports"
    return (
        report_dir / f"{name}_tranche_plan.v1.json",
        report_dir / f"{name}_tranche_plan.v1.md",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--papers-dir", type=Path)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    payload = build_expansion_tranche_plan(
        allowlist_path=args.allowlist,
        manifest_path=args.manifest,
        papers_dir=args.papers_dir,
        target_count=args.target_count,
        batch_size=args.batch_size,
        apply=args.apply,
    )
    default_report_json, default_report_md = _default_report_paths(payload)
    report_json = args.report_json or default_report_json
    report_md = args.report_md or default_report_md
    _write_json(report_json, payload)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "counts": payload["counts"]}, indent=2))
    return 0 if payload["status"] in {"ready", "applied"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
