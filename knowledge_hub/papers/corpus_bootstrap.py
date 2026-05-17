"""Explicit paper corpus artifact acquisition helper.

This module is intentionally operator-triggered only. It reads the corpus
manifest, downloads selected missing local corpus artifacts when explicitly
allowed, verifies manifest hashes before promotion, and does not write SQLite
rows or rebuild downstream derivatives.
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from knowledge_hub.application.corpus_artifacts import (
    DEFAULT_CORPUS_MANIFEST_PATH,
    LOCAL_CORPUS_TIERS,
    REPO_FIXTURE_TIER,
    corpus_entry_ref,
    corpus_manifest_entries,
    inspect_corpus_artifact,
    load_corpus_manifest,
    public_corpus_artifact_diagnostic,
)


SCHEMA_ID = "knowledge-hub.corpus-bootstrap.result.v1"
SUCCESS_STATUSES = {"already_present", "planned_download", "downloaded", "skipped_repo_fixture"}


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


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, tuple):
        return [_clean_text(item) for item in value if _clean_text(item)]
    text = _clean_text(value)
    return [text] if text else []


def _normalize_hash(value: Any) -> str:
    text = _clean_text(value).lower()
    if text and not text.startswith("sha256:"):
        text = f"sha256:{text}"
    return text


def _configured_papers_dir(config: Any, override: str | Path | None = None) -> Path | None:
    if override not in (None, ""):
        return Path(override).expanduser()
    raw = ""
    if hasattr(config, "get_nested"):
        raw = _clean_text(config.get_nested("storage", "papers_dir", default=""))
    if not raw:
        raw = _clean_text(getattr(config, "papers_dir", ""))
    if not raw:
        return None
    return Path(raw).expanduser()


def _candidate_names(entry: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("expectedFilename", "fileName", "filename"):
        text = _clean_text(entry.get(key))
        if text:
            names.append(text)
    for key in ("expectedFilenames", "fileNames", "filenames", "pdfCandidates"):
        names.extend(_as_list(entry.get(key)))
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        basename = Path(name).name
        if basename and basename not in seen:
            seen.add(basename)
            deduped.append(basename)
    return deduped


def _source_ids(entry: dict[str, Any]) -> list[str]:
    return _as_list(entry.get("sourceIds")) or _as_list(entry.get("sourceId"))


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}", size


def _expected_byte_length(entry: dict[str, Any]) -> int | None:
    raw = entry.get("byteLength")
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _safe_target_ref(filename: str) -> str:
    return f"papers_dir/{Path(filename).name}"


def _repair_hints(entry: dict[str, Any]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for source_id in _source_ids(entry):
        hints.append(
            {
                "sourceId": source_id,
                "command": ["khub", "paper", "repair-source", "--paper-id", source_id, "--dry-run", "--json"],
            }
        )
    return hints


def _entry_summary(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifactId": corpus_entry_ref(entry),
        "sourceIds": _source_ids(entry),
        "corpusTier": _clean_text(entry.get("corpusTier")) or "local_corpus",
        "expectedSourceContentHash": _normalize_hash(entry.get("expectedSourceContentHash")),
        "expectedByteLength": entry.get("byteLength"),
        "provenanceUrl": _clean_text(entry.get("provenanceUrl")),
    }


def _download_to_path(url: str, target: Path, *, timeout: float) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _valid_download_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _select_entries(
    entries: list[dict[str, Any]],
    *,
    artifact_ids: list[str],
    source_ids: list[str],
    all_artifacts: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if all_artifacts:
        return entries, []

    wanted_artifacts = {_clean_text(item).casefold(): _clean_text(item) for item in artifact_ids if _clean_text(item)}
    wanted_sources = {_clean_text(item).casefold(): _clean_text(item) for item in source_ids if _clean_text(item)}
    selected: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    matched_artifacts: set[str] = set()
    matched_sources: set[str] = set()
    for entry in entries:
        ref = corpus_entry_ref(entry)
        ref_key = ref.casefold()
        entry_sources = {source.casefold(): source for source in _source_ids(entry)}
        match_artifact = ref_key in wanted_artifacts
        match_sources = set(entry_sources) & set(wanted_sources)
        if match_artifact:
            matched_artifacts.add(ref_key)
        matched_sources.update(match_sources)
        if (match_artifact or match_sources) and ref_key not in seen_refs:
            seen_refs.add(ref_key)
            selected.append(entry)

    errors: list[dict[str, Any]] = []
    for key, original in sorted(wanted_artifacts.items()):
        if key not in matched_artifacts:
            errors.append(
                {
                    "selectorType": "artifactId",
                    "selector": original,
                    "status": "missing_manifest_entry",
                    "reason": "artifact id has no corpus manifest entry",
                }
            )
    for key, original in sorted(wanted_sources.items()):
        if key not in matched_sources:
            errors.append(
                {
                    "selectorType": "sourceId",
                    "selector": original,
                    "status": "missing_manifest_entry",
                    "reason": "source id has no corpus manifest entry",
                }
            )
    return selected, errors


def _bootstrap_entry(
    entry: dict[str, Any],
    *,
    config: Any,
    papers_dir: Path | None,
    apply: bool,
    allow_network: bool,
    timeout: float,
) -> dict[str, Any]:
    summary = _entry_summary(entry)
    tier = str(summary["corpusTier"])
    if tier == REPO_FIXTURE_TIER:
        return {
            **summary,
            "status": "skipped_repo_fixture",
            "reason": "repo fixture artifacts are checked into the repository and are not acquired",
        }
    if tier not in LOCAL_CORPUS_TIERS:
        return {
            **summary,
            "status": "unsupported_corpus_tier",
            "reason": f"unsupported corpus tier: {tier}",
        }
    if papers_dir is None:
        return {
            **summary,
            "status": "papers_dir_unconfigured",
            "reason": "configured papers_dir is unavailable",
        }
    if not summary["expectedSourceContentHash"]:
        return {
            **summary,
            "status": "missing_expected_hash",
            "reason": "corpus manifest entry has no expectedSourceContentHash",
        }

    current = public_corpus_artifact_diagnostic(inspect_corpus_artifact(entry, config=config))
    if current.get("status") == "ok":
        return {
            **summary,
            "status": "already_present",
            "artifact": current,
            "targetPath": current.get("path"),
            "repairHints": _repair_hints(entry),
        }
    if current.get("status") == "hash_mismatch":
        return {
            **summary,
            "status": "hash_mismatch",
            "reason": "existing local corpus artifact hash does not match manifest; not replacing automatically",
            "artifact": current,
            "targetPath": current.get("path"),
        }

    candidate_names = _candidate_names(entry)
    if not candidate_names:
        return {
            **summary,
            "status": "missing_filename",
            "reason": "corpus manifest entry has no expected filename",
            "artifact": current,
        }
    filename = candidate_names[0]
    target_path = papers_dir / filename
    target_ref = _safe_target_ref(filename)
    provenance_url = str(summary["provenanceUrl"])
    if not provenance_url:
        return {
            **summary,
            "status": "missing_provenance_url",
            "reason": "corpus manifest entry has no provenanceUrl",
            "artifact": current,
            "targetPath": target_ref,
        }
    if not _valid_download_url(provenance_url):
        return {
            **summary,
            "status": "unsupported_provenance_url",
            "reason": "provenanceUrl must be http or https for acquisition",
            "artifact": current,
            "targetPath": target_ref,
        }
    if not apply:
        return {
            **summary,
            "status": "planned_download",
            "reason": "dry run; pass --apply --allow-network to acquire",
            "artifact": current,
            "targetPath": target_ref,
            "repairHints": _repair_hints(entry),
        }
    if not allow_network:
        return {
            **summary,
            "status": "network_not_allowed",
            "reason": "network acquisition requires --allow-network",
            "artifact": current,
            "targetPath": target_ref,
        }

    papers_dir.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target_path.name}.",
            suffix=".download",
            dir=papers_dir,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        _download_to_path(provenance_url, temp_path, timeout=timeout)
        observed_hash, observed_size = _sha256_file(temp_path)
        expected_hash = str(summary["expectedSourceContentHash"])
        expected_size = _expected_byte_length(entry)
        if expected_hash and observed_hash != expected_hash:
            return {
                **summary,
                "status": "hash_mismatch",
                "reason": "downloaded corpus artifact hash does not match manifest",
                "artifact": current,
                "targetPath": target_ref,
                "observedSourceContentHash": observed_hash,
                "observedByteLength": observed_size,
            }
        if expected_size is not None and observed_size != expected_size:
            return {
                **summary,
                "status": "byte_length_mismatch",
                "reason": "downloaded corpus artifact byte length does not match manifest",
                "artifact": current,
                "targetPath": target_ref,
                "observedSourceContentHash": observed_hash,
                "observedByteLength": observed_size,
            }
        temp_path.replace(target_path)
        temp_path = None
        verified = public_corpus_artifact_diagnostic(inspect_corpus_artifact(entry, config=config))
        if verified.get("status") != "ok":
            return {
                **summary,
                "status": "post_write_verification_failed",
                "reason": "downloaded artifact was written but did not pass manifest verification",
                "artifact": verified,
                "targetPath": target_ref,
            }
        return {
            **summary,
            "status": "downloaded",
            "artifact": verified,
            "targetPath": target_ref,
            "observedSourceContentHash": observed_hash,
            "observedByteLength": observed_size,
            "repairHints": _repair_hints(entry),
        }
    except requests.RequestException as exc:
        return {
            **summary,
            "status": "download_failed",
            "reason": f"download failed: {exc}",
            "artifact": current,
            "targetPath": target_ref,
        }
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def bootstrap_corpus_artifacts(
    *,
    config: Any,
    manifest_path: str | Path | None = None,
    papers_dir: str | Path | None = None,
    artifact_ids: list[str] | tuple[str, ...] | None = None,
    source_ids: list[str] | tuple[str, ...] | None = None,
    all_artifacts: bool = False,
    apply: bool = False,
    allow_network: bool = False,
    timeout: float = 60.0,
) -> dict[str, Any]:
    manifest = load_corpus_manifest(manifest_path)
    entries = corpus_manifest_entries(manifest)
    selected, selection_errors = _select_entries(
        entries,
        artifact_ids=list(artifact_ids or []),
        source_ids=list(source_ids or []),
        all_artifacts=bool(all_artifacts),
    )
    resolved_papers_dir = _configured_papers_dir(config, papers_dir)
    inspection_config = _PapersDirConfig(config, resolved_papers_dir) if resolved_papers_dir is not None else config
    if not selected and not selection_errors:
        selection_errors.append(
            {
                "selectorType": "selection",
                "selector": "",
                "status": "no_artifacts_selected",
                "reason": "select artifacts with --artifact-id, --source-id, or --all",
            }
        )
    items = [
        _bootstrap_entry(
            entry,
            config=inspection_config,
            papers_dir=resolved_papers_dir,
            apply=bool(apply),
            allow_network=bool(allow_network),
            timeout=float(timeout),
        )
        for entry in selected
    ]
    blocked_count = sum(1 for item in items if item.get("status") not in SUCCESS_STATUSES)
    status = "ok" if blocked_count == 0 and not selection_errors else "blocked"
    manifest_ref = str(Path(manifest.get("_manifestPath") or manifest_path or DEFAULT_CORPUS_MANIFEST_PATH).name)
    return {
        "schema": SCHEMA_ID,
        "generatedAt": _now_iso(),
        "status": status,
        "dryRun": not bool(apply),
        "networkAllowed": bool(allow_network),
        "manifestRef": manifest_ref,
        "selectedCount": len(selected),
        "counts": {
            "ok": len(items) - blocked_count,
            "blocked": blocked_count + len(selection_errors),
            "downloaded": sum(1 for item in items if item.get("status") == "downloaded"),
            "alreadyPresent": sum(1 for item in items if item.get("status") == "already_present"),
            "plannedDownload": sum(1 for item in items if item.get("status") == "planned_download"),
            "skippedRepoFixture": sum(1 for item in items if item.get("status") == "skipped_repo_fixture"),
        },
        "selectionErrors": selection_errors,
        "items": items,
    }
