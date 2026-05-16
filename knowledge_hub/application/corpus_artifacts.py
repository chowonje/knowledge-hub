"""Local corpus artifact manifest helpers.

The manifest describes source artifacts that may exist in the operator's
configured local corpus. It is metadata only: no network fetches, no DB writes,
and no evidence promotion happen here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_MANIFEST_PATH = REPO_ROOT / "eval" / "knowledgeos" / "fixtures" / "corpus_manifest.json"
CORPUS_MANIFEST_SCHEMA = "knowledge-hub.corpus-manifest.v1"
REPO_FIXTURE_TIER = "repo_fixture"
LOCAL_CORPUS_TIERS = {"local_corpus", "optional_local_corpus"}


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


def _configured_papers_dir(config: Any) -> Path | None:
    raw = ""
    if hasattr(config, "get_nested"):
        raw = _clean_text(config.get_nested("storage", "papers_dir", default=""))
    if not raw:
        raw = _clean_text(getattr(config, "papers_dir", ""))
    if not raw:
        return None
    return Path(raw).expanduser()


def _manifest_fixture_root(entry: dict[str, Any]) -> Path:
    manifest_path = _clean_text(entry.get("_manifestPath"))
    if manifest_path:
        return Path(manifest_path).expanduser().parent
    return DEFAULT_CORPUS_MANIFEST_PATH.parent


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


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}", size


def _path_ref(root_label: str, path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return f"{root_label}/{path.relative_to(root).as_posix()}"
        except ValueError:
            pass
    return f"{root_label}/{path.name}"


def _manifest_schema(payload: dict[str, Any]) -> str:
    return _clean_text(payload.get("schema")) or CORPUS_MANIFEST_SCHEMA


def load_corpus_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path or DEFAULT_CORPUS_MANIFEST_PATH).expanduser()
    if not manifest_path.exists():
        return {"schema": CORPUS_MANIFEST_SCHEMA, "artifacts": [], "_manifestPath": str(manifest_path)}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"schema": CORPUS_MANIFEST_SCHEMA, "artifacts": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"corpus manifest must be an object or list: {manifest_path}")
    schema = _manifest_schema(payload)
    if schema != CORPUS_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported corpus manifest schema {schema!r}: {manifest_path}")
    payload["schema"] = schema
    payload.setdefault("artifacts", [])
    payload["_manifestPath"] = str(manifest_path)
    return payload


def corpus_manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    manifest_path = _clean_text(manifest.get("_manifestPath"))
    for item in list(manifest.get("artifacts") or []):
        if not isinstance(item, dict):
            continue
        entry = dict(item or {})
        if manifest_path:
            entry["_manifestPath"] = manifest_path
        entries.append(entry)
    return entries


def corpus_entry_ref(entry: dict[str, Any]) -> str:
    artifact_id = _clean_text(entry.get("artifactId") or entry.get("id"))
    return artifact_id or _clean_text(entry.get("sourceId")) or "unknown"


def find_corpus_entry_for_source(source_id: str, manifest: dict[str, Any]) -> dict[str, Any] | None:
    wanted = _clean_text(source_id)
    if not wanted:
        return None
    wanted_folded = wanted.casefold()
    for entry in corpus_manifest_entries(manifest):
        aliases = [
            _clean_text(entry.get("sourceId")),
            *_as_list(entry.get("sourceIds")),
            *_as_list(entry.get("aliases")),
        ]
        if any(alias.casefold() == wanted_folded for alias in aliases if alias):
            return entry
    return None


def find_corpus_entry_for_artifact(artifact_id: str, manifest: dict[str, Any]) -> dict[str, Any] | None:
    wanted = _clean_text(artifact_id).casefold()
    if not wanted:
        return None
    for entry in corpus_manifest_entries(manifest):
        if corpus_entry_ref(entry).casefold() == wanted:
            return entry
    return None


def inspect_corpus_artifact(entry: dict[str, Any], *, config: Any) -> dict[str, Any]:
    tier = _clean_text(entry.get("corpusTier")) or "local_corpus"
    candidate_names = _candidate_names(entry)
    searched_paths: list[str] = []
    expected_hash = _normalize_hash(entry.get("expectedSourceContentHash"))
    expected_byte_length = entry.get("byteLength")
    base = {
        "artifactId": corpus_entry_ref(entry),
        "sourceIds": _as_list(entry.get("sourceIds")) or _as_list(entry.get("sourceId")),
        "corpusTier": tier,
        "expectedSourceContentHash": expected_hash,
        "expectedByteLength": expected_byte_length,
        "manifestEntryRef": corpus_entry_ref(entry),
        "searchedPaths": searched_paths,
        "minOffsetsRequired": bool(entry.get("minOffsetsRequired")),
    }
    search_roots: list[tuple[str, Path | None, list[str]]] = []
    if tier == REPO_FIXTURE_TIER:
        fixture_root = _manifest_fixture_root(entry)
        fixture_paths = _as_list(entry.get("fixturePath") or entry.get("repoFixturePath"))
        search_roots.append(("repo_fixture", fixture_root, fixture_paths or candidate_names))
    elif tier in LOCAL_CORPUS_TIERS:
        papers_dir = _configured_papers_dir(config)
        search_roots.append(("papers_dir", papers_dir, candidate_names))
        base["papersDirConfigured"] = papers_dir is not None
    else:
        return {**base, "status": "missing_artifact", "reason": f"unsupported corpus tier: {tier}"}
    if not any(root for _label, root, _names in search_roots):
        return {**base, "status": "missing_artifact", "reason": "configured corpus root is unavailable"}
    if not any(names for _label, _root, names in search_roots):
        return {**base, "status": "missing_artifact", "reason": "corpus manifest entry has no candidate filename"}
    for root_label, root, names in search_roots:
        if root is None:
            continue
        for name in names:
            raw_path = Path(name)
            candidate = raw_path if raw_path.is_absolute() else root / raw_path
            path_ref = _path_ref(root_label, candidate, root)
            searched_paths.append(path_ref)
            if not candidate.is_file():
                continue
            observed_hash, observed_size = _sha256_file(candidate)
            if expected_hash and observed_hash != expected_hash:
                return {
                    **base,
                    "status": "hash_mismatch",
                    "path": path_ref,
                    "filename": candidate.name,
                    "observedSourceContentHash": observed_hash,
                    "observedByteLength": observed_size,
                    "reason": "local corpus artifact hash does not match manifest",
                    "_resolvedPath": str(candidate),
                }
            return {
                **base,
                "status": "ok",
                "path": path_ref,
                "filename": candidate.name,
                "observedSourceContentHash": observed_hash,
                "observedByteLength": observed_size,
                "_resolvedPath": str(candidate),
            }
    return {**base, "status": "missing_artifact", "reason": "local corpus artifact was not found"}


def public_corpus_artifact_diagnostic(artifact: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict(artifact or {}).items() if not str(key).startswith("_")}


def inspect_corpus_requirement(
    requirement: dict[str, Any],
    *,
    manifest: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    artifact_id = _clean_text(requirement.get("artifactId"))
    entry = find_corpus_entry_for_artifact(artifact_id, manifest) if artifact_id else None
    if entry is None:
        return {
            "status": "missing_manifest_entry",
            "artifactId": artifact_id,
            "manifestEntryRef": artifact_id,
            "corpusTier": _clean_text(requirement.get("corpusTier")) or "local_corpus",
            "reason": "corpus requirement has no matching manifest entry",
        }
    merged = dict(entry)
    for key in ("corpusTier", "minOffsetsRequired"):
        if requirement.get(key) not in (None, ""):
            merged[key] = requirement[key]
    return inspect_corpus_artifact(merged, config=config)
