"""Report-only inventory of local paper source artifacts under papers_dir.

Scans configured local PDF/text source files, compares them against the public
corpus manifest, and emits schema-backed inventory rows. This helper does not
modify the manifest, download sources, scan vault content, or mutate DB/index
state.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from knowledge_hub.application.corpus_artifacts import (
    DEFAULT_CORPUS_MANIFEST_PATH,
    corpus_entry_ref,
    corpus_manifest_entries,
    load_corpus_manifest,
)
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.corpus_manifest_validation import validate_corpus_manifest


CORPUS_SOURCE_ARTIFACT_INVENTORY_SCHEMA_ID = (
    "knowledge-hub.paper.corpus-source-artifact-inventory.v1"
)

SOURCE_EXTENSIONS = {".pdf": "pdf", ".txt": "text"}
DERIVATIVE_SUBDIRS = frozenset(
    {
        "parsed",
        "structured_evidence",
        "structured_evidence_candidates",
        "summaries",
        "translated",
    }
)
SOURCE_SCAN_REL_DIRS = (
    "",
    "localpdf_pdfs",
    "localpdf_texts",
    "recovered_sources/arxiv",
    "recovered_sources/url",
)
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


class _PapersDirOverrideConfig:
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
        return Path(str(override)).expanduser()
    raw = ""
    if hasattr(config, "get_nested"):
        raw = _clean_text(config.get_nested("storage", "papers_dir", default=""))
    if not raw:
        raw = _clean_text(getattr(config, "papers_dir", ""))
    if not raw:
        return None
    return Path(raw).expanduser()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}", size


def _path_ref(root_label: str, path: Path, root: Path) -> str:
    try:
        return f"{root_label}/{path.relative_to(root).as_posix()}"
    except ValueError:
        return f"{root_label}/{path.name}"


def _candidate_names(entry: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("expectedFilename", "fileName", "filename"):
        text = _clean_text(entry.get(key))
        if text:
            names.append(Path(text).name)
    for key in ("expectedFilenames", "fileNames", "filenames", "pdfCandidates"):
        names.extend(Path(item).name for item in _as_list(entry.get(key)))
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name and name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _source_ids(entry: dict[str, Any]) -> list[str]:
    return _as_list(entry.get("sourceIds")) or _as_list(entry.get("sourceId"))


def _infer_source_id(filename: str, entry: dict[str, Any] | None) -> str:
    if entry is not None:
        source_ids = _source_ids(entry)
        if source_ids:
            return source_ids[0]
    stem = Path(filename).stem
    if _ARXIV_ID_RE.fullmatch(stem):
        return stem
    return stem


def _infer_artifact_id(source_id: str, entry: dict[str, Any] | None) -> str:
    if entry is not None:
        artifact_id = corpus_entry_ref(entry)
        if artifact_id and artifact_id != "unknown":
            return artifact_id
    normalized = source_id.replace(".", "_").replace("-", "_")
    if normalized.startswith("paper_"):
        return normalized
    return f"paper_{normalized}"


def _infer_provenance_url(source_id: str, entry: dict[str, Any] | None) -> str:
    if entry is not None:
        provenance = _clean_text(entry.get("provenanceUrl"))
        if provenance:
            return provenance
    if _ARXIV_ID_RE.fullmatch(source_id):
        return f"https://arxiv.org/pdf/{source_id}"
    return ""


def _manifest_indexes(manifest: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_filename: dict[str, list[dict[str, Any]]] = {}
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for entry in corpus_manifest_entries(manifest):
        for name in _candidate_names(entry):
            by_filename.setdefault(name, []).append(entry)
        expected_hash = _normalize_hash(entry.get("expectedSourceContentHash"))
        if expected_hash:
            by_hash.setdefault(expected_hash, []).append(entry)
    return by_filename, by_hash


def _iter_source_files(papers_dir: Path) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    for rel_dir in SOURCE_SCAN_REL_DIRS:
        if rel_dir:
            if rel_dir in DERIVATIVE_SUBDIRS:
                continue
            scan_root = papers_dir / rel_dir
        else:
            scan_root = papers_dir
        if not scan_root.is_dir():
            continue
        for path in sorted(scan_root.iterdir()):
            if not path.is_file():
                continue
            source_type = SOURCE_EXTENSIONS.get(path.suffix.lower())
            if source_type is None:
                continue
            discovered.append((_path_ref("papers_dir", path, papers_dir), path))
    return discovered


def _registration_status(
    *,
    observed_hash: str,
    matches: list[dict[str, Any]],
) -> tuple[str, bool, list[str]]:
    warnings: list[str] = []
    if not matches:
        return "unregistered_available", False, warnings
    if len(matches) > 1:
        warnings.append("duplicate_manifest_filename")
    entry = matches[0]
    expected_hash = _normalize_hash(entry.get("expectedSourceContentHash"))
    if not expected_hash:
        return "already_registered", True, warnings + ["manifest_expected_hash_missing"]
    if expected_hash != observed_hash:
        return "hash_mismatch_registered", True, warnings + ["manifest_hash_mismatch"]
    return "already_registered", True, warnings


def build_corpus_source_artifact_inventory(
    *,
    config: Any,
    manifest_path: str | Path | None = None,
    papers_dir: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_corpus_manifest(manifest_path)
    resolved_papers_dir = _configured_papers_dir(config, papers_dir)
    by_filename, by_hash = _manifest_indexes(manifest)

    items: list[dict[str, Any]] = []
    if resolved_papers_dir is None or not resolved_papers_dir.is_dir():
        status = "blocked"
        scan_error = "configured papers_dir is unavailable"
    else:
        status = "ok"
        scan_error = ""
        for corpus_location_ref, path in _iter_source_files(resolved_papers_dir):
            source_type = SOURCE_EXTENSIONS[path.suffix.lower()]
            observed_hash, byte_length = _sha256_file(path)
            filename = path.name
            filename_matches = list(by_filename.get(filename) or [])
            hash_matches = list(by_hash.get(observed_hash) or [])
            entry = filename_matches[0] if filename_matches else (hash_matches[0] if len(hash_matches) == 1 else None)
            source_id = _infer_source_id(filename, entry)
            registration_status, manifest_registered, warnings = _registration_status(
                observed_hash=observed_hash,
                matches=filename_matches,
            )
            if not filename_matches and len(hash_matches) == 1:
                registration_status = "already_registered"
                manifest_registered = True
                entry = hash_matches[0]
                source_id = _infer_source_id(filename, entry)
            elif not filename_matches and len(hash_matches) > 1:
                registration_status = "ambiguous"
                manifest_registered = False
                warnings.append("duplicate_manifest_hash")
            item = {
                "sourceId": source_id,
                "inferredArtifactId": _infer_artifact_id(source_id, entry),
                "filename": filename,
                "corpusLocationRef": corpus_location_ref,
                "sourceType": source_type,
                "byteLength": byte_length,
                "sha256": observed_hash,
                "provenanceUrl": _infer_provenance_url(source_id, entry),
                "manifestRegistered": manifest_registered,
                "registrationStatus": registration_status,
                "manifestArtifactId": corpus_entry_ref(entry) if entry is not None else "",
                "warnings": sorted(set(warnings)),
            }
            items.append(item)

    validation = validate_corpus_manifest(
        config=_PapersDirOverrideConfig(config, resolved_papers_dir) if resolved_papers_dir else config,
        manifest_path=manifest_path,
        papers_dir=resolved_papers_dir,
        check_artifacts=True,
        check_parsed=False,
    )
    registration_counts = Counter(item["registrationStatus"] for item in items)
    payload: dict[str, Any] = {
        "schema": CORPUS_SOURCE_ARTIFACT_INVENTORY_SCHEMA_ID,
        "status": status if scan_error else "ok",
        "generatedAt": _now_iso(),
        "manifestRef": Path(
            str(manifest.get("_manifestPath") or manifest_path or DEFAULT_CORPUS_MANIFEST_PATH)
        ).name,
        "papersDirConfigured": resolved_papers_dir is not None,
        "checks": {
            "networkUsed": False,
            "databaseMutation": False,
            "indexMutation": False,
            "vaultScan": False,
            "manifestMutation": False,
            "sourceRegistrationMutation": False,
            "parsedArtifactWrite": False,
            "derivativeSubdirsExcluded": sorted(DERIVATIVE_SUBDIRS),
            "scanRoots": [f"papers_dir/{rel or ''}".rstrip("/") for rel in SOURCE_SCAN_REL_DIRS],
        },
        "counts": {
            "inventoryRows": len(items),
            "alreadyRegisteredRows": int(registration_counts.get("already_registered") or 0),
            "unregisteredAvailableRows": int(registration_counts.get("unregistered_available") or 0),
            "hashMismatchRegisteredRows": int(registration_counts.get("hash_mismatch_registered") or 0),
            "ambiguousRows": int(registration_counts.get("ambiguous") or 0),
            "manifestRows": int((validation.get("counts") or {}).get("manifestRows") or 0),
            "manifestSourceAvailableRows": int((validation.get("counts") or {}).get("sourceAvailableRows") or 0),
            "manifestSourceMissingRows": int((validation.get("counts") or {}).get("sourceMissingRows") or 0),
            "manifestHashMissingRows": int((validation.get("counts") or {}).get("hashMissingRows") or 0),
            "manifestHashMismatchRows": int((validation.get("counts") or {}).get("hashMismatchRows") or 0),
            "manifestMetadataOnlyRows": int((validation.get("counts") or {}).get("metadataOnlyRows") or 0),
            "schemaViolationCount": 0,
        },
        "manifestValidationStatus": validation.get("status"),
        "items": items,
        "schemaViolations": [],
    }
    if scan_error:
        payload["status"] = "blocked"
        payload["scanError"] = scan_error
    validation_result = validate_payload(payload, CORPUS_SOURCE_ARTIFACT_INVENTORY_SCHEMA_ID, strict=True)
    if not validation_result.ok:
        payload["status"] = "blocked"
        payload["counts"]["schemaViolationCount"] = len(validation_result.errors)
        payload["schemaViolations"] = list(validation_result.errors)
    return payload


def render_corpus_source_artifact_inventory_markdown(payload: dict[str, Any]) -> str:
    counts = dict(payload.get("counts") or {})
    lines = [
        "# Corpus Source Artifact Inventory",
        "",
        f"- status: `{payload.get('status')}`",
        f"- manifest: `{payload.get('manifestRef', '')}`",
        f"- inventory rows: `{counts.get('inventoryRows', 0)}`",
        f"- already registered: `{counts.get('alreadyRegisteredRows', 0)}`",
        f"- unregistered available: `{counts.get('unregisteredAvailableRows', 0)}`",
        f"- hash mismatch registered: `{counts.get('hashMismatchRegisteredRows', 0)}`",
        f"- manifest source missing: `{counts.get('manifestSourceMissingRows', 0)}`",
        f"- manifest hash missing: `{counts.get('manifestHashMissingRows', 0)}`",
        f"- manifest hash mismatch: `{counts.get('manifestHashMismatchRows', 0)}`",
        "",
        "## Unregistered Available Candidates",
    ]
    unregistered = [
        item
        for item in list(payload.get("items") or [])
        if item.get("registrationStatus") == "unregistered_available"
    ]
    if not unregistered:
        lines.append("- none")
    else:
        for item in unregistered[:50]:
            lines.append(
                f"- `{item.get('filename')}` sourceId=`{item.get('sourceId')}` "
                f"sha256=`{item.get('sha256')}` bytes=`{item.get('byteLength')}`"
            )
        if len(unregistered) > 50:
            lines.append(f"- ... and {len(unregistered) - 50} more")
    return "\n".join(lines) + "\n"


__all__ = [
    "CORPUS_SOURCE_ARTIFACT_INVENTORY_SCHEMA_ID",
    "build_corpus_source_artifact_inventory",
    "render_corpus_source_artifact_inventory_markdown",
]
