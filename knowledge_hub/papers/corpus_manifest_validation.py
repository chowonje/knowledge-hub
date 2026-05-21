"""Corpus manifest validation for paper source artifacts.

This helper is report-only. It checks the repo-controlled manifest against the
configured local corpus without downloading, repairing sources, rebuilding
derivatives, indexing, or creating evidence.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

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
from knowledge_hub.core.schema_validator import validate_payload


CORPUS_MANIFEST_VALIDATION_SCHEMA_ID = "knowledge-hub.paper.corpus-manifest-validation.v1"
SUPPORTED_TIERS = set(LOCAL_CORPUS_TIERS) | {REPO_FIXTURE_TIER}
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


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


def _parsed_artifact_status(
    *,
    papers_dir: Path | None,
    source_ids: list[str],
    source_artifact_status: str,
) -> dict[str, Any]:
    if source_artifact_status != "available":
        return {
            "status": "source_unavailable",
            "sourceId": source_ids[0] if source_ids else "",
            "manifestRef": "",
        }
    if papers_dir is None or not source_ids:
        return {
            "status": "not_checked",
            "sourceId": source_ids[0] if source_ids else "",
            "manifestRef": "",
        }
    source_id = source_ids[0]
    manifest_path = papers_dir / "parsed" / source_id / "manifest.json"
    manifest_ref = f"papers_dir/parsed/{source_id}/manifest.json"
    return {
        "status": "available" if manifest_path.is_file() else "parsed_missing",
        "sourceId": source_id,
        "manifestRef": manifest_ref,
    }


def _classify_entry(
    entry: dict[str, Any],
    *,
    config: Any,
    papers_dir: Path | None,
    check_artifacts: bool,
    check_parsed: bool,
    duplicate_artifact: bool,
    duplicate_sources: set[str],
) -> dict[str, Any]:
    artifact_id = corpus_entry_ref(entry)
    source_ids = _source_ids(entry)
    tier = _clean_text(entry.get("corpusTier")) or "local_corpus"
    expected_hash = _normalize_hash(entry.get("expectedSourceContentHash"))
    candidate_names = _candidate_names(entry)
    blockers: list[str] = []
    warnings: list[str] = []

    if duplicate_artifact:
        blockers.append("duplicate_artifact_id")
    for source_id in source_ids:
        if source_id in duplicate_sources:
            blockers.append("duplicate_source_id")
            break
    if not artifact_id or artifact_id == "unknown":
        blockers.append("artifact_id_missing")
    if not source_ids:
        blockers.append("source_id_missing")
    if tier not in SUPPORTED_TIERS:
        blockers.append("unsupported_corpus_tier")
    if expected_hash and not _SHA256_RE.fullmatch(expected_hash):
        blockers.append("expected_source_content_hash_invalid")

    if not expected_hash and not candidate_names:
        source_status = "metadata_only"
        blockers.append("metadata_only_no_source_artifact")
        warnings.append("metadata_only_no_source_artifact_declared")
        artifact = {}
    elif not expected_hash:
        source_status = "hash_missing"
        blockers.append("expected_source_content_hash_missing")
        artifact = {}
    elif not candidate_names:
        source_status = "source_missing"
        blockers.append("expected_filename_missing")
        artifact = {}
    elif not check_artifacts:
        source_status = "not_checked"
        artifact = {
            "expectedSourceContentHash": expected_hash,
            "expectedFilename": candidate_names[0],
        }
    else:
        artifact = public_corpus_artifact_diagnostic(inspect_corpus_artifact(entry, config=config))
        inspection_status = _clean_text(artifact.get("status"))
        if inspection_status == "ok":
            source_status = "available"
        elif inspection_status == "hash_mismatch":
            source_status = "hash_mismatch"
            blockers.append("source_content_hash_mismatch")
        elif inspection_status == "missing_artifact":
            source_status = "source_missing"
            blockers.append("source_artifact_missing")
        else:
            source_status = inspection_status or "source_missing"
            blockers.append(source_status)

    parsed = {"status": "not_checked", "sourceId": source_ids[0] if source_ids else "", "manifestRef": ""}
    if check_parsed:
        parsed = _parsed_artifact_status(
            papers_dir=papers_dir,
            source_ids=source_ids,
            source_artifact_status=source_status,
        )

    return {
        "artifactId": artifact_id,
        "sourceIds": source_ids,
        "corpusTier": tier,
        "expectedFilename": candidate_names[0] if candidate_names else "",
        "expectedSourceContentHash": expected_hash,
        "expectedByteLength": entry.get("byteLength"),
        "provenanceUrl": _clean_text(entry.get("provenanceUrl")),
        "sourceArtifactStatus": source_status,
        "artifact": artifact,
        "parsedArtifactStatus": parsed["status"],
        "parsedArtifact": parsed,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
    }


def validate_corpus_manifest(
    *,
    config: Any,
    manifest_path: str | Path | None = None,
    papers_dir: str | Path | None = None,
    check_artifacts: bool = True,
    check_parsed: bool = True,
) -> dict[str, Any]:
    manifest = load_corpus_manifest(manifest_path)
    entries = corpus_manifest_entries(manifest)
    resolved_papers_dir = _configured_papers_dir(config, papers_dir)
    inspection_config = (
        _PapersDirOverrideConfig(config, resolved_papers_dir)
        if resolved_papers_dir is not None
        else config
    )

    artifact_refs = [corpus_entry_ref(entry) for entry in entries]
    artifact_counts = Counter(ref for ref in artifact_refs if ref)
    source_counts = Counter(source_id for entry in entries for source_id in _source_ids(entry))
    duplicate_sources = {source_id for source_id, count in source_counts.items() if count > 1}

    items = [
        _classify_entry(
            entry,
            config=inspection_config,
            papers_dir=resolved_papers_dir,
            check_artifacts=check_artifacts,
            check_parsed=check_parsed,
            duplicate_artifact=artifact_counts[corpus_entry_ref(entry)] > 1,
            duplicate_sources=duplicate_sources,
        )
        for entry in entries
    ]

    source_counts_by_status = Counter(item["sourceArtifactStatus"] for item in items)
    parsed_counts_by_status = Counter(item["parsedArtifactStatus"] for item in items)
    blocker_rows = [item for item in items if item.get("blockers")]
    status = "blocked" if blocker_rows else "ok"
    payload: dict[str, Any] = {
        "schema": CORPUS_MANIFEST_VALIDATION_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "manifestRef": Path(
            str(manifest.get("_manifestPath") or manifest_path or DEFAULT_CORPUS_MANIFEST_PATH)
        ).name,
        "checks": {
            "artifactsChecked": bool(check_artifacts),
            "parsedArtifactsChecked": bool(check_parsed),
            "networkUsed": False,
            "databaseMutation": False,
            "indexMutation": False,
            "vaultScan": False,
            "sourceRegistrationMutation": False,
            "parsedArtifactWrite": False,
            "evidencePromotion": False,
        },
        "counts": {
            "manifestRows": len(items),
            "sourceAvailableRows": int(source_counts_by_status.get("available") or 0),
            "sourceMissingRows": int(source_counts_by_status.get("source_missing") or 0),
            "metadataOnlyRows": int(source_counts_by_status.get("metadata_only") or 0),
            "hashMissingRows": int(source_counts_by_status.get("hash_missing") or 0),
            "hashMismatchRows": int(source_counts_by_status.get("hash_mismatch") or 0),
            "sourceNotCheckedRows": int(source_counts_by_status.get("not_checked") or 0),
            "parsedAvailableRows": int(parsed_counts_by_status.get("available") or 0),
            "parsedMissingRows": int(parsed_counts_by_status.get("parsed_missing") or 0),
            "parsedSourceUnavailableRows": int(parsed_counts_by_status.get("source_unavailable") or 0),
            "parsedNotCheckedRows": int(parsed_counts_by_status.get("not_checked") or 0),
            "duplicateArtifactIdRows": sum(1 for ref in artifact_refs if artifact_counts[ref] > 1),
            "duplicateSourceIdRows": sum(
                1 for item in items if any(source in duplicate_sources for source in item["sourceIds"])
            ),
            "blockerRows": len(blocker_rows),
            "schemaViolationCount": 0,
        },
        "sourceCoverageStatus": "complete" if not blocker_rows else "blocked",
        "parsedCoverageStatus": (
            "not_checked"
            if not check_parsed
            else "complete"
            if parsed_counts_by_status.get("parsed_missing", 0) == 0
            and parsed_counts_by_status.get("source_unavailable", 0) == 0
            else "incomplete"
        ),
        "items": items,
        "schemaViolations": [],
    }
    validation = validate_payload(payload, CORPUS_MANIFEST_VALIDATION_SCHEMA_ID, strict=True)
    if not validation.ok:
        payload["status"] = "blocked"
        payload["counts"]["schemaViolationCount"] = len(validation.errors)
        payload["schemaViolations"] = list(validation.errors)
    return payload


def render_corpus_manifest_validation_markdown(payload: dict[str, Any]) -> str:
    counts = dict(payload.get("counts") or {})
    lines = [
        "# Corpus Manifest Validation",
        "",
        f"- status: `{payload.get('status')}`",
        f"- manifest rows: `{counts.get('manifestRows', 0)}`",
        f"- source available: `{counts.get('sourceAvailableRows', 0)}`",
        f"- source missing: `{counts.get('sourceMissingRows', 0)}`",
        f"- hash missing: `{counts.get('hashMissingRows', 0)}`",
        f"- hash mismatch: `{counts.get('hashMismatchRows', 0)}`",
        f"- parsed available: `{counts.get('parsedAvailableRows', 0)}`",
        f"- parsed missing: `{counts.get('parsedMissingRows', 0)}`",
        "",
        "## Blockers",
    ]
    blockers = [item for item in list(payload.get("items") or []) if item.get("blockers")]
    if not blockers:
        lines.append("- none")
    else:
        for item in blockers:
            lines.append(
                f"- `{item.get('artifactId')}` source=`{item.get('sourceArtifactStatus')}` "
                f"parsed=`{item.get('parsedArtifactStatus')}` blockers={json.dumps(item.get('blockers') or [])}"
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "CORPUS_MANIFEST_VALIDATION_SCHEMA_ID",
    "render_corpus_manifest_validation_markdown",
    "validate_corpus_manifest",
]
