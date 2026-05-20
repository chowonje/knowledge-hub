"""Apply-gated URL source recovery for parsed artifact coverage blockers.

The helper recovers only URL-identified source PDFs whose canonical PDF URL can
be reconstructed conservatively from existing registry metadata.  Recovered
PDFs are written under ``papers_dir/recovered_sources/url`` before the paper
registry is updated to that local source path.

It does not write to the previously registered external/vault path, scan or
write the vault, materialize parsed artifacts, route parsers, create evidence,
or touch indexes/reembedding.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from urllib import request
from urllib.parse import urlparse

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_coverage_batch_report import (
    DEFAULT_MAX_SOURCE_PDF_BYTES,
    _counter_items,
    _safe_relative,
)
from knowledge_hub.papers.parsed_artifact_source_recovery_feasibility import (
    build_parsed_artifact_source_recovery_feasibility,
)


PARSED_ARTIFACT_URL_SOURCE_RECOVERY_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-url-source-recovery.v1"
)

DEFAULT_URL_SOURCE_RECOVERY_LIMIT = 10
DownloadFn = Callable[[str, Path, float], dict[str, Any]]
ProbeFn = Callable[[str, float], dict[str, Any]]

_ACL_ANTHOLOGY_ID_RE = re.compile(r"^\d{4}\.[A-Za-z0-9-]+\.\d+[A-Za-z]?$")
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_pdf(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return False
    with path.open("rb") as handle:
        return b"%PDF" in handle.read(1024)


def _download_pdf(url: str, target_path: Path, timeout_seconds: float) -> dict[str, Any]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    request_obj = request.Request(
        url,
        headers={
            "User-Agent": "knowledge-hub parsed-artifact-url-source-recovery/1.0",
        },
    )
    with tempfile.NamedTemporaryFile(
        prefix=f".{target_path.name}.",
        suffix=".download",
        dir=str(target_path.parent),
        delete=False,
    ) as temp_handle:
        temp_path = Path(temp_handle.name)
        try:
            with request.urlopen(request_obj, timeout=timeout_seconds) as response:  # noqa: S310 - URL is allowlist-resolved.
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temp_handle.write(chunk)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    if not _looks_like_pdf(temp_path):
        temp_path.unlink(missing_ok=True)
        raise ValueError("downloaded artifact is not a PDF")
    temp_path.replace(target_path)
    return {
        "sizeBytes": target_path.stat().st_size,
        "sha256": _sha256_file(target_path),
    }


def _probe_url(url: str, timeout_seconds: float) -> dict[str, Any]:
    request_obj = request.Request(
        url,
        method="HEAD",
        headers={
            "User-Agent": "knowledge-hub parsed-artifact-url-source-recovery/1.0",
        },
    )
    with request.urlopen(request_obj, timeout=timeout_seconds) as response:  # noqa: S310 - URL is allowlist-resolved.
        content_type = str(response.headers.get("Content-Type") or "")
        content_length = str(response.headers.get("Content-Length") or "0")
        try:
            size_bytes = int(content_length)
        except ValueError:
            size_bytes = 0
        return {
            "ok": True,
            "statusCode": int(getattr(response, "status", 0) or 0),
            "contentType": content_type,
            "contentLength": size_bytes,
            "looksLikePdf": "pdf" in content_type.casefold() and size_bytes > 0,
        }


def _download_error_result(error: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "errorType": type(error).__name__,
        "error": _clean_text(error),
    }


def _target_path(*, papers_dir: str | Path, paper_id: str) -> Path:
    safe_name = _SAFE_FILENAME_RE.sub("_", _clean_text(paper_id)).strip("._-") or "paper"
    return Path(str(papers_dir)).expanduser() / "recovered_sources" / "url" / f"{safe_name}.pdf"


def _safe_source_artifact(path: Path, *, papers_dir: str | Path) -> dict[str, Any]:
    return {
        "kind": "pdf",
        "exists": bool(path.exists()),
        "isFile": bool(path.exists() and path.is_file()),
        "sizeBytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "path": _safe_relative(path, root=papers_dir),
    }


def _old_source_artifact(row: dict[str, Any], *, papers_dir: str | Path) -> dict[str, Any]:
    raw_pdf = _clean_text(row.get("pdf_path") or row.get("pdfPath"))
    path = Path(raw_pdf).expanduser() if raw_pdf else None
    return {
        "kind": "pdf" if raw_pdf else "",
        "exists": bool(path and path.exists()),
        "isFile": bool(path and path.is_file()) if path and path.exists() else False,
        "sizeBytes": path.stat().st_size if path and path.exists() and path.is_file() else 0,
        "path": _safe_relative(path, root=papers_dir) if path else "",
    }


def _select_url_candidates(feasibility_report: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in list(feasibility_report.get("sourceMissingRows") or [])
        if _clean_text(dict(row).get("reacquisitionStatus")) == "url_identifier_reacquisition_candidate"
    ]
    rows.sort(key=lambda item: _clean_text(item.get("paperId")))
    return rows[: max(0, int(limit or 0))] if limit else []


def _acl_pdf_url_from_metadata(candidate: dict[str, Any], row: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    paper_id = _clean_text(candidate.get("paperId"))
    title = _clean_text(candidate.get("paperTitle") or (row or {}).get("title"))
    registered_pdf_name = Path(_clean_text((row or {}).get("pdf_path"))).name
    registered_pdf_stem = registered_pdf_name.removesuffix(".pdf") if registered_pdf_name.endswith(".pdf") else ""
    acl_id = title if _ACL_ANTHOLOGY_ID_RE.match(title) else registered_pdf_stem
    if paper_id.startswith("https___aclanthology_org_") and _ACL_ANTHOLOGY_ID_RE.match(acl_id):
        return (
            f"https://aclanthology.org/{acl_id}.pdf",
            {
                "method": "acl_anthology_id_from_title_or_registered_pdf_name",
                "sourceId": acl_id,
                "confidence": "high",
                "reason": "paper_id_host_and_acl_identifier_match",
            },
        )
    return (
        "",
        {
            "method": "unresolved",
            "sourceId": "",
            "confidence": "none",
            "reason": "unsupported_url_identifier_shape",
        },
    )


def _is_allowed_source_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.netloc == "aclanthology.org" and parsed.path.endswith(".pdf")


def _source_url_presence(*, resolved: bool, probe_network: bool, probe_result: dict[str, Any]) -> str:
    if not resolved:
        return "unresolved"
    if not probe_network:
        return "resolved_unprobed"
    if probe_result.get("ok") is True and probe_result.get("looksLikePdf") is True:
        return "confirmed_pdf"
    return "probe_failed_or_not_pdf"


def _mutation_policy(*, apply: bool, probe_network: bool) -> dict[str, Any]:
    return {
        "applyRequested": bool(apply),
        "applyPerformed": bool(apply),
        "sourceUrlProbe": bool(probe_network),
        "sourceDownload": bool(apply),
        "sourcePathRewrite": bool(apply),
        "sourceRegistrationMutation": bool(apply),
        "parsedArtifactWrite": False,
        "parserRouting": False,
        "strictEvidence": False,
        "citationEvidence": False,
        "runtimeEvidence": False,
        "sourceSpanCreated": False,
        "databaseMutation": bool(apply),
        "indexMutation": False,
        "reindexOrReembed": False,
        "vaultScan": False,
        "vaultWrite": False,
        "answerIntegration": False,
        "manualBlockerResolution": False,
    }


def _mutation_counters(items: list[dict[str, Any]]) -> dict[str, int]:
    downloaded = sum(1 for item in items if item.get("downloadPerformed") is True)
    registered = sum(1 for item in items if item.get("sourceRegistrationMutationPerformed") is True)
    return {
        "parsedArtifactWriteRows": 0,
        "sourceSpanCreatedRows": 0,
        "strictEvidenceRows": 0,
        "citationEvidenceRows": 0,
        "runtimeEvidenceRows": 0,
        "databaseMutationRows": registered,
        "indexMutationRows": 0,
        "reembedRows": 0,
        "vaultReadRows": 0,
        "vaultWriteRows": 0,
        "answerIntegrationRows": 0,
        "parserRoutingRows": 0,
        "manualBlockerResolutionRows": 0,
        "sourceDownloadRows": downloaded,
        "sourcePathRewriteRows": registered,
        "sourceRegistrationMutationRows": registered,
    }


def _status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    statuses = [_clean_text(item.get("status")) for item in items]
    return {
        "planned": statuses.count("planned"),
        "recovered": statuses.count("recovered"),
        "blocked": statuses.count("blocked"),
        "failed": statuses.count("failed"),
        "skippedExistingRegistered": statuses.count("skipped_existing_registered"),
    }


def _selected_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [_clean_text(row.get("paperId")) for row in rows if _clean_text(row.get("paperId"))]


def _overall_status(counts: dict[str, int], *, apply: bool) -> str:
    if counts.get("failed", 0):
        return "partial" if counts.get("recovered", 0) else "failed"
    if counts.get("blocked", 0):
        return "partial" if counts.get("recovered", 0) or counts.get("planned", 0) else "blocked"
    if apply:
        return "applied" if counts.get("recovered", 0) or counts.get("skippedExistingRegistered", 0) else "blocked"
    return "ready" if counts.get("planned", 0) else "blocked"


def _item_for_candidate(
    candidate: dict[str, Any],
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    apply: bool,
    overwrite: bool,
    probe_network: bool,
    timeout_seconds: float,
    download_fn: DownloadFn,
    probe_fn: ProbeFn,
) -> dict[str, Any]:
    paper_id = _clean_text(candidate.get("paperId"))
    row = sqlite_db.get_paper(paper_id) if hasattr(sqlite_db, "get_paper") else None
    target = _target_path(papers_dir=papers_dir, paper_id=paper_id)
    source_url, resolution = _acl_pdf_url_from_metadata(candidate, row if isinstance(row, dict) else None)
    resolved = bool(source_url and _is_allowed_source_url(source_url))
    probe_result: dict[str, Any] = {}
    if resolved and probe_network:
        try:
            probe_result = probe_fn(source_url, timeout_seconds)
        except Exception as error:
            probe_result = _download_error_result(error)
    base = {
        "paperId": paper_id,
        "paperTitle": _clean_text(candidate.get("paperTitle") or (row.get("title") if isinstance(row, dict) else "")),
        "sourceUrl": source_url,
        "sourceUrlResolution": resolution,
        "sourceUrlProbeAttempted": bool(resolved and probe_network),
        "sourceUrlProbeResult": probe_result,
        "sourceUrlPresence": _source_url_presence(
            resolved=resolved,
            probe_network=probe_network,
            probe_result=probe_result,
        ),
        "targetSourceArtifact": _safe_source_artifact(target, papers_dir=papers_dir),
        "previousSourceArtifact": _old_source_artifact(row, papers_dir=papers_dir) if isinstance(row, dict) else {
            "kind": "",
            "exists": False,
            "isFile": False,
            "sizeBytes": 0,
            "path": "",
        },
        "downloadAttempted": False,
        "downloadPerformed": False,
        "sourceRegistrationMutationAttempted": False,
        "sourceRegistrationMutationPerformed": False,
        "vaultWriteAttempted": False,
        "databaseMutationAttempted": False,
        "databaseMutationPerformed": False,
        "downloadResult": {},
    }
    if not isinstance(row, dict) or not row:
        return {
            **base,
            "status": "blocked",
            "reason": "paper_not_registered",
            "action": "none",
        }
    if not resolved:
        return {
            **base,
            "status": "blocked",
            "reason": "source_url_unresolved_or_not_allowlisted",
            "action": "none",
        }
    if probe_network and base["sourceUrlPresence"] != "confirmed_pdf":
        return {
            **base,
            "status": "blocked",
            "reason": "source_url_probe_failed_or_not_pdf",
            "action": "none",
        }
    if target.exists() and not overwrite and _clean_text(row.get("pdf_path")) == str(target):
        return {
            **base,
            "status": "skipped_existing_registered",
            "reason": "target_source_already_registered",
            "action": "none",
        }
    if not apply:
        return {
            **base,
            "status": "planned",
            "reason": "apply_required",
            "action": "download_to_recovered_sources_and_register_pdf_path",
        }
    if target.exists() and not overwrite and not _looks_like_pdf(target):
        return {
            **base,
            "status": "blocked",
            "reason": "target_exists_but_not_pdf",
            "action": "none",
        }
    try:
        downloaded = False
        download_result: dict[str, Any]
        if target.exists() and not overwrite and _looks_like_pdf(target):
            download_result = {
                "sizeBytes": target.stat().st_size,
                "sha256": _sha256_file(target),
                "reusedExistingTarget": True,
            }
        else:
            download_result = download_fn(source_url, target, timeout_seconds)
            if not _looks_like_pdf(target):
                raise ValueError("downloaded artifact is not a PDF")
            downloaded = True
        download_result = {**download_result, "ok": True}
        updated = dict(row)
        updated["pdf_path"] = str(target)
        sqlite_db.upsert_paper(updated)
        return {
            **base,
            "status": "recovered",
            "reason": "ok",
            "action": "download_to_recovered_sources_and_register_pdf_path",
            "targetSourceArtifact": _safe_source_artifact(target, papers_dir=papers_dir),
            "downloadAttempted": downloaded,
            "downloadPerformed": downloaded,
            "sourceRegistrationMutationAttempted": True,
            "sourceRegistrationMutationPerformed": True,
            "databaseMutationAttempted": True,
            "databaseMutationPerformed": True,
            "downloadResult": download_result,
        }
    except Exception as error:
        return {
            **base,
            "status": "failed",
            "reason": f"source_recovery_failed:{type(error).__name__}:{_clean_text(error)}",
            "action": "download_to_recovered_sources_and_register_pdf_path",
            "downloadAttempted": True,
            "downloadPerformed": False,
            "sourceRegistrationMutationAttempted": False,
            "sourceRegistrationMutationPerformed": False,
            "databaseMutationAttempted": False,
            "databaseMutationPerformed": False,
            "downloadResult": _download_error_result(error),
        }


def build_parsed_artifact_url_source_recovery(
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    report_name: str = "parsed-artifact-url-source-recovery",
    limit: int = DEFAULT_URL_SOURCE_RECOVERY_LIMIT,
    max_source_pdf_bytes: int = DEFAULT_MAX_SOURCE_PDF_BYTES,
    apply: bool = False,
    overwrite: bool = False,
    probe_network: bool = False,
    timeout_seconds: float = 30.0,
    delay_seconds: float = 0.0,
    generated_at: str | None = None,
    baseline_command_report: dict[str, Any] | None = None,
    baseline_command_report_path: str | Path | None = None,
    sqlite_hash_before_cli: str = "",
    sqlite_hash_after_cli: str = "",
    download_fn: DownloadFn | None = None,
    probe_fn: ProbeFn | None = None,
) -> dict[str, Any]:
    """Plan or apply a bounded URL PDF recovery tranche."""

    effective_limit = max(0, int(limit or 0))
    effective_max_bytes = max(0, int(max_source_pdf_bytes or 0))
    effective_delay_seconds = max(0.0, float(delay_seconds or 0.0))
    effective_download = download_fn or _download_pdf
    effective_probe = probe_fn or _probe_url
    feasibility = build_parsed_artifact_source_recovery_feasibility(
        sqlite_db=sqlite_db,
        papers_dir=papers_dir,
        report_name=f"{report_name}-input-source-recovery-feasibility",
        max_source_pdf_bytes=effective_max_bytes,
        generated_at=generated_at,
        baseline_command_report=baseline_command_report,
        baseline_command_report_path=baseline_command_report_path,
        sqlite_hash_before_cli=sqlite_hash_before_cli,
        sqlite_hash_after_cli=sqlite_hash_after_cli,
    )
    candidates = _select_url_candidates(feasibility, limit=effective_limit)
    items: list[dict[str, Any]] = []
    for candidate in candidates:
        item = _item_for_candidate(
            candidate,
            sqlite_db=sqlite_db,
            papers_dir=papers_dir,
            apply=apply,
            overwrite=overwrite,
            probe_network=probe_network,
            timeout_seconds=timeout_seconds,
            download_fn=effective_download,
            probe_fn=effective_probe,
        )
        items.append(item)
        if (
            apply
            and effective_delay_seconds
            and item.get("downloadAttempted") is True
            and item.get("downloadPerformed") is True
        ):
            time.sleep(effective_delay_seconds)
    counts = _status_counts(items)
    recovered_ids = [
        _clean_text(item.get("paperId"))
        for item in items
        if _clean_text(item.get("status")) in {"recovered", "skipped_existing_registered"}
    ]
    payload = {
        "schema": PARSED_ARTIFACT_URL_SOURCE_RECOVERY_SCHEMA_ID,
        "status": _overall_status(counts, apply=apply),
        "generatedAt": generated_at or _utc_now(),
        "report": {
            "name": report_name,
            "limit": effective_limit,
            "maxSourcePdfBytes": effective_max_bytes,
            "selectionRule": "url_identifier_reacquisition_candidate rows sorted by paperId ascending, bounded by limit; currently allowlists ACL Anthology PDF identifiers only",
            "targetRoot": "papers_dir/recovered_sources/url",
            "applyRequested": bool(apply),
            "overwrite": bool(overwrite),
            "probeNetwork": bool(probe_network),
            "timeoutSeconds": float(timeout_seconds),
            "delaySeconds": effective_delay_seconds,
            "nonScope": [
                "vault_scan_or_write",
                "write_to_previous_external_or_vault_pdf_path",
                "parsed_artifact_write",
                "parser_routing",
                "strict_or_citation_or_runtime_evidence",
                "source_span_creation",
                "index_or_reembed",
                "answer_integration",
                "manual_blocker_resolution",
                "unsupported_url_manual_resolution",
            ],
        },
        "baseline": dict(feasibility.get("baseline") or {}),
        "inputRecoverySummary": {
            "sourcePdfMissingRows": int(feasibility.get("sourceMissingRecoverySummary", {}).get("sourcePdfMissingRows") or 0),
            "arxivPdfReacquisitionCandidateRows": int(
                (feasibility.get("reacquisitionCandidatePaperIdsByStatus", {}).get("arxiv_pdf_reacquisition_candidate") or {}).get("count")
                or 0
            ),
            "urlIdentifierReacquisitionCandidateRows": int(
                (feasibility.get("reacquisitionCandidatePaperIdsByStatus", {}).get("url_identifier_reacquisition_candidate") or {}).get("count")
                or 0
            ),
            "manualLookupRequiredRows": int(feasibility.get("sourceMissingRecoverySummary", {}).get("manualLookupRequiredRows") or 0),
        },
        "selectedCandidatePaperIds": _selected_ids(candidates),
        "unselectedCandidatePaperIds": _selected_ids(
            _select_url_candidates(feasibility, limit=10_000_000)[effective_limit:]
        )
        if effective_limit
        else _selected_ids(_select_url_candidates(feasibility, limit=10_000_000)),
        "recoveredPaperIds": recovered_ids,
        "items": items,
        "counts": counts,
        "statusTaxonomy": _counter_items(Counter(_clean_text(item.get("status")) for item in items)),
        "sourceUrlPresenceTaxonomy": _counter_items(Counter(_clean_text(item.get("sourceUrlPresence")) for item in items)),
        "expectedCoverageChangeIfMaterialized": {
            "missingParsedArtifactsBefore": int(feasibility.get("baseline", {}).get("missingParsedArtifacts") or 0),
            "sourceRecoveredRows": len(recovered_ids),
            "expectedMissingParsedArtifactsReductionAfterSeparateMaterialization": len(recovered_ids),
            "expectedMissingParsedArtifactsAfterSeparateMaterialization": max(
                0,
                int(feasibility.get("baseline", {}).get("missingParsedArtifacts") or 0) - len(recovered_ids),
            ),
        },
        "nextRecommendedTranche": {
            "name": "parsed-artifact-materialization-for-recovered-url-sources",
            "candidateCount": len(recovered_ids),
            "paperIds": recovered_ids,
            "rationale": "run parsed materialization only for successfully recovered and registered local URL PDFs",
        },
        "mutationPolicy": _mutation_policy(apply=apply, probe_network=probe_network),
        "mutationCounters": _mutation_counters(items),
        "warnings": list(feasibility.get("warnings") or []),
    }
    validation = validate_payload(payload, PARSED_ARTIFACT_URL_SOURCE_RECOVERY_SCHEMA_ID, strict=True)
    if not validation.ok:
        raise ValueError(
            "parsed artifact URL source recovery schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    return payload


def write_parsed_artifact_url_source_recovery(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write JSON, Markdown, and recovered-ID artifacts."""

    validation = validate_payload(report, PARSED_ARTIFACT_URL_SOURCE_RECOVERY_SCHEMA_ID, strict=True)
    if not validation.ok:
        raise ValueError(
            "parsed artifact URL source recovery schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-url-source-recovery.json"
    summary_path = root / "parsed-artifact-url-source-recovery.md"
    selected_ids_path = root / "selected-url-source-recovery-paper-ids.txt"
    recovered_ids_path = root / "recovered-url-source-paper-ids.txt"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_ids_path.write_text("\n".join(report.get("selectedCandidatePaperIds") or []) + "\n", encoding="utf-8")
    recovered_ids_path.write_text("\n".join(report.get("recoveredPaperIds") or []) + "\n", encoding="utf-8")
    summary_path.write_text(_render_markdown_summary(report), encoding="utf-8")
    return {
        "reportJsonPath": str(report_path),
        "reportMarkdownPath": str(summary_path),
        "selectedCandidateIdsPath": str(selected_ids_path),
        "recoveredPaperIdsPath": str(recovered_ids_path),
    }


def _render_markdown_summary(report: dict[str, Any]) -> str:
    baseline = dict(report.get("baseline") or {})
    counts = dict(report.get("counts") or {})
    expected = dict(report.get("expectedCoverageChangeIfMaterialized") or {})
    lines = [
        "# Parsed Artifact URL Source Recovery",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- apply requested: {dict(report.get('report') or {}).get('applyRequested')}",
        f"- network probe: {dict(report.get('report') or {}).get('probeNetwork')}",
        f"- baseline scannedPapers: {baseline.get('scannedPapers', 0)}",
        f"- baseline missingParsedArtifacts: {baseline.get('missingParsedArtifacts', 0)}",
        f"- selected candidates: {len(report.get('selectedCandidatePaperIds') or [])}",
        f"- recovered sources: {len(report.get('recoveredPaperIds') or [])}",
        f"- planned: {counts.get('planned', 0)}",
        f"- failed: {counts.get('failed', 0)}",
        f"- expected missingParsedArtifacts reduction after separate materialization: {expected.get('expectedMissingParsedArtifactsReductionAfterSeparateMaterialization', 0)}",
        "",
        "## Selected Paper IDs",
        "",
    ]
    lines.extend(f"- `{paper_id}`" for paper_id in list(report.get("selectedCandidatePaperIds") or []))
    lines.extend(["", "## Recovered Paper IDs", ""])
    lines.extend(f"- `{paper_id}`" for paper_id in list(report.get("recoveredPaperIds") or []))
    lines.extend(["", "## Source URL Presence", ""])
    for item in list(report.get("sourceUrlPresenceTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Status Taxonomy", ""])
    for item in list(report.get("statusTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Mutation Counters", ""])
    for key, value in sorted(dict(report.get("mutationCounters") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_URL_SOURCE_RECOVERY_LIMIT",
    "PARSED_ARTIFACT_URL_SOURCE_RECOVERY_SCHEMA_ID",
    "build_parsed_artifact_url_source_recovery",
    "write_parsed_artifact_url_source_recovery",
]
