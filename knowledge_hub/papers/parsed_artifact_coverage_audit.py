"""Report-only parsed artifact coverage audit for eval-critical paper corpora."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.application.corpus_artifacts import (
    corpus_entry_ref,
    corpus_manifest_entries,
    inspect_corpus_artifact,
    public_corpus_artifact_diagnostic,
)
from knowledge_hub.papers.extraction_diagnostics import load_paper_diagnostic


PARSED_ARTIFACT_COVERAGE_AUDIT_SCHEMA_ID = "knowledge-hub.paper.parsed-artifact-coverage-audit.v1"
DEFAULT_INCLUDED_CORPUS_TIERS = ("local_corpus", "repo_fixture")
_MISSING_PARSED_REASONS = {"parsed_artifact_missing", "parsed_document_missing"}
_UNKNOWN_PARSED_REASONS = {"parsed_artifact_unreadable"}


class _PapersDirConfig:
    def __init__(self, papers_dir: str | Path):
        self.papers_dir = str(papers_dir)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        if keys == ("storage", "papers_dir"):
            return self.papers_dir
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_clean_text(item) for item in value if _clean_text(item)]
    text = _clean_text(value)
    return [text] if text else []


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_hash(value: Any) -> str:
    text = _clean_text(value).lower()
    if text and not text.startswith("sha256:"):
        text = f"sha256:{text}"
    return text


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}", size


def _safe_relative(path: str | Path, *, root: str | Path, prefix: str = "papers_dir") -> str:
    candidate = Path(str(path)).expanduser()
    root_path = Path(str(root)).expanduser()
    try:
        rel = candidate.resolve().relative_to(root_path.resolve())
        return str(Path(prefix) / rel)
    except Exception:
        return str(Path("external") / (candidate.name or "source"))


def _source_artifact_for_paper(
    paper: dict[str, Any],
    *,
    papers_dir: str | Path,
    expected_hash: str,
    expected_byte_length: int | None,
) -> dict[str, Any]:
    raw_pdf = _clean_text(paper.get("pdf_path") or paper.get("pdfPath"))
    raw_text = _clean_text(paper.get("text_path") or paper.get("textPath"))
    source_path: Path | None = None
    source_kind = ""
    if raw_pdf:
        source_path = Path(raw_pdf).expanduser()
        source_kind = "pdf"
    elif raw_text:
        source_path = Path(raw_text).expanduser()
        source_kind = "text"

    base: dict[str, Any] = {
        "kind": source_kind,
        "exists": bool(source_path and source_path.is_file()),
        "path": _safe_relative(source_path, root=papers_dir) if source_path else "",
        "expectedSourceContentHash": expected_hash,
        "expectedByteLength": expected_byte_length,
    }
    if not source_path:
        return {**base, "status": "missing_source", "reason": "registered_source_pdf_missing"}
    if source_kind != "pdf":
        return {**base, "status": "missing_source", "reason": "registered_source_pdf_required"}
    if not source_path.is_file():
        return {**base, "status": "missing_source", "reason": "registered_source_pdf_missing"}

    try:
        observed_hash, observed_size = _sha256_file(source_path)
    except Exception:
        return {**base, "status": "unknown_status", "reason": "registered_source_hash_unreadable"}

    result = {
        **base,
        "filename": source_path.name,
        "observedSourceContentHash": observed_hash,
        "observedByteLength": observed_size,
    }
    if expected_hash and observed_hash != expected_hash:
        return {**result, "status": "hash_mismatch", "reason": "registered_source_hash_mismatch"}
    if expected_byte_length is not None and observed_size != expected_byte_length:
        return {**result, "status": "hash_mismatch", "reason": "registered_source_byte_length_mismatch"}
    return {**result, "status": "ok", "reason": "registered_source_pdf_ok"}


def _paper_id_from_row(row: dict[str, Any], fallback: str = "") -> str:
    return _clean_text(row.get("arxiv_id") or row.get("paper_id") or row.get("paperId") or fallback)


def _lookup_registered_paper(sqlite_db: Any, source_ids: list[str]) -> tuple[dict[str, Any], str]:
    if not hasattr(sqlite_db, "get_paper"):
        return {}, ""
    for source_id in source_ids:
        row = sqlite_db.get_paper(source_id)
        if isinstance(row, dict) and row:
            return row, source_id
    return {}, ""


def _parsed_status(parsed_diagnostic: dict[str, Any]) -> tuple[str, list[str]]:
    diagnostic = dict(parsed_diagnostic.get("diagnostic") or {})
    reasons = [_clean_text(item) for item in list(diagnostic.get("degradationReasons") or []) if _clean_text(item)]
    reason_set = set(reasons)
    if reason_set & _MISSING_PARSED_REASONS:
        return "missing_parsed", reasons
    if reason_set & _UNKNOWN_PARSED_REASONS:
        return "unknown_status", reasons
    if bool(diagnostic.get("extractionDegraded")):
        return "degraded", reasons
    return "ok", reasons


def _coverage_status(
    *,
    corpus_artifact: dict[str, Any],
    paper_registered: bool,
    registered_source_artifact: dict[str, Any],
    parsed_status: str,
) -> str:
    corpus_status = _clean_text(corpus_artifact.get("status"))
    source_status = _clean_text(registered_source_artifact.get("status"))
    if corpus_status == "hash_mismatch" or source_status == "hash_mismatch":
        return "hash_mismatch"
    if corpus_status in {"missing_artifact", "missing_source"}:
        return "missing_source"
    if not paper_registered:
        return "unknown_status"
    if source_status == "missing_source":
        return "missing_source"
    if source_status == "unknown_status" or corpus_status not in {"", "ok"}:
        return "unknown_status"
    if parsed_status == "missing_parsed":
        return "missing_parsed"
    if parsed_status == "unknown_status":
        return "unknown_status"
    return "ready"


def _next_action(status: str, *, corpus_artifact: dict[str, Any], registered_source_artifact: dict[str, Any]) -> str:
    if status == "hash_mismatch":
        return "resolve_source_hash_mismatch_before_materialization"
    if status == "missing_source":
        if _clean_text(corpus_artifact.get("status")) == "ok":
            return "attach_registered_source_pdf_before_materialization"
        return "recover_required_local_source_artifact_before_materialization"
    if status == "missing_parsed":
        return "run_explicit_materialize_parsed_dry_run_for_source_ready_paper"
    if status == "unknown_status":
        if not _clean_text(registered_source_artifact.get("status")):
            return "resolve_paper_registration_or_manifest_mapping"
        return "inspect_unknown_source_or_parsed_artifact_status"
    return "none"


def _next_bottleneck(counts: dict[str, int]) -> str:
    if counts.get("hashMismatch", 0):
        return "source_hash_mismatch_blocks_trustworthy_parsed_materialization"
    if counts.get("missingSource", 0):
        return "missing_or_unattached_source_pdfs_block_parsed_materialization"
    if counts.get("missingParsed", 0):
        return "source_ready_papers_need_explicit_materialize_parsed_dry_run"
    if counts.get("unknownStatus", 0):
        return "registration_or_artifact_status_unknown_blocks_coverage_claim"
    if counts.get("parsedDegraded", 0):
        return "coverage_ready_but_parser_quality_degradation_remains"
    return "none"


def _entry_matches_filter(entry: dict[str, Any], paper_ids: set[str]) -> bool:
    if not paper_ids:
        return True
    aliases = {
        _clean_text(entry.get("artifactId")),
        _clean_text(entry.get("sourceId")),
        *_as_list(entry.get("sourceIds")),
        *_as_list(entry.get("aliases")),
    }
    return bool({item.casefold() for item in aliases if item} & paper_ids)


def build_parsed_artifact_coverage_audit(
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    corpus_manifest: dict[str, Any],
    paper_ids: list[str] | tuple[str, ...] | None = None,
    included_corpus_tiers: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Join corpus source requirements with registered sources and parsed artifacts."""

    requested_ids = [_clean_text(item) for item in list(paper_ids or []) if _clean_text(item)]
    requested_filter = {item.casefold() for item in requested_ids}
    included_tiers = tuple(included_corpus_tiers or DEFAULT_INCLUDED_CORPUS_TIERS)
    included_tier_set = {item.casefold() for item in included_tiers if _clean_text(item)}

    entries = [
        entry
        for entry in corpus_manifest_entries(corpus_manifest)
        if (_clean_text(entry.get("corpusTier")) or "local_corpus").casefold() in included_tier_set
        and _entry_matches_filter(entry, requested_filter)
    ]

    items: list[dict[str, Any]] = []
    warnings: list[str] = []
    corpus_config = _PapersDirConfig(papers_dir)
    for entry in entries:
        source_ids = _as_list(entry.get("sourceIds")) or _as_list(entry.get("sourceId"))
        artifact_id = corpus_entry_ref(entry)
        corpus_artifact = public_corpus_artifact_diagnostic(inspect_corpus_artifact(entry, config=corpus_config))
        expected_hash = _normalize_hash(entry.get("expectedSourceContentHash"))
        expected_byte_length = _safe_int(entry.get("byteLength"))
        paper, matched_source_id = _lookup_registered_paper(sqlite_db, source_ids)
        paper_registered = bool(paper)
        paper_id = _paper_id_from_row(paper, matched_source_id or (source_ids[0] if source_ids else ""))
        parsed_diagnostic: dict[str, Any] = {}
        parsed_status = "unknown_status"
        parsed_reasons: list[str] = []
        registered_source_artifact: dict[str, Any] = {}
        if paper_registered:
            registered_source_artifact = _source_artifact_for_paper(
                paper,
                papers_dir=papers_dir,
                expected_hash=expected_hash,
                expected_byte_length=expected_byte_length,
            )
            parsed_diagnostic = load_paper_diagnostic(paper=paper, papers_dir=papers_dir)
            parsed_status, parsed_reasons = _parsed_status(parsed_diagnostic)
        else:
            registered_source_artifact = {
                "status": "",
                "kind": "",
                "exists": False,
                "path": "",
                "reason": "paper_not_registered",
                "expectedSourceContentHash": expected_hash,
                "expectedByteLength": expected_byte_length,
            }
            parsed_reasons = ["paper_not_registered"]

        coverage_status = _coverage_status(
            corpus_artifact=corpus_artifact,
            paper_registered=paper_registered,
            registered_source_artifact=registered_source_artifact,
            parsed_status=parsed_status,
        )
        blockers = []
        for reason in (
            _clean_text(corpus_artifact.get("reason")),
            _clean_text(registered_source_artifact.get("reason")),
            *parsed_reasons,
        ):
            if reason and reason not in blockers and (
                coverage_status != "ready" or reason in _MISSING_PARSED_REASONS or reason in _UNKNOWN_PARSED_REASONS
            ):
                blockers.append(reason)
        items.append(
            {
                "artifactId": artifact_id,
                "sourceIds": source_ids,
                "corpusTier": _clean_text(entry.get("corpusTier")) or "local_corpus",
                "paperId": paper_id,
                "paperTitle": _clean_text(paper.get("title") or paper.get("paperTitle")) if paper_registered else "",
                "paperRegistered": paper_registered,
                "matchedSourceId": matched_source_id,
                "coverageStatus": coverage_status,
                "sourceStatus": registered_source_artifact.get("status") or "unknown_status",
                "parsedStatus": parsed_status,
                "parsedDegraded": parsed_status == "degraded",
                "corpusArtifact": corpus_artifact,
                "registeredSourceArtifact": registered_source_artifact,
                "parsedDiagnostic": parsed_diagnostic,
                "blockers": blockers,
                "recommendedNextAction": _next_action(
                    coverage_status,
                    corpus_artifact=corpus_artifact,
                    registered_source_artifact=registered_source_artifact,
                ),
            }
        )

    status_counts = Counter(str(item.get("coverageStatus") or "unknown_status") for item in items)
    parsed_degraded_count = sum(1 for item in items if bool(item.get("parsedDegraded")))
    counts = {
        "totalCorpusArtifacts": len(entries),
        "registeredPapers": sum(1 for item in items if bool(item.get("paperRegistered"))),
        "ready": status_counts.get("ready", 0),
        "missingSource": status_counts.get("missing_source", 0),
        "missingParsed": status_counts.get("missing_parsed", 0),
        "hashMismatch": status_counts.get("hash_mismatch", 0),
        "unknownStatus": status_counts.get("unknown_status", 0),
        "parsedDegraded": parsed_degraded_count,
    }
    return {
        "schema": PARSED_ARTIFACT_COVERAGE_AUDIT_SCHEMA_ID,
        "status": "ready",
        "generatedAt": _now_iso(),
        "scope": "eval_critical_corpus_manifest",
        "request": {
            "paperIds": requested_ids,
            "includedCorpusTiers": list(included_tiers),
            "reportOnly": True,
        },
        "safety": {
            "vaultScan": False,
            "externalDownload": False,
            "dbMutation": False,
            "indexMutation": False,
            "reindex": False,
            "reembed": False,
            "canonicalParsedArtifactWrite": False,
            "privatePathLeakAllowed": False,
        },
        "counts": counts,
        "nextParsedCoverageBottleneck": _next_bottleneck(counts),
        "items": items,
        "warnings": warnings,
    }


def render_parsed_artifact_coverage_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact Coverage Audit",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Scope: `{report.get('scope')}`",
        f"- Total corpus artifacts: `{counts.get('totalCorpusArtifacts', 0)}`",
        f"- Ready: `{counts.get('ready', 0)}`",
        f"- Missing source: `{counts.get('missingSource', 0)}`",
        f"- Missing parsed: `{counts.get('missingParsed', 0)}`",
        f"- Hash mismatch: `{counts.get('hashMismatch', 0)}`",
        f"- Unknown status: `{counts.get('unknownStatus', 0)}`",
        f"- Parsed degraded: `{counts.get('parsedDegraded', 0)}`",
        f"- Next parsed coverage bottleneck: `{report.get('nextParsedCoverageBottleneck')}`",
        "",
        "## Safety",
        "",
        "Report-only audit. No vault scan, external download, DB/index mutation, reindex, reembed, or canonical parsed artifact write.",
        "",
        "## Blocking Items",
        "",
    ]
    blocking = [item for item in list(report.get("items") or []) if item.get("coverageStatus") != "ready"]
    if not blocking:
        lines.append("- none")
    else:
        for item in blocking:
            blockers = ", ".join(str(reason) for reason in list(item.get("blockers") or [])) or "-"
            lines.append(
                f"- `{item.get('artifactId')}` paper=`{item.get('paperId') or '-'}` "
                f"status=`{item.get('coverageStatus')}` next=`{item.get('recommendedNextAction')}` "
                f"blockers=`{blockers}`"
            )
    lines.extend(["", "## Coverage Items", ""])
    for item in list(report.get("items") or []):
        lines.append(
            f"- `{item.get('artifactId')}` sourceIds=`{', '.join(item.get('sourceIds') or [])}` "
            f"coverage=`{item.get('coverageStatus')}` source=`{item.get('sourceStatus')}` "
            f"parsed=`{item.get('parsedStatus')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_coverage_audit_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "parsed-artifact-coverage-audit.json"
    markdown_path = root / "parsed-artifact-coverage-audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_parsed_artifact_coverage_audit_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def build_blocked_parsed_artifact_coverage_audit(*, reason: str, paper_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "schema": PARSED_ARTIFACT_COVERAGE_AUDIT_SCHEMA_ID,
        "status": "blocked",
        "generatedAt": _now_iso(),
        "scope": "eval_critical_corpus_manifest",
        "request": {
            "paperIds": list(paper_ids or []),
            "includedCorpusTiers": list(DEFAULT_INCLUDED_CORPUS_TIERS),
            "reportOnly": True,
        },
        "safety": {
            "vaultScan": False,
            "externalDownload": False,
            "dbMutation": False,
            "indexMutation": False,
            "reindex": False,
            "reembed": False,
            "canonicalParsedArtifactWrite": False,
            "privatePathLeakAllowed": False,
        },
        "counts": {
            "totalCorpusArtifacts": 0,
            "registeredPapers": 0,
            "ready": 0,
            "missingSource": 0,
            "missingParsed": 0,
            "hashMismatch": 0,
            "unknownStatus": 0,
            "parsedDegraded": 0,
        },
        "nextParsedCoverageBottleneck": "audit_blocked",
        "items": [],
        "warnings": [reason],
    }


__all__ = [
    "DEFAULT_INCLUDED_CORPUS_TIERS",
    "PARSED_ARTIFACT_COVERAGE_AUDIT_SCHEMA_ID",
    "build_blocked_parsed_artifact_coverage_audit",
    "build_parsed_artifact_coverage_audit",
    "render_parsed_artifact_coverage_audit_markdown",
    "write_parsed_artifact_coverage_audit_reports",
]
