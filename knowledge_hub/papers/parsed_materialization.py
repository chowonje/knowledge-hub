"""Operator-only parsed artifact materialization helpers.

This module intentionally stays below indexing/document-memory surfaces.  It
only plans or writes parser artifacts under ``papers_dir/parsed/<paper_id>/``
from already-registered local source PDFs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_hub.papers.pymupdf_adapter import PyMuPDFAdapter


PARSED_MATERIALIZATION_SCHEMA_ID = "knowledge-hub.paper.parsed-materialization.result.v1"

_SUPPORTED_PARSERS = {"pymupdf"}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _paper_id_from_row(row: dict[str, Any], requested_id: str) -> str:
    return _clean_text(row.get("arxiv_id") or row.get("paper_id") or row.get("paperId") or requested_id)


def _safe_relative(path: str | Path, *, root: str | Path, prefix: str = "papers_dir") -> str:
    candidate = Path(str(path)).expanduser()
    root_path = Path(str(root)).expanduser()
    try:
        rel = candidate.resolve().relative_to(root_path.resolve())
        return str(Path(prefix) / rel)
    except Exception:
        name = candidate.name or "external"
        return str(Path("external") / name)


def _parsed_paths(*, papers_dir: str | Path, paper_id: str) -> dict[str, Path]:
    artifact_dir = Path(str(papers_dir)).expanduser() / "parsed" / paper_id
    return {
        "artifactDir": artifact_dir,
        "documentMarkdownPath": artifact_dir / "document.md",
        "documentJsonPath": artifact_dir / "document.json",
        "manifestPath": artifact_dir / "manifest.json",
    }


def _safe_parsed_paths(*, papers_dir: str | Path, paper_id: str) -> dict[str, str]:
    paths = _parsed_paths(papers_dir=papers_dir, paper_id=paper_id)
    return {key: _safe_relative(value, root=papers_dir) for key, value in paths.items()}


def _artifact_state(paths: dict[str, Path]) -> dict[str, bool]:
    return {
        "artifactDirExists": bool(paths["artifactDir"].exists()),
        "documentMarkdownExists": bool(paths["documentMarkdownPath"].exists()),
        "documentJsonExists": bool(paths["documentJsonPath"].exists()),
        "manifestExists": bool(paths["manifestPath"].exists()),
    }


def _artifact_complete(state: dict[str, bool]) -> bool:
    return bool(state.get("documentJsonExists") and state.get("manifestExists"))


def _source_artifact_for_paper(paper: dict[str, Any]) -> tuple[Path | None, str]:
    raw_pdf = _clean_text(paper.get("pdf_path") or paper.get("pdfPath"))
    if raw_pdf:
        return Path(raw_pdf).expanduser(), "pdf"
    raw_text = _clean_text(paper.get("text_path") or paper.get("textPath"))
    if raw_text:
        return Path(raw_text).expanduser(), "text"
    return None, ""


def _count_items(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "planned": 0,
        "materialized": 0,
        "blocked": 0,
        "failed": 0,
        "skippedExisting": 0,
    }
    for item in items:
        status = str(item.get("status") or "")
        if status == "planned":
            counts["planned"] += 1
        elif status == "materialized":
            counts["materialized"] += 1
        elif status == "blocked":
            counts["blocked"] += 1
        elif status == "failed":
            counts["failed"] += 1
        elif status == "skipped_existing":
            counts["skippedExisting"] += 1
    return counts


def _overall_status(counts: dict[str, int]) -> str:
    if counts.get("failed", 0):
        return "failed"
    if counts.get("blocked", 0) and (counts.get("materialized", 0) or counts.get("planned", 0)):
        return "partial"
    if counts.get("blocked", 0):
        return "blocked"
    return "ok"


def _materialization_item(
    *,
    paper_id: str,
    title: str,
    parser: str,
    papers_dir: str | Path,
    source_path: Path | None,
    source_kind: str,
    before: dict[str, bool],
    after: dict[str, bool] | None = None,
    status: str,
    reason: str,
    action: str,
) -> dict[str, Any]:
    source_exists = bool(source_path and source_path.exists())
    parsed_paths = _safe_parsed_paths(papers_dir=papers_dir, paper_id=paper_id)
    return {
        "paperId": paper_id,
        "paperTitle": title,
        "parser": parser,
        "status": status,
        "action": action,
        "reason": reason,
        "sourceArtifact": {
            "kind": source_kind,
            "exists": source_exists,
            "path": _safe_relative(source_path, root=papers_dir) if source_path else "",
        },
        "parsedArtifacts": {
            "paths": parsed_paths,
            "before": dict(before),
            "after": dict(after if after is not None else before),
        },
        "writes": [
            parsed_paths["documentMarkdownPath"],
            parsed_paths["documentJsonPath"],
            parsed_paths["manifestPath"],
        ]
        if status in {"planned", "materialized"}
        else [],
    }


def materialize_parsed_artifacts(
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    paper_ids: list[str] | tuple[str, ...],
    parser: str = "pymupdf",
    apply: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Plan or materialize parsed artifacts for an explicit paper allowlist."""

    requested_ids = [_clean_text(item) for item in list(paper_ids or []) if _clean_text(item)]
    parser_token = _clean_text(parser).casefold() or "pymupdf"
    items: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not requested_ids:
        warnings.append("paper_id_required")
    if parser_token not in _SUPPORTED_PARSERS:
        warnings.append("unsupported_parser")

    for requested_id in requested_ids:
        paper = sqlite_db.get_paper(requested_id) if hasattr(sqlite_db, "get_paper") else None
        if not isinstance(paper, dict) or not paper:
            paths = _parsed_paths(papers_dir=papers_dir, paper_id=requested_id)
            items.append(
                _materialization_item(
                    paper_id=requested_id,
                    title="",
                    parser=parser_token,
                    papers_dir=papers_dir,
                    source_path=None,
                    source_kind="",
                    before=_artifact_state(paths),
                    status="blocked",
                    reason="paper_not_registered",
                    action="none",
                )
            )
            continue

        paper_id = _paper_id_from_row(paper, requested_id)
        title = _clean_text(paper.get("title") or paper.get("paperTitle"))
        paths = _parsed_paths(papers_dir=papers_dir, paper_id=paper_id)
        before = _artifact_state(paths)
        source_path, source_kind = _source_artifact_for_paper(paper)

        if parser_token not in _SUPPORTED_PARSERS:
            items.append(
                _materialization_item(
                    paper_id=paper_id,
                    title=title,
                    parser=parser_token,
                    papers_dir=papers_dir,
                    source_path=source_path,
                    source_kind=source_kind,
                    before=before,
                    status="blocked",
                    reason="unsupported_parser",
                    action="none",
                )
            )
            continue

        if _artifact_complete(before) and not overwrite:
            items.append(
                _materialization_item(
                    paper_id=paper_id,
                    title=title,
                    parser=parser_token,
                    papers_dir=papers_dir,
                    source_path=source_path,
                    source_kind=source_kind,
                    before=before,
                    status="skipped_existing",
                    reason="parsed_artifact_exists",
                    action="none",
                )
            )
            continue

        if source_kind != "pdf":
            reason = "source_pdf_missing" if not source_kind else "text_source_unsupported"
            items.append(
                _materialization_item(
                    paper_id=paper_id,
                    title=title,
                    parser=parser_token,
                    papers_dir=papers_dir,
                    source_path=source_path,
                    source_kind=source_kind,
                    before=before,
                    status="blocked",
                    reason=reason,
                    action="none",
                )
            )
            continue
        if source_path is None or not source_path.exists():
            items.append(
                _materialization_item(
                    paper_id=paper_id,
                    title=title,
                    parser=parser_token,
                    papers_dir=papers_dir,
                    source_path=source_path,
                    source_kind=source_kind,
                    before=before,
                    status="blocked",
                    reason="source_pdf_missing",
                    action="none",
                )
            )
            continue

        if not apply:
            items.append(
                _materialization_item(
                    paper_id=paper_id,
                    title=title,
                    parser=parser_token,
                    papers_dir=papers_dir,
                    source_path=source_path,
                    source_kind=source_kind,
                    before=before,
                    status="planned",
                    reason="apply_required",
                    action="materialize_parsed_artifacts",
                )
            )
            continue

        try:
            adapter = PyMuPDFAdapter(papers_dir=str(papers_dir))
            adapter.ensure_artifacts(
                paper_id=paper_id,
                pdf_path=str(source_path),
                refresh=bool(overwrite),
                allow_ocr=False,
            )
            after = _artifact_state(paths)
            if not _artifact_complete(after):
                items.append(
                    _materialization_item(
                        paper_id=paper_id,
                        title=title,
                        parser=parser_token,
                        papers_dir=papers_dir,
                        source_path=source_path,
                        source_kind=source_kind,
                        before=before,
                        after=after,
                        status="failed",
                        reason="parsed_artifact_incomplete_after_apply",
                        action="materialize_parsed_artifacts",
                    )
                )
                continue
            items.append(
                _materialization_item(
                    paper_id=paper_id,
                    title=title,
                    parser=parser_token,
                    papers_dir=papers_dir,
                    source_path=source_path,
                    source_kind=source_kind,
                    before=before,
                    after=after,
                    status="materialized",
                    reason="ok",
                    action="materialize_parsed_artifacts",
                )
            )
        except Exception as error:
            items.append(
                _materialization_item(
                    paper_id=paper_id,
                    title=title,
                    parser=parser_token,
                    papers_dir=papers_dir,
                    source_path=source_path,
                    source_kind=source_kind,
                    before=before,
                    status="failed",
                    reason=f"parser_failed: {_clean_text(error)}",
                    action="materialize_parsed_artifacts",
                )
            )

    counts = _count_items(items)
    status = _overall_status(counts)
    if warnings and status == "ok":
        status = "blocked"
    return {
        "schema": PARSED_MATERIALIZATION_SCHEMA_ID,
        "status": status,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "request": {
            "paperIds": requested_ids,
            "parser": parser_token,
            "apply": bool(apply),
            "overwrite": bool(overwrite),
        },
        "counts": counts,
        "items": items,
        "warnings": warnings,
    }


__all__ = [
    "PARSED_MATERIALIZATION_SCHEMA_ID",
    "materialize_parsed_artifacts",
]
