"""Report-only arXiv source/TeX availability audit.

This helper checks whether target arXiv papers have source archives that expose
TeX structure useful for later Section/Table/Equation/Figure candidate work.
It can fetch arXiv e-print sources only when ``--allow-network`` is explicit.

The audit is candidate-only: it does not mutate SQLite, scan vault content,
write canonical parsed artifacts, reindex, reembed, route parsers, or create
strict/runtime evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import gzip
import json
from pathlib import Path
import re
import tarfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ARXIV_SOURCE_TEX_AVAILABILITY_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.arxiv-source-tex-availability-audit.v1"
)

DEFAULT_PAPER_IDS = ["1706.03762", "1506.02640", "2005.14165"]
DEFAULT_PARSED_ROOT = Path.home() / ".khub" / "papers" / "parsed"
DEFAULT_MINERU_SOURCE_ALIGNMENT_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-17"
    / "mineru-source-alignment-helper"
    / "mineru-source-alignment-report.json"
)

_WHITESPACE_RE = re.compile(r"\s+")
_BEGIN_ENV_RE = re.compile(r"\\begin\s*\{\s*([^}]+?)\s*\}", re.DOTALL)
_COMMAND_RE_BY_NAME = {
    "section": re.compile(r"\\section\*?\s*\{", re.DOTALL),
    "subsection": re.compile(r"\\subsection\*?\s*\{", re.DOTALL),
    "subsubsection": re.compile(r"\\subsubsection\*?\s*\{", re.DOTALL),
    "caption": re.compile(r"\\caption(?:\[[^\]]*\])?\s*\{", re.DOTALL),
}
_TARGET_ENV_GROUP_BY_NAME = {
    "equation": "equation",
    "equation*": "equation",
    "align": "equation",
    "align*": "equation",
    "eqnarray": "equation",
    "eqnarray*": "equation",
    "gather": "equation",
    "gather*": "equation",
    "multline": "equation",
    "multline*": "equation",
    "table": "table",
    "table*": "table",
    "tabular": "tabular",
    "tabular*": "tabular",
    "figure": "figure",
    "figure*": "figure",
}
_MINERU_TYPE_BY_TEX_TYPE = {
    "section": {"section_candidate"},
    "subsection": {"section_candidate"},
    "subsubsection": {"section_candidate"},
    "figure_caption": {"figure_caption_candidate"},
    "table_caption": {"table_candidate"},
    "caption": {"figure_caption_candidate", "table_candidate"},
}
_TEX_EXTENSIONS = {".tex", ".ltx", ".bbl"}
_MAX_TEX_BYTES = 5_000_000
_MAX_ARCHIVE_BYTES = 50_000_000


@dataclass(frozen=True)
class _SourceFile:
    path: str
    text: str
    bytes_len: int


@dataclass(frozen=True)
class _TextMatch:
    status: str
    method: str
    chars_start: int | None
    chars_end: int | None
    confidence: float
    reason: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_text(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "").strip())


def _fold_text(value: Any) -> str:
    return _clean_text(value).casefold()


def _source_url(paper_id: str) -> str:
    return f"https://arxiv.org/e-print/{paper_id}"


def _candidate_cache_names(paper_id: str) -> list[str]:
    safe = paper_id.replace("/", "_")
    return [
        f"{safe}-source",
        f"{safe}-source.tar",
        f"{safe}-source.tar.gz",
        f"{safe}.tar",
        f"{safe}.tar.gz",
        f"{safe}.gz",
        f"{safe}.tex",
    ]


def _existing_source_path(source_cache_dir: str | Path | None, paper_id: str) -> Path | None:
    if not source_cache_dir:
        return None
    root = Path(str(source_cache_dir)).expanduser()
    for name in _candidate_cache_names(paper_id):
        path = root / name
        if path.exists() and path.is_file():
            return path
    return None


def _download_source(paper_id: str, output_dir: Path, timeout_seconds: int) -> tuple[str, Path | None, str]:
    source_dir = output_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    destination = source_dir / f"{paper_id}-source"
    request = Request(_source_url(paper_id), headers={"User-Agent": "knowledge-hub-arxiv-source-audit/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            data = response.read(_MAX_ARCHIVE_BYTES + 1)
            if len(data) > _MAX_ARCHIVE_BYTES:
                return "failed", None, "source_archive_exceeded_size_limit"
            destination.write_bytes(data)
            return "downloaded", destination, ""
    except HTTPError as exc:
        if exc.code == 404:
            return "unavailable", None, "arxiv_source_not_available_404"
        return "failed", None, f"http_error_{exc.code}"
    except URLError as exc:
        return "failed", None, f"url_error:{exc.reason}"
    except Exception as exc:
        return "failed", None, f"download_failed:{type(exc).__name__}"


def _safe_member_name(name: str) -> bool:
    if not name or name.startswith("/") or ".." in Path(name).parts:
        return False
    return True


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_tar_tex_files(data: bytes) -> list[_SourceFile]:
    files: list[_SourceFile] = []
    try:
        with tarfile.open(fileobj=BytesIO(data), mode="r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not _safe_member_name(member.name):
                    continue
                suffix = Path(member.name).suffix.casefold()
                if suffix not in _TEX_EXTENSIONS:
                    continue
                if int(member.size or 0) > _MAX_TEX_BYTES:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                raw = handle.read(_MAX_TEX_BYTES + 1)
                if len(raw) > _MAX_TEX_BYTES:
                    continue
                files.append(_SourceFile(member.name, _decode_bytes(raw), len(raw)))
    except tarfile.TarError:
        return []
    return files


def _read_source_files(path: Path) -> tuple[str, list[_SourceFile], str]:
    try:
        data = path.read_bytes()
    except Exception:
        return "unreadable", [], "source_file_unreadable"
    if len(data) > _MAX_ARCHIVE_BYTES:
        return "too_large", [], "source_file_exceeded_size_limit"

    tar_files = _read_tar_tex_files(data)
    if tar_files:
        return "tar", tar_files, ""

    try:
        decompressed = gzip.decompress(data)
    except OSError:
        decompressed = b""
    if decompressed:
        tar_files = _read_tar_tex_files(decompressed)
        if tar_files:
            return "gzip_tar", tar_files, ""
        if len(decompressed) <= _MAX_TEX_BYTES:
            return "gzip_single_file", [_SourceFile(path.name, _decode_bytes(decompressed), len(decompressed))], ""

    suffix = path.suffix.casefold()
    if suffix in _TEX_EXTENSIONS or b"\\section" in data or b"\\begin" in data:
        return "single_file", [_SourceFile(path.name, _decode_bytes(data[:_MAX_TEX_BYTES]), min(len(data), _MAX_TEX_BYTES))], ""
    return "unknown", [], "no_tex_files_detected"


def _find_balanced_brace_end(text: str, open_brace_index: int) -> int | None:
    depth = 0
    escaped = False
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _command_args(text: str, command: str) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    pattern = _COMMAND_RE_BY_NAME[command]
    for match in pattern.finditer(text):
        open_index = match.end() - 1
        close_index = _find_balanced_brace_end(text, open_index)
        if close_index is None:
            continue
        raw = text[open_index + 1 : close_index]
        cleaned = _clean_text(re.sub(r"\\[a-zA-Z]+\*?", "", raw).replace("{", "").replace("}", ""))
        if cleaned:
            rows.append((match.start(), close_index + 1, cleaned))
    return rows


def _nearest_environment(text: str, offset: int) -> str:
    prefix = text[:offset]
    begins = list(_BEGIN_ENV_RE.finditer(prefix))
    for match in reversed(begins[-20:]):
        env = match.group(1).strip()
        end_token = f"\\end{{{env}}}"
        if end_token not in prefix[match.end() :]:
            return env
    return ""


def _extract_structure_rows(paper_id: str, files: list[_SourceFile]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 1
    for file in files:
        text = file.text
        for command in ("section", "subsection", "subsubsection"):
            for start, end, value in _command_args(text, command):
                rows.append(
                    {
                        "structure_row_id": f"arxiv-source-structure:{paper_id}:{index:04d}",
                        "paper_id": paper_id,
                        "source_file": file.path,
                        "structure_type": command,
                        "tex_command": f"\\{command}",
                        "tex_environment": "",
                        "tex_chars_start": start,
                        "tex_chars_end": end,
                        "candidate_text": value,
                    }
                )
                index += 1
        for start, end, value in _command_args(text, "caption"):
            env = _nearest_environment(text, start)
            if env.startswith("table"):
                structure_type = "table_caption"
            elif env.startswith("figure"):
                structure_type = "figure_caption"
            else:
                structure_type = "caption"
            rows.append(
                {
                    "structure_row_id": f"arxiv-source-structure:{paper_id}:{index:04d}",
                    "paper_id": paper_id,
                    "source_file": file.path,
                    "structure_type": structure_type,
                    "tex_command": "\\caption",
                    "tex_environment": env,
                    "tex_chars_start": start,
                    "tex_chars_end": end,
                    "candidate_text": value,
                }
            )
            index += 1
        for match in _BEGIN_ENV_RE.finditer(text):
            env = match.group(1).strip()
            group = _TARGET_ENV_GROUP_BY_NAME.get(env)
            if not group:
                continue
            rows.append(
                {
                    "structure_row_id": f"arxiv-source-structure:{paper_id}:{index:04d}",
                    "paper_id": paper_id,
                    "source_file": file.path,
                    "structure_type": f"{group}_environment",
                    "tex_command": "\\begin",
                    "tex_environment": env,
                    "tex_chars_start": match.start(),
                    "tex_chars_end": match.end(),
                    "candidate_text": "",
                }
            )
            index += 1
    return rows


def _find_all(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    starts: list[int] = []
    cursor = text.find(needle)
    while cursor >= 0:
        starts.append(cursor)
        cursor = text.find(needle, cursor + 1)
    return starts


def _align_text(canonical_text: str, candidate_text: str) -> _TextMatch:
    cleaned = _clean_text(candidate_text)
    if not canonical_text or not cleaned:
        return _TextMatch("blocked", "none", None, None, 0.0, "missing_canonical_or_candidate_text")
    exact_starts = _find_all(canonical_text, cleaned)
    if len(exact_starts) == 1:
        start = exact_starts[0]
        return _TextMatch("aligned", "exact", start, start + len(cleaned), 0.99, "single_exact_text_match")
    if len(exact_starts) > 1:
        return _TextMatch("ambiguous", "exact", None, None, 0.2, "ambiguous_exact_text_match")
    folded_text = _fold_text(canonical_text)
    folded_candidate = _fold_text(cleaned)
    folded_starts = _find_all(folded_text, folded_candidate)
    if len(folded_starts) == 1:
        # This maps approximately because case-folding can alter length; keep non-strict.
        start = folded_starts[0]
        return _TextMatch("aligned", "normalized", start, start + len(folded_candidate), 0.8, "single_normalized_text_match")
    if len(folded_starts) > 1:
        return _TextMatch("ambiguous", "normalized", None, None, 0.18, "ambiguous_normalized_text_match")
    return _TextMatch("failed", "none", None, None, 0.0, "text_not_found_in_canonical_markdown")


def _canonical_markdown(parsed_root: Path, paper_id: str) -> str:
    path = parsed_root / paper_id / "document.md"
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _mineru_candidates(path: str | Path | None) -> list[dict[str, Any]]:
    payload = _read_json(path)
    return [dict(item) for item in list(payload.get("candidates") or []) if isinstance(item, dict)]


def _mineru_link(row: dict[str, Any], candidates_by_paper: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    paper_id = str(row.get("paper_id") or "")
    text = str(row.get("candidate_text") or "")
    structure_type = str(row.get("structure_type") or "")
    allowed_types = _MINERU_TYPE_BY_TEX_TYPE.get(structure_type, set())
    if not text or not allowed_types:
        return {"status": "blocked", "method": "none", "candidateIds": [], "bboxCount": 0, "reason": "no_text_or_no_matching_mineru_type"}
    matches: list[dict[str, Any]] = []
    folded = _fold_text(text)
    for candidate in candidates_by_paper.get(paper_id, []):
        if str(candidate.get("candidate_type") or "") not in allowed_types:
            continue
        candidate_text = str(candidate.get("candidate_text") or "")
        if candidate_text == text or _fold_text(candidate_text) == folded:
            matches.append(candidate)
    if len(matches) == 1:
        mineru_candidate = matches[0]
        bbox = ((mineru_candidate.get("mineruCandidate") or {}).get("bbox") or [])
        return {
            "status": "linked",
            "method": "text_match",
            "candidateIds": [str(mineru_candidate.get("candidate_id") or "")],
            "bboxCount": 1 if bbox else 0,
            "reason": "single_mineru_candidate_text_match",
        }
    if len(matches) > 1:
        return {
            "status": "ambiguous",
            "method": "text_match",
            "candidateIds": [str(item.get("candidate_id") or "") for item in matches[:10]],
            "bboxCount": sum(1 for item in matches if ((item.get("mineruCandidate") or {}).get("bbox") or [])),
            "reason": "multiple_mineru_candidate_text_matches",
        }
    return {"status": "failed", "method": "none", "candidateIds": [], "bboxCount": 0, "reason": "no_mineru_candidate_text_match"}


def _row_with_alignment(
    row: dict[str, Any],
    canonical_text: str,
    candidates_by_paper: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    alignment = _align_text(canonical_text, str(row.get("candidate_text") or ""))
    mineru = _mineru_link(row, candidates_by_paper)
    strict_blockers = [
        "source_structure_candidate_only",
        "tex_offsets_are_not_canonical_source_spans",
        "strict_promotion_requires_later_explicit_tranche",
        "runtime_promotion_disabled_for_tranche",
    ]
    if alignment.status != "aligned":
        strict_blockers.append("canonical_text_alignment_not_available")
    if alignment.method != "exact":
        strict_blockers.append("non_exact_or_missing_canonical_alignment")
    if mineru["status"] != "linked":
        strict_blockers.append("mineru_layout_link_not_unique")
    return {
        **row,
        "canonical_alignment_status": alignment.status,
        "canonical_alignment_method": alignment.method,
        "canonical_chars_start": alignment.chars_start,
        "canonical_chars_end": alignment.chars_end,
        "canonical_alignment_confidence": alignment.confidence,
        "canonical_alignment_reason": alignment.reason,
        "mineru_layout_link_status": mineru["status"],
        "mineru_layout_link_method": mineru["method"],
        "mineru_candidate_ids": mineru["candidateIds"],
        "mineru_bbox_link_count": mineru["bboxCount"],
        "mineru_layout_link_reason": mineru["reason"],
        "evidence_tier": "source_structure_candidate_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": [
            "source_structure_rows_are_candidate_only",
            "tex_source_structure_is_not_runtime_evidence",
            "this_audit_does_not_write_canonical_parsed_artifacts",
        ],
    }


def _paper_row(
    *,
    paper_id: str,
    allow_network: bool,
    output_dir: Path,
    parsed_root: Path,
    source_cache_dir: str | Path | None,
    timeout_seconds: int,
    candidates_by_paper: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    existing = _existing_source_path(source_cache_dir, paper_id)
    fetch_status = "cached" if existing else "skipped_network_disabled"
    source_path = existing
    failure_reason = ""
    if source_path is None and allow_network:
        fetch_status, source_path, failure_reason = _download_source(paper_id, output_dir, timeout_seconds)
    elif source_path is None:
        failure_reason = "network_not_allowed_and_no_cached_source"

    source_format = ""
    tex_files: list[_SourceFile] = []
    parse_reason = ""
    if source_path is not None:
        source_format, tex_files, parse_reason = _read_source_files(source_path)
        if parse_reason and not failure_reason:
            failure_reason = parse_reason

    canonical_text = _canonical_markdown(parsed_root, paper_id)
    raw_structure_rows = _extract_structure_rows(paper_id, tex_files)
    structure_rows = [
        _row_with_alignment(row, canonical_text, candidates_by_paper)
        for row in raw_structure_rows
    ]
    by_type = Counter(str(row.get("structure_type") or "") for row in structure_rows)
    aligned = sum(1 for row in structure_rows if row.get("canonical_alignment_status") == "aligned")
    mineru_linked = sum(1 for row in structure_rows if row.get("mineru_layout_link_status") == "linked")
    paper = {
        "paper_id": paper_id,
        "arxiv_source_url": _source_url(paper_id),
        "fetch_status": fetch_status,
        "failure_reason": failure_reason,
        "source_archive_path": str(source_path or ""),
        "source_format": source_format,
        "tex_file_count": len(tex_files),
        "tex_total_bytes": sum(item.bytes_len for item in tex_files),
        "tex_files": [{"path": item.path, "bytes": item.bytes_len} for item in tex_files[:20]],
        "canonical_document_present": bool(canonical_text),
        "mineru_candidate_count": len(candidates_by_paper.get(paper_id, [])),
        "structure_row_count": len(structure_rows),
        "canonical_aligned_structure_rows": aligned,
        "mineru_layout_linked_rows": mineru_linked,
        "strict_eligible_rows": 0,
        "citation_grade_rows": 0,
        "runtime_evidence_rows": 0,
        "byStructureType": dict(by_type),
        "evidence_tier": "arxiv_source_tex_availability_audit_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }
    return paper, structure_rows


def _counts(papers: list[dict[str, Any]], structure_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_fetch = Counter(str(paper.get("fetch_status") or "") for paper in papers)
    by_format = Counter(str(paper.get("source_format") or "") for paper in papers)
    by_structure = Counter(str(row.get("structure_type") or "") for row in structure_rows)
    by_alignment = Counter(str(row.get("canonical_alignment_status") or "") for row in structure_rows)
    by_mineru = Counter(str(row.get("mineru_layout_link_status") or "") for row in structure_rows)
    return {
        "paperCount": len(papers),
        "sourceAvailablePapers": sum(1 for paper in papers if int(paper.get("tex_file_count") or 0) > 0),
        "sourceDownloadedPapers": by_fetch.get("downloaded", 0),
        "sourceCachedPapers": by_fetch.get("cached", 0),
        "sourceUnavailablePapers": by_fetch.get("unavailable", 0),
        "sourceSkippedNetworkDisabledPapers": by_fetch.get("skipped_network_disabled", 0),
        "texFilePapers": sum(1 for paper in papers if int(paper.get("tex_file_count") or 0) > 0),
        "structureRows": len(structure_rows),
        "sectionCommandRows": by_structure.get("section", 0),
        "subsectionCommandRows": by_structure.get("subsection", 0),
        "subsubsectionCommandRows": by_structure.get("subsubsection", 0),
        "equationEnvironmentRows": by_structure.get("equation_environment", 0),
        "tableEnvironmentRows": by_structure.get("table_environment", 0),
        "tabularEnvironmentRows": by_structure.get("tabular_environment", 0),
        "figureEnvironmentRows": by_structure.get("figure_environment", 0),
        "captionCommandRows": by_structure.get("caption", 0) + by_structure.get("figure_caption", 0) + by_structure.get("table_caption", 0),
        "canonicalAlignedRows": by_alignment.get("aligned", 0),
        "mineruLayoutLinkedRows": by_mineru.get("linked", 0),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "byFetchStatus": dict(by_fetch),
        "bySourceFormat": dict(by_format),
        "byStructureType": dict(by_structure),
        "byCanonicalAlignmentStatus": dict(by_alignment),
        "byMineruLayoutLinkStatus": dict(by_mineru),
    }


def build_arxiv_source_tex_availability_audit(
    *,
    paper_ids: list[str] | None = None,
    output_dir: str | Path,
    allow_network: bool = False,
    parsed_root: str | Path = DEFAULT_PARSED_ROOT,
    source_cache_dir: str | Path | None = None,
    mineru_source_alignment_report: str | Path | None = DEFAULT_MINERU_SOURCE_ALIGNMENT_REPORT,
    timeout_seconds: int = 45,
) -> dict[str, Any]:
    """Build a report-only arXiv source/TeX availability audit."""

    target_paper_ids = [str(item).strip() for item in (paper_ids or DEFAULT_PAPER_IDS) if str(item).strip()]
    root = Path(str(output_dir)).expanduser()
    parsed_root_path = Path(str(parsed_root)).expanduser()
    mineru_candidates = _mineru_candidates(mineru_source_alignment_report)
    candidates_by_paper: dict[str, list[dict[str, Any]]] = {}
    for candidate in mineru_candidates:
        candidates_by_paper.setdefault(str(candidate.get("paper_id") or ""), []).append(candidate)

    papers: list[dict[str, Any]] = []
    all_structure_rows: list[dict[str, Any]] = []
    for paper_id in target_paper_ids:
        paper, rows = _paper_row(
            paper_id=paper_id,
            allow_network=allow_network,
            output_dir=root,
            parsed_root=parsed_root_path,
            source_cache_dir=source_cache_dir,
            timeout_seconds=timeout_seconds,
            candidates_by_paper=candidates_by_paper,
        )
        papers.append(paper)
        all_structure_rows.extend(rows)

    counts = _counts(papers, all_structure_rows)
    status = "ok" if target_paper_ids else "blocked"
    if not target_paper_ids:
        decision = "blocked_no_target_papers"
    elif int(counts.get("sourceAvailablePapers") or 0) > 0:
        decision = "tex_source_candidates_available_report_only"
    else:
        decision = "no_tex_source_candidates_available"
    return {
        "schema": ARXIV_SOURCE_TEX_AVAILABILITY_AUDIT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "paperIds": target_paper_ids,
            "allowNetwork": bool(allow_network),
            "parsedRoot": str(parsed_root_path),
            "sourceCacheDir": str(Path(str(source_cache_dir)).expanduser()) if source_cache_dir else "",
            "mineruSourceAlignmentReport": str(Path(str(mineru_source_alignment_report)).expanduser())
            if mineru_source_alignment_report
            else "",
            "timeoutSeconds": int(timeout_seconds),
        },
        "counts": counts,
        "gate": {
            "auditReady": bool(target_paper_ids),
            "networkUsed": bool(allow_network) and int(counts.get("sourceDownloadedPapers") or 0) > 0,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "recommendedNextTranche": "tex_structure_candidate_alignment_audit"
            if int(counts.get("sourceAvailablePapers") or 0) > 0
            else "parsed_artifact_coverage_or_source_acquisition",
        },
        "policy": {
            "reportOnly": True,
            "sourceStructureCandidateOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "arxiv_tex_structure_rows_are_candidate_only",
            "tex_offsets_are_not_runtime_source_spans",
            "strict_or_runtime_promotion_requires_a_separate_explicit_tranche",
        ],
        "papers": papers,
        "structureRows": all_structure_rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_arxiv_source_tex_availability_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# arXiv Source / TeX Availability Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Papers: `{int(counts.get('paperCount') or 0)}`",
        f"- Source available papers: `{int(counts.get('sourceAvailablePapers') or 0)}`",
        f"- Structure rows: `{int(counts.get('structureRows') or 0)}`",
        f"- Canonical aligned rows: `{int(counts.get('canonicalAlignedRows') or 0)}`",
        f"- MinerU layout linked rows: `{int(counts.get('mineruLayoutLinkedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This audit is report-only. It does not create strict evidence, route parsers, write canonical parsed artifacts, mutate DB state, scan vault content, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By fetch status: `{json.dumps(counts.get('byFetchStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By source format: `{json.dumps(counts.get('bySourceFormat') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By structure type: `{json.dumps(counts.get('byStructureType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By canonical alignment: `{json.dumps(counts.get('byCanonicalAlignmentStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By MinerU layout link: `{json.dumps(counts.get('byMineruLayoutLinkStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Papers",
        "",
    ]
    for paper in list(report.get("papers") or []):
        lines.extend(
            [
                f"### {paper.get('paper_id', '')}",
                "",
                f"- Fetch status: `{paper.get('fetch_status', '')}`",
                f"- Source format: `{paper.get('source_format', '')}`",
                f"- TeX files: `{int(paper.get('tex_file_count') or 0)}`",
                f"- Structure rows: `{int(paper.get('structure_row_count') or 0)}`",
                f"- Canonical aligned rows: `{int(paper.get('canonical_aligned_structure_rows') or 0)}`",
                f"- MinerU linked rows: `{int(paper.get('mineru_layout_linked_rows') or 0)}`",
                f"- By structure type: `{json.dumps(paper.get('byStructureType') or {}, ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_arxiv_source_tex_availability_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "arxiv-source-tex-availability-report.json"
    summary_path = root / "arxiv-source-tex-availability-summary.json"
    markdown_path = root / "arxiv-source-tex-availability-audit.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_arxiv_source_tex_availability_audit_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only arXiv source/TeX availability audit.")
    parser.add_argument("--paper-id", action="append", default=[], help="Target arXiv paper id; can be repeated.")
    parser.add_argument("--allow-network", action="store_true", help="Allow fetching arXiv e-print source archives.")
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT))
    parser.add_argument("--source-cache-dir", default="")
    parser.add_argument("--mineru-source-alignment-report", default=str(DEFAULT_MINERU_SOURCE_ALIGNMENT_REPORT))
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--output-dir", required=True, help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_arxiv_source_tex_availability_audit(
        paper_ids=args.paper_id or DEFAULT_PAPER_IDS,
        output_dir=args.output_dir,
        allow_network=bool(args.allow_network),
        parsed_root=args.parsed_root,
        source_cache_dir=args.source_cache_dir or None,
        mineru_source_alignment_report=args.mineru_source_alignment_report or None,
        timeout_seconds=args.timeout_seconds,
    )
    paths = write_arxiv_source_tex_availability_audit_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARXIV_SOURCE_TEX_AVAILABILITY_AUDIT_SCHEMA_ID",
    "build_arxiv_source_tex_availability_audit",
    "render_arxiv_source_tex_availability_audit_markdown",
    "write_arxiv_source_tex_availability_audit_reports",
]
