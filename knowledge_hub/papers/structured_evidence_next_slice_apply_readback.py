"""Apply/readback helper for the next Structured Evidence slice.

The helper consumes the schema-backed next-slice candidate report, applies a
small explicit section_text_offset tranche, and immediately reads the written
records back through the same strict trace gates used by the first vertical
slice. It does not expand the corpus manifest, run answers, promote citation
grade/runtime evidence, or touch table/equation parsers.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.application.corpus_artifacts import (
    DEFAULT_CORPUS_MANIFEST_PATH,
    corpus_entry_ref,
    find_corpus_entry_for_source,
    inspect_corpus_artifact,
    load_corpus_manifest,
)
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.config import Config
from knowledge_hub.papers.priority_corpus_300_freeze import (
    STRUCTURED_EVIDENCE_NEXT_SLICE_SCHEMA_ID,
)
from knowledge_hub.papers.structured_evidence_vertical_slice_implementation import (
    DEFERRED_EVIDENCE_TYPES,
    STRUCTURED_EVIDENCE_VERTICAL_SLICE_IMPLEMENTATION_SCHEMA_ID,
    _clean,
    _figure_caption_readback,
    _greenfield_section,
    _pilot_readback,
    _sanitize_for_public,
)


STRUCTURED_EVIDENCE_NEXT_SLICE_APPLY_READBACK_SCHEMA_ID = (
    "knowledge-hub.paper.structured-evidence-next-slice-apply-readback.v1"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEXT_SLICE_CANDIDATE_REPORT_PATH = PROJECT_ROOT / "eval" / "knowledgeos" / "reports" / (
    "structured_evidence_next_slice_candidate_report.v1.json"
)
RUN_ID = "structured-evidence-next-slice-20260521"
READBACK_MODES = {
    "1506.02640": "figure_caption_readback",
}


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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _project_ref(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _configured_papers_dir(config: Any, override: str | Path | None = None) -> Path:
    if override not in (None, ""):
        return Path(str(override)).expanduser()
    raw = ""
    if hasattr(config, "get_nested"):
        raw = _clean(config.get_nested("storage", "papers_dir", default=""))
    if not raw:
        raw = _clean(getattr(config, "papers_dir", ""))
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ("." + "khub") / "papers"


def _resolve_source_pdf(
    *,
    entry: dict[str, Any],
    config: Any,
    papers_dir: Path,
) -> tuple[Path | None, str, list[str], dict[str, Any]]:
    expected_hash = _clean(entry.get("expectedSourceContentHash"))
    if expected_hash and not expected_hash.startswith("sha256:"):
        expected_hash = f"sha256:{expected_hash}"
    inspection = inspect_corpus_artifact(entry, config=_PapersDirConfig(config, papers_dir))
    blockers: list[str] = []
    if inspection.get("status") != "ok":
        blockers.append(f"source_artifact_{inspection.get('status') or 'blocked'}")
        return None, expected_hash, blockers, inspection
    resolved = _clean(inspection.get("_resolvedPath"))
    if not resolved:
        blockers.append("source_artifact_resolved_path_missing")
        return None, expected_hash, blockers, inspection
    return Path(resolved), expected_hash, blockers, inspection


def _selected_greenfield_ids(report: dict[str, Any], *, selected_greenfield_rows: int) -> list[str]:
    ids: list[str] = []
    for row in report.get("selectedGreenfieldCandidates") or []:
        source_id = _clean(row.get("sourceId"))
        if source_id and source_id not in ids:
            ids.append(source_id)
        if len(ids) >= selected_greenfield_rows:
            break
    return ids


def _selected_readback_ids(report: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for row in report.get("readbackCandidates") or []:
        source_id = _clean(row.get("sourceId"))
        if source_id and source_id not in ids:
            ids.append(source_id)
    return ids


def _post_apply_readback(
    *,
    source_id: str,
    entry: dict[str, Any],
    papers_dir: Path,
    pdf_path: Path,
    expected_hash: str,
) -> dict[str, Any]:
    readback = _pilot_readback(
        source_id=source_id,
        entry=entry,
        papers_dir=papers_dir,
        pdf_path=pdf_path,
        expected_hash=expected_hash,
    )
    readback["mode"] = "greenfield_apply_readback"
    return readback


def build_structured_evidence_next_slice_apply_readback(
    *,
    config: Any | None = None,
    manifest_path: str | Path = DEFAULT_CORPUS_MANIFEST_PATH,
    candidate_report_path: str | Path = DEFAULT_NEXT_SLICE_CANDIDATE_REPORT_PATH,
    papers_dir: str | Path | None = None,
    apply: bool = False,
    selected_greenfield_rows: int = 7,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    config = config or Config()
    resolved_papers_dir = _configured_papers_dir(config, papers_dir)
    manifest = load_corpus_manifest(manifest_path)
    candidate_report = _read_json(candidate_report_path)
    schema = _clean(candidate_report.get("schema"))
    if schema != STRUCTURED_EVIDENCE_NEXT_SLICE_SCHEMA_ID:
        raise ValueError(f"unsupported candidate report schema: {schema}")

    readback_ids = _selected_readback_ids(candidate_report)
    greenfield_ids = _selected_greenfield_ids(
        candidate_report,
        selected_greenfield_rows=selected_greenfield_rows,
    )
    readback_rows: list[dict[str, Any]] = []
    greenfield_rows: list[dict[str, Any]] = []

    for source_id in readback_ids:
        entry = find_corpus_entry_for_source(source_id, manifest)
        if entry is None:
            readback_rows.append({"sourceId": source_id, "status": "blocked", "blockers": ["corpus_entry_missing"]})
            continue
        pdf_path, expected_hash, blockers, _inspection = _resolve_source_pdf(
            entry=entry,
            config=config,
            papers_dir=resolved_papers_dir,
        )
        if blockers or pdf_path is None:
            readback_rows.append(
                {
                    "sourceId": source_id,
                    "artifactId": corpus_entry_ref(entry),
                    "status": "blocked",
                    "expectedSourceContentHash": expected_hash,
                    "blockers": blockers,
                }
            )
            continue
        if READBACK_MODES.get(source_id) == "figure_caption_readback":
            row = _figure_caption_readback(
                source_id=source_id,
                entry=entry,
                papers_dir=resolved_papers_dir,
                pdf_path=pdf_path,
                expected_hash=expected_hash,
            )
        else:
            row = _pilot_readback(
                source_id=source_id,
                entry=entry,
                papers_dir=resolved_papers_dir,
                pdf_path=pdf_path,
                expected_hash=expected_hash,
            )
        readback_rows.append(_sanitize_for_public(row))

    for source_id in greenfield_ids:
        entry = find_corpus_entry_for_source(source_id, manifest)
        if entry is None:
            greenfield_rows.append({"sourceId": source_id, "status": "blocked", "blockers": ["corpus_entry_missing"]})
            continue
        pdf_path, expected_hash, blockers, _inspection = _resolve_source_pdf(
            entry=entry,
            config=config,
            papers_dir=resolved_papers_dir,
        )
        if blockers or pdf_path is None:
            greenfield_rows.append(
                {
                    "sourceId": source_id,
                    "artifactId": corpus_entry_ref(entry),
                    "status": "blocked",
                    "expectedSourceContentHash": expected_hash,
                    "blockers": blockers,
                }
            )
            continue
        row = _greenfield_section(
            source_id=source_id,
            entry=entry,
            papers_dir=resolved_papers_dir,
            pdf_path=pdf_path,
            expected_hash=expected_hash,
            apply=apply,
            run_id=run_id,
        )
        if apply and row.get("status") == "generated":
            post_apply = _post_apply_readback(
                source_id=source_id,
                entry=entry,
                papers_dir=resolved_papers_dir,
                pdf_path=pdf_path,
                expected_hash=expected_hash,
            )
            row["postApplyReadback"] = {
                "status": post_apply.get("status"),
                "strictEvidenceCount": (post_apply.get("existingRecords") or {}).get("strictEvidenceCount"),
                "sourceSpanCount": (post_apply.get("existingRecords") or {}).get("sourceSpanCount"),
                "traceValidation": post_apply.get("traceValidation"),
                "blockers": post_apply.get("blockers") or [],
            }
            if post_apply.get("status") != "pass":
                row.setdefault("blockers", []).append("post_apply_readback_failed")
                row["status"] = "blocked"
        greenfield_rows.append(_sanitize_for_public(row))

    counts = Counter()
    counts["readbackRows"] = len(readback_rows)
    counts["greenfieldSelectedRows"] = len(greenfield_rows)
    for row in readback_rows:
        if row.get("status") == "pass":
            counts["readbackPassRows"] += 1
        else:
            counts["readbackBlockedRows"] += 1
    for row in greenfield_rows:
        generated = row.get("generatedRecords") if isinstance(row.get("generatedRecords"), dict) else {}
        counts["generatedSourceSpanRecords"] += int(generated.get("sourceSpanCount") or 0)
        counts["generatedStrictEvidenceRecords"] += int(generated.get("strictEvidenceCount") or 0)
        if generated.get("applied"):
            counts["appliedGreenfieldRows"] += 1
        if row.get("status") == "generated":
            counts["greenfieldGeneratedRows"] += 1
        else:
            counts["greenfieldBlockedRows"] += 1
        post = row.get("postApplyReadback") if isinstance(row.get("postApplyReadback"), dict) else {}
        if post.get("status") == "pass":
            counts["postApplyReadbackPassRows"] += 1
    baseline_counts = dict(candidate_report.get("counts") or {})
    baseline_strict = int(baseline_counts.get("strictCoveredRows") or 0)
    for key in (
        "readbackPassRows",
        "readbackBlockedRows",
        "greenfieldGeneratedRows",
        "greenfieldBlockedRows",
        "generatedSourceSpanRecords",
        "generatedStrictEvidenceRecords",
        "appliedGreenfieldRows",
        "postApplyReadbackPassRows",
    ):
        counts.setdefault(key, 0)
    counts["baselineStrictCoveredRows"] = baseline_strict
    counts["strictCoveredRowsAfterLocalApply"] = baseline_strict + int(counts.get("appliedGreenfieldRows") or 0)
    counts["manifestRows"] = int(baseline_counts.get("manifestRows") or 0)

    payload: dict[str, Any] = {
        "schema": STRUCTURED_EVIDENCE_NEXT_SLICE_APPLY_READBACK_SCHEMA_ID,
        "generatedAt": _now_iso(),
        "runId": run_id,
        "apply": bool(apply),
        "status": "ready" if not (counts.get("readbackBlockedRows") or counts.get("greenfieldBlockedRows")) else "blocked",
        "scopeNote": (
            "Apply/readback tranche for the next Structured Evidence slice. Only section_text_offset "
            "greenfield rows are generated; table/equation/runtime/citation-grade work remains closed."
        ),
        "inputs": {
            "corpusManifest": _project_ref(manifest_path),
            "candidateReport": _project_ref(candidate_report_path),
            "candidateReportSchema": schema,
            "papersDirRef": "papers_dir",
        },
        "selection": {
            "readbackSourceIds": readback_ids,
            "greenfieldSourceIds": greenfield_ids,
            "greenfieldSelectionPolicy": "first N selectedGreenfieldCandidates from the schema-backed next-slice report",
        },
        "policy": {
            "runtimeAnswerIntegration": False,
            "complexQaEval": False,
            "manifestMutation": False,
            "tableEquationParser": False,
            "citationGradePromotion": False,
            "runtimeEvidencePromotion": False,
            "externalDownload": False,
            "vaultScan": False,
            "databaseMutation": False,
            "indexMutation": False,
            "sourceContentHashAuthority": "corpus_manifest.expectedSourceContentHash",
        },
        "deferredEvidenceTypes": list(DEFERRED_EVIDENCE_TYPES),
        "counts": dict(counts),
        "readbackRows": readback_rows,
        "greenfieldRows": greenfield_rows,
    }
    validation = validate_payload(payload, STRUCTURED_EVIDENCE_NEXT_SLICE_APPLY_READBACK_SCHEMA_ID, strict=True)
    payload["schemaValidation"] = {
        "ok": validation.ok and validation.schema_found and not validation.errors,
        "schemaFound": bool(validation.schema_found),
        "errors": list(validation.errors),
    }
    if not payload["schemaValidation"]["ok"]:
        payload["status"] = "blocked"
    return payload


def render_structured_evidence_next_slice_apply_readback_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Structured Evidence Next Slice Apply/Readback",
        "",
        f"- generatedAt: {report.get('generatedAt', '')}",
        f"- runId: {report.get('runId', '')}",
        f"- apply: {json.dumps(report.get('apply'))}",
        f"- status: {report.get('status', '')}",
        f"- manifestRows: {int(counts.get('manifestRows') or 0)}",
        f"- baselineStrictCoveredRows: {int(counts.get('baselineStrictCoveredRows') or 0)}",
        f"- strictCoveredRowsAfterLocalApply: {int(counts.get('strictCoveredRowsAfterLocalApply') or 0)}",
        f"- readbackPassRows: {int(counts.get('readbackPassRows') or 0)}",
        f"- greenfieldGeneratedRows: {int(counts.get('greenfieldGeneratedRows') or 0)}",
        f"- appliedGreenfieldRows: {int(counts.get('appliedGreenfieldRows') or 0)}",
        f"- postApplyReadbackPassRows: {int(counts.get('postApplyReadbackPassRows') or 0)}",
        "",
        "## Greenfield Rows",
        "",
    ]
    for row in list(report.get("greenfieldRows") or []):
        generated = row.get("generatedRecords") if isinstance(row.get("generatedRecords"), dict) else {}
        post = row.get("postApplyReadback") if isinstance(row.get("postApplyReadback"), dict) else {}
        lines.append(
            f"- `{row.get('sourceId')}` status=`{row.get('status')}` "
            f"applied=`{generated.get('applied')}` postReadback=`{post.get('status', 'not_run')}`"
        )
    lines.extend(["", "## Readback Rows", ""])
    for row in list(report.get("readbackRows") or []):
        existing = row.get("existingRecords") if isinstance(row.get("existingRecords"), dict) else {}
        lines.append(
            f"- `{row.get('sourceId')}` status=`{row.get('status')}` "
            f"strictEvidence={existing.get('strictEvidenceCount')}"
        )
    lines.extend(["", "## Deferred", ""])
    for item in list(report.get("deferredEvidenceTypes") or []):
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "RUN_ID",
    "STRUCTURED_EVIDENCE_NEXT_SLICE_APPLY_READBACK_SCHEMA_ID",
    "build_structured_evidence_next_slice_apply_readback",
    "render_structured_evidence_next_slice_apply_readback_markdown",
]
