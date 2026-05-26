"""Read-only dirty-checkout inventory for the v0.1 text-evidence RC line."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import utc_now_iso

TEXT_EVIDENCE_CANONICAL_DIRTY_INVENTORY_SCHEMA_ID = (
    "knowledge-hub.paper.text-evidence-canonical-dirty-inventory.v1"
)

PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _private_path_leak_count(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return 1 if PRIVATE_PATH_RE.search(encoded) else 0


def _run_text(args: Sequence[str], *, cwd: Path | None = None, timeout: int = 8) -> str:
    try:
        result = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.rstrip("\n")


def _git_text(repo: Path, *args: str) -> str:
    return _run_text(["git", "-C", str(repo), *args])


def _status_lines(repo: Path) -> list[str]:
    output = _git_text(repo, "status", "--short")
    return [line for line in output.splitlines() if line.strip()]


def _path_from_status_line(line: str) -> tuple[str, str]:
    status = line[:2]
    path = line[3:].strip() if len(line) > 3 else ""
    if " -> " in path:
        path = path.split(" -> ", 1)[1].strip()
    return status, path


def _path_kind(status_code: str) -> str:
    if status_code == "??":
        return "untracked"
    if status_code.strip():
        return "tracked_dirty"
    return "unknown"


def _classify_path(path: str) -> tuple[str, str, str]:
    lower = path.lower()
    if path.startswith(("tasks/", "reviews/", "worklog/", "artifacts/")):
        return (
            "workspace_process_record",
            "exclude_from_public_rc_or_move_to_workspace_records",
            "local process records are not product behavior",
        )
    if (
        "research_object" in lower
        or path.startswith("docs/research_objects/")
        or path.startswith("knowledge_hub/research_objects/")
        or path.startswith("eval/knowledgeos/fixtures/research_objects/")
        or path.startswith("tests/fixtures/research_objects/")
    ):
        return (
            "research_objects_side_stack",
            "hold_outside_text_evidence_rc",
            "research-object planning is outside the current text-evidence RC scope",
        )
    if "provider_hint" in lower or "provider_policy" in lower:
        return (
            "provider_hint_side_stack",
            "hold_outside_text_evidence_rc",
            "provider hint shadow work is outside the current text-evidence RC scope",
        )
    if (
        "evidence" in lower
        or "answer_contract" in lower
        or "answer_verification" in lower
        or "source-ledger" in lower
        or "prepared-source" in lower
        or "prepared_source" in lower
    ):
        return (
            "evidence_spine_or_source_contract_stack",
            "review_as_separate_clean_replay_candidate",
            "evidence/source contract work needs its own clean replay and verification scope",
        )
    if "pymupdf" in lower or "parsed_artifact" in lower or "paper_table" in lower:
        return (
            "parser_artifact_side_stack",
            "hold_for_parser_track",
            "parser/artifact repair is deferred from the text-only RC convergence line",
        )
    if path.startswith(("knowledge_hub/interfaces/", "knowledge_hub/mcp/", "scripts/check_release_smoke.py")):
        return (
            "cli_mcp_public_surface_stack",
            "compare_against_public_operator_cleanup_before_replay",
            "CLI/MCP changes overlap public surface rules and need explicit replay review",
        )
    if path.startswith(
        (
            "knowledge_hub/library/",
            "knowledge_hub/project/",
            "knowledge_hub/papers/manager.py",
            "knowledge_hub/papers/source_text.py",
            "knowledge_hub/papers/vault_links.py",
            "knowledge_hub/vault/indexer.py",
            "knowledge_hub/web/ingest.py",
        )
    ):
        return (
            "source_ingest_or_library_stack",
            "review_as_separate_clean_replay_candidate",
            "source ingest/library/vault-link changes need their own clean replay scope",
        )
    if path.startswith(("knowledge_hub/ai/", "knowledge_hub/application/", "knowledge_hub/domain/ai_papers/")):
        return (
            "answer_runtime_or_query_stack",
            "review_after_text_evidence_rc_candidate",
            "answer/runtime/query changes are broader than the report-only text-evidence RC line",
        )
    if path.startswith(("eval/knowledgeos/", "tests/")):
        return (
            "eval_or_test_support_stack",
            "map_to_owning_feature_before_replay",
            "eval/test changes must follow the owning clean feature branch",
        )
    if path.startswith(("docs/", "README.md", "CHANGELOG.md")):
        return (
            "docs_governance_stack",
            "review_for_record_sync_only",
            "docs/governance edits need reconciliation with current product state",
        )
    if path.startswith(("knowledge_hub/infrastructure/", "knowledge_hub/core/", "pyproject.toml")):
        return (
            "infrastructure_or_core_stack",
            "review_as_separate_clean_replay_candidate",
            "core/infrastructure changes have cross-cutting blast radius",
        )
    return (
        "unknown",
        "manual_classification_required",
        "path does not match a known convergence bucket",
    )


def _row_from_status_line(line: str) -> dict[str, Any]:
    status_code, path = _path_from_status_line(line)
    bucket, rc_disposition, reason = _classify_path(path)
    return {
        "statusCode": status_code,
        "path": path,
        "pathKind": _path_kind(status_code),
        "bucket": bucket,
        "rcDisposition": rc_disposition,
        "reason": reason,
    }


def _canonical_state(canonical_repo: Path | None) -> dict[str, Any]:
    if canonical_repo is None or not canonical_repo.exists():
        return {"available": False, "repoRef": "canonical_product_checkout", "branch": "", "head": "", "dirtyRows": 0}
    return {
        "available": True,
        "repoRef": "canonical_product_checkout",
        "branch": _git_text(canonical_repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "head": _git_text(canonical_repo, "rev-parse", "--short", "HEAD"),
        "dirtyRows": len(_status_lines(canonical_repo)),
    }


def build_text_evidence_canonical_dirty_inventory_report(
    *,
    canonical_repo: Path | None,
    status_lines: Sequence[str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    lines = list(status_lines) if status_lines is not None else (_status_lines(canonical_repo) if canonical_repo else [])
    rows = [_row_from_status_line(line) for line in lines]
    bucket_counts = dict(sorted(Counter(row["bucket"] for row in rows).items()))
    disposition_counts = dict(sorted(Counter(row["rcDisposition"] for row in rows).items()))
    unknown_rows = sum(1 for row in rows if row["bucket"] == "unknown")
    untracked_rows = sum(1 for row in rows if row["pathKind"] == "untracked")
    tracked_dirty_rows = sum(1 for row in rows if row["pathKind"] == "tracked_dirty")

    report: dict[str, Any] = {
        "schema": TEXT_EVIDENCE_CANONICAL_DIRTY_INVENTORY_SCHEMA_ID,
        "status": "ready",
        "generatedAt": generated_at or utc_now_iso(),
        "scope": {
            "writes": "report_only",
            "canonicalCheckoutEdited": False,
            "fileContentReadRows": 0,
            "vaultScanRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "externalDownloadRows": 0,
            "mergeRows": 0,
            "cherryPickRows": 0,
            "worktreeDeletionRows": 0,
        },
        "canonicalCheckout": _canonical_state(canonical_repo),
        "dirtyRows": len(rows),
        "trackedDirtyRows": tracked_dirty_rows,
        "untrackedRows": untracked_rows,
        "unknownRows": unknown_rows,
        "bucketCounts": bucket_counts,
        "dispositionCounts": disposition_counts,
        "rows": rows,
        "nextAction": "decide_keep_drop_or_clean_replay_per_bucket_before_public_rc",
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "mergeRows": 0,
            "cherryPickRows": 0,
            "worktreeDeletionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
            "strictEvidencePromotionRows": 0,
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "reportHash": "",
        "warnings": [
            "This inventory reads git status only; it does not inspect file content.",
            "Do not replay canonical dirty files into the text-evidence RC without bucket-level review.",
        ],
        "schemaErrors": [],
    }
    report["privatePathLeakRows"] = _private_path_leak_count(report)
    if report["privatePathLeakRows"]:
        report["status"] = "blocked"
    payload_for_hash = dict(report)
    payload_for_hash["reportHash"] = ""
    report["reportHash"] = _sha256_json(payload_for_hash)
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Text Evidence Canonical Dirty Inventory",
        "",
        f"- status: `{report.get('status')}`",
        f"- dirtyRows: `{report.get('dirtyRows')}`",
        f"- trackedDirtyRows: `{report.get('trackedDirtyRows')}`",
        f"- untrackedRows: `{report.get('untrackedRows')}`",
        f"- unknownRows: `{report.get('unknownRows')}`",
        f"- privatePathLeakRows: `{report.get('privatePathLeakRows')}`",
        f"- nextAction: `{report.get('nextAction')}`",
        "",
        "## Bucket Counts",
        "",
        "| bucket | rows |",
        "|---|---:|",
    ]
    for bucket, count in dict(report.get("bucketCounts") or {}).items():
        lines.append(f"| `{bucket}` | `{count}` |")
    lines.extend(
        [
            "",
            "## Disposition Counts",
            "",
            "| disposition | rows |",
            "|---|---:|",
        ]
    )
    for disposition, count in dict(report.get("dispositionCounts") or {}).items():
        lines.append(f"| `{disposition}` | `{count}` |")
    lines.extend(
        [
            "",
            "## Dirty Rows",
            "",
            "| status | bucket | disposition | path |",
            "|---|---|---|---|",
        ]
    )
    for row in report.get("rows", []):
        lines.append(
            f"| `{row.get('statusCode')}` | `{row.get('bucket')}` | "
            f"`{row.get('rcDisposition')}` | `{row.get('path')}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


__all__ = [
    "TEXT_EVIDENCE_CANONICAL_DIRTY_INVENTORY_SCHEMA_ID",
    "build_text_evidence_canonical_dirty_inventory_report",
    "render_markdown_report",
    "write_report",
]
