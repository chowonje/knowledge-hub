#!/usr/bin/env python3
"""Build priority corpus candidate-to-source-artifact join report (report-only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload

DEFAULT_LEDGER = PROJECT_ROOT / "eval/knowledgeos/fixtures/priority_corpus_candidate_ledger.v1.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "eval/knowledgeos/fixtures/corpus_manifest.json"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "eval/knowledgeos/reports/priority_corpus_source_join_report.v1.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "eval/knowledgeos/reports/priority_corpus_source_join_report.v1.md"
DEFAULT_ALLOWLIST = (
    PROJECT_ROOT / "eval/knowledgeos/fixtures/priority_corpus_manifest_expansion_allowlist.v1.json"
)

JOIN_SCHEMA = "knowledge-hub.priority-corpus-source-join-report.v1"
ALLOWLIST_SCHEMA = "knowledge-hub.priority-corpus-manifest-expansion-allowlist.v1"
TIER_ORDER = {
    "eval_critical": 0,
    "foundational": 1,
    "recent_ai": 2,
    "local_corpus_candidate": 3,
}
_ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}$")
_NORM_RE = re.compile(r"[^a-z0-9]+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_hash(value: Any) -> str:
    text = _clean(value).lower()
    if text and not text.startswith("sha256:"):
        text = f"sha256:{text}"
    return text


def _normalize_label(value: Any) -> str:
    return _NORM_RE.sub("", _clean(value).casefold())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_schema_valid(payload: dict[str, Any], schema_id: str) -> None:
    validation = validate_payload(payload, schema_id, strict=True)
    if not validation.ok or validation.errors:
        joined = "; ".join(validation.errors)
        raise ValueError(f"{schema_id} validation failed: {joined}")


def _project_ref(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _default_papers_dir() -> Path:
    return Path.home() / ("." + "khub") / "papers"


def _run_inventory() -> dict[str, Any]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "knowledge_hub.interfaces.cli.main",
            "paper",
            "corpus-source-artifact-inventory",
            "--json",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def _manifest_indexes(manifest: dict[str, Any]) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    by_source: dict[str, dict] = {}
    by_filename: dict[str, dict] = {}
    by_hash: dict[str, dict] = {}
    for entry in manifest.get("artifacts") or []:
        for sid in entry.get("sourceIds") or []:
            by_source[sid] = entry
        filename = _clean(entry.get("expectedFilename"))
        if filename:
            by_filename[filename] = entry
        expected_hash = _normalize_hash(entry.get("expectedSourceContentHash"))
        if expected_hash:
            by_hash[expected_hash] = entry
    return by_source, by_filename, by_hash


def _map_inventory_to_candidate(
    candidate: dict[str, Any],
    inventory_items: list[dict[str, Any]],
    *,
    by_filename: dict[str, dict],
    by_hash: dict[str, dict],
) -> tuple[list[dict[str, Any]], str]:
    source_id = candidate["source_id"]
    direct = [item for item in inventory_items if item.get("sourceId") == source_id]
    if direct:
        return direct, "direct_source_id"

    manifest_filename = _clean(candidate.get("manifest_expected_filename"))
    if manifest_filename:
        filename_matches = [item for item in inventory_items if item.get("filename") == manifest_filename]
        if filename_matches:
            return filename_matches, "manifest_expected_filename"

    manifest_entry = by_filename.get(manifest_filename) if manifest_filename else None
    if manifest_entry is None:
        for entry in by_filename.values():
            if source_id in (entry.get("sourceIds") or []):
                manifest_entry = entry
                break
    if manifest_entry:
        expected_hash = _normalize_hash(manifest_entry.get("expectedSourceContentHash"))
        if expected_hash:
            hash_matches = [item for item in inventory_items if item.get("sha256") == expected_hash]
            if hash_matches:
                return hash_matches, "manifest_expected_hash"

    title = _clean(candidate.get("title"))
    if len(title) >= 12:
        normalized_title = _normalize_label(title)
        title_matches: list[dict[str, Any]] = []
        for item in inventory_items:
            filename = _clean(item.get("filename"))
            stem = Path(filename).stem
            normalized_filename = _normalize_label(stem)
            if normalized_title and (
                normalized_title in normalized_filename or normalized_filename in normalized_title
            ):
                title_matches.append(item)
        if title_matches:
            return title_matches, "title_filename_bridge"

    return [], "unmatched"


def _canonicalize_artifact_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, bool, list[str]]:
    warnings: list[str] = []
    if not rows:
        return None, False, warnings

    pdfs = [row for row in rows if row.get("sourceType") == "pdf"]
    texts = [row for row in rows if row.get("sourceType") == "text"]
    chosen_pool = pdfs if pdfs else texts
    if pdfs and texts:
        warnings.append("pdf_preferred_over_text")

    unique_hashes = sorted({row.get("sha256") for row in chosen_pool if row.get("sha256")})
    if len(unique_hashes) > 1:
        warnings.append("ambiguous_multiple_hashes")
        return None, True, warnings

    if len(chosen_pool) > 1:
        warnings.append("duplicate_file_rows_canonicalized")

    chosen = sorted(chosen_pool, key=lambda row: (_clean(row.get("filename")), _clean(row.get("corpusLocationRef"))))[0]
    return chosen, False, warnings


def _parsed_manifest_exists(source_id: str, papers_dir: Path) -> bool:
    candidates = [
        papers_dir / "parsed" / source_id / "manifest.json",
        papers_dir / "parsed" / source_id.replace(".", "_").replace("-", "_") / "manifest.json",
    ]
    return any(path.is_file() for path in candidates)


def build_join_report(
    *,
    ledger_path: Path,
    manifest_path: Path,
    inventory_payload: dict[str, Any],
    papers_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = _load_json(ledger_path)
    manifest = _load_json(manifest_path)
    candidates = list(ledger.get("candidates") or [])
    inventory_items = list(inventory_payload.get("items") or [])
    by_source, by_filename, by_hash = _manifest_indexes(manifest)

    resolved_papers_dir = papers_dir or _default_papers_dir()

    join_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    ambiguous_count = 0
    parsed_missing_count = 0
    matched_count = 0

    for candidate in candidates:
        source_id = candidate["source_id"]
        manifest_entry = by_source.get(source_id)
        in_manifest = candidate.get("current_manifest_status") == "in_manifest" or manifest_entry is not None
        mapped_rows, join_method = _map_inventory_to_candidate(
            candidate,
            inventory_items,
            by_filename=by_filename,
            by_hash=by_hash,
        )
        artifact, is_ambiguous, artifact_warnings = _canonicalize_artifact_rows(mapped_rows)

        expected_hash = _normalize_hash(manifest_entry.get("expectedSourceContentHash")) if manifest_entry else ""
        observed_hash = _normalize_hash(artifact.get("sha256")) if artifact else ""
        byte_length = artifact.get("byteLength") if artifact else None

        join_status = "source_missing"
        if is_ambiguous:
            join_status = "ambiguous"
            ambiguous_count += 1
        elif artifact is None and join_method == "unmatched":
            if not _clean(source_id) or (not _clean(candidate.get("title")) and not manifest_entry):
                join_status = "metadata_only"
            else:
                join_status = "source_missing"
        elif artifact is None:
            join_status = "metadata_only"
        elif not observed_hash or byte_length in (None, ""):
            join_status = "hash_missing"
        elif expected_hash and observed_hash != expected_hash:
            join_status = "hash_mismatch"
        elif observed_hash and byte_length is not None:
            join_status = "available"
            matched_count += 1
        else:
            join_status = "metadata_only"

        parsed_present = False
        parsed_missing = False
        if join_status == "available":
            parsed_present = _parsed_manifest_exists(source_id, resolved_papers_dir)
            parsed_missing = not parsed_present
            if parsed_missing:
                parsed_missing_count += 1

        status_counts[join_status] += 1

        join_rows.append(
            {
                "source_id": source_id,
                "title": candidate.get("title"),
                "year": candidate.get("year"),
                "candidate_tier": candidate.get("candidate_tier"),
                "current_manifest_status": "in_manifest" if in_manifest else "not_in_manifest",
                "join_method": join_method,
                "join_status": join_status,
                "parsed_status": (
                    "parsed_present"
                    if parsed_present
                    else ("parsed_missing" if join_status == "available" else "not_applicable")
                ),
                "expected_source_content_hash": expected_hash or None,
                "observed_sha256": observed_hash or None,
                "byte_length": byte_length,
                "canonical_filename": _clean(artifact.get("filename")) if artifact else None,
                "corpus_location_ref": _clean(artifact.get("corpusLocationRef")) if artifact else None,
                "source_type": artifact.get("sourceType") if artifact else None,
                "mapped_inventory_file_rows": len(mapped_rows),
                "warnings": sorted(set(artifact_warnings)),
                "expansion_allowlist_eligible": False,
            }
        )

    allowlist_rows: list[dict[str, Any]] = []
    for row in join_rows:
        benign_warnings = {"pdf_preferred_over_text", "duplicate_file_rows_canonicalized"}
        warnings = set(row["warnings"])
        eligible = (
            row["current_manifest_status"] == "not_in_manifest"
            and row["join_status"] == "available"
            and warnings.issubset(benign_warnings)
        )
        if row["join_status"] == "ambiguous":
            eligible = False
        if "ambiguous_multiple_hashes" in row["warnings"]:
            eligible = False
        row["expansion_allowlist_eligible"] = eligible
        if eligible:
            allowlist_rows.append(row)

    allowlist_rows.sort(
        key=lambda row: (
            TIER_ORDER.get(row["candidate_tier"], 9),
            -(row.get("year") or 0),
            row["source_id"],
        )
    )

    first_tranche = allowlist_rows[:50]
    tier_counts = Counter(row["candidate_tier"] for row in allowlist_rows)

    report = {
        "schema": JOIN_SCHEMA,
        "generated_at": _now_iso(),
        "scope_note": (
            "Report-only join of priority corpus candidate ledger rows against local source "
            "artifact inventory file rows. Does not mutate corpus_manifest.json, download "
            "sources, or register manifest entries."
        ),
        "inputs": {
            "candidate_ledger": _project_ref(ledger_path),
            "corpus_manifest": _project_ref(manifest_path),
            "inventory_schema": inventory_payload.get("schema"),
            "inventory_generated_at": inventory_payload.get("generatedAt"),
        },
        "inventory_summary": {
            "inventory_file_rows": int((inventory_payload.get("counts") or {}).get("inventoryRows") or 0),
            "inventory_unique_source_ids": len({item.get("sourceId") for item in inventory_items}),
            "note": "inventory_file_rows is file-row count, not unique paper/source_id count",
        },
        "join_rules": {
            "join_key": "candidate.source_id",
            "inventory_canonicalization": [
                "deduplicate inventory file rows per candidate source_id",
                "prefer pdf over text when both exist",
                "mark ambiguous when multiple pdf hashes disagree",
            ],
            "manifest_rows_excluded_from_expansion_allowlist": True,
        },
        "counts": {
            "total_candidate_rows": len(candidates),
            "unique_source_id_count": len({row["source_id"] for row in join_rows}),
            "matched_to_artifact_count": matched_count,
            "unmatched_or_source_missing_count": int(status_counts.get("source_missing") or 0),
            "metadata_only_count": int(status_counts.get("metadata_only") or 0),
            "hash_missing_count": int(status_counts.get("hash_missing") or 0),
            "hash_mismatch_count": int(status_counts.get("hash_mismatch") or 0),
            "ambiguous_count": ambiguous_count,
            "available_count": int(status_counts.get("available") or 0),
            "parsed_missing_count": parsed_missing_count,
            "already_in_manifest_count": sum(1 for row in join_rows if row["current_manifest_status"] == "in_manifest"),
            "expansion_allowlist_count": len(allowlist_rows),
            "first_tranche_recommended_count": len(first_tranche),
            "expansion_allowlist_by_tier": dict(sorted(tier_counts.items(), key=lambda item: TIER_ORDER.get(item[0], 9))),
            "join_status_breakdown": dict(sorted(status_counts.items())),
        },
        "rows": join_rows,
        "first_tranche_recommendation": [
            {
                "source_id": row["source_id"],
                "title": row.get("title"),
                "year": row.get("year"),
                "candidate_tier": row.get("candidate_tier"),
                "observed_sha256": row.get("observed_sha256"),
                "byte_length": row.get("byte_length"),
                "canonical_filename": row.get("canonical_filename"),
            }
            for row in first_tranche
        ],
    }

    allowlist_payload = {
        "schema": ALLOWLIST_SCHEMA,
        "generated_at": report["generated_at"],
        "scope_note": (
            "Expansion allowlist only. Rows are not registered in corpus_manifest.json at generation time. "
            "All rows are not_in_manifest, available, non-ambiguous, and source-byte verified via inventory join."
        ),
        "source_report": _project_ref(DEFAULT_REPORT_JSON),
        "counts": {
            "allowlist_rows": len(allowlist_rows),
            "first_tranche_recommended_rows": len(first_tranche),
            "by_candidate_tier": dict(sorted(tier_counts.items(), key=lambda item: TIER_ORDER.get(item[0], 9))),
            "already_in_manifest_excluded": report["counts"]["already_in_manifest_count"],
        },
        "first_tranche_size_target": "30-50",
        "allowlist": [
            {
                "source_id": row["source_id"],
                "title": row.get("title"),
                "year": row.get("year"),
                "candidate_tier": row.get("candidate_tier"),
                "join_method": row.get("join_method"),
                "observed_sha256": row.get("observed_sha256"),
                "byte_length": row.get("byte_length"),
                "canonical_filename": row.get("canonical_filename"),
                "corpus_location_ref": row.get("corpus_location_ref"),
                "provenance_url": next(
                    (
                        candidate.get("provenance_url")
                        for candidate in candidates
                        if candidate["source_id"] == row["source_id"]
                    ),
                    None,
                ),
                "parsed_status": row.get("parsed_status"),
            }
            for row in allowlist_rows
        ],
        "first_tranche_recommendation": report["first_tranche_recommendation"],
    }

    return report, allowlist_payload


def render_markdown(report: dict[str, Any]) -> str:
    counts = report.get("counts") or {}
    inv = report.get("inventory_summary") or {}
    lines = [
        "# Priority Corpus Source Join Report",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        "## Scope",
        "",
        report.get("scope_note", ""),
        "",
        "## Inventory Input (file rows, not paper count)",
        "",
        f"- Inventory file rows: **{inv.get('inventory_file_rows', 0)}**",
        f"- Inventory unique source ids (unjoined): **{inv.get('inventory_unique_source_ids', 0)}**",
        "",
        "## Join Result",
        "",
        f"- Total candidate rows: **{counts.get('total_candidate_rows', 0)}**",
        f"- Unique source_id count: **{counts.get('unique_source_id_count', 0)}**",
        f"- Matched to artifact (`available`): **{counts.get('matched_to_artifact_count', 0)}**",
        f"- Source missing: **{counts.get('unmatched_or_source_missing_count', 0)}**",
        f"- Metadata only: **{counts.get('metadata_only_count', 0)}**",
        f"- Hash missing: **{counts.get('hash_missing_count', 0)}**",
        f"- Hash mismatch: **{counts.get('hash_mismatch_count', 0)}**",
        f"- Ambiguous: **{counts.get('ambiguous_count', 0)}**",
        f"- Parsed missing (available only): **{counts.get('parsed_missing_count', 0)}**",
        f"- Already in manifest (excluded from allowlist): **{counts.get('already_in_manifest_count', 0)}**",
        "",
        "## Expansion Allowlist",
        "",
        f"- Allowlist rows: **{counts.get('expansion_allowlist_count', 0)}**",
        f"- First tranche recommended: **{counts.get('first_tranche_recommended_count', 0)}**",
        "",
        "### Allowlist by tier",
        "",
    ]
    for tier, value in (counts.get("expansion_allowlist_by_tier") or {}).items():
        lines.append(f"- `{tier}`: **{value}**")
    lines.extend(["", "## First Tranche Recommendation (up to 50)", ""])
    for row in report.get("first_tranche_recommendation") or []:
        title = _clean(row.get("title")).replace("|", "\\|")[:70].rstrip()
        lines.append(
            f"- `{row.get('source_id')}` ({row.get('candidate_tier')}, {row.get('year') or 'n/a'}) — {title}"
        )
    lines.extend(
        [
            "",
            "## Status Breakdown",
            "",
        ]
    )
    for status, value in (counts.get("join_status_breakdown") or {}).items():
        lines.append(f"- `{status}`: **{value}**")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    args = parser.parse_args()

    inventory_payload = _run_inventory()
    report, allowlist_payload = build_join_report(
        ledger_path=args.ledger,
        manifest_path=args.manifest,
        inventory_payload=inventory_payload,
    )
    _assert_schema_valid(report, JOIN_SCHEMA)
    _assert_schema_valid(allowlist_payload, ALLOWLIST_SCHEMA)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.allowlist.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    args.allowlist.write_text(json.dumps(allowlist_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
