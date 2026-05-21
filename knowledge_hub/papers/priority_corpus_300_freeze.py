"""Report-only 300-row priority corpus freeze and evidence-slice planning.

This module locks the current corpus/source-artifact state into schema-backed
reports. It does not mutate the corpus manifest, download source artifacts,
promote structured evidence, run answers, or scan vault content.
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
    corpus_manifest_entries,
    load_corpus_manifest,
)
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.config import Config
from knowledge_hub.papers.corpus_manifest_validation import validate_corpus_manifest


PRIORITY_CORPUS_300_FREEZE_SCHEMA_ID = "knowledge-hub.priority-corpus-300-freeze-report.v1"
STRUCTURED_EVIDENCE_NEXT_SLICE_SCHEMA_ID = (
    "knowledge-hub.paper.structured-evidence-next-slice-candidate-report.v1"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOIN_REPORT_PATH = PROJECT_ROOT / "eval" / "knowledgeos" / "reports" / (
    "priority_corpus_source_join_report.v1.json"
)
DEFAULT_ALLOWLIST_PATH = PROJECT_ROOT / "eval" / "knowledgeos" / "fixtures" / (
    "priority_corpus_manifest_expansion_allowlist.v1.json"
)

TIER_ORDER = {
    "eval_critical": 0,
    "foundational": 1,
    "recent_ai": 2,
    "local_corpus_candidate": 3,
}
MINIMAL_EVIDENCE_TYPES = ("section_text_offset", "figure_caption_text")
DEFERRED_EVIDENCE_TYPES = ("table_cell_numeric", "equation_citation", "appendix_table_lookup")
OPERATOR_LOCAL_RUN_IDS_EXCLUDED = frozenset({"structured-evidence-vertical-slice-20260521"})
GREENFIELD_CARRYOVER_SOURCE_IDS = ("2005.11401", "1512.03385")
DISCOVERY_PRIORITY_SOURCE_IDS = (
    "2201.11903",
    "2312.00752",
    "2310.11511",
    "2404.16130",
    "1810.04805",
    "2410.05779",
    "1502.03167",
    "alexnet-2012",
    "2007.01282",
)
RAG_AGENT_PRIORITY_SOURCE_IDS = (
    "2410.11414",
    "2309.15217",
    "2312.10997",
    "2403.14403",
    "2407.02485",
    "2308.08155",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _load_json(path: str | Path) -> dict[str, Any]:
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


def _source_id_for_entry(entry: dict[str, Any]) -> str:
    source_ids = [str(item) for item in (entry.get("sourceIds") or []) if _clean(item)]
    if source_ids:
        return _clean(source_ids[0])
    return _clean(entry.get("sourceId"))


def _manifest_source_ids(manifest: dict[str, Any]) -> set[str]:
    return {
        source_id
        for entry in corpus_manifest_entries(manifest)
        for source_id in [_source_id_for_entry(entry)]
        if source_id
    }


def _source_rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_clean(row.get("source_id")): row for row in rows if _clean(row.get("source_id"))}


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(_clean(row.get(key)) or "unknown" for row in rows).items()))


def _public_join_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": row.get("source_id"),
        "title": row.get("title"),
        "year": row.get("year"),
        "candidate_tier": row.get("candidate_tier"),
        "join_status": row.get("join_status"),
        "current_manifest_status": row.get("current_manifest_status"),
        "parsed_status": row.get("parsed_status"),
        "warnings": list(row.get("warnings") or []),
    }


def _schema_validation(schema_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    validation = validate_payload(payload, schema_id, strict=True)
    return {
        "ok": validation.ok and validation.schema_found and not validation.errors,
        "schemaFound": bool(validation.schema_found),
        "errors": list(validation.errors),
    }


def build_priority_corpus_300_freeze_report(
    *,
    config: Any | None = None,
    manifest_path: str | Path = DEFAULT_CORPUS_MANIFEST_PATH,
    join_report_path: str | Path = DEFAULT_JOIN_REPORT_PATH,
    allowlist_path: str | Path = DEFAULT_ALLOWLIST_PATH,
    papers_dir: str | Path | None = None,
    lower_bound_target: int = 300,
    upper_bound_target: int = 500,
) -> dict[str, Any]:
    """Build a report that freezes the current verified corpus lower bound."""

    config = config or Config()
    resolved_papers_dir = _configured_papers_dir(config, papers_dir)
    manifest = load_corpus_manifest(manifest_path)
    manifest_source_ids = _manifest_source_ids(manifest)
    join_report = _load_json(join_report_path)
    allowlist = _load_json(allowlist_path)
    join_rows = list(join_report.get("rows") or [])
    join_by_source = _source_rows_by_id(join_rows)
    allowlist_rows = list(allowlist.get("allowlist") or [])

    validation_report = validate_corpus_manifest(
        config=config,
        manifest_path=manifest_path,
        papers_dir=resolved_papers_dir,
        check_artifacts=True,
        check_parsed=True,
    )
    validation_counts = dict(validation_report.get("counts") or {})

    manifest_join_rows = [join_by_source[source_id] for source_id in sorted(manifest_source_ids) if source_id in join_by_source]
    manifest_bad_join_rows = [row for row in manifest_join_rows if row.get("join_status") != "available"]
    source_missing_rows = [row for row in join_rows if row.get("join_status") == "source_missing"]
    ambiguous_rows = [row for row in join_rows if row.get("join_status") == "ambiguous"]
    hold_rows = source_missing_rows + ambiguous_rows
    allowlist_source_ids = {_clean(row.get("source_id")) for row in allowlist_rows}
    manifest_ids_in_allowlist = sorted(manifest_source_ids & allowlist_source_ids)
    hold_ids_in_manifest = sorted(
        _clean(row.get("source_id")) for row in hold_rows if _clean(row.get("source_id")) in manifest_source_ids
    )
    hold_ids_in_allowlist = sorted(
        _clean(row.get("source_id")) for row in hold_rows if _clean(row.get("source_id")) in allowlist_source_ids
    )

    join_counts = dict(join_report.get("counts") or {})
    verified_available_total = int(
        join_counts.get("available_count")
        or join_counts.get("matched_to_artifact_count")
        or len([row for row in join_rows if row.get("join_status") == "available"])
    )
    manifest_rows = len(manifest_source_ids)
    remaining_verified_allowlist = len(allowlist_rows)
    current_ledger_candidate_rows = int(join_counts.get("total_candidate_rows") or len(join_rows))
    current_ledger_ceiling_if_all_holds_resolved = verified_available_total + len(hold_rows)
    additional_verified_needed_for_500 = max(0, upper_bound_target - verified_available_total)
    additional_new_candidates_needed_after_holds = max(
        0,
        upper_bound_target - current_ledger_ceiling_if_all_holds_resolved,
    )

    payload: dict[str, Any] = {
        "schema": PRIORITY_CORPUS_300_FREEZE_SCHEMA_ID,
        "generatedAt": _now_iso(),
        "status": "locked" if validation_report.get("status") == "ok" and manifest_rows >= lower_bound_target else "blocked",
        "scopeNote": (
            "Report-only freeze of the verified priority corpus lower bound. The corpus manifest is "
            "not mutated; source_missing and ambiguous candidates remain hold-only."
        ),
        "inputs": {
            "corpusManifest": _project_ref(manifest_path),
            "prioritySourceJoinReport": _project_ref(join_report_path),
            "priorityManifestExpansionAllowlist": _project_ref(allowlist_path),
            "papersDirRef": "papers_dir",
        },
        "targets": {
            "lowerBoundTargetRows": lower_bound_target,
            "upperBoundTargetRows": upper_bound_target,
            "lowerBoundReached": manifest_rows >= lower_bound_target,
            "upperBoundReachedWithVerifiedAvailable": verified_available_total >= upper_bound_target,
            "currentLedgerCandidateRows": current_ledger_candidate_rows,
            "immediateManifestCeilingWithVerifiedAvailable": verified_available_total,
            "currentLedgerCeilingIfAllHoldsResolved": current_ledger_ceiling_if_all_holds_resolved,
            "additionalVerifiedAvailableNeededForUpperBound": additional_verified_needed_for_500,
            "additionalNewCandidateRowsNeededForUpperBoundAfterAllCurrentHoldsResolved": (
                additional_new_candidates_needed_after_holds
            ),
        },
        "counts": {
            "manifestRows": manifest_rows,
            "sourceAvailableRows": int(validation_counts.get("sourceAvailableRows") or 0),
            "sourceMissingRowsInManifest": int(validation_counts.get("sourceMissingRows") or 0),
            "hashMissingRowsInManifest": int(validation_counts.get("hashMissingRows") or 0),
            "hashMismatchRowsInManifest": int(validation_counts.get("hashMismatchRows") or 0),
            "parsedAvailableRows": int(validation_counts.get("parsedAvailableRows") or 0),
            "parsedMissingRows": int(validation_counts.get("parsedMissingRows") or 0),
            "blockerRows": int(validation_counts.get("blockerRows") or 0),
            "verifiedAvailableCandidateRows": verified_available_total,
            "alreadyInManifestRows": int(join_counts.get("already_in_manifest_count") or manifest_rows),
            "remainingVerifiedAllowlistRows": remaining_verified_allowlist,
            "sourceMissingHoldRows": len(source_missing_rows),
            "ambiguousHoldRows": len(ambiguous_rows),
            "holdOrExcludeRows": len(hold_rows),
            "manifestRowsWithNonAvailableJoinStatus": len(manifest_bad_join_rows),
            "manifestRowsStillInAllowlist": len(manifest_ids_in_allowlist),
            "holdRowsInManifest": len(hold_ids_in_manifest),
            "holdRowsInAllowlist": len(hold_ids_in_allowlist),
        },
        "tierBreakdown": {
            "manifestByTier": _count_by(manifest_join_rows, "candidate_tier"),
            "remainingAllowlistByTier": dict(allowlist.get("counts", {}).get("by_candidate_tier") or {}),
            "sourceMissingHoldByTier": _count_by(source_missing_rows, "candidate_tier"),
            "ambiguousHoldByTier": _count_by(ambiguous_rows, "candidate_tier"),
        },
        "lockRules": {
            "registeredAvailableRule": (
                "Only rows with local source bytes, SHA-256, byteLength, and resolver-visible source artifacts "
                "may be registered as available manifest rows."
            ),
            "holdRule": "source_missing and ambiguous rows are report-only holds until bytes/hash/operator selection is verified.",
            "parsedRule": "parsed_missing is reported separately from source artifact availability.",
            "publicPathRule": "Public reports use `papers_dir/...` refs only and do not include absolute local paths.",
        },
        "safetyChecks": {
            "manifestMutation": False,
            "networkUsed": False,
            "vaultScan": False,
            "sourceMissingPromoted": False,
            "ambiguousPromoted": False,
            "sourceMissingRowsExcluded": len(hold_ids_in_manifest) == 0 and not any(
                _clean(row.get("source_id")) in allowlist_source_ids for row in source_missing_rows
            ),
            "ambiguousRowsExcluded": not any(row.get("join_status") == "ambiguous" for row in manifest_bad_join_rows)
            and not any(_clean(row.get("source_id")) in allowlist_source_ids for row in ambiguous_rows),
            "hashMismatchGreen": int(validation_counts.get("hashMismatchRows") or 0) > 0,
            "hashMissingGreen": int(validation_counts.get("hashMissingRows") or 0) > 0,
        },
        "holdRows": [_public_join_row(row) for row in sorted(hold_rows, key=lambda item: (_clean(item.get("join_status")), _clean(item.get("source_id"))))],
        "manifestBadJoinRows": [_public_join_row(row) for row in manifest_bad_join_rows],
        "manifestRowsStillInAllowlist": manifest_ids_in_allowlist,
        "holdRowsInManifest": hold_ids_in_manifest,
        "holdRowsInAllowlist": hold_ids_in_allowlist,
    }
    payload["schemaValidation"] = _schema_validation(PRIORITY_CORPUS_300_FREEZE_SCHEMA_ID, payload)
    if not payload["schemaValidation"]["ok"]:
        payload["status"] = "blocked"
    return payload


def _jsonl_counts(path: Path) -> dict[str, int]:
    total = 0
    operator_local = 0
    malformed = 0
    if not path.is_file():
        return {"total": 0, "operatorLocalExcluded": 0, "publicReviewable": 0, "malformed": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            item = json.loads(line)
        except Exception:
            malformed += 1
            continue
        if _clean(item.get("runId")) in OPERATOR_LOCAL_RUN_IDS_EXCLUDED:
            operator_local += 1
    return {
        "total": total,
        "operatorLocalExcluded": operator_local,
        "publicReviewable": max(0, total - operator_local),
        "malformed": malformed,
    }


def _parsed_document_summary(papers_dir: Path, source_id: str) -> dict[str, Any]:
    document_path = papers_dir / "parsed" / source_id / "document.json"
    if not document_path.is_file():
        return {
            "documentJsonRef": f"papers_dir/parsed/{source_id}/document.json",
            "documentJsonExists": False,
            "elementCount": 0,
            "figureArtifactCount": 0,
        }
    try:
        document = json.loads(document_path.read_text(encoding="utf-8"))
    except Exception:
        document = {}
    return {
        "documentJsonRef": f"papers_dir/parsed/{source_id}/document.json",
        "documentJsonExists": True,
        "elementCount": len(list(document.get("elements") or [])),
        "figureArtifactCount": len(list(document.get("figure_artifacts") or [])),
    }


def _recommended_evidence_types(parsed: dict[str, Any]) -> list[str]:
    evidence_types = ["section_text_offset"]
    if int(parsed.get("figureArtifactCount") or 0) > 0:
        evidence_types.append("figure_caption_text")
    return evidence_types


def _evidence_status(strict_counts: dict[str, int]) -> str:
    if strict_counts["total"] <= 0:
        return "missing"
    if strict_counts["publicReviewable"] > 0:
        return "public_reviewable"
    return "operator_local_only"


def _store_counts(store: dict[str, Any]) -> dict[str, int]:
    return {
        "total": int(store.get("total") or 0),
        "operatorLocalExcluded": int(store.get("operatorLocalExcluded") or 0),
        "publicReviewable": int(store.get("publicReviewable") or 0),
        "malformed": int(store.get("malformed") or 0),
    }


def _evidence_candidate_public_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed = dict(row.get("parsedDocument") or {})
    parsed.pop("documentJsonRef", None)
    return {
        "sourceId": row.get("sourceId"),
        "artifactId": row.get("artifactId"),
        "title": row.get("title"),
        "year": row.get("year"),
        "candidateTier": row.get("candidateTier"),
        "sourceArtifactStatus": row.get("sourceArtifactStatus"),
        "parsedArtifactStatus": row.get("parsedArtifactStatus"),
        "parsedDocument": parsed,
        "sourceSpanStore": _store_counts(row.get("sourceSpanStore") or {}),
        "strictEvidenceStore": _store_counts(row.get("strictEvidenceStore") or {}),
        "structuredEvidenceStatus": row.get("structuredEvidenceStatus"),
        "recommendedEvidenceTypes": list(row.get("recommendedEvidenceTypes") or []),
        "eligibleForGreenfieldSlice": bool(row.get("eligibleForGreenfieldSlice")),
        "selectionReason": row.get("selectionReason"),
    }


def _greenfield_rank_bucket(row: dict[str, Any]) -> tuple[int, int]:
    source_id = row["sourceId"]
    if source_id in GREENFIELD_CARRYOVER_SOURCE_IDS:
        return (0, GREENFIELD_CARRYOVER_SOURCE_IDS.index(source_id))
    if source_id in DISCOVERY_PRIORITY_SOURCE_IDS:
        return (1, DISCOVERY_PRIORITY_SOURCE_IDS.index(source_id))
    if source_id in RAG_AGENT_PRIORITY_SOURCE_IDS:
        return (2, RAG_AGENT_PRIORITY_SOURCE_IDS.index(source_id))
    return (3, TIER_ORDER.get(row["candidateTier"], 9))


def build_structured_evidence_next_slice_candidate_report(
    *,
    config: Any | None = None,
    manifest_path: str | Path = DEFAULT_CORPUS_MANIFEST_PATH,
    join_report_path: str | Path = DEFAULT_JOIN_REPORT_PATH,
    papers_dir: str | Path | None = None,
    greenfield_target_rows: int = 30,
) -> dict[str, Any]:
    """Build a report-only candidate list for the next structured evidence slice."""

    config = config or Config()
    resolved_papers_dir = _configured_papers_dir(config, papers_dir)
    manifest = load_corpus_manifest(manifest_path)
    join_report = _load_json(join_report_path)
    join_by_source = _source_rows_by_id(list(join_report.get("rows") or []))
    validation_report = validate_corpus_manifest(
        config=config,
        manifest_path=manifest_path,
        papers_dir=resolved_papers_dir,
        check_artifacts=True,
        check_parsed=True,
    )
    validation_by_source = {
        _clean((item.get("sourceIds") or [""])[0]): item for item in validation_report.get("items") or []
    }

    rows: list[dict[str, Any]] = []
    for entry in corpus_manifest_entries(manifest):
        source_id = _source_id_for_entry(entry)
        if not source_id:
            continue
        join_row = join_by_source.get(source_id, {})
        validation_item = validation_by_source.get(source_id, {})
        source_span_counts = _jsonl_counts(
            resolved_papers_dir / "structured_evidence" / "source_span" / f"{source_id}.jsonl"
        )
        strict_counts = _jsonl_counts(
            resolved_papers_dir / "structured_evidence" / "strict_evidence" / f"{source_id}.jsonl"
        )
        parsed = _parsed_document_summary(resolved_papers_dir, source_id)
        evidence_status = _evidence_status(strict_counts)
        row = {
            "sourceId": source_id,
            "artifactId": corpus_entry_ref(entry),
            "title": join_row.get("title") or Path(_clean(entry.get("expectedFilename"))).stem,
            "year": join_row.get("year"),
            "candidateTier": join_row.get("candidate_tier") or "unknown",
            "sourceArtifactStatus": validation_item.get("sourceArtifactStatus") or "unknown",
            "parsedArtifactStatus": validation_item.get("parsedArtifactStatus") or "unknown",
            "parsedDocument": parsed,
            "sourceSpanStore": {
                "storeRef": f"papers_dir/structured_evidence/source_span/{source_id}.jsonl",
                **source_span_counts,
            },
            "strictEvidenceStore": {
                "storeRef": f"papers_dir/structured_evidence/strict_evidence/{source_id}.jsonl",
                **strict_counts,
            },
            "structuredEvidenceStatus": evidence_status,
            "recommendedEvidenceTypes": _recommended_evidence_types(parsed),
            "eligibleForGreenfieldSlice": (
                validation_item.get("sourceArtifactStatus") == "available"
                and validation_item.get("parsedArtifactStatus") == "available"
                and evidence_status in {"missing", "operator_local_only"}
            ),
            "selectionReason": "",
        }
        reason_parts = []
        if row["candidateTier"] in {"eval_critical", "foundational"}:
            reason_parts.append(f"{row['candidateTier']} corpus row")
        if parsed.get("figureArtifactCount"):
            reason_parts.append("figure artifacts present")
        if evidence_status == "missing":
            reason_parts.append("no strict evidence store yet")
        elif evidence_status == "operator_local_only":
            reason_parts.append("generated-only local side effect exists; carry over as greenfield")
        else:
            reason_parts.append("existing strict evidence readback candidate")
        row["selectionReason"] = "; ".join(reason_parts)
        rows.append(row)

    readback_candidates = [
        row
        for row in rows
        if row["structuredEvidenceStatus"] == "public_reviewable"
    ]
    readback_candidates.sort(
        key=lambda row: (
            TIER_ORDER.get(row["candidateTier"], 9),
            row["structuredEvidenceStatus"] != "public_reviewable",
            row["sourceId"],
        )
    )
    greenfield_candidates = [row for row in rows if row["eligibleForGreenfieldSlice"]]
    greenfield_candidates.sort(
        key=lambda row: (
            _greenfield_rank_bucket(row),
            -(row.get("year") or 0),
            -int((row.get("parsedDocument") or {}).get("figureArtifactCount") or 0),
            row["sourceId"],
        )
    )
    selected_greenfield = greenfield_candidates[:greenfield_target_rows]

    status_counts = Counter(row["structuredEvidenceStatus"] for row in rows)
    tier_counts = Counter(row["candidateTier"] for row in selected_greenfield)
    payload: dict[str, Any] = {
        "schema": STRUCTURED_EVIDENCE_NEXT_SLICE_SCHEMA_ID,
        "generatedAt": _now_iso(),
        "status": "ready" if selected_greenfield else "blocked",
        "scopeNote": (
            "Report-only next-slice candidate list for structured evidence over the 300-row verified corpus. "
            "It does not create SourceSpan or StrictEvidence records."
        ),
        "inputs": {
            "corpusManifest": _project_ref(manifest_path),
            "prioritySourceJoinReport": _project_ref(join_report_path),
            "papersDirRef": "papers_dir",
        },
        "selectionPolicy": {
            "greenfieldTargetRows": greenfield_target_rows,
            "rankOrder": [
                "generated-only carry-over rows from the checked-in apply=false vertical slice",
                "legacy eval-critical discovery priorities",
                "RAG/agent-relevant local-corpus priorities",
                "eval_critical",
                "foundational",
                "recent_ai",
                "local_corpus_candidate",
                "year descending",
                "figure caption availability",
                "source_id",
            ],
            "excludeFromGreenfield": [
                "source artifact unavailable",
                "parsed artifact unavailable",
                "public-reviewable strict evidence store rows",
            ],
            "strictCoverageRule": (
                "Strict coverage counts only public-reviewable strict_evidence JSONL rows; "
                "operator-local runId=structured-evidence-vertical-slice-20260521 rows remain greenfield carry-over."
            ),
        },
        "counts": {
            "manifestRows": len(rows),
            "sourceAvailableRows": int((validation_report.get("counts") or {}).get("sourceAvailableRows") or 0),
            "parsedAvailableRows": int((validation_report.get("counts") or {}).get("parsedAvailableRows") or 0),
            "strictEvidenceStoreRowsRaw": sum(1 for row in rows if row["strictEvidenceStore"]["total"] > 0),
            "strictEvidenceStoreRowsPublicReviewable": sum(
                1 for row in rows if row["strictEvidenceStore"]["publicReviewable"] > 0
            ),
            "strictEvidenceStoreRowsOperatorLocalOnly": sum(
                1
                for row in rows
                if row["strictEvidenceStore"]["total"] > 0 and row["strictEvidenceStore"]["publicReviewable"] == 0
            ),
            "strictCoveredRows": sum(1 for row in rows if row["strictEvidenceStore"]["publicReviewable"] > 0),
            "strictCoveragePct": (
                round(
                    100 * sum(1 for row in rows if row["strictEvidenceStore"]["publicReviewable"] > 0) / len(rows),
                    2,
                )
                if rows
                else 0.0
            ),
            "greenfieldEligibleRows": len(greenfield_candidates),
            "selectedGreenfieldRows": len(selected_greenfield),
            "readbackCandidateRows": len(readback_candidates),
            "greenfieldCarryoverRows": sum(1 for row in rows if row["structuredEvidenceStatus"] == "operator_local_only"),
            "structuredEvidenceStatusBreakdown": dict(sorted(status_counts.items())),
            "selectedGreenfieldByTier": dict(sorted(tier_counts.items(), key=lambda item: TIER_ORDER.get(item[0], 9))),
        },
        "minimalEvidenceTypes": list(MINIMAL_EVIDENCE_TYPES),
        "deferredEvidenceTypes": list(DEFERRED_EVIDENCE_TYPES),
        "readbackCandidates": [_evidence_candidate_public_row(row) for row in readback_candidates],
        "selectedGreenfieldCandidates": [_evidence_candidate_public_row(row) for row in selected_greenfield],
        "allRows": [_evidence_candidate_public_row(row) for row in rows],
        "policy": {
            "reportOnly": True,
            "manifestMutation": False,
            "sourceSpanWrite": False,
            "strictEvidenceWrite": False,
            "runtimeAnswerIntegration": False,
            "citationGradePromotion": False,
            "tableEquationParser": False,
            "vaultScan": False,
            "externalDownload": False,
        },
    }
    payload["schemaValidation"] = _schema_validation(STRUCTURED_EVIDENCE_NEXT_SLICE_SCHEMA_ID, payload)
    if not payload["schemaValidation"]["ok"]:
        payload["status"] = "blocked"
    return payload


def render_priority_corpus_300_freeze_markdown(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    targets = payload.get("targets") or {}
    lines = [
        "# Priority Corpus 300 Freeze Report",
        "",
        f"Generated: `{payload.get('generatedAt')}`",
        "",
        f"- status: `{payload.get('status')}`",
        f"- manifest rows: **{counts.get('manifestRows', 0)}**",
        f"- source available rows: **{counts.get('sourceAvailableRows', 0)}**",
        f"- parsed available rows: **{counts.get('parsedAvailableRows', 0)}**",
        f"- manifest blocker rows: **{counts.get('blockerRows', 0)}**",
        "",
        "## Target Status",
        "",
        f"- 300 lower bound reached: **{targets.get('lowerBoundReached')}**",
        f"- verified available candidate universe: **{counts.get('verifiedAvailableCandidateRows', 0)}**",
        f"- remaining verified allowlist rows: **{counts.get('remainingVerifiedAllowlistRows', 0)}**",
        f"- additional verified candidates needed for 500: **{targets.get('additionalVerifiedAvailableNeededForUpperBound', 0)}**",
        f"- current ledger ceiling if all holds resolve: **{targets.get('currentLedgerCeilingIfAllHoldsResolved', 0)}**",
        f"- new candidates still needed for 500 after all current holds resolve: **{targets.get('additionalNewCandidateRowsNeededForUpperBoundAfterAllCurrentHoldsResolved', 0)}**",
        "",
        "## Hold / Exclude",
        "",
        f"- source_missing hold rows: **{counts.get('sourceMissingHoldRows', 0)}**",
        f"- ambiguous hold rows: **{counts.get('ambiguousHoldRows', 0)}**",
        f"- hold rows in manifest: **{counts.get('holdRowsInManifest', 0)}**",
        f"- hold rows in allowlist: **{counts.get('holdRowsInAllowlist', 0)}**",
        "",
        "## Safety",
        "",
        "- Manifest mutation: **false**",
        "- Network used: **false**",
        "- Vault scan: **false**",
        "- source_missing / ambiguous rows promoted: **false**",
        "",
    ]
    return "\n".join(lines)


def render_structured_evidence_next_slice_markdown(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    lines = [
        "# Structured Evidence Next Slice Candidate Report",
        "",
        f"Generated: `{payload.get('generatedAt')}`",
        "",
        f"- status: `{payload.get('status')}`",
        f"- manifest rows: **{counts.get('manifestRows', 0)}**",
        f"- strict covered rows: **{counts.get('strictCoveredRows', 0)}**",
        f"- strict coverage pct: **{counts.get('strictCoveragePct', 0)}**",
        f"- operator-local generated-only rows: **{counts.get('strictEvidenceStoreRowsOperatorLocalOnly', 0)}**",
        f"- greenfield eligible rows: **{counts.get('greenfieldEligibleRows', 0)}**",
        f"- selected greenfield rows: **{counts.get('selectedGreenfieldRows', 0)}**",
        "",
        "## Selected Greenfield Candidates",
        "",
    ]
    for row in payload.get("selectedGreenfieldCandidates") or []:
        evidence_types = ", ".join(f"`{item}`" for item in row.get("recommendedEvidenceTypes") or [])
        lines.append(
            f"- `{row.get('sourceId')}` ({row.get('candidateTier')}, {row.get('year') or 'n/a'}) — {evidence_types}"
        )
    lines.extend(["", "## Readback Candidates", ""])
    for row in payload.get("readbackCandidates") or []:
        lines.append(
            f"- `{row.get('sourceId')}` status=`{row.get('structuredEvidenceStatus')}` "
            f"strict_records={row.get('strictEvidenceStore', {}).get('total', 0)}"
        )
    lines.extend(["", "## Deferred", ""])
    for item in payload.get("deferredEvidenceTypes") or []:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "PRIORITY_CORPUS_300_FREEZE_SCHEMA_ID",
    "STRUCTURED_EVIDENCE_NEXT_SLICE_SCHEMA_ID",
    "build_priority_corpus_300_freeze_report",
    "build_structured_evidence_next_slice_candidate_report",
    "render_priority_corpus_300_freeze_markdown",
    "render_structured_evidence_next_slice_markdown",
]
