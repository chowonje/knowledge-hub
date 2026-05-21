from __future__ import annotations

import hashlib
import json
from pathlib import Path

from knowledge_hub.papers.priority_corpus_300_freeze import (
    PRIORITY_CORPUS_300_FREEZE_SCHEMA_ID,
    STRUCTURED_EVIDENCE_NEXT_SLICE_SCHEMA_ID,
    build_priority_corpus_300_freeze_report,
    build_structured_evidence_next_slice_candidate_report,
)


class _ConfigWithPapersDir:
    def __init__(self, papers_dir: Path):
        self.papers_dir = str(papers_dir)

    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        if tuple(args) == ("storage", "papers_dir"):
            return self.papers_dir
        return default


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_source(papers_dir: Path, filename: str, content: bytes) -> tuple[str, int]:
    papers_dir.mkdir(parents=True, exist_ok=True)
    (papers_dir / filename).write_bytes(content)
    return "sha256:" + hashlib.sha256(content).hexdigest(), len(content)


def _touch_parsed(papers_dir: Path, source_id: str, *, figure_count: int = 0) -> None:
    parsed_dir = papers_dir / "parsed" / source_id
    parsed_dir.mkdir(parents=True, exist_ok=True)
    _write_json(parsed_dir / "manifest.json", {"parser": "fixture"})
    _write_json(
        parsed_dir / "document.json",
        {
            "elements": [{"type": "paragraph", "text": "fixture"}],
            "figure_artifacts": [{"caption": "caption"} for _ in range(figure_count)],
        },
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_priority_corpus_freeze_blocks_hold_rows_from_manifest_and_allowlist(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    source_meta = {
        source_id: _write_source(papers_dir, f"{source_id}.pdf", f"%PDF {source_id}".encode())
        for source_id in ("paper-a", "paper-b", "paper-c")
    }
    manifest_path = _write_json(
        tmp_path / "corpus_manifest.json",
        {
            "schema": "knowledge-hub.corpus-manifest.v1",
            "artifacts": [
                {
                    "artifactId": "paper-a",
                    "sourceIds": ["paper-a"],
                    "expectedFilename": "paper-a.pdf",
                    "expectedSourceContentHash": source_meta["paper-a"][0],
                    "byteLength": source_meta["paper-a"][1],
                    "corpusTier": "local_corpus",
                },
                {
                    "artifactId": "paper-b",
                    "sourceIds": ["paper-b"],
                    "expectedFilename": "paper-b.pdf",
                    "expectedSourceContentHash": source_meta["paper-b"][0],
                    "byteLength": source_meta["paper-b"][1],
                    "corpusTier": "local_corpus",
                },
                {
                    "artifactId": "paper-c",
                    "sourceIds": ["paper-c"],
                    "expectedFilename": "paper-c.pdf",
                    "expectedSourceContentHash": source_meta["paper-c"][0],
                    "byteLength": source_meta["paper-c"][1],
                    "corpusTier": "local_corpus",
                },
            ],
        },
    )
    for source_id in ("paper-a", "paper-b", "paper-c"):
        _touch_parsed(papers_dir, source_id)
    join_report_path = _write_json(
        tmp_path / "join.json",
        {
            "schema": "knowledge-hub.priority-corpus-source-join-report.v1",
            "counts": {
                "available_count": 4,
                "matched_to_artifact_count": 4,
                "already_in_manifest_count": 3,
                "expansion_allowlist_count": 1,
            },
            "rows": [
                {
                    "source_id": source_id,
                    "title": source_id,
                    "candidate_tier": "eval_critical",
                    "join_status": "available",
                    "current_manifest_status": "in_manifest",
                    "parsed_status": "parsed_present",
                    "warnings": [],
                }
                for source_id in ("paper-a", "paper-b", "paper-c")
            ]
            + [
                {
                    "source_id": "paper-d",
                    "title": "paper-d",
                    "candidate_tier": "recent_ai",
                    "join_status": "available",
                    "current_manifest_status": "not_in_manifest",
                    "parsed_status": "parsed_present",
                    "warnings": [],
                },
                {
                    "source_id": "paper-missing",
                    "title": "missing",
                    "candidate_tier": "recent_ai",
                    "join_status": "source_missing",
                    "current_manifest_status": "not_in_manifest",
                    "parsed_status": "not_applicable",
                    "warnings": [],
                },
                {
                    "source_id": "paper-ambiguous",
                    "title": "ambiguous",
                    "candidate_tier": "foundational",
                    "join_status": "ambiguous",
                    "current_manifest_status": "not_in_manifest",
                    "parsed_status": "not_applicable",
                    "warnings": ["ambiguous_multiple_hashes"],
                },
            ],
        },
    )
    allowlist_path = _write_json(
        tmp_path / "allowlist.json",
        {
            "schema": "knowledge-hub.priority-corpus-manifest-expansion-allowlist.v1",
            "counts": {"by_candidate_tier": {"recent_ai": 1}},
            "allowlist": [{"source_id": "paper-d"}],
        },
    )

    payload = build_priority_corpus_300_freeze_report(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
        join_report_path=join_report_path,
        allowlist_path=allowlist_path,
        papers_dir=papers_dir,
        lower_bound_target=3,
        upper_bound_target=5,
    )

    assert payload["schema"] == PRIORITY_CORPUS_300_FREEZE_SCHEMA_ID
    assert payload["status"] == "locked"
    assert payload["counts"]["manifestRows"] == 3
    assert payload["counts"]["holdRowsInManifest"] == 0
    assert payload["counts"]["holdRowsInAllowlist"] == 0
    assert payload["counts"]["sourceMissingHoldRows"] == 1
    assert payload["counts"]["ambiguousHoldRows"] == 1
    assert payload["targets"]["additionalVerifiedAvailableNeededForUpperBound"] == 1
    assert payload["schemaValidation"]["ok"] is True


def test_structured_evidence_next_slice_excludes_existing_strict_evidence(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    source_meta = {
        source_id: _write_source(papers_dir, f"{source_id}.pdf", f"%PDF {source_id}".encode())
        for source_id in ("paper-a", "paper-b", "paper-c")
    }
    manifest_path = _write_json(
        tmp_path / "corpus_manifest.json",
        {
            "schema": "knowledge-hub.corpus-manifest.v1",
            "artifacts": [
                {
                    "artifactId": f"paper-{source_id}",
                    "sourceIds": [source_id],
                    "expectedFilename": f"{source_id}.pdf",
                    "expectedSourceContentHash": source_meta[source_id][0],
                    "byteLength": source_meta[source_id][1],
                    "corpusTier": "local_corpus",
                }
                for source_id in ("paper-a", "paper-b", "paper-c")
            ],
        },
    )
    for source_id in ("paper-a", "paper-b", "paper-c"):
        _touch_parsed(papers_dir, source_id, figure_count=1 if source_id == "paper-b" else 0)
    _write_jsonl(
        papers_dir / "structured_evidence" / "strict_evidence" / "paper-a.jsonl",
        [{"recordId": "strict-a", "runId": "committed"}],
    )
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-a.jsonl",
        [{"recordId": "span-a", "runId": "committed"}],
    )
    join_report_path = _write_json(
        tmp_path / "join.json",
        {
            "schema": "knowledge-hub.priority-corpus-source-join-report.v1",
            "rows": [
                {
                    "source_id": "paper-a",
                    "title": "A",
                    "year": 2026,
                    "candidate_tier": "eval_critical",
                    "join_status": "available",
                    "current_manifest_status": "in_manifest",
                    "parsed_status": "parsed_present",
                    "warnings": [],
                },
                {
                    "source_id": "paper-b",
                    "title": "B",
                    "year": 2025,
                    "candidate_tier": "eval_critical",
                    "join_status": "available",
                    "current_manifest_status": "in_manifest",
                    "parsed_status": "parsed_present",
                    "warnings": [],
                },
                {
                    "source_id": "paper-c",
                    "title": "C",
                    "year": 2026,
                    "candidate_tier": "recent_ai",
                    "join_status": "available",
                    "current_manifest_status": "in_manifest",
                    "parsed_status": "parsed_present",
                    "warnings": [],
                },
            ],
        },
    )

    payload = build_structured_evidence_next_slice_candidate_report(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
        join_report_path=join_report_path,
        papers_dir=papers_dir,
        greenfield_target_rows=2,
    )

    assert payload["schema"] == STRUCTURED_EVIDENCE_NEXT_SLICE_SCHEMA_ID
    assert payload["status"] == "ready"
    assert payload["schemaValidation"]["ok"] is True
    assert payload["counts"]["strictEvidenceStoreRowsPublicReviewable"] == 1
    assert [row["sourceId"] for row in payload["readbackCandidates"]] == ["paper-a"]
    assert [row["sourceId"] for row in payload["selectedGreenfieldCandidates"]] == ["paper-b", "paper-c"]
    assert "figure_caption_text" in payload["selectedGreenfieldCandidates"][0]["recommendedEvidenceTypes"]
    assert payload["policy"]["strictEvidenceWrite"] is False


def test_structured_evidence_next_slice_reports_all_readback_candidates(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    source_ids = [f"paper-{idx:02d}" for idx in range(12)]
    source_meta = {
        source_id: _write_source(papers_dir, f"{source_id}.pdf", f"%PDF {source_id}".encode())
        for source_id in source_ids
    }
    manifest_path = _write_json(
        tmp_path / "corpus_manifest.json",
        {
            "schema": "knowledge-hub.corpus-manifest.v1",
            "artifacts": [
                {
                    "artifactId": source_id,
                    "sourceIds": [source_id],
                    "expectedFilename": f"{source_id}.pdf",
                    "expectedSourceContentHash": source_meta[source_id][0],
                    "byteLength": source_meta[source_id][1],
                    "corpusTier": "local_corpus",
                }
                for source_id in source_ids
            ],
        },
    )
    for source_id in source_ids:
        _touch_parsed(papers_dir, source_id)
        _write_jsonl(
            papers_dir / "structured_evidence" / "strict_evidence" / f"{source_id}.jsonl",
            [{"recordId": f"strict-{source_id}", "runId": "committed"}],
        )
    join_report_path = _write_json(
        tmp_path / "join.json",
        {
            "schema": "knowledge-hub.priority-corpus-source-join-report.v1",
            "rows": [
                {
                    "source_id": source_id,
                    "title": source_id,
                    "year": 2026,
                    "candidate_tier": "eval_critical",
                    "join_status": "available",
                    "current_manifest_status": "in_manifest",
                    "parsed_status": "parsed_present",
                    "warnings": [],
                }
                for source_id in source_ids
            ],
        },
    )

    payload = build_structured_evidence_next_slice_candidate_report(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
        join_report_path=join_report_path,
        papers_dir=papers_dir,
        greenfield_target_rows=1,
    )

    assert payload["counts"]["readbackCandidateRows"] == len(source_ids)
    assert [row["sourceId"] for row in payload["readbackCandidates"]] == source_ids


def test_structured_evidence_next_slice_carries_over_operator_local_side_effects(
    tmp_path: Path,
) -> None:
    papers_dir = tmp_path / "papers"
    source_meta = {
        source_id: _write_source(papers_dir, f"{source_id}.pdf", f"%PDF {source_id}".encode())
        for source_id in ("2005.11401", "1512.03385", "paper-new")
    }
    manifest_path = _write_json(
        tmp_path / "corpus_manifest.json",
        {
            "schema": "knowledge-hub.corpus-manifest.v1",
            "artifacts": [
                {
                    "artifactId": f"paper-{source_id}",
                    "sourceIds": [source_id],
                    "expectedFilename": f"{source_id}.pdf",
                    "expectedSourceContentHash": source_meta[source_id][0],
                    "byteLength": source_meta[source_id][1],
                    "corpusTier": "local_corpus",
                }
                for source_id in ("2005.11401", "1512.03385", "paper-new")
            ],
        },
    )
    for source_id in ("2005.11401", "1512.03385", "paper-new"):
        _touch_parsed(papers_dir, source_id)
    for source_id in ("2005.11401", "1512.03385"):
        _write_jsonl(
            papers_dir / "structured_evidence" / "strict_evidence" / f"{source_id}.jsonl",
            [{"recordId": f"strict-{source_id}", "runId": "structured-evidence-vertical-slice-20260521"}],
        )
    join_report_path = _write_json(
        tmp_path / "join.json",
        {
            "schema": "knowledge-hub.priority-corpus-source-join-report.v1",
            "rows": [
                {
                    "source_id": source_id,
                    "title": source_id,
                    "year": 2020,
                    "candidate_tier": "eval_critical",
                    "join_status": "available",
                    "current_manifest_status": "in_manifest",
                    "parsed_status": "parsed_present",
                    "warnings": [],
                }
                for source_id in ("2005.11401", "1512.03385", "paper-new")
            ],
        },
    )

    payload = build_structured_evidence_next_slice_candidate_report(
        config=_ConfigWithPapersDir(papers_dir),
        manifest_path=manifest_path,
        join_report_path=join_report_path,
        papers_dir=papers_dir,
        greenfield_target_rows=3,
    )

    assert payload["counts"]["strictCoveredRows"] == 0
    assert payload["counts"]["greenfieldCarryoverRows"] == 2
    assert payload["readbackCandidates"] == []
    assert [row["sourceId"] for row in payload["selectedGreenfieldCandidates"][:2]] == [
        "2005.11401",
        "1512.03385",
    ]
