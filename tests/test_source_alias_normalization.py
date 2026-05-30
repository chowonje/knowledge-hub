from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.source_alias_normalization import (
    SOURCE_ALIAS_NORMALIZATION_CASE_SCHEMA_ID,
    SOURCE_ALIAS_NORMALIZATION_REPORT_SCHEMA_ID,
    build_source_alias_normalization_report,
    write_report,
)


def _row(report: dict, case_id: str) -> dict:
    return next(item for item in report["rows"] if item["caseId"] == case_id)


def test_source_alias_report_blocks_short_alias_direct_lookup_without_context() -> None:
    report = build_source_alias_normalization_report(generated_at="2026-05-26T00:00:00Z")

    assert report["status"] == "ready"
    assert report["caseRows"] == 10
    assert report["blockedShortAliasRows"] == 1
    assert report["conceptOnlyRows"] == 2
    assert report["discoverOnlyRows"] == 2
    assert report["contextualAliasResolvedRows"] == 3
    assert report["explicitIdResolvedRows"] == 1
    assert report["explicitTitleContextRows"] == 1
    assert report["unsafeDirectAliasRows"] == 0
    assert report["expectationFailureRows"] == 0
    assert validate_payload(report, SOURCE_ALIAS_NORMALIZATION_REPORT_SCHEMA_ID, strict=True).ok
    for case in report["rows"]:
        assert validate_payload(case, SOURCE_ALIAS_NORMALIZATION_CASE_SCHEMA_ID, strict=True).ok


def test_bare_short_alias_lookup_is_not_direct_source_resolution() -> None:
    report = build_source_alias_normalization_report(generated_at="2026-05-26T00:00:00Z")
    row = _row(report, "gpt-bare-lookup-blocked")

    assert row["disposition"] == "blocked_short_alias_no_context"
    assert row["sourceResolutionPolicy"] == "no_direct_source_resolution"
    assert row["directShortAliasResolutionAllowed"] is False
    assert row["blockerReason"] == "short_alias_lookup_requires_explicit_id_or_title"
    assert row["unsafeDirectAliasResolution"] is False


def test_contextual_compare_allows_bounded_alias_resolution_only_inside_pair() -> None:
    report = build_source_alias_normalization_report(generated_at="2026-05-26T00:00:00Z")
    row = _row(report, "cnn-vit-contextual-compare")

    assert row["family"] == "paper_compare"
    assert row["disposition"] == "contextual_alias_resolved"
    assert row["sourceResolutionPolicy"] == "contextual_resolution_only"
    assert row["directShortAliasResolutionAllowed"] is True
    assert "paired_context_term" in row["contextSignals"]
    assert "ViT" in row["contextTerms"]


def test_concept_explainer_keeps_representative_ids_out_of_direct_alias_scope() -> None:
    report = build_source_alias_normalization_report(generated_at="2026-05-26T00:00:00Z")
    row = _row(report, "cnn-concept-representative-only")

    assert row["family"] == "concept_explainer"
    assert row["queryPlanResolvedPaperIdRows"] >= 0
    assert row["disposition"] == "concept_only"
    assert row["sourceResolutionPolicy"] == "representative_candidates_only"
    assert row["directShortAliasResolutionAllowed"] is False


def test_report_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_source_alias_normalization_report(generated_at="2026-05-26T00:00:00Z")
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_report(report, json_path=json_path, markdown_path=md_path)

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert json.loads(json_path.read_text(encoding="utf-8"))["privatePathLeakRows"] == 0
