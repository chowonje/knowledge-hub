from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from knowledge_hub.application.eval_cases import (
    build_eval_case_from_csv_row,
    create_eval_case,
    import_eval_cases_from_csv,
    list_eval_cases,
    read_eval_case_registry,
    write_eval_case_registry,
)
from knowledge_hub.core.schema_validator import validate_payload


def test_create_eval_case_generates_deterministic_id() -> None:
    now = datetime.fromisoformat("2026-04-29T00:00:00+00:00")
    left = create_eval_case(
        lane="paper_default_eval",
        source_type="paper",
        scenario_type="concept_explainer",
        query="CNN을 쉽게 설명해줘",
        expected_family="concept_explainer",
        tags=["seed"],
        now=now,
    )
    right = create_eval_case(
        lane=" paper_default_eval ",
        source_type="paper",
        scenario_type="concept_explainer",
        query=" CNN을 쉽게 설명해줘 ",
        expected_family="concept_explainer",
        tags=["different"],
        now=now,
    )
    different = create_eval_case(
        lane="paper_default_eval",
        source_type="paper",
        scenario_type="paper_lookup",
        query="CNN을 쉽게 설명해줘",
        expected_family="paper_lookup",
        now=now,
    )

    assert validate_payload(left, left["schema"], strict=True).ok
    assert validate_payload(right, right["schema"], strict=True).ok
    assert left["evalCaseId"] == right["evalCaseId"]
    assert left["evalCaseId"] != different["evalCaseId"]


def test_build_eval_case_from_csv_row_normalizes_regression_fields() -> None:
    now = datetime.fromisoformat("2026-04-29T00:00:00+00:00")
    case = build_eval_case_from_csv_row(
        {
            "query": "Deep Residual Learning 논문 설명해줘",
            "source": "paper",
            "eval_bucket": "lookup",
            "expected_family": "paper_lookup",
            "expected_top1_or_set": "Deep_Residual_Learning_efbb7871|1512.03385",
            "expected_answer_mode": "paper_scoped_answer",
            "expected_match_count": "1",
            "expected_scope_applied": "1",
            "allowed_fallback": "paper_scoped_no_result",
        },
        lane="paper_regression_eval",
        source_path="/tmp/paper_regression_eval.csv",
        row_number=2,
        now=now,
    )

    assert validate_payload(case, case["schema"], strict=True).ok
    assert case["scenarioType"] == "lookup"
    assert case["expectedSourceScope"] == "scoped"
    assert case["expectedEvidencePolicy"] == "scoped_evidence_required"
    assert case["successCriteria"]["expectedMatchCount"] == 1
    assert case["successCriteria"]["expectedScopeApplied"] is True
    assert case["provenance"]["rowNumber"] == 2
    assert case["provenance"]["rawRow"]["expected_scope_applied"] == "1"


def test_write_and_list_eval_cases_round_trip(tmp_path: Path) -> None:
    registry_path = tmp_path / "eval_cases.jsonl"
    now = datetime.fromisoformat("2026-04-29T00:00:00+00:00")
    cases = [
        create_eval_case(
            lane="paper_default_eval",
            source_type="paper",
            scenario_type="concept_explainer",
            query="CNN을 쉽게 설명해줘",
            expected_family="concept_explainer",
            now=now,
        ),
        create_eval_case(
            lane="paper_regression_eval",
            source_type="paper",
            scenario_type="lookup",
            query="Deep Residual Learning 논문 설명해줘",
            expected_family="paper_lookup",
            expected_source_scope="scoped",
            now=now,
        ),
    ]

    write_eval_case_registry(cases, registry_path=registry_path, now=now)
    rows, warnings, resolved_path = read_eval_case_registry(registry_path=registry_path)
    payload = list_eval_cases(registry_path=registry_path, lane="paper_default_eval")

    assert resolved_path == registry_path
    assert warnings == []
    assert len(rows) == 2
    assert rows[0]["lane"] == "paper_default_eval"
    assert rows[1]["lane"] == "paper_regression_eval"
    assert validate_payload(payload, payload["schema"], strict=True).ok
    assert payload["status"] == "ok"
    assert payload["recordCount"] == 1
    assert payload["totalRecordCount"] == 2
    assert payload["items"][0]["query"] == "CNN을 쉽게 설명해줘"


def test_import_eval_cases_from_csv_upserts_and_preserves_created_at(tmp_path: Path) -> None:
    csv_path = tmp_path / "paper_default_eval_queries_v1.csv"
    csv_path.write_text(
        (
            "query,source,expected_family,expected_top1_or_set,expected_answer_mode,allowed_fallback\n"
            "CNN을 쉽게 설명해줘,paper,concept_explainer,"
            "alexnet-2012|ImageNet Classification with Deep Convolutional Neural Networks,"
            "representative_paper_explainer_beginner,planner_retry\n"
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "eval_cases.jsonl"
    first_now = datetime.fromisoformat("2026-04-29T00:00:00+00:00")
    second_now = datetime.fromisoformat("2026-04-30T00:00:00+00:00")

    first = import_eval_cases_from_csv(
        csv_path=csv_path,
        lane="paper_default_eval",
        registry_path=registry_path,
        now=first_now,
    )
    second = import_eval_cases_from_csv(
        csv_path=csv_path,
        lane="paper_default_eval",
        registry_path=registry_path,
        now=second_now,
    )

    assert validate_payload(first, first["schema"], strict=True).ok
    assert validate_payload(second, second["schema"], strict=True).ok
    assert first["createdCount"] == 1
    assert first["updatedCount"] == 0
    assert second["createdCount"] == 0
    assert second["updatedCount"] == 1

    rows = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["createdAt"] == "2026-04-29T00:00:00+00:00"
    assert rows[0]["updatedAt"] == "2026-04-30T00:00:00+00:00"
    assert rows[0]["expectedSourceScope"] == "unspecified"
    assert rows[0]["successCriteria"]["allowedFallback"] == "planner_retry"
