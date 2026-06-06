from __future__ import annotations

import csv
from pathlib import Path

from knowledge_hub.application.eval_center import build_eval_center_summary


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_canonical_eval_query_csvs_have_no_extra_fields():
    queries_dir = Path(__file__).resolve().parents[1] / "eval" / "knowledgeos" / "queries"
    failures: list[str] = []
    for path in sorted(queries_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                extras = row.get(None)
                if extras:
                    failures.append(f"{path.name}: row {row_number} has {len(extras)} extra field(s)")
    assert failures == []


def test_build_eval_center_summary_warns_on_query_csv_extra_fields(tmp_path: Path):
    runs_root = tmp_path / "eval" / "knowledgeos" / "runs"
    queries_dir = tmp_path / "eval" / "knowledgeos" / "queries"
    runs_root.mkdir(parents=True)
    _write_text(queries_dir / "user_answer_eval_queries_v1.csv", 'query,source\n"a,b",paper,extra,second-extra\n')

    payload = build_eval_center_summary(
        runs_root=runs_root,
        queries_dir=queries_dir,
        failure_bank_path=tmp_path / "missing_failure_bank.jsonl",
        repo_root=tmp_path,
        generated_at="2026-04-26T00:00:00+00:00",
    )

    expected_warning = "user_answer_eval_queries_v1.csv: row 2 has 2 extra field(s)"
    query_item = payload["queryInventory"]["items"][0]
    assert expected_warning in payload["warnings"]
    assert query_item["parseWarnings"] == ["row 2 has 2 extra field(s)"]
