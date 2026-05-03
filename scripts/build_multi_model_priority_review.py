#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASE_FIELDS = [
    "query",
    "source",
    "query_type",
    "temporal_query",
    "expected_primary_source",
    "expected_answer_style",
    "difficulty",
    "no_result",
    "top1_title",
    "top1_source_type",
    "top1_excerpt",
    "answer_status",
    "memory_route_applied",
    "memory_prefilter_reason",
    "temporal_route_applied",
    "insufficient_reasons",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def _as_bool_str(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes"}:
        return "1"
    return "0"


def _safe_float(value: str) -> float:
    try:
        return float(str(value or "").strip())
    except Exception:
        return 0.0


def _priority_bundle(base: dict[str, str], evaluator_rows: dict[str, dict[str, str]]) -> tuple[int, str]:
    labels = {name: str(row.get("pred_label") or "").strip() for name, row in evaluator_rows.items()}
    wrong_eras = {name: _as_bool_str(row.get("pred_wrong_era", "")) for name, row in evaluator_rows.items()}
    abstains = {name: _as_bool_str(row.get("pred_should_abstain", "")) for name, row in evaluator_rows.items()}
    confidences = {name: _safe_float(row.get("pred_confidence", "")) for name, row in evaluator_rows.items()}

    reasons: list[str] = []
    score = 0

    distinct_labels = {v for v in labels.values() if v}
    if len(distinct_labels) > 1:
        score += 5
        reasons.append("label_disagreement")

    if "1" in wrong_eras.values():
        score += 4
        reasons.append("wrong_era_flagged")

    distinct_abstains = {v for v in abstains.values() if v}
    if len(distinct_abstains) > 1:
        score += 4
        reasons.append("abstain_disagreement")

    if any(v == "partial" for v in labels.values()):
        score += 3
        reasons.append("partial_present")

    if any(v < 0.7 for v in confidences.values() if v > 0):
        score += 2
        reasons.append("low_confidence_present")

    top1_present = bool(str(base.get("top1_title") or "").strip() or str(base.get("top1_excerpt") or "").strip())
    if top1_present and "1" in abstains.values():
        score += 2
        reasons.append("abstain_with_top1")

    if _as_bool_str(base.get("temporal_query", "")) == "1":
        score += 1
        reasons.append("temporal_query")

    if _as_bool_str(base.get("no_result", "")) == "1":
        score += 1
        reasons.append("no_result")

    return score, "|".join(reasons)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a multi-model priority review queue from judged CSVs.")
    parser.add_argument("--base", required=True, help="Base machine-eval CSV path")
    parser.add_argument(
        "--evaluator",
        action="append",
        default=[],
        help="Evaluator spec in the form name=path/to/judged.csv; repeatable",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    base_path = Path(args.base).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_rows = _read_csv(base_path)
    evaluators: list[tuple[str, Path]] = []
    for item in args.evaluator:
        if "=" not in item:
            raise SystemExit(f"Invalid --evaluator value: {item!r}. Expected name=path")
        name, raw_path = item.split("=", 1)
        evaluators.append((name.strip(), Path(raw_path).expanduser()))

    evaluator_tables = {name: _read_csv(path) for name, path in evaluators}
    for name, rows in evaluator_tables.items():
        if len(rows) != len(base_rows):
            raise SystemExit(
                f"Evaluator {name!r} row count mismatch: base={len(base_rows)} evaluator={len(rows)}"
            )

    fieldnames = BASE_FIELDS + [
        "priority_score",
        "priority_reasons",
    ]
    for name, _ in evaluators:
        fieldnames.extend(
            [
                f"{name}_pred_label",
                f"{name}_pred_wrong_era",
                f"{name}_pred_should_abstain",
                f"{name}_pred_confidence",
                f"{name}_pred_reason",
            ]
        )
    fieldnames.extend(
        [
            "final_label",
            "final_wrong_era",
            "final_should_abstain",
            "review_notes",
        ]
    )

    output_rows: list[dict[str, str]] = []
    for idx, base_row in enumerate(base_rows):
        evaluator_rows = {name: evaluator_tables[name][idx] for name, _ in evaluators}
        priority_score, priority_reasons = _priority_bundle(base_row, evaluator_rows)
        out_row = {field: str(base_row.get(field) or "").strip() for field in BASE_FIELDS}
        out_row["priority_score"] = str(priority_score)
        out_row["priority_reasons"] = priority_reasons
        for name, row in evaluator_rows.items():
            out_row[f"{name}_pred_label"] = str(row.get("pred_label") or "").strip()
            out_row[f"{name}_pred_wrong_era"] = str(row.get("pred_wrong_era") or "").strip()
            out_row[f"{name}_pred_should_abstain"] = str(row.get("pred_should_abstain") or "").strip()
            out_row[f"{name}_pred_confidence"] = str(row.get("pred_confidence") or "").strip()
            out_row[f"{name}_pred_reason"] = str(row.get("pred_reason") or "").strip()
        out_row["final_label"] = ""
        out_row["final_wrong_era"] = ""
        out_row["final_should_abstain"] = ""
        out_row["review_notes"] = ""
        output_rows.append(out_row)

    output_rows.sort(
        key=lambda row: (
            -int(row.get("priority_score") or 0),
            row.get("source") or "",
            row.get("query") or "",
        )
    )

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote multi-model priority review queue: {out_path} ({len(output_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
