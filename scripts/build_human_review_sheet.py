#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a human-review sheet from machine-eval CSV.")
    parser.add_argument("--machine-eval", required=True, help="Machine eval CSV path")
    parser.add_argument("--out", required=True, help="Output human review CSV path")
    args = parser.parse_args()

    machine_path = Path(args.machine_eval).expanduser()
    out_path = Path(args.out).expanduser()
    rows = list(csv.DictReader(machine_path.open("r", encoding="utf-8")))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query",
        "source",
        "query_type",
        "temporal_query",
        "expected_primary_source",
        "expected_answer_style",
        "difficulty",
        "no_result",
        "pred_label",
        "pred_wrong_era",
        "pred_should_abstain",
        "pred_confidence",
        "pred_reason",
        "final_label",
        "final_wrong_era",
        "final_should_abstain",
        "review_notes",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "query": str(row.get("query") or "").strip(),
                    "source": str(row.get("source") or "").strip(),
                    "query_type": str(row.get("query_type") or "").strip(),
                    "temporal_query": str(row.get("temporal_query") or "").strip(),
                    "expected_primary_source": str(row.get("expected_primary_source") or "").strip(),
                    "expected_answer_style": str(row.get("expected_answer_style") or "").strip(),
                    "difficulty": str(row.get("difficulty") or "").strip(),
                    "no_result": str(row.get("no_result") or "").strip(),
                    "pred_label": str(row.get("pred_label") or "").strip(),
                    "pred_wrong_era": str(row.get("pred_wrong_era") or "").strip(),
                    "pred_should_abstain": str(row.get("pred_should_abstain") or "").strip(),
                    "pred_confidence": str(row.get("pred_confidence") or "").strip(),
                    "pred_reason": str(row.get("pred_reason") or "").strip(),
                    "final_label": "",
                    "final_wrong_era": "",
                    "final_should_abstain": "",
                    "review_notes": "",
                }
            )

    print(f"Wrote human-review sheet: {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
