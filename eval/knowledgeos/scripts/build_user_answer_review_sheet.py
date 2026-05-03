#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a human-review sheet from user-answer eval CSV.")
    parser.add_argument("--machine-eval", required=True, help="Collected user-answer eval CSV path")
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
        "expected_primary_source",
        "expected_answer_style",
        "difficulty",
        "review_bucket",
        "answer_status",
        "answer_preview",
        "answer_text",
        "needs_caution",
        "verification_status",
        "verification_summary",
        "source_count",
        "source_titles",
        "source_refs",
        "runtime_used",
        "answer_route",
        "router_provider",
        "router_model",
        "latency_ms",
        "top_k",
        "retrieval_mode",
        "answer_backend",
        "answer_backend_model",
        "packet_ref",
        "pred_label",
        "pred_groundedness",
        "pred_usefulness",
        "pred_readability",
        "pred_source_accuracy",
        "pred_should_abstain",
        "pred_confidence",
        "pred_reason",
        "judge_provider",
        "judge_model",
        "final_label",
        "final_groundedness",
        "final_usefulness",
        "final_readability",
        "final_source_accuracy",
        "final_should_abstain",
        "final_notes",
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
                    "expected_primary_source": str(row.get("expected_primary_source") or "").strip(),
                    "expected_answer_style": str(row.get("expected_answer_style") or "").strip(),
                    "difficulty": str(row.get("difficulty") or "").strip(),
                    "review_bucket": str(row.get("review_bucket") or "").strip(),
                    "answer_status": str(row.get("answer_status") or "").strip(),
                    "answer_preview": str(row.get("answer_preview") or "").strip(),
                    "answer_text": str(row.get("answer_text") or "").strip(),
                    "needs_caution": str(row.get("needs_caution") or "").strip(),
                    "verification_status": str(row.get("verification_status") or "").strip(),
                    "verification_summary": str(row.get("verification_summary") or "").strip(),
                    "source_count": str(row.get("source_count") or "").strip(),
                    "source_titles": str(row.get("source_titles") or "").strip(),
                    "source_refs": str(row.get("source_refs") or "").strip(),
                    "runtime_used": str(row.get("runtime_used") or "").strip(),
                    "answer_route": str(row.get("answer_route") or "").strip(),
                    "router_provider": str(row.get("router_provider") or "").strip(),
                    "router_model": str(row.get("router_model") or "").strip(),
                    "latency_ms": str(row.get("latency_ms") or "").strip(),
                    "top_k": str(row.get("top_k") or "").strip(),
                    "retrieval_mode": str(row.get("retrieval_mode") or "").strip(),
                    "answer_backend": str(row.get("answer_backend") or "").strip(),
                    "answer_backend_model": str(row.get("answer_backend_model") or "").strip(),
                    "packet_ref": str(row.get("packet_ref") or "").strip(),
                    "pred_label": str(row.get("pred_label") or "").strip(),
                    "pred_groundedness": str(row.get("pred_groundedness") or "").strip(),
                    "pred_usefulness": str(row.get("pred_usefulness") or "").strip(),
                    "pred_readability": str(row.get("pred_readability") or "").strip(),
                    "pred_source_accuracy": str(row.get("pred_source_accuracy") or "").strip(),
                    "pred_should_abstain": str(row.get("pred_should_abstain") or "").strip(),
                    "pred_confidence": str(row.get("pred_confidence") or "").strip(),
                    "pred_reason": str(row.get("pred_reason") or "").strip(),
                    "judge_provider": str(row.get("judge_provider") or "").strip(),
                    "judge_model": str(row.get("judge_model") or "").strip(),
                    "final_label": "",
                    "final_groundedness": "",
                    "final_usefulness": "",
                    "final_readability": "",
                    "final_source_accuracy": "",
                    "final_should_abstain": "",
                    "final_notes": "",
                }
            )

    print(f"Wrote user-answer human-review sheet: {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
