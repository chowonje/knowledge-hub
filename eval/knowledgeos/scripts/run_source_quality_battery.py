#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceBatterySpec:
    name: str
    collector_script: str
    queries: str
    output_name: str


SOURCE_BATTERY_SPECS = (
    SourceBatterySpec(
        name="paper",
        collector_script="eval/knowledgeos/scripts/collect_paper_default_eval.py",
        queries="eval/knowledgeos/queries/paper_default_eval_queries_v1.csv",
        output_name="paper_default_eval.csv",
    ),
    SourceBatterySpec(
        name="vault",
        collector_script="eval/knowledgeos/scripts/collect_vault_default_eval.py",
        queries="eval/knowledgeos/queries/vault_default_eval_queries_v1.csv",
        output_name="vault_default_eval.csv",
    ),
    SourceBatterySpec(
        name="web",
        collector_script="eval/knowledgeos/scripts/collect_web_default_eval.py",
        queries="eval/knowledgeos/queries/knowledgeos_eval_queries_100_v1.csv",
        output_name="web_default_eval.csv",
    ),
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        {str(key): _clean_text(value) for key, value in row.items()}
        for row in csv.DictReader(path.open("r", encoding="utf-8-sig", newline=""))
    ]


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _int_token(row: dict[str, str], key: str) -> int:
    text = _clean_text(row.get(key))
    if not text:
        return 0
    return int(float(text))


def _float_token(row: dict[str, str], key: str) -> float | None:
    text = _clean_text(row.get(key))
    if not text:
        return None
    return round(float(text), 6)


def summarize_source_rows(source: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    family_matches = sum(1 for row in rows if _clean_text(row.get("family_match")) == "1")
    no_results = sum(1 for row in rows if _clean_text(row.get("no_result")) == "1")
    summary: dict[str, Any] = {
        "source": source,
        "rows": total,
        "route_correctness": _ratio(family_matches, total),
        "no_result_rate": _ratio(no_results, total),
    }
    runtime_used_counter = Counter(_clean_text(row.get("actual_runtime_used")) for row in rows if _clean_text(row.get("actual_runtime_used")))
    fallback_reason_counter = Counter(
        _clean_text(row.get("actual_fallback_reason")) for row in rows if _clean_text(row.get("actual_fallback_reason"))
    )
    summary["runtime_used_counts"] = dict(sorted(runtime_used_counter.items()))
    summary["fallback_reason_counts"] = dict(sorted(fallback_reason_counter.items()))
    summary["legacy_runtime_rate"] = _ratio(runtime_used_counter.get("legacy", 0), total)
    summary["capability_missing_rate"] = _ratio(fallback_reason_counter.get("ask_v2_capability_missing", 0), total)
    summary["forced_legacy_rate"] = _ratio(fallback_reason_counter.get("ask_v2_not_used", 0), total)

    if source == "paper":
        paper_rows = [row for row in rows if _clean_text(row.get("citation_support_match"))]
        supported = sum(1 for row in paper_rows if _clean_text(row.get("citation_support_match")) == "1")
        summary["citation_correctness_soft"] = _ratio(supported, len(paper_rows))

    if source == "vault":
        citation_count = sum(_int_token(row, "citation_count") for row in rows)
        stale_count = sum(_int_token(row, "stale_citation_count") for row in rows)
        abstain_rows = [row for row in rows if _clean_text(row.get("expected_answer_mode")).casefold() == "abstain"]
        abstention_ok = sum(
            1
            for row in abstain_rows
            if _clean_text(row.get("no_result")) == "1" or _clean_text(row.get("answer_mode_match")) == "1"
        )
        summary["stale_citation_rate"] = _ratio(stale_count, citation_count)
        summary["abstention_correctness_soft"] = _ratio(abstention_ok, len(abstain_rows))

    if source == "web":
        temporal_rows = [row for row in rows if _clean_text(row.get("query_type")).casefold() == "temporal"]
        recency_rows = [row for row in temporal_rows if _clean_text(row.get("recency_violation"))]
        violations = sum(1 for row in recency_rows if _clean_text(row.get("recency_violation")) == "1")
        ages = [value for row in temporal_rows if (value := _float_token(row, "latest_source_age_days")) is not None]
        summary["recency_violation_soft"] = _ratio(violations, len(recency_rows))
        summary["latest_source_age_days_p50"] = None if not ages else sorted(ages)[len(ages) // 2]

    return summary


def _hard_gates(
    summaries: dict[str, dict[str, Any]],
    *,
    route_threshold: float,
    vault_stale_threshold: float,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "route_correctness": {},
        "vault_stale_citation_rate": {},
    }
    for source, summary in summaries.items():
        value = summary.get("route_correctness")
        results["route_correctness"][source] = {
            "value": value,
            "threshold": route_threshold,
            "passed": value is not None and float(value) >= route_threshold,
        }
    vault_value = summaries.get("vault", {}).get("stale_citation_rate")
    results["vault_stale_citation_rate"]["vault"] = {
        "value": vault_value,
        "threshold": vault_stale_threshold,
        "passed": vault_value is not None and float(vault_value) <= vault_stale_threshold,
    }
    return results


def _soft_gates(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "paper_citation_correctness": summaries.get("paper", {}).get("citation_correctness_soft"),
        "vault_abstention_correctness": summaries.get("vault", {}).get("abstention_correctness_soft"),
        "web_recency_violation": summaries.get("web", {}).get("recency_violation_soft"),
    }


def _run_collector(
    repo_root: Path,
    spec: SourceBatterySpec,
    *,
    out_path: Path,
    gate_mode: str,
    top_k: int,
    retrieval_mode: str,
) -> None:
    command = [
        sys.executable,
        str(repo_root / spec.collector_script),
        "--queries",
        str(repo_root / spec.queries),
        "--out",
        str(out_path),
        "--top-k",
        str(top_k),
        "--mode",
        retrieval_mode,
        "--gate-mode",
        gate_mode,
    ]
    if gate_mode == "stub_hard":
        command.append("--stub-llm")
    subprocess.run(command, cwd=repo_root, check=True)


def _relative_output_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return os.path.relpath(path, repo_root)


def build_battery_summary(
    repo_root: Path,
    *,
    run_dir: Path,
    gate_mode: str,
    top_k: int,
    retrieval_mode: str,
    route_threshold: float,
    vault_stale_threshold: float,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    per_source: dict[str, dict[str, Any]] = {}
    outputs: dict[str, str] = {}
    for spec in SOURCE_BATTERY_SPECS:
        out_path = run_dir / spec.output_name
        _run_collector(
            repo_root,
            spec,
            out_path=out_path,
            gate_mode=gate_mode,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        outputs[spec.name] = _relative_output_path(out_path, repo_root)
        per_source[spec.name] = summarize_source_rows(spec.name, _read_rows(out_path))
    runtime_used_counter = Counter()
    fallback_reason_counter = Counter()
    total_rows = 0
    for summary in per_source.values():
        total_rows += int(summary.get("rows") or 0)
        runtime_used_counter.update(dict(summary.get("runtime_used_counts") or {}))
        fallback_reason_counter.update(dict(summary.get("fallback_reason_counts") or {}))
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate_mode": gate_mode,
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "outputs": outputs,
        "per_source": per_source,
        "hard_gates": _hard_gates(
            per_source,
            route_threshold=route_threshold,
            vault_stale_threshold=vault_stale_threshold,
        ),
        "soft_gates": _soft_gates(per_source),
        "runtime_readiness": {
            "global": {
                "rows": total_rows,
                "runtime_used_counts": dict(sorted(runtime_used_counter.items())),
                "fallback_reason_counts": dict(sorted(fallback_reason_counter.items())),
                "legacy_runtime_rate": _ratio(runtime_used_counter.get("legacy", 0), total_rows),
                "capability_missing_rate": _ratio(fallback_reason_counter.get("ask_v2_capability_missing", 0), total_rows),
                "forced_legacy_rate": _ratio(fallback_reason_counter.get("ask_v2_not_used", 0), total_rows),
            }
        },
    }
    (run_dir / "source_quality_battery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the source-specific nightly quality battery.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--gate-mode", default="stub_hard", choices=["standard", "stub_hard", "live_smoke"])
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--route-threshold", type=float, default=0.8)
    parser.add_argument("--vault-stale-threshold", type=float, default=0.0)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if _clean_text(args.run_dir)
        else repo_root / "eval/knowledgeos/runs" / f"source_quality_battery_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    summary = build_battery_summary(
        repo_root,
        run_dir=run_dir,
        gate_mode=str(args.gate_mode),
        top_k=int(args.top_k),
        retrieval_mode=str(args.mode),
        route_threshold=float(args.route_threshold),
        vault_stale_threshold=float(args.vault_stale_threshold),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
