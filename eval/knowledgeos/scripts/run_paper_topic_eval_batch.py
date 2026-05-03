#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(handle)]


def _run_json(cmd: list[str], cwd: Path, timeout_sec: int) -> dict[str, Any]:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(cmd)}\nSTDERR:\n{completed.stderr.strip()}"
        )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(f"command returned empty stdout: {' '.join(cmd)}")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse JSON from command: {' '.join(cmd)}\nSTDOUT:\n{stdout[:2000]}") from exc


def _titles(items: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for item in items:
        title = str(item.get("title") or "").strip()
        if title:
            out.append(title)
    return out


def _ids(items: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for item in items:
        paper_id = str(item.get("paperId") or item.get("paper_id") or "").strip()
        if paper_id:
            out.append(paper_id)
    return out


def _extract_labs_row(query_row: dict[str, str], payload: dict[str, Any], elapsed_sec: float) -> dict[str, str]:
    selected = list(payload.get("selectedPapers") or [])
    excluded = list(payload.get("excludedPapers") or [])
    citations = list(payload.get("citations") or [])
    verification = dict(payload.get("verification") or {})
    diagnostics = dict(payload.get("selectionDiagnostics") or {})
    return {
        "query_id": query_row["query_id"],
        "query": query_row["query"],
        "runner": "labs_paper_topic_synthesize",
        "status": str(payload.get("status") or ""),
        "answer_text": str(payload.get("topicSummary") or ""),
        "candidate_count": str(len(payload.get("candidatePapers") or [])),
        "selected_count": str(len(selected)),
        "selected_titles_json": json.dumps(_titles(selected), ensure_ascii=False),
        "selected_paper_ids_json": json.dumps(_ids(selected), ensure_ascii=False),
        "excluded_titles_json": json.dumps(_titles(excluded), ensure_ascii=False),
        "citation_titles_json": json.dumps(_titles(citations), ensure_ascii=False),
        "architecture_groups_json": json.dumps(payload.get("architectureGroups") or [], ensure_ascii=False),
        "comparison_points_json": json.dumps(payload.get("comparisonPoints") or [], ensure_ascii=False),
        "limitations_json": json.dumps(payload.get("limitations") or [], ensure_ascii=False),
        "gaps_json": json.dumps(payload.get("gaps") or [], ensure_ascii=False),
        "verification_status": str(verification.get("status") or ""),
        "verification_summary": str(verification.get("summary") or ""),
        "judge_fallback_used": "1" if diagnostics.get("judge", {}).get("fallbackUsed") else "0",
        "synthesis_fallback_used": "1" if diagnostics.get("synthesis", {}).get("fallbackUsed") else "0",
        "warnings_json": json.dumps(payload.get("warnings") or [], ensure_ascii=False),
        "elapsed_sec": f"{elapsed_sec:.2f}",
    }


def _extract_ask_row(query_row: dict[str, str], payload: dict[str, Any], elapsed_sec: float) -> dict[str, str]:
    citations = list(payload.get("citations") or [])
    paper_scope = dict(payload.get("paperAnswerScope") or {})
    verification = dict(payload.get("verification") or {})
    return {
        "query_id": query_row["query_id"],
        "query": query_row["query"],
        "runner": "ask_paper_baseline",
        "status": str(payload.get("status") or ""),
        "answer_text": str(payload.get("answer") or ""),
        "candidate_count": str(len(payload.get("sources") or [])),
        "selected_count": str(len(citations)),
        "selected_titles_json": json.dumps(_titles(citations), ensure_ascii=False),
        "selected_paper_ids_json": json.dumps(_ids(citations), ensure_ascii=False),
        "excluded_titles_json": json.dumps([], ensure_ascii=False),
        "citation_titles_json": json.dumps(_titles(citations), ensure_ascii=False),
        "architecture_groups_json": json.dumps([], ensure_ascii=False),
        "comparison_points_json": json.dumps([], ensure_ascii=False),
        "limitations_json": json.dumps([], ensure_ascii=False),
        "gaps_json": json.dumps(payload.get("evidencePacket", {}).get("insufficientEvidenceReasons") or [], ensure_ascii=False),
        "verification_status": str(verification.get("status") or ""),
        "verification_summary": str(verification.get("summary") or ""),
        "judge_fallback_used": "0",
        "synthesis_fallback_used": "0",
        "warnings_json": json.dumps(
            [
                {
                    "paperAnswerScopeApplied": bool(paper_scope.get("applied")),
                    "paperAnswerScopeReason": str(paper_scope.get("reason") or ""),
                    "queryIntent": str(paper_scope.get("queryIntent") or ""),
                }
            ],
            ensure_ascii=False,
        ),
        "elapsed_sec": f"{elapsed_sec:.2f}",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "query_id",
        "query",
        "runner",
        "status",
        "answer_text",
        "candidate_count",
        "selected_count",
        "selected_titles_json",
        "selected_paper_ids_json",
        "excluded_titles_json",
        "citation_titles_json",
        "architecture_groups_json",
        "comparison_points_json",
        "limitations_json",
        "gaps_json",
        "verification_status",
        "verification_summary",
        "judge_fallback_used",
        "synthesis_fallback_used",
        "warnings_json",
        "elapsed_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper-topic eval queries against labs candidate and ask baseline.")
    parser.add_argument(
        "--query-csv",
        default="eval/knowledgeos/queries/knowledgeos_paper_topic_eval_queries_20_v1.csv",
        help="Path to the paper-topic eval CSV",
    )
    parser.add_argument(
        "--config",
        default="exports/config_bge_m3_local.yaml",
        help="Knowledge Hub config path",
    )
    parser.add_argument(
        "--output-prefix",
        default="eval/knowledgeos/runs/knowledgeos_paper_topic_eval_v1",
        help="Output prefix without extension",
    )
    parser.add_argument("--timeout-sec", type=int, default=600, help="Per-command timeout")
    parser.add_argument(
        "--candidate-llm-mode",
        default="local",
        choices=["auto", "local", "mini", "strong", "fallback-only"],
        help="labs paper topic-synthesize llm mode",
    )
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Only run labs candidate path and skip ask baseline",
    )
    parser.add_argument(
        "--candidate-allow-external",
        action="store_true",
        help="Allow external provider usage for labs candidate runs",
    )
    parser.add_argument(
        "--candidate-provider",
        default="",
        help="Optional provider override for labs candidate runs",
    )
    parser.add_argument(
        "--candidate-model",
        default="",
        help="Optional model override for labs candidate runs",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    query_csv = (repo_root / args.query_csv).resolve()
    config_path = (repo_root / args.config).resolve()
    output_prefix = (repo_root / args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    query_rows = _read_rows(query_csv)
    if not query_rows:
        raise SystemExit(f"no query rows found in {query_csv}")

    python = sys.executable
    candidate_jsonl: list[dict[str, Any]] = []
    baseline_jsonl: list[dict[str, Any]] = []
    candidate_csv_rows: list[dict[str, str]] = []
    baseline_csv_rows: list[dict[str, str]] = []

    for index, row in enumerate(query_rows, start=1):
        query = row["query"]
        print(f"[{index}/{len(query_rows)}] candidate {row['query_id']} :: {query}", flush=True)
        candidate_cmd = [
            python,
            "-m",
            "knowledge_hub.interfaces.cli.main",
            "-c",
            str(config_path),
            "labs",
            "paper",
            "topic-synthesize",
            query,
            "--llm-mode",
            args.candidate_llm_mode,
            "--json",
        ]
        if args.candidate_allow_external:
            candidate_cmd.insert(-1, "--allow-external")
        if str(args.candidate_provider or "").strip():
            candidate_cmd.extend(["--provider", str(args.candidate_provider).strip()])
        if str(args.candidate_model or "").strip():
            candidate_cmd.extend(["--model", str(args.candidate_model).strip()])
        started = time.perf_counter()
        candidate_payload = _run_json(candidate_cmd, repo_root, args.timeout_sec)
        candidate_elapsed = time.perf_counter() - started
        candidate_jsonl.append(
            {
                "query_id": row["query_id"],
                "query": query,
                "runner": "labs_paper_topic_synthesize",
                "payload": candidate_payload,
                "elapsed_sec": candidate_elapsed,
            }
        )
        candidate_csv_rows.append(_extract_labs_row(row, candidate_payload, candidate_elapsed))

        if args.candidate_only:
            continue

        print(f"[{index}/{len(query_rows)}] baseline {row['query_id']} :: {query}", flush=True)
        ask_cmd = [
            python,
            "-m",
            "knowledge_hub.interfaces.cli.main",
            "-c",
            str(config_path),
            "ask",
            query,
            "-s",
            "paper",
            "--mode",
            "hybrid",
            "--alpha",
            "0.7",
            "--memory-route-mode",
            "prefilter",
            "--paper-memory-mode",
            "prefilter",
            "--json",
        ]
        started = time.perf_counter()
        baseline_payload = _run_json(ask_cmd, repo_root, args.timeout_sec)
        baseline_elapsed = time.perf_counter() - started
        baseline_jsonl.append(
            {
                "query_id": row["query_id"],
                "query": query,
                "runner": "ask_paper_baseline",
                "payload": baseline_payload,
                "elapsed_sec": baseline_elapsed,
            }
        )
        baseline_csv_rows.append(_extract_ask_row(row, baseline_payload, baseline_elapsed))

    _write_jsonl(output_prefix.with_name(output_prefix.name + "_candidate.jsonl"), candidate_jsonl)
    _write_csv(output_prefix.with_name(output_prefix.name + "_candidate.csv"), candidate_csv_rows)
    if not args.candidate_only:
        _write_jsonl(output_prefix.with_name(output_prefix.name + "_baseline.jsonl"), baseline_jsonl)
        _write_csv(output_prefix.with_name(output_prefix.name + "_baseline.csv"), baseline_csv_rows)

    summary = {
        "query_csv": str(query_csv),
        "config": str(config_path),
        "candidate_llm_mode": args.candidate_llm_mode,
        "candidate_count": len(candidate_csv_rows),
        "baseline_count": len(baseline_csv_rows),
        "candidate_csv": str(output_prefix.with_name(output_prefix.name + "_candidate.csv")),
        "candidate_jsonl": str(output_prefix.with_name(output_prefix.name + "_candidate.jsonl")),
        "baseline_csv": str(output_prefix.with_name(output_prefix.name + "_baseline.csv")) if not args.candidate_only else "",
        "baseline_jsonl": str(output_prefix.with_name(output_prefix.name + "_baseline.jsonl")) if not args.candidate_only else "",
    }
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
