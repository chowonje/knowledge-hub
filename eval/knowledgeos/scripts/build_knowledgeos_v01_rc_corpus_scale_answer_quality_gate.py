#!/usr/bin/env python3
"""Build the KnowledgeOS v0.1 RC corpus-scale answer quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.complex_qa_abstain_baseline_runner import build_complex_qa_abstain_baseline
from knowledge_hub.papers.complex_qa_seed_pack import build_complex_qa_seed_pack
from knowledge_hub.papers.complex_qa_strict_evidence_answer_quality_dry_run import (
    build_complex_qa_strict_evidence_answer_quality_dry_run,
)
from knowledge_hub.papers.complex_qa_structured_evidence_comparison_runner import (
    build_complex_qa_structured_evidence_comparison,
)
from knowledge_hub.papers.knowledgeos_v01_rc_corpus_scale_answer_quality_gate import (
    DEFAULT_LABS_QUALITY_REPORT,
    DEFAULT_NO_ANSWER_REPORT,
    DEFAULT_POST_MERGE_REPORT,
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_SCHEMA_ID,
    build_knowledgeos_v01_rc_corpus_scale_answer_quality_gate,
    write_knowledgeos_v01_rc_corpus_scale_answer_quality_gate,
)


DEFAULT_CORPUS_MANIFEST = Path("eval/knowledgeos/fixtures/corpus_manifest.json")
DEFAULT_JSON_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_gate.v1.json"
DEFAULT_MD_PATH = PROJECT_ROOT / "eval/knowledgeos/reports/knowledgeos_v01_rc_corpus_scale_answer_quality_gate.v1.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_complex_qa_chain(corpus_manifest: Path) -> dict[str, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="khub-corpus-scale-quality-") as tmp_name:
        tmp = Path(tmp_name)
        seed = build_complex_qa_seed_pack(corpus_manifest=corpus_manifest, target_paper_count=20)
        seed_path = tmp / "complex-paper-qa-seed-pack.json"
        _write_json(seed_path, seed)

        baseline = build_complex_qa_abstain_baseline(seed_pack_report=seed_path)
        baseline_path = tmp / "complex-qa-abstain-baseline-runner.json"
        _write_json(baseline_path, baseline)

        comparison = build_complex_qa_structured_evidence_comparison(
            seed_pack_report=seed_path,
            abstain_baseline_report=baseline_path,
        )
        comparison_path = tmp / "complex-qa-structured-evidence-comparison-runner.json"
        _write_json(comparison_path, comparison)

        dry_run = build_complex_qa_strict_evidence_answer_quality_dry_run(
            comparison_report=comparison_path,
        )
        return {
            "seed": seed,
            "baseline": baseline,
            "comparison": comparison,
            "dryRun": dry_run,
        }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-merge-report", type=Path, default=DEFAULT_POST_MERGE_REPORT)
    parser.add_argument("--no-answer-report", type=Path, default=DEFAULT_NO_ANSWER_REPORT)
    parser.add_argument("--labs-quality-report", type=Path, default=DEFAULT_LABS_QUALITY_REPORT)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    chain = _build_complex_qa_chain(args.corpus_manifest)
    report = build_knowledgeos_v01_rc_corpus_scale_answer_quality_gate(
        post_merge_report_path=args.post_merge_report,
        no_answer_report_path=args.no_answer_report,
        labs_quality_report_path=args.labs_quality_report,
        seed_pack_report=chain["seed"],
        abstain_baseline_report=chain["baseline"],
        structured_evidence_comparison_report=chain["comparison"],
        answer_quality_dry_run_report=chain["dryRun"],
    )
    validation = validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "KnowledgeOS v0.1 RC corpus-scale answer quality gate schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    paths = {}
    if not args.no_write_report:
        paths = write_knowledgeos_v01_rc_corpus_scale_answer_quality_gate(
            report,
            report_json=args.report_json,
            report_md=args.report_md,
        )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "status": report.get("status"),
                    "decision": report.get("decision"),
                    "nextRecommendedTranche": report.get("nextRecommendedTranche"),
                    "qualityGateDecision": report.get("qualityGateDecision"),
                    "counts": report.get("counts"),
                    "gate": report.get("gate"),
                    "paths": paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
