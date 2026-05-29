from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

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
    HELD_DECISION,
    KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_SCHEMA_ID,
    build_knowledgeos_v01_rc_corpus_scale_answer_quality_gate,
    write_knowledgeos_v01_rc_corpus_scale_answer_quality_gate,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(root: Path, *, count: int = 20) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    seed_ids = [
        "2005.11401",
        "2007.01282",
        "2404.16130",
        "2410.05779",
        "alexnet-2012",
        "2010.11929",
        "1706.03762",
        "2312.00752",
        "1810.04805",
        "2005.14165",
        "1312.5602",
        "1707.06347",
        "1406.2661",
        "2006.11239",
        "1512.03385",
        "2310.11511",
        "1502.03167",
        "1409.3215",
        "2201.11903",
        "2501.12948",
    ][:count]
    artifacts = [
        {
            "artifactId": f"paper_{paper_id.replace('.', '_').replace('-', '_')}",
            "sourceIds": [paper_id],
            "expectedFilename": f"Seed Paper {paper_id}.pdf",
            "expectedSourceContentHash": f"sha256:{index:064d}",
            "corpusTier": "local_corpus",
        }
        for index, paper_id in enumerate(seed_ids, start=1)
    ]
    path = root / "corpus_manifest.json"
    path.write_text(
        json.dumps({"schema": "knowledge-hub.corpus-manifest.v1", "artifacts": artifacts}),
        encoding="utf-8",
    )
    return path


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _chain(tmp_path: Path, *, paper_count: int = 20) -> dict[str, dict[str, Any]]:
    manifest = _write_manifest(tmp_path / "input", count=paper_count)
    seed = build_complex_qa_seed_pack(corpus_manifest=manifest, target_paper_count=20)
    seed_path = _write(tmp_path / "seed.json", seed)
    baseline = build_complex_qa_abstain_baseline(seed_pack_report=seed_path)
    baseline_path = _write(tmp_path / "baseline.json", baseline)
    comparison = build_complex_qa_structured_evidence_comparison(
        seed_pack_report=seed_path,
        abstain_baseline_report=baseline_path,
    )
    comparison_path = _write(tmp_path / "comparison.json", comparison)
    dry_run = build_complex_qa_strict_evidence_answer_quality_dry_run(comparison_report=comparison_path)
    return {
        "seed_pack_report": seed,
        "abstain_baseline_report": baseline,
        "structured_evidence_comparison_report": comparison,
        "answer_quality_dry_run_report": dry_run,
    }


def _build(tmp_path: Path, **updates: Any) -> dict[str, Any]:
    inputs = {
        "post_merge_report": _json(DEFAULT_POST_MERGE_REPORT),
        "no_answer_report": _json(DEFAULT_NO_ANSWER_REPORT),
        "labs_quality_report": _json(DEFAULT_LABS_QUALITY_REPORT),
        "generated_at": "2026-05-29T00:00:00Z",
        **_chain(tmp_path),
    }
    inputs.update(updates)
    return build_knowledgeos_v01_rc_corpus_scale_answer_quality_gate(**inputs)


def test_corpus_scale_gate_ready_but_held_without_live_quality_execution(tmp_path: Path) -> None:
    report = _build(tmp_path)

    assert report["status"] == "ready"
    assert report["decision"] == HELD_DECISION
    assert report["nextRecommendedTranche"] == "corpus_scale_answer_quality_live_runner_design"
    assert report["qualityGateDecision"]["corpusScaleGate"] == "held"
    assert report["counts"]["seedPaperRows"] == 20
    assert report["counts"]["seedQuestionRows"] == 50
    assert report["counts"]["abstainNoAnswerPassRows"] == 50
    assert report["counts"]["defaultOffNoAnswerPassRows"] == 3
    assert report["counts"]["labsQualityPassRows"] == 4
    assert report["counts"]["strictEvidenceAvailableRows"] == 0
    assert report["counts"]["plannedAnswerQualityDryRunRows"] == 0
    assert report["counts"]["liveAnswerExecutionRows"] == 0
    assert report["counts"]["answerQualityMeasuredRows"] == 0
    assert report["counts"]["corpusScaleAnswerQualityGateGreenRows"] == 0
    assert report["counts"]["corpusScaleAnswerQualityGateHeldRows"] == 1
    assert report["counts"]["publicDefaultPromotionReadyRows"] == 0
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["schemaViolationCount"] == 0
    assert report["gate"]["seedBreadthReady"] is True
    assert report["gate"]["noAnswerSafetyReady"] is True
    assert report["gate"]["corpusScaleAnswerQualityGateGreen"] is False
    assert report["gate"]["publicDefaultPromotionAllowed"] is False
    assert "corpus_scale_live_answer_quality_execution_missing" in report["qualityGateDecision"]["gateBlockers"]
    assert validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_corpus_scale_gate_blocks_when_seed_breadth_is_too_small(tmp_path: Path) -> None:
    chain = _chain(tmp_path)
    seed = deepcopy(chain["seed_pack_report"])
    seed["counts"]["paperRows"] = 19
    report = _build(tmp_path, seed_pack_report=seed)

    assert report["status"] == "blocked"
    assert "complex_qa_seed_pack_less_than_20_papers" in report["gate"]["semanticViolations"]


def test_corpus_scale_gate_blocks_when_no_answer_smoke_not_ready(tmp_path: Path) -> None:
    no_answer = _json(DEFAULT_NO_ANSWER_REPORT)
    no_answer["status"] = "blocked"
    report = _build(tmp_path, no_answer_report=no_answer)

    assert report["status"] == "blocked"
    assert "default_off_no_answer_smoke_not_ready" in report["gate"]["semanticViolations"]


def test_corpus_scale_gate_blocks_when_labs_quality_has_failures(tmp_path: Path) -> None:
    labs = _json(DEFAULT_LABS_QUALITY_REPORT)
    labs["counts"] = deepcopy(labs["counts"])
    labs["counts"]["qualityPassRows"] = 3
    report = _build(tmp_path, labs_quality_report=labs)

    assert report["status"] == "blocked"
    assert "labs_opt_in_quality_eval_runner_not_all_passed" in report["gate"]["semanticViolations"]


def test_corpus_scale_gate_blocks_unsafe_counter_in_upstream_report(tmp_path: Path) -> None:
    chain = _chain(tmp_path)
    dry_run = deepcopy(chain["answer_quality_dry_run_report"])
    dry_run["counts"]["databaseMutationRows"] = 1
    report = _build(tmp_path, answer_quality_dry_run_report=dry_run)

    assert report["status"] == "blocked"
    assert "unsafe_counter_nonzero:report5:databaseMutationRows" in report["gate"]["semanticViolations"]


def test_corpus_scale_gate_blocks_private_path_marker(tmp_path: Path) -> None:
    marker = "/" + "Users" + "/example/private"
    labs = _json(DEFAULT_LABS_QUALITY_REPORT)
    labs["warnings"] = list(labs.get("warnings") or []) + [marker]
    report = _build(tmp_path, labs_quality_report=labs)

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "corpus_scale_answer_quality_gate_private_path_marker" in report["gate"]["semanticViolations"]


def test_corpus_scale_gate_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build(tmp_path)

    paths = write_knowledgeos_v01_rc_corpus_scale_answer_quality_gate(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# KnowledgeOS v0.1 RC Corpus-Scale Answer Quality Gate"
    )
    assert validate_payload(
        parsed,
        KNOWLEDGEOS_V01_RC_CORPUS_SCALE_ANSWER_QUALITY_GATE_SCHEMA_ID,
        strict=True,
    ).ok
