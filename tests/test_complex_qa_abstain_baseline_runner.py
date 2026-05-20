from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.complex_qa_abstain_baseline_runner import (
    COMPLEX_QA_ABSTAIN_BASELINE_SCHEMA_ID,
    build_complex_qa_abstain_baseline,
    write_complex_qa_abstain_baseline_reports,
)
from knowledge_hub.papers.complex_qa_seed_pack import build_complex_qa_seed_pack


def _write_manifest(root: Path, count: int = 19) -> Path:
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
    path.write_text(json.dumps({"schema": "knowledge-hub.corpus-manifest.v1", "artifacts": artifacts}), encoding="utf-8")
    return path


def _write_seed_pack(tmp_path: Path) -> Path:
    manifest = _write_manifest(tmp_path / "input", count=19)
    seed_pack = build_complex_qa_seed_pack(corpus_manifest=manifest)
    path = tmp_path / "complex-paper-qa-seed-pack.json"
    path.write_text(json.dumps(seed_pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def test_complex_qa_abstain_baseline_counts_and_schema_validation(tmp_path: Path) -> None:
    seed_pack_report = _write_seed_pack(tmp_path)

    payload = build_complex_qa_abstain_baseline(seed_pack_report=seed_pack_report)

    assert payload["schema"] == COMPLEX_QA_ABSTAIN_BASELINE_SCHEMA_ID
    assert validate_payload(payload, COMPLEX_QA_ABSTAIN_BASELINE_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["paperRows"] == 20
    assert payload["counts"]["questionRows"] == 50
    assert payload["counts"]["expectedNoAnswerRows"] == 17
    assert payload["counts"]["blockedUntilStructuredEvidenceRows"] == 33
    assert payload["counts"]["expectedNoAnswerOrBlockedRows"] == 50
    assert payload["counts"]["answerableRows"] == 0
    assert payload["counts"]["abstainExpectedRows"] == 50
    assert payload["counts"]["abstainBaselinePassRows"] == 50
    assert payload["counts"]["abstainBaselineFailRows"] == 0
    assert payload["counts"]["unsafeAnsweredRows"] == 0
    assert payload["counts"]["schemaViolationCount"] == 0
    assert payload["counts"]["abstainNoAnswerPassRate"] == 1.0
    assert payload["counts"]["byBaselineVerdict"] == {
        "pass_blocked_until_structured_evidence": 33,
        "pass_expected_no_answer": 17,
    }


def test_complex_qa_abstain_baseline_keeps_answer_and_mutation_paths_disabled(tmp_path: Path) -> None:
    seed_pack_report = _write_seed_pack(tmp_path)

    payload = build_complex_qa_abstain_baseline(seed_pack_report=seed_pack_report)

    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["contractOnlyStaticBaseline"] is True
    assert payload["policy"]["questionExecutionRun"] is False
    assert payload["policy"]["answerGenerationRun"] is False
    assert payload["policy"]["llmCalls"] is False
    assert payload["policy"]["answerPathChanged"] is False
    assert payload["policy"]["answerPathInvoked"] is False
    assert payload["policy"]["searchIndexQueried"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert payload["policy"]["indexMutation"] is False
    assert payload["policy"]["reindexOrReembed"] is False
    assert payload["policy"]["vaultScan"] is False
    assert payload["policy"]["runtimeEvidenceCreated"] is False
    assert payload["policy"]["citationEvidenceCreated"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False
    assert payload["counts"]["answerGeneratedRows"] == 0
    assert payload["counts"]["llmCallRows"] == 0
    assert payload["counts"]["answerPathInvokedRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["counts"]["indexMutationRows"] == 0
    assert payload["counts"]["vaultScanRows"] == 0
    assert payload["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert payload["counts"]["citationEvidenceCreatedRows"] == 0
    assert payload["counts"]["strictEvidenceCreatedRows"] == 0
    for row in payload["rows"]:
        assert row["strictEvidenceAvailable"] is False
        assert row["answerGenerated"] is False
        assert row["unsafeAnswered"] is False
        assert row["llmCall"] is False
        assert row["answerPathInvoked"] is False
        assert row["strictMissingBlockers"]
        assert row["requiredEvidenceContract"]
        assert row["riskNotes"]


def test_complex_qa_abstain_baseline_writer_outputs_schema_valid_report_and_summary(tmp_path: Path) -> None:
    seed_pack_report = _write_seed_pack(tmp_path)
    payload = build_complex_qa_abstain_baseline(seed_pack_report=seed_pack_report)

    report_paths = write_complex_qa_abstain_baseline_reports(payload, tmp_path / "reports")

    assert set(report_paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(report_paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(report_paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(report_paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, COMPLEX_QA_ABSTAIN_BASELINE_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["questionRows"] == 50
    assert summary["nextRecommendedTranche"] == "structured evidence gated complex QA comparison runner"
    assert "Report-only static baseline" in markdown
