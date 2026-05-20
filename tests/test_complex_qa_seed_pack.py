from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.complex_qa_seed_pack import (
    COMPLEX_QA_SEED_PACK_SCHEMA_ID,
    build_complex_qa_seed_pack,
    write_complex_qa_seed_pack_reports,
)


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


def test_complex_qa_seed_pack_generates_20_papers_50_questions_and_validates_schema(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, count=19)

    payload = build_complex_qa_seed_pack(corpus_manifest=manifest)

    assert payload["schema"] == COMPLEX_QA_SEED_PACK_SCHEMA_ID
    assert validate_payload(payload, COMPLEX_QA_SEED_PACK_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["paperRows"] == 20
    assert payload["counts"]["questionRows"] == 50
    assert payload["counts"]["schemaViolationCount"] == 0
    assert payload["counts"]["expectedNoAnswerOrBlockedRows"] == 50
    assert payload["counts"]["answerableRows"] == 0
    assert payload["counts"]["tableNumericQuestionRows"] > 0
    assert payload["counts"]["equationCitationQuestionRows"] > 0
    assert payload["counts"]["figureCaptionQuestionRows"] > 0
    assert payload["counts"]["methodComparisonQuestionRows"] > 0
    assert payload["counts"]["limitationQuestionRows"] > 0
    assert payload["counts"]["appendixTableLookupQuestionRows"] > 0


def test_complex_qa_seed_pack_questions_include_required_contract_fields(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, count=20)

    payload = build_complex_qa_seed_pack(corpus_manifest=manifest)

    for question in payload["questions"]:
        assert question["questionId"]
        assert question.get("paperId") or question.get("paperIds")
        assert question["expectedEvidenceType"]
        assert question["answerabilityExpectation"] in {"answerable", "expected_no_answer", "blocked_until_structured_evidence"}
        assert question["requiredEvidenceContract"]["contractId"]
        assert question["requiredEvidenceContract"]["mustHave"]
        assert question["requiredEvidenceContract"]["blockedIfMissing"]
        assert question["riskNotes"]


def test_complex_qa_seed_pack_policy_keeps_report_only_no_mutation_surface(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, count=20)

    payload = build_complex_qa_seed_pack(corpus_manifest=manifest)

    assert payload["policy"]["reportOnly"] is True
    assert payload["policy"]["answerGenerationRun"] is False
    assert payload["policy"]["llmCalls"] is False
    assert payload["policy"]["answerPathChanged"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert payload["policy"]["indexMutation"] is False
    assert payload["policy"]["reindexOrReembed"] is False
    assert payload["policy"]["vaultScan"] is False
    assert payload["policy"]["runtimeEvidenceCreated"] is False
    assert payload["policy"]["citationEvidenceCreated"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False
    assert payload["counts"]["llmCallRows"] == 0
    assert payload["counts"]["databaseMutationRows"] == 0
    assert payload["counts"]["vaultScanRows"] == 0
    assert payload["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert payload["counts"]["citationEvidenceCreatedRows"] == 0


def test_complex_qa_seed_pack_writer_outputs_schema_valid_report_and_summary(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "input", count=20)
    payload = build_complex_qa_seed_pack(corpus_manifest=manifest)

    report_paths = write_complex_qa_seed_pack_reports(payload, tmp_path / "reports")

    assert set(report_paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(report_paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(report_paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(report_paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, COMPLEX_QA_SEED_PACK_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["questionRows"] == 50
    assert summary["nextRecommendedTranche"] == "complex QA abstain baseline runner"
    assert "Report-only seed pack" in markdown
