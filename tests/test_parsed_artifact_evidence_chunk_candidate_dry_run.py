from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_dry_run import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_candidate_dry_run,
)
from knowledge_hub.papers.paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review import (
    PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_CONTRACT_REVIEW_SCHEMA_ID,
    READY_DECISION as CONTRACT_READY_DECISION,
)


CONTRACT_REF = "eval/knowledgeos/reports/paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review.v1.json"


def _contract_review() -> dict[str, object]:
    zero_fields = {
        "runtimeRouteWriteRows": 0,
        "runtimeConfigMutationRows": 0,
        "operationalSearchIndexQueryRows": 0,
        "runtimeVisibleRows": 0,
        "answerVisibleRows": 0,
        "answerGenerationRows": 0,
        "strictEvidenceRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "answerableWithoutTextEvidenceRows": 0,
        "candidateStoreWriteRows": 0,
        "sourceSpanCreatedRows": 0,
        "sourceSpanCandidateCreatedRows": 0,
        "parsedArtifactEvidenceChunkCreatedRows": 0,
        "embeddingCallRows": 0,
        "embeddingVectorWriteRows": 0,
        "vectorIndexWriteRows": 0,
        "productionVectorIndexWriteRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "parserExecutionRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "graphDbWriteRows": 0,
        "ontologyWriteRows": 0,
        "memoryCardWriteRows": 0,
        "clusterWriteRows": 0,
        "vaultScanRows": 0,
        "externalDownloadRows": 0,
        "branchDeletionRows": 0,
        "githubPrMutationRows": 0,
    }
    return {
        "schema": PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_CONTRACT_REVIEW_SCHEMA_ID,
        "status": "ready",
        "decision": CONTRACT_READY_DECISION,
        "gate": {"passed": True},
        "counts": {
            "allowedArtifactTypeRows": 5,
            "disallowedSourceRows": 5,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
            **zero_fields,
        },
    }


def _write_parsed_artifact(
    papers_dir: Path,
    *,
    paper_id: str = "paper-1",
    source_hash: str = "sha256:source",
    document_text: str | None = None,
    manifest_hash: str | None = None,
    document_hash: str | None = None,
) -> None:
    text = document_text or (
        "## Page 1\n\n"
        "1 Introduction This paper introduces a careful local evidence runtime for research workflows. "
        "The method keeps evidence inspectable and avoids unsupported synthesis when source backing is absent.\n\n"
        "## Page 2\n\n"
        "2 Methods The runtime extracts paragraph candidates from parsed artifacts and records source hashes, "
        "character offsets, excerpts, and snippet hashes for later answerability review."
    )
    paper_dir = papers_dir / "parsed" / paper_id
    paper_dir.mkdir(parents=True)
    (paper_dir / "document.md").write_text(text, encoding="utf-8")
    first = "1 Introduction This paper introduces a careful local evidence runtime for research workflows. The method keeps evidence inspectable and avoids unsupported synthesis when source backing is absent."
    second = "2 Methods The runtime extracts paragraph candidates from parsed artifacts and records source hashes, character offsets, excerpts, and snippet hashes for later answerability review."
    document = {
        "markdown_text": text,
        "parser_meta": {
            "sourceContentHash": document_hash or source_hash,
            "source_content_hash": document_hash or source_hash,
        },
        "elements": [
            {"type": "paragraph", "page": 1, "heading_path": ["Page 1"], "text": first},
            {"type": "paragraph", "page": 2, "heading_path": ["Page 2"], "text": second},
        ],
    }
    (paper_dir / "document.json").write_text(json.dumps(document), encoding="utf-8")
    manifest = {
        "paper_id": paper_id,
        "sourceContentHash": manifest_hash or source_hash,
        "source_content_hash": manifest_hash or source_hash,
        "parser_meta": {
            "parser": "pymupdf",
            "sourceContentHash": manifest_hash or source_hash,
            "source_content_hash": manifest_hash or source_hash,
        },
    }
    (paper_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _build(tmp_path: Path, contract: dict[str, object] | None = None) -> dict[str, object]:
    return build_parsed_artifact_evidence_chunk_candidate_dry_run(
        contract_review=contract or _contract_review(),
        source_contract_review_ref=CONTRACT_REF,
        papers_dir=tmp_path / "papers",
        generated_at="2026-05-29T00:00:00Z",
    )


def test_candidate_dry_run_scans_parsed_artifacts_and_emits_candidate_rows(tmp_path: Path) -> None:
    _write_parsed_artifact(tmp_path / "papers")

    report = _build(tmp_path)

    assert report["schema"] == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["nextRecommendedTranche"] == "parsed_artifact_evidence_chunk_candidate_canary_apply_readback"
    assert report["counts"]["parsedArtifactRows"] == 1
    assert report["counts"]["selectedCandidateRows"] >= 2
    assert report["counts"]["paragraphCandidateRows"] >= 1
    assert report["counts"]["sectionCandidateRows"] >= 1
    assert report["counts"]["answerEvidenceCandidateRows"] == report["counts"]["selectedCandidateRows"]
    assert report["counts"]["answerabilityCandidateRows"] == report["counts"]["selectedCandidateRows"]
    assert report["counts"]["answerVisibleRows"] == 0
    assert report["counts"]["strictEvidenceRows"] == 0
    assert report["counts"]["databaseMutationRows"] == 0
    assert report["gate"]["passed"] is True

    first = report["candidateRows"][0]
    assert first["sourceRef"] == "papers_dir/parsed/paper-1/document.md"
    assert first["sourceContentHash"] == "sha256:source"
    assert first["spanLocator"].startswith("chars:")
    assert first["candidateOnly"] is True
    assert first["strictEvidence"] is False

    validation = validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_candidate_dry_run_blocks_if_contract_review_is_not_ready(tmp_path: Path) -> None:
    _write_parsed_artifact(tmp_path / "papers")
    contract = copy.deepcopy(_contract_review())
    contract["status"] = "blocked"
    contract["decision"] = "blocked"
    contract["gate"] = {"passed": False}

    report = _build(tmp_path, contract)

    assert report["status"] == "blocked"
    assert report["counts"]["selectedCandidateRows"] == 0
    assert "contract_review_not_ready" in report["technicalBlockers"]
    assert "contract_review_invalid_decision" in report["technicalBlockers"]
    assert "contract_review_gate_not_passed" in report["technicalBlockers"]


def test_candidate_dry_run_blocks_paper_with_missing_hash(tmp_path: Path) -> None:
    _write_parsed_artifact(tmp_path / "papers", source_hash="")

    report = _build(tmp_path)

    assert report["status"] == "blocked"
    assert report["counts"]["blockedMissingSourceHashRows"] == 1
    assert report["counts"]["selectedCandidateRows"] == 0


def test_candidate_dry_run_blocks_paper_with_hash_mismatch(tmp_path: Path) -> None:
    _write_parsed_artifact(
        tmp_path / "papers",
        manifest_hash="sha256:manifest",
        document_hash="sha256:document",
    )

    report = _build(tmp_path)

    assert report["status"] == "blocked"
    assert report["counts"]["blockedSourceHashMismatchRows"] == 1
    assert report["counts"]["selectedCandidateRows"] == 0


def test_candidate_dry_run_blocks_missing_document_json(tmp_path: Path) -> None:
    _write_parsed_artifact(tmp_path / "papers")
    (tmp_path / "papers" / "parsed" / "paper-1" / "document.json").unlink()

    report = _build(tmp_path)

    assert report["status"] == "blocked"
    assert report["counts"]["blockedMissingDocumentJsonRows"] == 1
    assert report["counts"]["selectedCandidateRows"] == 0


def test_candidate_dry_run_caps_rows_per_paper_and_total_rows(tmp_path: Path) -> None:
    for index in range(4):
        _write_parsed_artifact(tmp_path / "papers", paper_id=f"paper-{index}", source_hash=f"sha256:{index}")

    report = build_parsed_artifact_evidence_chunk_candidate_dry_run(
        contract_review=_contract_review(),
        source_contract_review_ref=CONTRACT_REF,
        papers_dir=tmp_path / "papers",
        max_rows_per_paper=1,
        max_total_rows=2,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["counts"]["selectedCandidateRows"] == 2
    assert report["counts"]["heldCandidateRows"] == 2


def test_candidate_dry_run_keeps_private_paths_out_of_report(tmp_path: Path) -> None:
    _write_parsed_artifact(tmp_path / "papers")
    manifest_path = tmp_path / "papers" / "parsed" / "paper-1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parser_meta"]["source_pdf"] = "Mobile Documents/private-paper.pdf"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _build(tmp_path)
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert report["status"] == "ready"
    assert "Mobile Documents/private-paper.pdf" not in rendered
