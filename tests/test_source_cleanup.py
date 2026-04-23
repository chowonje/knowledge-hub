from __future__ import annotations

import csv
import json
from pathlib import Path

from knowledge_hub.papers.source_cleanup import (
    apply_source_cleanup_plan,
    build_source_cleanup_plan,
    write_source_cleanup_artifacts,
)


class _FakeSQLite:
    def __init__(self):
        self.rows = {
            "Batch_Normalization_c72acd36": {
                "arxiv_id": "Batch_Normalization_c72acd36",
                "title": "Batch Normalization",
                "pdf_path": "/tmp/bad-bn.pdf",
                "text_path": "",
            },
            "Deep_Residual_Learning_efbb7871": {
                "arxiv_id": "Deep_Residual_Learning_efbb7871",
                "title": "Deep Residual Learning",
                "pdf_path": "/tmp/bad-deep-residual.pdf",
                "text_path": "",
            },
            "localpdf_Attention_is_All_you_Need_AI_Papers_lega_7398ae8b16": {
                "arxiv_id": "localpdf_Attention_is_All_you_Need_AI_Papers_lega_7398ae8b16",
                "title": "Attention is All you Need AI_Papers legacy",
                "pdf_path": "/tmp/bad-attention.pdf",
                "text_path": "/tmp/bad-attention.txt",
            },
            "1512.03385": {
                "arxiv_id": "1512.03385",
                "title": "Deep Residual Learning for Image Recognition",
                "pdf_path": "/tmp/good-deep-residual.pdf",
                "text_path": "/tmp/good-deep-residual.txt",
            },
            "1502.03167": {
                "arxiv_id": "1502.03167",
                "title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
                "pdf_path": "/tmp/good-bn.pdf",
                "text_path": "/tmp/good-bn.txt",
            },
            "1409.3215": {
                "arxiv_id": "1409.3215",
                "title": "Sequence to Sequence Learning with Neural Networks",
                "pdf_path": "/tmp/good-seq2seq.pdf",
                "text_path": "/tmp/good-seq2seq.txt",
            },
            "1706.03762": {
                "arxiv_id": "1706.03762",
                "title": "Attention is All you Need",
                "pdf_path": "/tmp/good-attention.pdf",
                "text_path": "/tmp/good-attention.txt",
            },
            "NeRF_1b9d0d11": {
                "arxiv_id": "NeRF_1b9d0d11",
                "title": "NeRF",
                "pdf_path": "/tmp/good-nerf.pdf",
                "text_path": "",
            },
            "Sequence_to_Sequence_Learning_0e5054e7": {
                "arxiv_id": "Sequence_to_Sequence_Learning_0e5054e7",
                "title": "Sequence to Sequence Learning",
                "pdf_path": "/tmp/bad-seq2seq.pdf",
                "text_path": "",
            },
        }
        self.upserts: list[dict[str, str]] = []

    def get_paper(self, paper_id: str):
        return dict(self.rows.get(paper_id) or {}) or None

    def upsert_paper(self, payload):
        self.rows[str(payload["arxiv_id"])] = dict(payload)
        self.upserts.append(dict(payload))


def test_build_source_cleanup_plan_assigns_expected_actions():
    sqlite_db = _FakeSQLite()
    rows = [
        {
            "paperId": "Batch_Normalization_c72acd36",
            "title": "Batch Normalization",
            "oldPdfPath": "/tmp/bad-bn.pdf",
            "oldTextPath": "",
            "recommendedParser": "raw",
        },
        {
            "paperId": "Deep_Residual_Learning_efbb7871",
            "title": "Deep Residual Learning",
            "oldPdfPath": "/tmp/bad-deep-residual.pdf",
            "oldTextPath": "",
            "recommendedParser": "raw",
        },
        {
            "paperId": "NeRF_1b9d0d11",
            "title": "NeRF",
            "oldPdfPath": "/tmp/good-nerf.pdf",
            "oldTextPath": "",
            "recommendedParser": "raw",
        },
        {
            "paperId": "Sequence_to_Sequence_Learning_0e5054e7",
            "title": "Sequence to Sequence Learning",
            "oldPdfPath": "/tmp/bad-seq2seq.pdf",
            "oldTextPath": "",
            "recommendedParser": "raw",
        },
        {
            "paperId": "localpdf_Attention_is_All_you_Need_AI_Papers_lega_7398ae8b16",
            "title": "Attention is All you Need AI_Papers legacy",
            "oldPdfPath": "/tmp/bad-attention.pdf",
            "oldTextPath": "/tmp/bad-attention.txt",
            "recommendedParser": "opendataloader",
        },
        {
            "paperId": "2503.07891",
            "title": "Gemini Embedding: Generalizable Embeddings from Gemini",
            "oldPdfPath": "/tmp/dino.pdf",
            "oldTextPath": "/tmp/dino.txt",
            "recommendedParser": "opendataloader",
        },
        {
            "paperId": "2005.14165",
            "title": "Language Models are Few-Shot Learners",
            "oldPdfPath": "/tmp/gpt3.pdf",
            "oldTextPath": "/tmp/gpt3.txt",
            "recommendedParser": "opendataloader",
        },
    ]
    decisions = build_source_cleanup_plan(rows, sqlite_db=sqlite_db)
    by_id = {item["paperId"]: item for item in decisions}
    assert by_id["Batch_Normalization_c72acd36"]["action"] == "relink_to_canonical"
    assert by_id["Batch_Normalization_c72acd36"]["status"] == "resolved"
    assert by_id["Batch_Normalization_c72acd36"]["newPdfPath"] == "/tmp/good-bn.pdf"
    assert by_id["Deep_Residual_Learning_efbb7871"]["action"] == "relink_to_canonical"
    assert by_id["Deep_Residual_Learning_efbb7871"]["status"] == "resolved"
    assert by_id["Deep_Residual_Learning_efbb7871"]["newPdfPath"] == "/tmp/good-deep-residual.pdf"
    assert by_id["NeRF_1b9d0d11"]["action"] == "keep_current_source"
    assert by_id["Sequence_to_Sequence_Learning_0e5054e7"]["action"] == "relink_to_canonical"
    assert by_id["Sequence_to_Sequence_Learning_0e5054e7"]["newPdfPath"] == "/tmp/good-seq2seq.pdf"
    assert by_id["localpdf_Attention_is_All_you_Need_AI_Papers_lega_7398ae8b16"]["action"] == "relink_to_canonical"
    assert by_id["localpdf_Attention_is_All_you_Need_AI_Papers_lega_7398ae8b16"]["status"] == "resolved"
    assert by_id["localpdf_Attention_is_All_you_Need_AI_Papers_lega_7398ae8b16"]["newPdfPath"] == "/tmp/good-attention.pdf"
    assert by_id["2503.07891"]["action"] == "exclude_until_manual_fix"
    assert by_id["2005.14165"]["action"] == "keep_current_source"


def test_apply_source_cleanup_plan_relinks_paths():
    sqlite_db = _FakeSQLite()
    decisions = build_source_cleanup_plan(
        [
            {
                "paperId": "Batch_Normalization_c72acd36",
                "title": "Batch Normalization",
                "oldPdfPath": "/tmp/bad-bn.pdf",
                "oldTextPath": "",
                "recommendedParser": "raw",
            },
            {
                "paperId": "Deep_Residual_Learning_efbb7871",
                "title": "Deep Residual Learning",
                "oldPdfPath": "/tmp/bad-deep-residual.pdf",
                "oldTextPath": "",
                "recommendedParser": "raw",
            },
            {
                "paperId": "Sequence_to_Sequence_Learning_0e5054e7",
                "title": "Sequence to Sequence Learning",
                "oldPdfPath": "/tmp/bad-seq2seq.pdf",
                "oldTextPath": "",
                "recommendedParser": "raw",
            },
            {
                "paperId": "localpdf_Attention_is_All_you_Need_AI_Papers_lega_7398ae8b16",
                "title": "Attention is All you Need AI_Papers legacy",
                "oldPdfPath": "/tmp/bad-attention.pdf",
                "oldTextPath": "/tmp/bad-attention.txt",
                "recommendedParser": "opendataloader",
            }
        ],
        sqlite_db=sqlite_db,
    )
    summary = apply_source_cleanup_plan(sqlite_db=sqlite_db, decisions=decisions)
    assert summary["applied"] == 4
    updated_bn = sqlite_db.get_paper("Batch_Normalization_c72acd36")
    assert updated_bn["pdf_path"] == "/tmp/good-bn.pdf"
    assert updated_bn["text_path"] == "/tmp/good-bn.txt"
    updated_deep = sqlite_db.get_paper("Deep_Residual_Learning_efbb7871")
    assert updated_deep["pdf_path"] == "/tmp/good-deep-residual.pdf"
    assert updated_deep["text_path"] == "/tmp/good-deep-residual.txt"
    updated_seq2seq = sqlite_db.get_paper("Sequence_to_Sequence_Learning_0e5054e7")
    assert updated_seq2seq["pdf_path"] == "/tmp/good-seq2seq.pdf"
    assert updated_seq2seq["text_path"] == "/tmp/good-seq2seq.txt"
    updated = sqlite_db.get_paper("localpdf_Attention_is_All_you_Need_AI_Papers_lega_7398ae8b16")
    assert updated["pdf_path"] == "/tmp/good-attention.pdf"
    assert updated["text_path"] == "/tmp/good-attention.txt"


def test_write_source_cleanup_artifacts_filters_blocked_pass_b_ids(tmp_path: Path):
    decisions = [
        {
            "paperId": "a",
            "title": "A",
            "action": "relink_to_canonical",
            "status": "resolved",
            "oldPdfPath": "",
            "oldTextPath": "",
            "newPdfPath": "/tmp/a.pdf",
            "newTextPath": "/tmp/a.txt",
            "canonicalPaperId": "ca",
            "canonicalTitle": "CA",
            "resolutionReason": "ok",
            "recommendedParser": "raw",
        },
        {
            "paperId": "b",
            "title": "B",
            "action": "exclude_until_manual_fix",
            "status": "blocked_manual_source_needed",
            "oldPdfPath": "",
            "oldTextPath": "",
            "newPdfPath": "",
            "newTextPath": "",
            "canonicalPaperId": "",
            "canonicalTitle": "",
            "resolutionReason": "blocked",
            "recommendedParser": "opendataloader",
        },
    ]
    artifact_paths = write_source_cleanup_artifacts(
        artifact_dir=tmp_path,
        decisions=decisions,
        pass_b_ids=["a", "b", "c"],
    )
    clean_ids = Path(artifact_paths["passBCleanIds"]).read_text(encoding="utf-8").splitlines()
    assert clean_ids == ["a", "c"]
    summary = json.loads(Path(artifact_paths["summary"]).read_text(encoding="utf-8"))
    assert summary["passBExcludedCount"] == 1
    with Path(artifact_paths["resolutionsCsv"]).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
