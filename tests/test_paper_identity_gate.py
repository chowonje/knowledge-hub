"""Tranche B: parse-acceptance page-1 identity gate.

The 2026-06-11 parsed-store audit found 13 papers whose on-disk PDF is a
different document than the registered metadata (e.g. 1207.0580 "dropout"
holding the AlphaFold Nature article). These tests pin the gate that rejects
such artifacts at parse acceptance using two page-1 signals: arXiv watermark
id and registered-title token containment.
"""

from __future__ import annotations

import pytest

from knowledge_hub.papers.identity_gate import (
    IDENTITY_GATE_FAIL_REASON_PREFIX,
    PaperIdentityGateError,
    check_page1_identity,
    enforce_parse_identity,
)
from knowledge_hub.papers.parsed_materialization import materialize_parsed_artifacts

DROPOUT_TITLE = "Improving neural networks by preventing co-adaptation of feature detectors"

# Page-1 double of the wrong on-disk PDF observed in the audit for 1207.0580.
ALPHAFOLD_PAGE1 = """
Highly accurate protein structure prediction with AlphaFold
John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov
Nature 596, 583-589 (2021)
Proteins are essential to life, and understanding their structure can
facilitate a mechanistic understanding of their function. Through an enormous
experimental effort the structures of around 100,000 unique proteins have
been determined, but this represents a small fraction of the billions of
known protein sequences. Here we provide the first computational method that
can regularly predict protein structures with atomic accuracy.
"""

DROPOUT_PAGE1 = """
arXiv:1207.0580v1 [cs.NE] 3 Jul 2012
Improving neural networks by preventing co-adaptation of feature detectors
G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, R. R. Salakhutdinov
When a large feedforward neural network is trained on a small training set,
it typically performs poorly on held-out test data. This "overfitting" is
greatly reduced by randomly omitting half of the feature detectors on each
training case.
"""


def test_wrong_document_fails_title_containment():
    result = check_page1_identity(
        page1_text=ALPHAFOLD_PAGE1,
        registered_title=DROPOUT_TITLE,
        expected_arxiv_id="1207.0580",
    )
    assert result.status == "fail"
    assert result.reason.startswith("title_containment_too_low")
    assert result.ok is False


def test_correct_document_passes_watermark():
    result = check_page1_identity(
        page1_text=DROPOUT_PAGE1,
        registered_title=DROPOUT_TITLE,
        expected_arxiv_id="1207.0580",
    )
    assert result.status == "pass"
    assert result.reason == "watermark_match"


def test_correct_document_passes_title_without_watermark():
    result = check_page1_identity(
        page1_text=DROPOUT_PAGE1.replace("arXiv:1207.0580v1 [cs.NE] 3 Jul 2012", ""),
        registered_title=DROPOUT_TITLE,
        expected_arxiv_id="1207.0580",
    )
    assert result.status == "pass"
    assert result.reason == "title_token_containment"


def test_watermark_mismatch_fails():
    page1 = "arXiv:2508.10104v1 [cs.CV]\nDINOv3\nSelf-supervised vision transformers at scale."
    result = check_page1_identity(
        page1_text=page1,
        registered_title="Rich feature hierarchies for accurate object detection",
        expected_arxiv_id="1311.2524",
    )
    assert result.status == "fail"
    assert result.reason.startswith("watermark_mismatch")


def test_watermark_match_overrides_stale_registered_title():
    page1 = (
        "arXiv:2505.05849v4 [cs.CR]\n"
        "AgentVigil: Generic Black-Box Red-teaming for Indirect Prompt Injection"
    )
    result = check_page1_identity(
        page1_text=page1,
        registered_title="Totally Unrelated Stale Words Everywhere Honestly",
        expected_arxiv_id="2505.05849",
    )
    assert result.status == "pass"
    assert result.reason == "watermark_match"


def test_placeholder_title_without_watermark_is_inconclusive():
    result = check_page1_identity(
        page1_text="Some scanned page with no recognizable header text at all.",
        registered_title="arXiv 2401.17043",
        expected_arxiv_id="",
    )
    assert result.status == "inconclusive"
    assert result.ok is True


def test_empty_page_text_is_inconclusive():
    result = check_page1_identity(
        page1_text="",
        registered_title=DROPOUT_TITLE,
        expected_arxiv_id="1207.0580",
    )
    assert result.status == "inconclusive"


def test_enforce_raises_for_wrong_document():
    with pytest.raises(PaperIdentityGateError) as exc_info:
        enforce_parse_identity(
            paper_id="1207.0580",
            pdf_path="/nonexistent/never-read.pdf",
            registered_title=DROPOUT_TITLE,
            page1_text=ALPHAFOLD_PAGE1,
        )
    error = exc_info.value
    assert error.paper_id == "1207.0580"
    assert IDENTITY_GATE_FAIL_REASON_PREFIX in str(error)


def test_enforce_returns_result_for_correct_document():
    result = enforce_parse_identity(
        paper_id="1207.0580",
        pdf_path="/nonexistent/never-read.pdf",
        registered_title=DROPOUT_TITLE,
        page1_text=DROPOUT_PAGE1,
    )
    assert result.status == "pass"


class _FakeSqliteDb:
    def __init__(self, paper: dict):
        self._paper = paper

    def get_paper(self, paper_id: str):
        return dict(self._paper) if paper_id == self._paper["arxiv_id"] else None


class _FakeAdapter:
    """Stands in for PyMuPDFAdapter; writes the three parsed artifacts."""

    def __init__(self, *, papers_dir):
        from pathlib import Path

        self.papers_dir = Path(str(papers_dir))

    def ensure_artifacts(self, *, paper_id, pdf_path, refresh=False, allow_ocr=True):
        artifact_dir = self.papers_dir / "parsed" / paper_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "document.md").write_text("# parsed", encoding="utf-8")
        (artifact_dir / "document.json").write_text("{}", encoding="utf-8")
        (artifact_dir / "manifest.json").write_text(
            '{"sourceContentHash": "sha256:fake-for-test"}', encoding="utf-8"
        )
        return None


def _seed_pdf(tmp_path):
    pdf_path = tmp_path / "1207.0580.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake-bytes-for-identity-gate-test")
    return pdf_path


def test_materialize_blocks_wrong_document(tmp_path, monkeypatch):
    import knowledge_hub.papers.parsed_materialization as pm

    pdf_path = _seed_pdf(tmp_path)
    db = _FakeSqliteDb(
        {"arxiv_id": "1207.0580", "title": DROPOUT_TITLE, "pdf_path": str(pdf_path)}
    )
    monkeypatch.setattr(pm, "extract_first_page_text", lambda path, **kwargs: ALPHAFOLD_PAGE1)
    monkeypatch.setattr(pm, "PyMuPDFAdapter", _FakeAdapter)

    payload = materialize_parsed_artifacts(
        sqlite_db=db,
        papers_dir=tmp_path / "papers",
        paper_ids=["1207.0580"],
        apply=True,
    )
    item = payload["items"][0]
    assert item["status"] == "blocked"
    assert item["reason"].startswith(IDENTITY_GATE_FAIL_REASON_PREFIX)
    assert not (tmp_path / "papers" / "parsed" / "1207.0580" / "document.md").exists()


def test_materialize_accepts_matching_document(tmp_path, monkeypatch):
    import knowledge_hub.papers.parsed_materialization as pm

    pdf_path = _seed_pdf(tmp_path)
    db = _FakeSqliteDb(
        {"arxiv_id": "1207.0580", "title": DROPOUT_TITLE, "pdf_path": str(pdf_path)}
    )
    monkeypatch.setattr(pm, "extract_first_page_text", lambda path, **kwargs: DROPOUT_PAGE1)
    monkeypatch.setattr(pm, "PyMuPDFAdapter", _FakeAdapter)

    payload = materialize_parsed_artifacts(
        sqlite_db=db,
        papers_dir=tmp_path / "papers",
        paper_ids=["1207.0580"],
        apply=True,
    )
    item = payload["items"][0]
    assert item["status"] == "materialized"
    assert (tmp_path / "papers" / "parsed" / "1207.0580" / "document.md").exists()
