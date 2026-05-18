from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tarfile

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.arxiv_source_tex_availability_audit import (
    ARXIV_SOURCE_TEX_AVAILABILITY_AUDIT_SCHEMA_ID,
    build_arxiv_source_tex_availability_audit,
    write_arxiv_source_tex_availability_audit_reports,
)


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_tar(root: Path, paper_id: str, tex: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{paper_id}-source.tar.gz"
    payload = tex.encode("utf-8")
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("main.tex")
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    return path


def _mineru_report(root: Path, paper_id: str) -> Path:
    return _write_json(
        root,
        "mineru.json",
        {
            "schema": "knowledge-hub.paper.mineru-source-alignment-audit.v1",
            "candidates": [
                {
                    "candidate_id": f"{paper_id}:section:0001",
                    "paper_id": paper_id,
                    "candidate_type": "section_candidate",
                    "candidate_text": "Introduction",
                    "mineruCandidate": {"bbox": [1, 2, 3, 4]},
                },
                {
                    "candidate_id": f"{paper_id}:figure:0001",
                    "paper_id": paper_id,
                    "candidate_type": "figure_caption_candidate",
                    "candidate_text": "Model overview.",
                    "mineruCandidate": {"bbox": [5, 6, 7, 8]},
                },
            ],
        },
    )


def test_arxiv_source_tex_audit_extracts_cached_tex_structure_without_runtime_authority(tmp_path: Path) -> None:
    paper_id = "1234.5678"
    _write_tar(
        tmp_path / "cache",
        paper_id,
        r"""
        \section{Introduction}
        Text.
        \subsection{Background}
        \begin{equation} a=b \end{equation}
        \begin{figure}
        \caption{Model overview.}
        \end{figure}
        \begin{table}
        \caption{Scores.}
        \begin{tabular}{cc} A & B \end{tabular}
        \end{table}
        """,
    )
    parsed = tmp_path / "parsed" / paper_id
    parsed.mkdir(parents=True)
    parsed.joinpath("document.md").write_text(
        "# Paper\n\n## Page 1\n\nIntroduction\n\nBackground\n\nModel overview.\n\nScores.\n",
        encoding="utf-8",
    )
    mineru = _mineru_report(tmp_path, paper_id)

    payload = build_arxiv_source_tex_availability_audit(
        paper_ids=[paper_id],
        output_dir=tmp_path / "out",
        allow_network=False,
        parsed_root=tmp_path / "parsed",
        source_cache_dir=tmp_path / "cache",
        mineru_source_alignment_report=mineru,
    )

    assert payload["schema"] == ARXIV_SOURCE_TEX_AVAILABILITY_AUDIT_SCHEMA_ID
    assert validate_payload(payload, ARXIV_SOURCE_TEX_AVAILABILITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["sourceAvailablePapers"] == 1
    assert payload["counts"]["sectionCommandRows"] == 1
    assert payload["counts"]["subsectionCommandRows"] == 1
    assert payload["counts"]["equationEnvironmentRows"] == 1
    assert payload["counts"]["equationEnvironmentTextRows"] == 1
    assert payload["counts"]["figureEnvironmentRows"] == 1
    assert payload["counts"]["tableEnvironmentRows"] == 1
    assert payload["counts"]["tabularEnvironmentRows"] == 1
    assert payload["counts"]["captionCommandRows"] == 2
    assert payload["counts"]["canonicalAlignedRows"] >= 4
    assert payload["counts"]["mineruLayoutLinkedRows"] == 2
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["policy"]["databaseMutation"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False
    row = next(item for item in payload["structureRows"] if item["candidate_text"] == "Introduction")
    assert row["canonical_alignment_status"] == "aligned"
    assert row["mineru_layout_link_status"] == "linked"
    assert row["strict_eligible"] is False
    assert "source_structure_candidate_only" in row["strict_blockers"]
    equation = next(item for item in payload["structureRows"] if item["structure_type"] == "equation_environment")
    assert equation["candidate_text"] == "a=b"
    assert equation["tex_chars_end"] > equation["tex_chars_start"]
    assert "equation_text_or_semantics_not_citation_grade" in equation["strict_blockers"]


def test_arxiv_source_tex_audit_requires_explicit_network_without_cached_source(tmp_path: Path) -> None:
    payload = build_arxiv_source_tex_availability_audit(
        paper_ids=["1234.5678"],
        output_dir=tmp_path / "out",
        allow_network=False,
        parsed_root=tmp_path / "parsed",
        source_cache_dir=tmp_path / "missing-cache",
        mineru_source_alignment_report=None,
    )

    assert validate_payload(payload, ARXIV_SOURCE_TEX_AVAILABILITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["sourceAvailablePapers"] == 0
    assert payload["counts"]["sourceSkippedNetworkDisabledPapers"] == 1
    assert payload["papers"][0]["fetch_status"] == "skipped_network_disabled"
    assert payload["papers"][0]["failure_reason"] == "network_not_allowed_and_no_cached_source"
    assert payload["gate"]["networkUsed"] is False
    assert payload["gate"]["runtimePromotionAllowed"] is False


def test_arxiv_source_tex_audit_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    payload = build_arxiv_source_tex_availability_audit(
        paper_ids=["1234.5678"],
        output_dir=tmp_path / "out",
        allow_network=False,
        parsed_root=tmp_path / "parsed",
        source_cache_dir=tmp_path / "missing-cache",
        mineru_source_alignment_report=None,
    )

    paths = write_arxiv_source_tex_availability_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, ARXIV_SOURCE_TEX_AVAILABILITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["runtimeEvidenceRows"] == 0
    assert "This audit is report-only" in markdown
