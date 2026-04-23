from __future__ import annotations

from knowledge_hub.application.ops_alerts import evaluate_paper_source_report_alerts
from knowledge_hub.application.paper_reports import build_paper_source_ops_report, verify_paper_source_state


class _FakeSQLite:
    def __init__(self, papers: dict[str, dict[str, str]]):
        self.papers = dict(papers)

    def get_paper(self, paper_id: str):
        return dict(self.papers.get(paper_id) or {})


def test_build_paper_source_ops_report_detects_pending_and_blocked_candidates():
    sqlite_db = _FakeSQLite(
        {
            "1502.03167": {
                "arxiv_id": "1502.03167",
                "title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
                "pdf_path": "/canonical/bn.pdf",
                "text_path": "/canonical/bn.txt",
            },
            "Batch_Normalization_c72acd36": {
                "arxiv_id": "Batch_Normalization_c72acd36",
                "title": "Batch Normalization",
                "pdf_path": "/wrong/bn.pdf",
                "text_path": "/wrong/bn.txt",
            },
            "localpdf_R_Graphics_Output_8b745a311c": {
                "arxiv_id": "localpdf_R_Graphics_Output_8b745a311c",
                "title": "R Graphics Output",
                "pdf_path": "/wrong/rgraphics.pdf",
                "text_path": "/wrong/rgraphics.txt",
            },
            "NeRF_1b9d0d11": {
                "arxiv_id": "NeRF_1b9d0d11",
                "title": "NeRF",
                "pdf_path": "/canonical/nerf.pdf",
                "text_path": "/canonical/nerf.txt",
            },
        }
    )

    report = build_paper_source_ops_report(sqlite_db, limit=20)

    assert report["status"] == "ok"
    assert report["counts"]["repairablePending"] == 1
    assert report["counts"]["blockedManual"] == 1
    assert report["counts"]["keepCurrentReviewed"] == 1
    batch_item = next(item for item in report["items"] if item["paperId"] == "Batch_Normalization_c72acd36")
    assert batch_item["needsRepair"] is True
    assert batch_item["canonicalPaperId"] == "1502.03167"
    blocked_item = next(item for item in report["items"] if item["paperId"] == "localpdf_R_Graphics_Output_8b745a311c")
    assert blocked_item["status"] == "blocked_manual_source_needed"


def test_evaluate_paper_source_report_alerts_emits_repair_actions():
    alerts, actions = evaluate_paper_source_report_alerts(
        counts={
            "repairablePending": 1,
            "blockedManual": 1,
            "blockedMissingCanonical": 0,
        },
        items=[
            {
                "paperId": "Deep_Residual_Learning_efbb7871",
                "title": "Deep Residual Learning",
                "needsRepair": True,
            }
        ],
    )

    assert {alert["code"] for alert in alerts} == {
        "paper_source_repair_pending",
        "paper_source_manual_fix_needed",
    }
    assert len(actions) == 1
    assert actions[0]["actionType"] == "repair_paper_source"
    assert actions[0]["paperId"] == "Deep_Residual_Learning_efbb7871"
    assert actions[0]["args"] == ["paper", "repair-source", "--paper-id", "Deep_Residual_Learning_efbb7871"]


def test_verify_paper_source_state_marks_aligned_after_relink():
    sqlite_db = _FakeSQLite(
        {
            "1502.03167": {
                "arxiv_id": "1502.03167",
                "title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
                "pdf_path": "/canonical/bn.pdf",
                "text_path": "/canonical/bn.txt",
            },
            "Batch_Normalization_c72acd36": {
                "arxiv_id": "Batch_Normalization_c72acd36",
                "title": "Batch Normalization",
                "pdf_path": "/canonical/bn.pdf",
                "text_path": "/canonical/bn.txt",
            },
        }
    )

    verification = verify_paper_source_state(sqlite_db, paper_id="Batch_Normalization_c72acd36")

    assert verification["status"] == "ok"
    assert verification["resolved"] is True
    assert verification["verificationStatus"] == "already_aligned"
    assert verification["item"]["alreadyAligned"] is True
