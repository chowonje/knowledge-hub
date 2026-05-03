from __future__ import annotations

from knowledge_hub.papers.memory_quality import evaluate_paper_memory_quality, summarize_quality_reports


def test_evaluate_paper_memory_quality_marks_weak_card_and_auxiliary_signals():
    report = evaluate_paper_memory_quality(
        title="Gemini Embedding",
        paper_core="DINOv3 improves image representation quality for self-supervised learning benchmarks.",
        method_core="",
        evidence_core="",
        limitations="Limited information provided in the extracted text.",
        diagnostics={
            "fallbackUsed": True,
            "textSanitation": {
                "weakContent": False,
                "translated": {"startsWithLatex": False},
                "raw": {"startsWithLatex": True},
            },
        },
    )

    assert report["weakCard"] is True
    assert "empty_method_core" in report["weakReasons"]
    assert "empty_evidence_core" in report["weakReasons"]
    assert "generic_limitation" in report["weakReasons"]
    assert "fallback_used" in report["weakReasons"]
    assert report["sourceStartsLatex"] is True
    assert report["semanticMismatchLikely"] is True


def test_summarize_quality_reports_aggregates_counts_and_rates():
    summary = summarize_quality_reports(
        [
            {
                "weakCard": True,
                "needsReview": True,
                "weakReasons": ["empty_method_core"],
                "auxiliaryReviewReasons": ["source_starts_latex"],
                "paperCoreHasLatex": False,
                "emptyMethodCore": True,
                "emptyEvidenceCore": False,
                "genericLimitation": False,
                "fallbackUsed": False,
                "sourceStartsLatex": True,
                "weakSanitizedContent": False,
                "semanticMismatchLikely": False,
            },
            {
                "weakCard": False,
                "needsReview": False,
                "weakReasons": [],
                "auxiliaryReviewReasons": [],
                "paperCoreHasLatex": False,
                "emptyMethodCore": False,
                "emptyEvidenceCore": False,
                "genericLimitation": False,
                "fallbackUsed": False,
                "sourceStartsLatex": False,
                "weakSanitizedContent": False,
                "semanticMismatchLikely": False,
            },
        ]
    )

    assert summary["weakCardCount"] == 1
    assert summary["weakCardRate"] == 0.5
    assert summary["needsReviewCount"] == 1
    assert summary["sourceStartsLatexCount"] == 1
    assert summary["weakReasonCounts"]["empty_method_core"] == 1
    assert summary["auxiliaryReviewReasonCounts"]["source_starts_latex"] == 1
