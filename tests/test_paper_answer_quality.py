"""Paper answer quality helper tests."""

from __future__ import annotations

from knowledge_hub.ai.paper_answer_quality import (
    PaperAnswerScopePlan,
    apply_paper_answer_scope,
    build_paper_answer_plan,
    build_paper_answer_quality_bundle,
    build_paper_citation_assembly,
    classify_paper_answer_query,
)


def _evidence_item(
    *,
    title: str,
    source_type: str,
    paper_id: str = "",
    excerpt: str = "",
    parent_id: str = "",
    score: float = 0.0,
):
    payload = {
        "title": title,
        "source_type": source_type,
        "excerpt": excerpt,
        "parent_id": parent_id,
        "score": score,
    }
    if paper_id:
        payload["paper_id"] = paper_id
    return payload


def test_classify_paper_answer_query_detects_lookup_and_analysis():
    assert classify_paper_answer_query("paper id 2602.16662") == "paper_lookup"
    assert classify_paper_answer_query("이 논문의 한계와 evaluation은?") == "paper_analysis"
    assert classify_paper_answer_query("what is retrieval?") == "general"


def test_build_paper_answer_plan_scopes_paper_queries_and_budgeting():
    plan = build_paper_answer_plan(
        "What is the contribution and limitation of this paper?",
        source_type="note",
        paper_memory_prefilter={"applied": True, "requestedMode": "prefilter", "matchedPaperIds": ["2602.16662"]},
        top_k=7,
    )

    assert isinstance(plan, PaperAnswerScopePlan)
    assert plan.paper_scoped is True
    assert plan.question_kind == "paper_analysis"
    assert plan.requested_paper_mode == "prefilter"
    assert plan.matched_paper_ids == ("2602.16662",)
    assert plan.evidence_budget == 2
    assert plan.citation_budget == 2
    assert plan.citation_style == "paper_scoped"
    assert plan.reason == "paper_memory_prefilter"
    assert plan.fallback_used is False

    payload = plan.to_dict()
    assert payload["paperScoped"] is True
    assert payload["matchedPaperIds"] == ["2602.16662"]
    assert payload["evidenceBudget"] == 2
    assert payload["citationBudget"] == 2


def test_apply_paper_answer_scope_keeps_matching_paper_evidence_only():
    plan = build_paper_answer_plan(
        "What is the contribution of this paper?",
        source_type="paper",
        paper_memory_prefilter={"applied": True, "matchedPaperIds": ["2602.16662"]},
        top_k=5,
    )
    evidence = [
        _evidence_item(
            title="Matching paper",
            source_type="paper",
            paper_id="2602.16662",
            excerpt="matching excerpt",
            score=0.9,
        ),
        _evidence_item(
            title="Duplicate matching paper",
            source_type="paper",
            paper_id="2602.16662",
            excerpt="duplicate excerpt",
            score=0.8,
        ),
        _evidence_item(
            title="Other paper",
            source_type="paper",
            paper_id="2602.00001",
            excerpt="other excerpt",
            score=0.7,
        ),
        _evidence_item(
            title="Vault note",
            source_type="vault",
            excerpt="vault excerpt",
            score=0.6,
        ),
    ]

    scoped, diagnostics = apply_paper_answer_scope(evidence, plan)

    assert len(scoped) == 1
    assert scoped[0]["paper_id"] == "2602.16662"
    assert diagnostics["applied"] is True
    assert diagnostics["fallbackUsed"] is False
    assert diagnostics["paperScoped"] is True
    assert diagnostics["keptEvidenceCount"] == 1
    assert diagnostics["droppedEvidenceCount"] == 3


def test_apply_paper_answer_scope_falls_back_when_no_matching_paper_evidence_exists():
    plan = build_paper_answer_plan(
        "What are the limitations of this paper?",
        source_type="paper",
        paper_memory_prefilter={"applied": True, "matchedPaperIds": ["2602.16662"]},
        top_k=3,
    )
    evidence = [
        _evidence_item(title="Vault note", source_type="vault", excerpt="vault excerpt", score=0.9),
        _evidence_item(title="Unmatched paper", source_type="paper", paper_id="2602.00001", excerpt="paper excerpt", score=0.8),
    ]

    scoped, diagnostics = apply_paper_answer_scope(evidence, plan)

    assert scoped == evidence[: plan.evidence_budget]
    assert diagnostics["applied"] is False
    assert diagnostics["fallbackUsed"] is True
    assert diagnostics["reason"] == "no_matching_paper_evidence"
    assert diagnostics["paperScoped"] is True


def test_build_paper_citation_assembly_dedupes_citations_and_renders_text():
    plan = build_paper_answer_plan(
        "What is the paper contribution?",
        source_type="paper",
        paper_memory_prefilter={"applied": True, "matchedPaperIds": ["2602.16662"]},
        top_k=4,
    )
    evidence = [
        _evidence_item(
            title="Paper A",
            source_type="paper",
            paper_id="2602.16662",
            excerpt="first excerpt",
            score=0.9,
        ),
        _evidence_item(
            title="Paper A duplicate",
            source_type="paper",
            paper_id="2602.16662",
            excerpt="duplicate excerpt",
            score=0.8,
        ),
        _evidence_item(
            title="Paper B",
            source_type="paper",
            paper_id="2602.00002",
            excerpt="second excerpt",
            score=0.7,
        ),
    ]

    citation_bundle = build_paper_citation_assembly(evidence, plan, max_citations=2)

    assert citation_bundle["citationStyle"] == "paper_scoped"
    assert citation_bundle["citationBudget"] == 2
    assert citation_bundle["usedEvidenceCount"] == 2
    assert len(citation_bundle["citations"]) == 2
    assert citation_bundle["citations"][0]["citationId"] == 1
    assert citation_bundle["citations"][0]["paperId"] == "2602.16662"
    assert citation_bundle["citations"][1]["paperId"] == "2602.00002"
    assert "Paper A" in citation_bundle["rendered"]
    assert "2602.16662" in citation_bundle["rendered"]


def test_build_paper_answer_quality_bundle_combines_plan_scope_and_citations():
    evidence = [
        _evidence_item(
            title="Paper A",
            source_type="paper",
            paper_id="2602.16662",
            excerpt="paper excerpt",
            score=0.95,
        ),
        _evidence_item(
            title="Vault note",
            source_type="vault",
            excerpt="vault excerpt",
            score=0.65,
        ),
    ]

    bundle = build_paper_answer_quality_bundle(
        "What is the contribution of this paper?",
        evidence,
        source_type="paper",
        paper_memory_prefilter={"applied": True, "matchedPaperIds": ["2602.16662"]},
        top_k=3,
    )

    assert set(bundle) == {"plan", "scopeDiagnostics", "scopedEvidence", "citationAssembly"}
    assert bundle["plan"]["paperScoped"] is True
    assert bundle["scopeDiagnostics"]["applied"] is True
    assert len(bundle["scopedEvidence"]) == 1
    assert bundle["citationAssembly"]["citations"][0]["paperId"] == "2602.16662"
