from __future__ import annotations

from knowledge_hub.interfaces.cli.ask_text_output import (
    render_paper_lookup_text,
    should_render_paper_lookup_text,
)


def test_paper_lookup_text_shows_identity_source_backed_excerpt_and_labels() -> None:
    # Given: a single-paper lookup has raw paper evidence and a verifier caution.
    payload = {
        "status": "ok",
        "answer": "The Transformer replaces recurrent sequence processing with attention.",
        "queryFrame": {"family": "paper_lookup"},
        "evidencePacket": {"answerable": True, "paperFamily": "paper_lookup"},
        "answerVerification": {
            "status": "caution",
            "supportedClaimCount": 0,
            "unsupportedClaimCount": 2,
            "uncertainClaimCount": 1,
            "needsCaution": True,
            "route": {"mode": "heuristic", "provider": "ollama", "model": "gemma4:e4b"},
        },
        "citations": [{"label": "S1", "title": "Attention Is All You Need", "target": "1706.03762"}],
        "sources": [
            {
                "title": "Attention Is All You Need",
                "source_type": "paper",
                "arxiv_id": "1706.03762",
                "retrieval_mode": "active-vector-paper",
                "evidence_kind": "raw_span",
                "citation_label": "S1",
                "excerpt": "The Transformer uses only attention mechanisms and removes recurrence and convolutions.",
            }
        ],
    }

    # When: the normal text renderer formats the paper lookup payload.
    rendered = render_paper_lookup_text("Explain the core idea.", payload)

    # Then: the first screen contains paper identity, inline labels, state, and raw evidence.
    assert should_render_paper_lookup_text("paper", payload) is True
    assert "The Transformer replaces recurrent sequence processing with attention. [S1]" in rendered
    assert "Paper: Attention Is All You Need (arXiv:1706.03762)" in rendered
    assert "Evidence state: verifier weakness" in rendered
    assert "Verdict: THIN - 0/3 claims verified, 2 unsupported, coverage unknown" in rendered
    assert "Verifier: lexical heuristic; Korean/English mismatch can make counts unreliable." in rendered
    assert "[S1] raw - The Transformer uses only attention mechanisms" in rendered
    assert "Citations: [S1]" in rendered
    assert "Chroma" not in rendered
    assert "vector" not in rendered.lower()
    assert "provider" not in rendered.lower()
    assert "model" not in rendered.lower()


def test_paper_lookup_text_labels_card_only_answers_as_insufficient() -> None:
    # Given: a paper lookup only selected paper-card-v2 hints.
    payload = {
        "status": "ok",
        "answer": "This answer came from a paper memory card.",
        "queryFrame": {"family": "paper_lookup"},
        "evidencePacket": {"answerable": True, "paperFamily": "paper_lookup"},
        "citations": [{"label": "S1", "title": "Attention Is All You Need", "target": "1706.03762"}],
        "sources": [
            {
                "title": "Attention Is All You Need",
                "source_type": "paper",
                "arxiv_id": "1706.03762",
                "retrieval_mode": "paper-card-v2",
                "evidence_kind": "memory_hint",
                "citation_label": "S1",
                "excerpt": "Card-derived introduction hint.",
            }
        ],
    }

    # When: the normal text renderer formats the payload.
    rendered = render_paper_lookup_text("Explain the paper.", payload)

    # Then: the user sees that the answer is not source-backed.
    assert "Evidence state: insufficient" in rendered
    assert "paper-card-v2 only" in rendered
    assert "Source-backed excerpts:" not in rendered
    assert "This answer came from a paper memory card. [S1]" in rendered


def test_paper_lookup_text_marks_partial_coverage_as_thin_verdict() -> None:
    # Given: a verified-looking paper answer has only partial claim coverage.
    payload = {
        "status": "ok",
        "answer": "The Transformer removes recurrence by using attention [S1].",
        "queryFrame": {"family": "paper_lookup"},
        "evidencePacket": {
            "answerable": True,
            "paperFamily": "paper_lookup",
            "coverage": {"status": "partial"},
        },
        "answerVerification": {
            "status": "verified",
            "supportedClaimCount": 1,
            "unsupportedClaimCount": 0,
            "uncertainClaimCount": 1,
            "needsCaution": False,
            "route": {"mode": "llm"},
        },
        "sources": [
            {
                "title": "Attention Is All You Need",
                "source_type": "paper",
                "arxiv_id": "1706.03762",
                "retrieval_mode": "active-vector-paper",
                "evidence_kind": "raw_span",
                "citation_label": "S1",
                "excerpt": "The Transformer avoids recurrence and instead relies on attention.",
            }
        ],
    }

    # When: the normal text renderer formats the payload.
    rendered = render_paper_lookup_text("What evidence supports removing recurrence?", payload)

    # Then: partial coverage is visible as a thin verdict without exposing internals.
    assert "Verdict: THIN - 1/2 claims verified, 0 unsupported, coverage partial" in rendered
    assert "Verifier: lexical heuristic" not in rendered
    assert "provider" not in rendered.lower()


def test_paper_lookup_text_marks_unanswerable_payload_as_insufficient() -> None:
    # Given: the ask path failed closed for a paper lookup.
    payload = {
        "status": "no_result",
        "answer": "근거가 부족해 답변할 수 없습니다.",
        "queryFrame": {"family": "paper_lookup"},
        "evidencePacket": {
            "answerable": False,
            "paperFamily": "paper_lookup",
            "insufficientEvidenceReasons": ["no_source_backed_evidence"],
        },
        "sources": [],
    }

    # When: the normal text renderer formats the payload.
    rendered = render_paper_lookup_text("Explain the paper.", payload)

    # Then: the user sees an explicit insufficient evidence state.
    assert "Evidence state: insufficient" in rendered
    assert "source-backed evidence is insufficient" in rendered


def test_paper_lookup_text_state_passes_when_source_backed_and_verifier_passed() -> None:
    # Given: a source-backed paper lookup has no verifier caution.
    payload = {
        "status": "ok",
        "answer": "The answer already cites the raw source [S1].",
        "queryFrame": {"family": "paper_lookup"},
        "evidencePacket": {"answerable": True, "paperFamily": "paper_lookup"},
        "answerVerification": {"status": "verified", "needsCaution": False},
        "sources": [
            {
                "title": "Attention Is All You Need",
                "source_type": "paper",
                "arxiv_id": "1706.03762",
                "retrieval_mode": "active-vector-paper",
                "evidence_kind": "raw_span",
                "citation_label": "S1",
                "excerpt": "The architecture relies entirely on attention.",
            }
        ],
    }

    # When: the normal text renderer formats the payload.
    rendered = render_paper_lookup_text("Explain the paper.", payload)

    # Then: the evidence state is source-backed and the existing inline label is preserved.
    assert "The answer already cites the raw source [S1]. [S1]" not in rendered
    assert "The answer already cites the raw source [S1]." in rendered
    assert "Evidence state: source-backed" in rendered
