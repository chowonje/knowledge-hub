"""N1: paper document-summary anchors must surface the raw source span (and its
provenance offsets) instead of the LLM contextual summary.

Regression guard for `AskV2Service._anchors_for_paper_document_summaries`:
1. when a unit carries a real `source_excerpt`, the anchor excerpt is that raw
   span (not the paraphrased `contextual_summary`) and is marked `raw_span`;
2. when no raw span exists, it falls back to the summary (anchor not dropped);
3. every emitted anchor is routed through the shared provenance enricher so it
   can carry a resolved span locator / char offsets / source_content_hash.
"""

from knowledge_hub.ai.ask_v2 import AskV2Service


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def list_document_memory_units(self, document_id, limit=24):
        return list(self._rows)


def _service(rows):
    svc = object.__new__(AskV2Service)  # bypass __init__: method only needs sqlite_db + enricher
    svc.sqlite_db = _FakeDB(rows)
    return svc


_CARD = {"paper_id": "1706.03762", "card_id": "card-1", "title": "Attention Is All You Need"}


def test_anchor_prefers_raw_source_excerpt_over_contextual_summary():
    rows = [
        {
            "unit_id": "u1",
            "unit_type": "document_summary",
            "source_excerpt": "We propose the Transformer, a model architecture eschewing recurrence.",
            "contextual_summary": "A paraphrased gloss invented by the summarizer.",
            "document_id": "paper:1706.03762",
        }
    ]
    svc = _service(rows)
    svc._enrich_anchor_provenance = lambda anchor, **_: anchor  # isolate assembly logic

    anchors = svc._anchors_for_paper_document_summaries(cards=[_CARD])

    assert len(anchors) == 1
    anchor = anchors[0]
    assert anchor["excerpt"] == "We propose the Transformer, a model architecture eschewing recurrence."
    assert anchor["evidence_kind"] == "raw_span"
    # the paraphrase must NOT be what gets cited
    assert "paraphrased gloss" not in anchor["excerpt"]


def test_anchor_falls_back_to_summary_when_no_raw_span():
    rows = [
        {
            "unit_id": "u2",
            "unit_type": "document_summary",
            "source_excerpt": "",
            "contextual_summary": "Only a contextual summary is available here.",
            "document_id": "paper:1706.03762",
        }
    ]
    svc = _service(rows)
    svc._enrich_anchor_provenance = lambda anchor, **_: anchor

    anchors = svc._anchors_for_paper_document_summaries(cards=[_CARD])

    assert len(anchors) == 1  # anchor is not dropped
    assert anchors[0]["excerpt"] == "Only a contextual summary is available here."
    assert anchors[0]["evidence_kind"] == "summary"


def test_anchor_is_routed_through_provenance_enricher():
    rows = [
        {
            "unit_id": "u3",
            "unit_type": "summary",
            "source_excerpt": "Self-attention relates positions of a single sequence.",
            "contextual_summary": "paraphrase",
            "document_id": "paper:1706.03762",
        }
    ]
    svc = _service(rows)

    calls = {"n": 0}

    def fake_enrich(anchor, *, card=None, fallback_source_hash=""):
        calls["n"] += 1
        enriched = dict(anchor)
        # the real enricher surfaces these from unit/provenance; assert we delegate to it
        enriched["source_content_hash"] = "HASH-FROM-ENRICHER"
        enriched["span_locator"] = "chars:0-52"
        return enriched

    svc._enrich_anchor_provenance = fake_enrich

    anchors = svc._anchors_for_paper_document_summaries(cards=[_CARD])

    assert calls["n"] == 1, "doc-summary anchors must be routed through _enrich_anchor_provenance"
    assert anchors[0]["source_content_hash"] == "HASH-FROM-ENRICHER"
    assert anchors[0]["span_locator"] == "chars:0-52"
    assert anchors[0]["excerpt"] == "Self-attention relates positions of a single sequence."
    assert anchors[0]["evidence_kind"] == "raw_span"
