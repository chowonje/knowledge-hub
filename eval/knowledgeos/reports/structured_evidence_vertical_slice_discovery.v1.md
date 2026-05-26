# Structured Evidence Vertical Slice Discovery

Generated: `2026-05-21T05:28:06.949329+00:00`

Discovery-only report for Structured Evidence vertical slice planning on the verified 100-row corpus manifest. Does not mutate manifests, create runtime evidence, integrate answers, scan vault content, or download external sources.

## Corpus / Parsed Coverage

- Corpus manifest rows: **100**
- Corpus source available rows: **100**
- Parsed manifest rows: **100**
- Parsed document.json rows: **100**
- SourceSpan store rows: **3**
- StrictEvidence store rows: **3**

## Recommended First Slice (5 papers)

- `1706.03762` (paper_1706_03762) — eval-critical manifest paper; existing source_span pilot records for readback; figure_artifacts present for caption slice; section-like paragraph coverage
- `2005.14165` (paper_2005_14165) — eval-critical manifest paper; existing source_span pilot records for readback; section-like paragraph coverage
- `1506.02640` (paper_1506_02640) — existing source_span pilot records for readback; figure_artifacts present for caption slice; section-like paragraph coverage
- `1512.03385` (paper_1512_03385) — eval-critical manifest paper; section-like paragraph coverage
- `2005.11401` (paper_2005_11401) — eval-critical manifest paper; greenfield RAG paper with verified source and parsed artifact; section-like paragraph coverage

## Minimal Evidence Types

- `section_text_offset`
- `figure_caption_text`

## Deferred Evidence Types

- `table_cell_numeric`
- `equation_citation`
- `appendix_table_lookup`
