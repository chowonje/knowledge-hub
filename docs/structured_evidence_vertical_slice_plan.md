# Structured Evidence Vertical Slice Plan

Date: 2026-05-21

## Objective

Start a **minimal Structured Evidence vertical slice** on top of the verified
100-row corpus manifest without expanding the manifest, integrating runtime
answers, or running the full complex QA eval.

This plan is discovery-first. Implementation is limited to report/readback
helpers and one small promotion/readback tranche.

## Preconditions (verified)

- `corpus_manifest.json`: `manifestRows=100`
- `corpus-manifest-validate`: `sourceAvailableRows=100`, `hashMismatchRows=0`
- Parsed derivative check in validator: `parsedMissingRows=0` (presence of
  `papers_dir/parsed/{source_id}/manifest.json`, not structured-evidence completeness)

## Current Structure

### 1. Parsed artifact layer

| Item | Contract |
| --- | --- |
| Root | `papers_dir/parsed/{source_id}/` |
| Files | `manifest.json`, `document.json`, `document.md`, optional `figures/` |
| Parser today | Primarily `pymupdf` via `knowledge_hub/papers/pymupdf_adapter.py` |
| `document.json` keys | `markdown_text`, `elements[]`, `parser_meta`, `figure_artifacts[]` |
| Element shape | Mostly `type=paragraph` with `page`, `heading_path`, `reading_order` |
| Tables / equations | **Not natively structured** in current PyMuPDF output |

Important gap: corpus manifest stores `expectedSourceContentHash`, but many
existing parsed manifests do not yet surface `sourceContentHash` at the parsed
manifest root. The vertical slice must treat **corpus manifest hash** as the
source-artifact authority and require parsed/structured records to preserve the
hash body in `sourceContentHash`.

### 2. Structured evidence stores

Local JSONL stores under `papers_dir/structured_evidence/`:

| Store | Path template | Purpose |
| --- | --- | --- |
| `source_span` | `{source_id}.jsonl` | Candidate promoted spans (`section`, `table`, `figure`) |
| `strict_evidence` | `{source_id}.jsonl` | Authority-bearing structured evidence records |
| `strict_evidence_eligibility` | `{source_id}.jsonl` | Eligibility gate records |
| `strict_evidence_citation_grade` | `{source_id}.jsonl` | Citation-grade gate (later) |
| `strict_evidence_runtime_binding` | `{source_id}.jsonl` | Runtime binding gate (later) |

Existing pilot coverage on the 100-row corpus: **3 papers only**

- `1706.03762`
- `1506.02640`
- `2005.14165`

### 3. Evidence packet / citation / trace schemas

| Layer | Schema / fixture | Notes |
| --- | --- | --- |
| Runtime evidence packet | `knowledge-hub.evidence-packet.v1` | Fixture: `docs/schemas/fixtures/evidence-packet.v1.fixture.json` |
| SourceSpan record | `knowledge-hub.paper.parsed-artifact-source-span-record.v1` | Contract in `parsed_artifact_source_span_store_contract.py` |
| StrictEvidence record | `knowledge-hub.paper.parsed-artifact-strict-evidence-record.v1` | Promoted from SourceSpan with `authority.text_offset` |
| Complex QA seed categories | table / equation / figure / method / limitation / appendix | All currently `blocked_until_structured_evidence` baseline |

Citation/trace path today:

```text
corpus_manifest.sourceIds[]
  -> corpus_manifest.artifactId
  -> corpus_manifest.expectedSourceContentHash
  -> local source PDF bytes
  -> parsed/{source_id}/manifest.json
  -> parsed/{source_id}/document.json
  -> structured_evidence/source_span/{source_id}.jsonl
  -> structured_evidence/strict_evidence/{source_id}.jsonl
  -> evidence-packet spans (runtime; out of scope)
```

## Selected Papers (first vertical slice = 5)

| source_id | artifactId | Role in slice | Why |
| --- | --- | --- | --- |
| `1706.03762` | `paper_1706_03762` | Pilot readback | Existing source_span + strict_evidence records; figure artifacts; eval-critical |
| `2005.11401` | `paper_2005_11401` | Greenfield promotion | Eval-critical RAG paper; parsed artifact present; no structured evidence yet |
| `1512.03385` | `paper_1512_03385` | Greenfield section slice | Eval-critical foundational paper; parsed artifact present |
| `1506.02640` | `paper_1506_02640` | Figure caption readback | Existing pilot records; strong `figure_artifacts` coverage |
| `2005.14165` | `paper_2005_14165` | Pilot readback | Existing pilot records; long-form paragraph coverage |

Machine-readable discovery output:

- `eval/knowledgeos/reports/structured_evidence_vertical_slice_discovery.v1.json`
- `eval/knowledgeos/reports/structured_evidence_vertical_slice_discovery.v1.md`

## Proposed Minimal Evidence Types

Start with only these two:

1. **`section_text_offset`**
   - Authority: `text_offset` over parsed paragraph text
   - Store path: `source_span` -> `strict_evidence`
   - Why first: existing pilot executors/design already cover section promotion

2. **`figure_caption_text`**
   - Authority: caption text linked to `figure_artifacts`
   - Store path: `source_span` -> `strict_evidence`
   - Why first: figure caption pilot exists; `1506.02640` and `1706.03762` have figure artifacts

Deferred until parser/layout work matures:

- `table_cell_numeric`
- `equation_citation`
- `appendix_table_lookup`

## Required Contract

Every promoted record in the first slice must carry:

- `paperId` == corpus `sourceIds[0]`
- `sourceContentHash` body matching corpus `expectedSourceContentHash`
- stable `artifactType`
- stable `locator` or `authority.text_offset`
- `runtimeEvidence=false`
- `citationGrade=false`
- no absolute local paths in public artifacts

Trace gates for the next tranche:

1. corpus hash matches local source bytes
2. parsed manifest + document.json exist
3. source_span record exists for target slice
4. strict_evidence record links back to source_span id(s)
5. readback report only; no runtime evidence packet emission

## Next Implementation Tranche (1–2 days)

1. Add report-only trace readback helper for the 5 selected papers.
2. Promote/readback **one greenfield** section slice on `2005.11401`.
3. Read back **one pilot** section slice and **one figure caption** slice on
   `1706.03762` or `1506.02640`.
4. Emit schema-valid JSON report; no answer integration.

## Explicitly Out of Scope

- corpus manifest expansion
- resolving `source_missing` / `ambiguous` join buckets
- runtime answer integration
- complex QA full eval execution
- citation-grade or runtime-binding promotion
- vault access
- external PDF download
- table/equation structured evidence in this tranche

## Existing Tests / Fixtures To Reuse

- `tests/test_complex_qa_seed_pack.py`
- `tests/test_parsed_artifact_source_span_candidate_executor.py`
- `tests/test_strict_evidence_figure_caption_pilot_executor_apply.py`
- `tests/test_complex_qa_structured_evidence_comparison_runner.py`
- `docs/schemas/fixtures/evidence-packet.v1.fixture.json`

## Discovery Helper

Report builder:

```bash
python eval/knowledgeos/scripts/build_structured_evidence_vertical_slice_discovery.py
```

Implementation module:

- `knowledge_hub/papers/structured_evidence_vertical_slice_discovery.py`
