# Context-Preserving Document Memory v1

Last updated: 2026-03-19

## Why this exists

`knowledge-hub` already has:
- section-aware chunking for markdown/html/code/plain text
- stable vault scope ids
- paper-local memory cards

It does **not** yet have a source-agnostic layer that stores a document as human-readable semantic units before retrieval chunks are derived.

This document defines the next architecture target:

`Document -> MemoryUnit -> RetrievalChunk`

The goal is not better generic chunking. The goal is:
- preserve context while segmenting
- store natural-language memory units that a human can read again
- keep retrieval chunks linked to those higher-level units
- support notes, papers, and crawled web documents with one model

## Current internal baseline

### Already implemented

- `knowledge_hub/core/chunking.py`
  - deterministic chunking
  - heading-aware markdown/html splitting
  - section title and section path retention
- `knowledge_hub/vault/parser.py`
  - vault parsing
  - stable document/section scope ids
  - section-aware chunk metadata
- `knowledge_hub/papers/memory_builder.py`
  - compact paper memory cards
  - human-readable summaries for papers only

### Missing

- source-agnostic `MemoryUnit` schema
- typed block/unit storage for vault, paper, and web
- document summary + section summary hierarchy
- retrieval path that narrows by summary/unit first, then drills down to chunks
- human-readable writeback for non-paper sources

## External research targets

These are the next high-value comparison targets for this problem.

### Adopt candidates

- `Unstructured`
  - why: typed element extraction (`Title`, `NarrativeText`, `List`, `Table`) and `by_title` chunking match the need for typed segmentation and section-preserving boundaries
  - likely use: optional parser adapter for paper/pdf/web sources
  - not for: replacing the whole ingestion pipeline
- `LlamaIndex DocumentSummaryIndex`
  - why: summary-first retrieval and document/node hierarchy match the desired `MemoryUnit` layer closely
  - likely use: retrieval pattern and storage shape inspiration
  - not for: introducing a separate LlamaIndex runtime as a core dependency
- `dsRAG AutoContext`
  - why: deterministic context headers for retrieval chunks map directly onto the `MemoryUnit -> RetrievalChunk` boundary
  - likely use: build `search_text` from `raw excerpt + context header` instead of indexing raw text alone
  - not for: introducing a separate dsRAG runtime or replacing the current retrieval stack
- `Structured Hierarchical Retrieval`
  - why: document summary plus structured metadata can act as a first-pass filter before section-level drill-down
  - likely use: future query-time narrowing with summary and metadata dictionaries
  - not for: adopting hosted vector infrastructure just to mirror the example stack
- `IDP`
  - why: header-preserving semantic block IDs and parent-child heading relations fit the `MemoryUnit` hierarchy directly
  - likely use: later internal semantic block ids and section-to-block retrieval stitching
  - not for: taking the full PDF-to-Markdown pipeline as a mandatory dependency in v1

### Hold candidates

- `Docling`
  - why: useful as a structured PDF-to-Markdown or PDF-to-block parser candidate
  - hold reason: parser quality and integration cost should be compared before adoption
- `Marker`
  - why: strong PDF extraction candidate
  - hold reason: similar role to Docling; evaluate as an interchangeable optional parser
- `semantic-chunker`
  - why: useful for semantic overlap and boundary refinement ideas
  - hold reason: secondary to `MemoryUnit` design because current gap is storage/modeling, not only splitting
- `S2 Chunking`
  - why: useful future reference when PDF layout complexity becomes the real bottleneck
  - hold reason: too heavy for v1 compared with internal segmentation plus optional parser adapters
### Evaluation-only

- `READoc`
  - why: parser quality benchmark
  - role: selection benchmark only
  - not for: runtime dependency

## GPT research prompts

### Prompt 1: document memory systems

```text
긴 문서를 단순히 잘게 자르는 것이 아니라, 문맥을 보존한 의미 단위로 나누고 각 단위를 사람이 다시 읽기 쉬운 자연어 메모리 형태로 저장하는 데 강한 프로젝트, 논문, 라이브러리를 GitHub와 Hugging Face 기준으로 찾아줘.

특히 다음 능력이 있는 것들을 우선 찾아줘:
1. typed document segmentation (title, paragraph, list, table, figure 등)
2. section-aware chunking
3. segment-level contextual summary 생성
4. retrieval chunk와 상위 semantic unit 연결 유지
5. local-first Python 파이프라인에 붙이기 쉬운 구조

나는 generic vector chunking보다 context-preserving memory formation에 더 관심이 있다.
각 후보마다:
- 무엇이 좋은지
- 무엇을 복사하면 안 되는지
- Python local-first stack에 붙일 때 비용이 어떤지
- core/supporting/labs 중 어디에 맞는지
를 정리해줘.
```

### Prompt 2: target stack comparison

```text
Unstructured, LlamaIndex DocumentSummaryIndex, Docling, Marker, semantic-chunker, READoc를 중심으로 조사해줘.

내 프로젝트는:
- Obsidian note
- paper ingestion
- crawled web article
- local vector retrieval
- ontology entities/claims/relations
을 가진 Python 기반 local-first knowledge system이다.

각 후보에 대해 다음만 답해줘:
1. 내 아키텍처에 바로 재사용할 수 있는 데이터 구조나 처리 단계는 무엇인가
2. 문서 -> 의미 단위(memory unit) -> retrieval chunk의 3단 구조를 만들 때 어떤 부분이 유용한가
3. human-readable memory persistence를 위해 어떤 메타데이터를 저장해야 하는가
4. 유지보수성 기준에서 채택/보류/기각 중 무엇이 맞는가
```

### Prompt 3: schema and retrieval design

```text
PDF/논문/Markdown/웹문서를 하나의 통일된 semantic unit schema로 저장하려고 한다.
이 목적에 맞는 설계 패턴, 오픈소스 구현, benchmark를 찾아줘.

중점 질문:
- 문서 전체 summary와 section summary를 어떻게 함께 저장하는가
- chunk보다 상위 memory unit을 먼저 저장하는 시스템이 있는가
- summary-first retrieval 또는 hierarchical retrieval 패턴은 무엇이 있는가
- parser 품질은 어떻게 평가하는가
- local-first Python 프로젝트에 붙일 때 가장 현실적인 조합은 무엇인가

결과는 다음 형식으로 정리해줘:
- candidate
- why it matters
- adopt / hold / reject
- concrete extraction pattern worth copying
```

## v1 architecture decision

### Canonical hierarchy

`Document -> MemoryUnit -> RetrievalChunk`

### Document

Canonical source record for one note, paper, or crawled article.

Required identity:
- `document_id`
- `source_type`
- `file_path` or canonical URL
- `title`
- `document_scope_id`
- provenance fields already used by the source pipeline

### MemoryUnit

Human-readable semantic unit derived from a document.

Required fields:
- `unit_id`
- `document_id`
- `unit_type`
- `section_path`
- `title`
- `contextual_summary`
- `source_excerpt`
- `context_header`
- `document_thesis`
- `parent_unit_id`
- `scope_id`
- `confidence`
- `provenance`

Recommended fields:
- `order_index`
- `content_type`
- `links`
- `tags`
- `claims`
- `concepts`

### RetrievalChunk

Indexing/search shard derived from a `MemoryUnit`, not from the raw document alone.

Required linkage:
- `chunk_id`
- `document_id`
- `unit_id`
- `section_path`
- `stable_scope_id`

Retrieval text rule:
- `search_text = raw excerpt + context header + contextual summary`
- keep `source_excerpt` raw and human-readable
- inject hierarchy and thesis only into the search representation

## Segmentation defaults

### Vault / Markdown

- use heading-aware segmentation first
- create `MemoryUnit` per section or subsection
- preserve section path and first heading title
- carry wiki links and tags into unit metadata

### Paper / PDF

- start with existing note/paper section structure if available
- if parser output exists, treat typed blocks as candidate unit seeds
- prioritize sections such as abstract, method, results, limitations
- do not require a heavy hosted parser in v1

### Web

- use heading + paragraph/list/table block boundaries
- preserve canonical URL and fetched provenance
- if no heading exists, build units from bounded narrative blocks

## Retrieval decision

### Default behavior

Summary-first retrieval is additive, not a replacement.

1. narrow candidate documents or units with summary-level representations
2. retrieve chunk-level evidence only from the narrowed units
3. preserve existing chunk retrieval as fallback
4. in `khub labs memory`, re-rank child units with query-aware and hierarchy-aware signals before showing related units

### Why

- reduces semantic drift
- keeps human-readable unit context attached to chunk evidence
- allows better writeback and inspection

## Human-readable persistence

For each `MemoryUnit`, persist:
- the natural-language summary
- the local source excerpt
- the section path
- the parent/child relation
- the provenance/confidence

This is the minimum contract needed to make the system readable by both humans and retrieval code.

## Evaluation loop

Before connecting document memory to the core retrieval runtime, use a small manual evaluation loop.

- query set: `docs/research/document-memory-eval-queries-v1.txt`
- runner: `scripts/eval_document_memory.py`
- operator guide: `docs/research/document-memory-eval.md`

This keeps the decision gate explicit: the labs path should prove interpretability and useful narrowing before it becomes a core retrieval input.

## PDF 5 follow-up judgment

The fifth review mostly confirms the current direction rather than changing it.

- keep now:
  - `Unstructured + DocumentSummaryIndex` as the main external mental model
  - `dsRAG AutoContext` for header augmentation
  - `Structured Hierarchical Retrieval` as the retrieval pattern to keep copying
- add next:
  - bounded downrank for metadata-heavy placeholder summaries
  - title-aware boosting for paper-title or entity-like queries so specific document names outrank generic topical notes
  - `RSE-lite` adjacent unit stitching on the labs path so the best unit can return a slightly longer semantic segment instead of a single isolated block
  - later true section-to-block stitching similar to dsRAG RSE, but only after current summary-first retrieval is stable
  - parser quality evaluation with `READoc` before adopting heavier PDF parsers
- still hold:
  - `S2 Chunking`, `Docling`, full `GraphRAG`

## Labs-only v1.1 refinement

The current executable surface is intentionally limited to `khub labs memory ...`.

- `khub search` and `khub ask` stay unchanged for now
- parser strategy remains internal-only in v1.1
- `context_header` is deterministic and derived from current metadata plus summaries
- `relatedUnits` should be query-aware, not just recent or structural listing
- core integration stays deferred until the labs path proves retrieval value

## Implementation sequence

1. Define `MemoryUnit` schema and persistence contract.
2. Add shared segmentation adapters for vault, paper, and web.
3. Add contextual summary generation per unit.
4. Add summary-first retrieval as an additive path.
5. Add human-readable writeback for units where the source workflow already supports writeback.

## Non-goals for v1

- full GraphRAG adoption
- replacing current chunk retrieval entirely
- hosted-only parser requirements
- new UI surfaces before the backend memory model is stable
- turning notebook workbench into a system of record

## Acceptance criteria

- the same document can be represented as a small set of meaningful `MemoryUnit`s
- each `MemoryUnit` has a human-readable `contextual_summary`
- retrieval can narrow by unit summary, then drill down to chunk evidence
- vault note, paper, and crawled web article can all map into the same `MemoryUnit` contract
- existing `search` and `ask` paths remain additive and inspectable
