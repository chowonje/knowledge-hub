# Source Alias Normalization

- status: `ready`
- caseRows: `10`
- blockedShortAliasRows: `1`
- conceptOnlyRows: `2`
- discoverOnlyRows: `2`
- contextualAliasResolvedRows: `3`
- explicitIdResolvedRows: `1`
- explicitTitleContextRows: `1`
- unsafeDirectAliasRows: `0`
- privatePathLeakRows: `0`

## Cases

| caseId | alias | family | disposition | policy | blocker |
|---|---:|---|---|---|---|
| `rag-concept-only` | `RAG` | `paper_discover` | `discover_only` | `shortlist_only` | `short_alias_discovery_not_single_source_scope` |
| `rag-discover-only` | `RAG` | `paper_discover` | `discover_only` | `shortlist_only` | `short_alias_discovery_not_single_source_scope` |
| `rag-self-rag-contextual-compare` | `RAG` | `paper_compare` | `contextual_alias_resolved` | `contextual_resolution_only` | `` |
| `gpt-bert-contextual-compare` | `GPT` | `paper_compare` | `contextual_alias_resolved` | `contextual_resolution_only` | `` |
| `cnn-concept-representative-only` | `CNN` | `concept_explainer` | `concept_only` | `representative_candidates_only` | `short_alias_requires_explicit_source_context` |
| `cnn-vit-contextual-compare` | `CNN` | `paper_compare` | `contextual_alias_resolved` | `contextual_resolution_only` | `` |
| `vlm-concept-only` | `VLM` | `concept_explainer` | `concept_only` | `representative_candidates_only` | `short_alias_requires_explicit_source_context` |
| `rag-explicit-title-context` | `RAG` | `paper_lookup` | `explicit_title_context` | `explicit_source_resolution_allowed` | `` |
| `rag-explicit-id-resolved` | `RAG` | `paper_lookup` | `explicit_id_resolved` | `explicit_source_resolution_allowed` | `` |
| `gpt-bare-lookup-blocked` | `GPT` | `paper_lookup` | `blocked_short_alias_no_context` | `no_direct_source_resolution` | `short_alias_lookup_requires_explicit_id_or_title` |
