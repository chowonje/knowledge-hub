# ADR: Store Authority Inventory

Date: 2026-04-24

## Status

Accepted for the Opus hardening split branch.

## Context

The evidence-first contract depends on a stable distinction between canonical material and rebuildable semantic projections. The persistence layer has many stores with similar "store" names, but they do not all have the same authority:

- canonical source and audit records can support evidence provenance;
- derivative rows can improve retrieval, ranking, and UI review, but must not become the end of a citation chain;
- operational queues and run logs are canonical for workflow audit, not for factual answer evidence;
- mixed stores need row-level authority rules before they can be promoted into stricter lifecycle enforcement.

## Decision

`EvidencePacket` spans may be built only from source-backed material with real source identity, source content hash, and span offsets. Semantic stores may be used as retrieval hints, filters, or UI projections unless their rows carry strict provenance back to source-backed spans.

The current store authority inventory is:

| Store module | Primary tables | Authority class | Evidence role | Lifecycle status |
| --- | --- | --- | --- | --- |
| `paper_store.py` | `papers` | canonical source metadata/content pointer | source authority | upstream invalidation hook present |
| `note_store.py` | `notes`, `tags`, `links` | canonical local source projection | source authority for vault-backed rows | upstream invalidation hook present |
| `crawl_pipeline_store.py` | crawl policies/jobs/records/checkpoints/metrics | operational canonical | not answer evidence | not lifecycle-managed |
| `event_store.py` | `ontology_events` | canonical audit/event log | audit only | append/audit semantics |
| `rag_answer_log_store.py` | `rag_answer_logs` | canonical answer audit | audit only; answer text is not evidence | not source lifecycle |
| `mcp_job_store.py` | `mcp_jobs` | operational canonical | not answer evidence | not lifecycle-managed |
| `ops_action_queue_store.py` | `ops_action_queue` | operational canonical | not answer evidence | not lifecycle-managed |
| `ops_action_receipt_store.py` | `ops_action_receipts` | operational canonical | not answer evidence | not lifecycle-managed |
| `sync_conflict_store.py` | `foundry_sync_pending_conflicts` | operational canonical | not answer evidence | not lifecycle-managed |
| `quality_mode_store.py` | `quality_mode_usage` | operational canonical | not answer evidence | not lifecycle-managed |
| `ko_note_store.py` | ko-note run/item/enrichment tables | operational canonical for pipeline runs | not answer evidence | not source lifecycle |
| `learning_store.py` | learning sessions/progress/events | canonical learner state | not answer evidence by itself | not source lifecycle |
| `document_memory_store.py` | `document_memory_units` | derivative materialized view | retrieval hint only unless strict source span is available | lifecycle v1 present |
| `paper_memory_store.py` | `paper_memory_cards` | derivative materialized view | retrieval hint/card UI | lifecycle v1 present |
| `paper_card_v2_store.py` | `paper_cards_v2`, anchors/refs | derivative materialized view | retrieval hint/card UI | lifecycle v1 present |
| `section_card_v1_store.py` | `paper_section_cards_v1` | derivative materialized view | retrieval hint/card UI | lifecycle v1 present |
| `claim_card_v1_store.py` | `claim_cards_v1`, refs/aliases | derivative materialized view | retrieval hint/card UI | lifecycle v1 present |
| `vault_card_v2_store.py` | `vault_cards_v2`, anchors/refs | derivative materialized view | retrieval hint/card UI | lifecycle v1 present |
| `web_card_v2_store.py` | `web_cards_v2`, anchors/refs | derivative materialized view | retrieval hint/card UI | lifecycle v1 present |
| `claim_store.py` | `claim_normalizations` | derivative projection | normalization signal only | follow-up lifecycle decision required |
| `memory_relation_store.py` | `memory_relations` | derivative projection | relation/ranking signal only | follow-up lifecycle decision required |
| `entity_resolution_store.py` | merge/split proposals | derivative proposal store | resolution workflow only | follow-up lifecycle decision required |
| `learning_graph_store.py` | graph nodes/edges/paths/resource links/pending/events | derivative graph projection plus graph event log | learning/navigation signal only | follow-up row-level split required |
| `ontology_store.py` | concepts/entities/claims/predicates/relations/pending/kg relations | mixed: curated canonical nodes plus derivative claims/relations/projections | optional lens only; not citation endpoint | follow-up row-level authority split required |
| `ontology_profile_store.py` | profile state/runtime/proposals/overlays | mixed governance/config plus derivative proposals | ontology routing/config only | follow-up row-level authority split required |
| `epistemic_store.py` | beliefs/decisions/outcomes | mixed: derivative beliefs plus canonical decisions/outcomes | epistemic signal only; not source evidence | follow-up row-level authority split required |

## Invariants

- Source-backed tables and append-only audit/event records can be canonical for their own domain, but answer citations still need source ids, source content hashes, and span offsets.
- Derivative cards, memory units, claims, graph edges, ontology projections, and epistemic beliefs are not citation endpoints.
- Stale derivative rows must not be emitted as evidence. They may be exposed through direct inspection or rebuild workflows.
- If a derivative row is used for answer generation, it must resolve back to strict source-backed spans before it reaches `EvidencePacket`.
- Mixed stores require row-level authority before they can be included in lifecycle introspection tests.

## Follow-Ups

1. Add a small machine-readable derivative inventory after the row-level mixed-store split is decided.
2. Add SQL introspection tests requiring derivative tables in that inventory to carry `source_content_hash`, `stale`, `stale_reason`, and `invalidated_at`.
3. Decide whether `claim_normalizations`, `memory_relations`, entity-resolution proposals, learning-graph projections, ontology claims/relations, and epistemic beliefs need lifecycle columns or should stay non-evidence projections with explicit retrieval-only contracts.
4. Record a card v1/v2 retirement plan separately; this ADR only defines authority, not schema deprecation timing.
