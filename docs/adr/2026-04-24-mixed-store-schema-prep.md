# ADR: Mixed Store Table-Level Authority Schema Prep

Date: 2026-04-24

## Status

Accepted for the `mixed-store-pr-a-20260424` tranche.

## Context

The store-authority inventory established the high-level distinction between canonical source/audit stores, derivative materialized views, mixed semantic stores, and operational queues. That inventory is still too coarse for the next lifecycle tranche:

- `ontology_store.py` contains multiple tables with different authority semantics;
- `epistemic_store.py` mixes derivative belief inputs with canonical decision/outcome records;
- `learning_graph_store.py` contains both derivative projections and operational/event tables;
- `memory_relation_store.py` has no row-level marker separating auto-derived edges from future manual links.

If lifecycle rules are added before the persistence layer can express `derived` vs `manual` vs `pending`, the next tranche will either over-invalidate manual rows or under-invalidate derived rows.

## Decision

Mixed-store authority is now fixed at the table level, with row-level origin markers where manual and derived rows may eventually coexist.

This tranche is intentionally mechanical:

- it adds schema vocabulary only;
- it does not change retrieval, ranking, answer generation, or stale filtering behavior;
- it leaves lifecycle expansion to a follow-up tranche once row-level authority markers are available.

The current authority split is:

| Table | Authority class | Schema prep |
| --- | --- | --- |
| `concepts` | derivative projection | `contributor_hashes` |
| `ontology_entities` | derivative projection for concept/entity materialization | `contributor_hashes` |
| `ontology_claims` | derivative projection | `origin` |
| `ontology_relations` | derivative projection | `origin` |
| `kg_relations` | derivative legacy projection | `origin` |
| `ontology_pending` | operational queue | no lifecycle columns |
| `beliefs` | mixed, derivative statement plus canonical review stance | `supersedes`, `superseded_by` |
| `decisions` | canonical decision history | `supersedes`, `superseded_by` |
| `outcomes` | canonical append-style observation log | no lifecycle columns |
| `learning_graph_edges` | derivative projection | `origin` |
| `learning_graph_resource_links` | derivative projection | `origin` |
| `learning_graph_pending` | operational queue | no lifecycle columns |
| `learning_graph_events` | canonical append-only log | no lifecycle columns |
| `memory_relations` | mixed future surface | `origin` |
| `entity_merge_proposals`, `entity_split_proposals` | operational workflow queue | no lifecycle columns |

## Column Contract

- `origin TEXT NOT NULL DEFAULT 'derived' CHECK(origin IN ('derived','manual','pending'))`
  - marks rows whose lifecycle may differ in later tranches;
  - defaults old and newly inserted rows to `derived` until explicit manual flows exist.
- `contributor_hashes TEXT NOT NULL DEFAULT '[]'`
  - reserves the contributor set needed for AND-gated invalidation on aggregated concept/entity projections.
- `supersedes TEXT NOT NULL DEFAULT ''`
- `superseded_by TEXT NOT NULL DEFAULT ''`
  - reserve the epistemic supersede chain without introducing a second generic stale language alongside belief `status`.

## Consequences

- Old databases can be upgraded in-place by rerunning store `ensure_schema()` methods.
- Existing rows backfill to safe defaults:
  - `origin='derived'`
  - `contributor_hashes='[]'`
  - `supersedes=''`
  - `superseded_by=''`
- Future lifecycle work can now distinguish:
  - generic derivative rows from future manual rows;
  - aggregated concept/entity projections from single-source derivatives;
  - epistemic supersession from generic derivative staleness.

## Non-Goals

- No retrieval stale filtering changes in this tranche.
- No new evidence/citation rules in this tranche.
- No automatic epistemic supersede behavior yet.
- No lifecycle rollout for operational queues or append-only event tables.

## Follow-Ups

1. Expand lifecycle only to tables that are still derivative after this split, starting with mixed-store retrieval signals (`memory_relations`, ontology relations, learning-graph edges/resource links).
2. Add retrieval-side stale filters only after the lifecycle rules are explicit per table or per origin.
3. Add answer-contract negative tests so ontology, epistemic, learning-graph, and memory-relation rows cannot become citation targets.
