# ADR: Decision Versioning And Evidence-First Gates

Date: 2026-04-24

## Status

Accepted for the `mixed-store-pr-d-20260424` tranche.

## Context

The previous mixed-store hardening work separated evidence from retrieval signals and added lifecycle filtering for several derivative tables, but three gaps remained:

1. `epistemic.beliefs` and `epistemic.decisions` still overwrote review state in-place even after `supersedes` / `superseded_by` columns were added.
2. `ontology_entities` concept rows had no live contributor set, so aggregated concept/entity projections could not use an AND-gated invalidation rule.
3. Answer verification and CI still allowed a structurally misleading success path where rejected-belief conflicts or signal-only grounding could look like ordinary grounded verification.

Without closing those gaps, the runtime would still say "evidence-first" while:

- erasing review history for belief/decision changes,
- over- or under-invalidating concept/entity projections,
- or letting retrieval-only semantic hints masquerade as grounded answer evidence.

## Decision

### 1. Epistemic reviews create successor rows

`belief_review` and `decision_review` now supersede the latest active row instead of overwriting it:

- the prior row remains in place;
- the prior row records `superseded_by=<new_id>`;
- belief rows additionally move to `status='superseded'`;
- the new row records `supersedes=<prior_id>`.

Default list surfaces now hide superseded rows:

- `list_beliefs(..., include_superseded=False)`
- `list_decisions(..., include_superseded=False)`

Explicit `show` by id still returns historical rows for audit/debug.

### 2. Concept/entity projections carry contributor hash sets

Concept-like ontology entities now accumulate `contributor_hashes` from derived ontology inputs:

- `ontology_claims` source hashes update contributor hashes on referenced concept entities;
- `ontology_relations` source hashes update contributor hashes on referenced concept entities;
- legacy `concepts` rows are updated too when the matching concept row exists.

Concept/entity stale invalidation is now AND-gated:

- single-source derivatives (`ontology_claims`, `ontology_relations`, `kg_relations`) still stale under the existing source-change rule;
- aggregated concept/entity rows stale only when the changed source hash intersects their contributor set **and** no live supporting derivative remains for any contributor hash in that set.

This avoids the old failure mode where one changed contributor could suppress an otherwise still-supported concept projection.

### 3. Verification must distinguish evidence from signals

Answer verification now records two structural signals:

- `rejectedBeliefConflictCount` / `contradictsRejectedBelief`
- `retrievalSignalCount` / `groundingEvidenceCount`

Runtime behavior is:

- if a rejected belief is contradicted **without** explicit conflict framing in the answer, verification fails with `reasonCode=contradicts_rejected_belief`;
- if retrieval input consists only of blocked mixed-store signals and no citation-grade evidence, verification fails with `reasonCode=signal_only_grounding`;
- if a rejected belief conflict is explicitly acknowledged, the conflict is preserved as verification metadata and warning context rather than silently erased.

### 4. Evidence-first gates become required CI surface

A dedicated required CI slice now runs:

- `tests/test_evidence_first_golden.py`
- `tests/test_answer_contracts_runtime.py`
- `tests/test_answer_verification_guards.py`
- `tests/test_epistemic_supersede.py`
- `tests/test_mixed_store_lifecycle.py`

The fixture-backed golden file lives at:

- `eval/knowledgeos/fixtures/evidence_first_golden_cases.json`

This keeps the required gate small, deterministic, and directly aligned with the evidence-first contract.

## Consequences

- Belief/decision history is now append-style at review time rather than overwrite-style.
- Callers that assume `review_*` preserves the same id must read the returned item and keep the successor id.
- Stale concept/entity rows disappear from default entity lists once all contributors are invalidated.
- Verification output now exposes stronger structural failure codes for rejected-belief conflicts and signal-only grounding.
- CI will fail earlier when mixed-store signals leak back into citation-grade behavior.

## Non-Goals

- No generic `stale` field is added to epistemic rows beyond the existing status/supersede model.
- No full ontology rebuild worker is introduced in this tranche.
- No broad eval harness promotion is attempted beyond the narrow evidence-first required gate.

## Follow-Ups

1. Extend supersede-aware UX surfaces so CLI/MCP review flows can optionally show both old and successor ids more explicitly.
2. Expand contributor-set invalidation beyond concept rows if later mixed entity types genuinely become aggregated projections.
3. Promote a small span-overlap retrieval golden set once the evidence-first gate remains stable in CI.
