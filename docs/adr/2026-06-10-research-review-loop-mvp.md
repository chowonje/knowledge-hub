# ADR: Research Review Loop MVP

Date: 2026-06-10

## Status

Accepted as the labs-first MVP direction.

## Context

The current default product promise remains retrieval-assistant-first:
`discover -> index -> search/ask -> evidence review`. That path is useful, but
answer fluency and answer-quality harnesses alone do not prove the user's
durable research memory is improving. A generated answer can cite sources and
still fail the more important product question: did the user review a claim,
learn what is weak, and produce a reusable judgment context?

The repo already has claim-card, claim-normalization, strict-report,
context-pack, review-card, and learning-review surfaces. The next product
tranche should connect these into a bounded review loop rather than creating a
parallel agent memory system or promoting a broader `khub ask` default.

## Decision

Define the next KnowledgeOS MVP as a labs-first Research Review Loop:

```text
source -> proposed claims -> evidence spans -> user review decisions
       -> weak concepts / open questions -> learning or judgment context pack
```

The governing rule is:

```text
Agent output is not memory. User-reviewed artifact is memory.
```

### Contract

- `proposed` artifacts are inspectable candidates, not canonical memory.
- A claim, evidence span, concept, question, or pack becomes canonical context
  only after an explicit user or operator review decision.
- Accepted and unsure decisions must preserve source ids, evidence span ids,
  source hashes or snippet hashes where available, reviewer id, review time,
  confidence, and reason.
- Rejected decisions stay inspectable so later agents can avoid reintroducing
  the same weak claim.
- External agents such as Hermes can produce report-only recommendations, but
  they must not write canonical memory. Only a reviewed local import or local
  decision record can cross the boundary.

### Reuse before new surface

The implementation should reuse the existing claim and review surfaces first:

- `khub labs claims extract-paper`
- `khub labs claims pending list`
- `khub labs claims strict-report`
- `khub labs claims normalize`
- `khub labs claims compare`
- `khub labs claims synthesize`
- `khub labs claim-cards metrics`
- existing paper review-card and context-pack records

A new `khub labs review-loop ...` surface is allowed only as a thin
composition layer over these existing stores and report contracts.

## Non-Goals

- No default `khub ask` promotion.
- No Chroma/vector repair as a substitute for user-visible review value.
- No graph, ontology, MCP, Foundry, Agent Gateway, or qwen default expansion in
  this tranche.
- No vault scan or vault write by default.
- No external-agent canonical memory writes.
- No claim-card or evidence promotion without source-backed spans and review
  decisions.

## Acceptance Criteria

The first human-test slice is one explicit paper or a small explicit paper set.
It is useful only if it can show all of the following:

- 3-7 proposed claims with direct source or span evidence.
- Clear blocked rows when evidence is missing or only weakly linked.
- At least three user review decisions across accept, reject, or unsure.
- A weak-concepts list grounded in rejected or unsure decisions.
- An open-questions list grounded in missing or disputed evidence.
- A final learning or judgment context pack that includes reviewed claim ids
  and excludes unreviewed proposed artifacts from canonical context.

## Consequences

- PR #209's answer-quality harness can remain a bounded gate, but it is not the
  next product center.
- Onboarding and quickstart work should wait until this loop has a believable
  human-test artifact.
- Future implementation should start report-only, then add an explicit apply or
  review-decision write path after the report shape is accepted.
- Durable records should use schema-backed JSON payloads and human-readable
  previews so the operator can inspect what became memory and what stayed
  proposed.

## References

- `docs/ARCHITECTURE.md`
- `docs/adr/2026-05-29-knowledgeos-product-definition.md`
- `docs/adr/2026-06-03-four-layer-agent-architecture.md`
- `docs/plans/research-review-loop-mvp-2026-06-10.md`
- `docs/schemas/research-review-loop-result.v1.json`
