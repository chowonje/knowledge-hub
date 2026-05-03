# Observation Action Rules

## Purpose

This document prevents the current observation stack from degrading into a passive dashboard.

The repo already has:

- nightly source-quality runs
- trend reports
- readiness reports
- observation verdict reports

What was missing is the rule that says:

> when does a green or unstable observation require a concrete next action?

This document is that rule.

## Scope

This status note covers four observation-to-action paths:

1. source-quality hard-gate promotion
1. source-quality detail-gate promotion
2. next gateway tranche start
3. cleanup or removal tranche start
4. product-direction review

It does **not** define new runtime behavior. It defines when existing reports must trigger the next decision or implementation step.

## Rule 1: Source-quality hard-gate promotion

### Signal

- latest source-quality observation verdict
- recent source-quality trend report

### Minimum observation window

- 7 consecutive source-quality runs with report-backed trend/readiness history

### Required condition

All of the following must hold across the full observation window:

- `paper route_correctness == 1.0`
- `vault route_correctness == 1.0`
- `web route_correctness == 1.0`
- `vault stale_citation_rate == 0`
- `paper/vault/web legacy_runtime_rate == 0`
- `paper/vault/web capability_missing_rate == 0`
- latest observation verdict is `ready_for_hard_gate_review`
- no active blocker equivalent to `web_route_correctness_not_stable`

### Action

Open one reviewable promotion change that:

- proposes the specific hard-gate subset
- keeps the rest of the observation layer trend-only
- records rollback conditions in the task note and project state

### Owner

- repo owner

### Stop condition

Do not open the promotion PR if the required window is incomplete or if any blocker reappears during the window.

### Rollback condition

If the promoted hard-gate metric drops below threshold during the first 7 days after promotion:

- revert the promotion change or set `ENFORCE_HARD_GATE=0` only as a temporary rollback
- record the reason in `docs/PROJECT_STATE.md` and the next observation window start date

## Rule 1.5: Source-quality detail-gate promotion

### Signal

- latest source-quality detail observation verdict
- recent source-quality trend report
- latest base source-quality hard gate result

### Minimum observation window

- 7 consecutive source-quality runs with numeric values for every candidate detail metric

### Required condition

All of the following must hold across the full observation window:

- base source-quality hard gate still passes
- paper `paper_citation_correctness >= 1.0`
- vault `vault_abstention_correctness >= 1.0`
- web `web_recency_violation <= 0.0`
- no candidate detail metric is unobserved or `None`

### Action

Open one reviewable promotion change that:

- promotes only the stable detail metrics
- leaves missing-coverage metrics in observation
- records the exact rollback threshold in project state

### Stop condition

Do not promote if any candidate metric lacks numeric coverage, even if the other candidates are stable.

### Rollback condition

If a promoted detail metric drops below threshold during the first 7 days after promotion:

- revert the detail-gate promotion or move that one metric back to observation
- record the failed metric and the next observation window start date

## Rule 2: Next gateway tranche start

### Signal

- actual use evidence for existing gateway surfaces
- especially successful end-to-end use of the current writeback lane

### Minimum observation window

- 14 days of normal project work

### Required condition

All of the following must be true:

- at least 5 successful uses of the existing gateway lane or its first real consumer
- revert rate for approved writeback results stays at or below 20 percent
- at least one bounded rejection event proves the current safety boundary is still active

### Action

Open exactly one next-tranche proposal for the gateway, choosing one of:

- first real consumer stabilization
- one bounded allowlist expansion
- one narrow external consumer surface

Do not propose more than one of those in the same tranche.

### Owner

- repo owner

### Stop condition

Do not open a gateway expansion tranche if use remains mostly theoretical, if the revert rate is too high, or if the current lane is not being used in real project work.

### Rollback condition

If the new consumer or expansion materially increases revert rate or approval friction in the first 2 weeks:

- stop rollout
- revert the tranche if possible
- return to the previous bounded lane

## Rule 3: Cleanup or removal tranche start

### Signal

- readiness report
- static callsite scan
- trend-backed usage evidence when applicable

### Minimum observation window

- 7 consecutive source-quality runs for readiness-report-backed cleanup/removal work

### Required condition

All of the following must hold:

- production/runtime callsites are reduced to the intended final minimal set
- tests are not the only remaining reason the surface exists
- no eval or scripts dependency remains, unless the surface is explicitly retained for observation only
- the latest readiness verdict is stable across the full 7-run window, not a one-run flip

### Action

Open one removal or cleanup tranche that:

- changes one legacy path at a time
- remains one-commit-revertable
- leaves a durable record in `CHANGELOG.md` and `docs/PROJECT_STATE.md`

### Owner

- repo owner

### Stop condition

Do not start cleanup just because a report is green once. The full 7-run readiness window must already be satisfied.

### Rollback condition

If regression appears in the first two weeks after cleanup:

- revert the tranche
- restore the previous path
- start a new observation window before attempting removal again

## Rule 4: Product-direction review

### Signal

At least two of the following are true at the same time:

- gateway real-consumer usage grows enough to challenge the current Core Runtime centered model
- gateway usage remains near zero despite available surfaces
- source-quality remains unstable for a prolonged period and requires renewed core-runtime investment
- measured `foundry-core` or adjacent bridge usage shows a mismatch between maintenance cost and actual value

### Minimum observation window

- 8 weeks

### Required condition

- two or more product-direction signals persist through the full observation window
- evidence is recorded in durable artifacts, not only remembered informally

### Action

Write one direction-review note that answers:

- keep current direction
- tighten scope
- expand the gateway
- reduce a low-value maintained surface

The result should become the next 90-day direction input.

### Owner

- repo owner

### Stop condition

Do not reopen product-direction debate every week. If a review concludes "keep direction," wait at least 4 more weeks before another review.

### Rollback condition

If a direction change is started and the evidence does not hold up:

- cancel the direction change
- record the failed trigger
- return to the previous direction until the next valid window

## Anti-patterns To Avoid

The following are considered observation failures:

1. **Green-without-action**
   - reports are green for weeks but no tranche opens
2. **Permanent observe-more**
   - the system keeps waiting because no one has written a minimum window or promotion rule
3. **Human-memory-only interpretation**
   - a report changes state but the working model in people’s heads does not

The fix for all three is the same:

- durable trigger rules
- named owner
- bounded observation window
- explicit next action or explicit no-action reason

## Current Application

As of this note:

- source-quality was promoted on 2026-04-21 for the local daily operating loop after the latest 7-run observation report reached `ready_for_hard_gate_review` with no blockers
- source-quality detail observation is automated but not promoted; current blocker is missing numeric coverage for vault abstention correctness
- gateway surfaces should not widen before a first real consumer produces operating evidence
- legacy-runtime removal readiness uses the dedicated 7-run report, not the older generic 28-day cleanup rule
- cleanup and removal should follow stable readiness, not a single green report

## Next

The next concrete use of this document should be:

1. pick the first real gateway consumer
2. run it long enough to produce operating evidence
3. use these rules to decide whether the next tranche is expansion, rollback, or hold
