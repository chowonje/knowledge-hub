# ADR: khub ask External-Generation Default Is Opt-In

Date: 2026-06-11

## Status

Accepted.

## Context

`khub ask` resolved its default `allow_external` from the configured summarization
provider: any non-local provider (e.g. `summarization.provider: openai`) silently made
external answer generation the default. With the current operator config this meant the
documented "local-first by default" posture and the actual default disagreed — a plain
`khub ask` would send P1/P2 context to the configured cloud provider unless the user
remembered `--no-allow-external`.

The 2026-06-11 architecture review flagged this as a posture/config contradiction, and
the adversarial verification pass additionally showed that local-only and external runs
exercise different enforcement regimes, so the default must be explicit and observable
rather than inferred.

## Decision

1. `khub ask`'s default `allow_external` is **False**. It is never inferred from
   provider configuration.
2. External generation by default requires an explicit `answer.allow_external_default: true`
   in config. The per-invocation `--allow-external / --no-allow-external` flags continue
   to override in both directions.
3. The effective policy remains visible on every ask: text output prints
   `allow_external=...`; JSON output carries `allowExternal`.

## Consequences

- With a cloud summarization provider configured but no explicit opt-in, plain
  `khub ask` now routes generation through local/codex-local paths and may abstain more
  often (the local heuristic verifier is stricter). This is consistent with
  "answerability is stricter than fluency" (ADR 2026-05-29).
- Eval/gate runs must record which regime (`allowExternal`) they ran under; results from
  different regimes are not comparable.
