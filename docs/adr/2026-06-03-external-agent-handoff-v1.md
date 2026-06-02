# ADR: External Agent Handoff v1

Date: 2026-06-03

## Status

Accepted as the v1 contract for report-only external agent packages.

## Context

External agents such as Hermes can help with public-source radar, review,
classification, or report drafting. They may run outside the local KnowledgeOS
workspace and may not share the same privacy, policy, or path boundaries.

The existing Hermes public source radar bundle demonstrates the safe shape:
sanitized seed input, a report-only prompt, generated report artifacts, and a
separate import manifest for operator review. The raw local seed remains
available only for explicitly local runs and is not part of the external default.

## Decision

External agents use a handoff package under:

```text
artifacts/<agent>/<YYYY-MM-DD>/
```

The package is report-only unless a reviewed import manifest is explicitly
processed by a Knowledge Hub operator.

## Directory Layout

```text
artifacts/<agent>/<YYYY-MM-DD>/
  manifest.json
  prompt.md
  input.sanitized.json
  input.readme.md
  report.md
  report.json
  selected.import-manifest.json
```

Not every package needs every file, but `manifest.json` must describe which
inputs and outputs are present.

## Manifest Required Fields

The canonical JSON Schema stub is:

```text
docs/schemas/knowledge-hub.external-agent-handoff.v1.json
```

The minimum required fields are:

- `agentKind`
- `mutationsAllowed`
- `khubCommandsAllowed`
- `vaultAccess`
- `importSteps`

Recommended fields are:

- `schema`
- `handoffId`
- `createdAt`
- `agentName`
- `artifactRoot`
- `sanitizedInputs`
- `outputs`
- `privacyGuards`
- `knowledgeHubRepoRef`
- `operatorNotes`

## Mutation Rules

Default:

- P0 and raw private material stay local and are not transmitted externally.
- `mutationsAllowed=false`
- `khubCommandsAllowed.allowed=false`
- `vaultAccess.allowed=false`
- `importSteps[].requiresOperatorApproval=true`

Allowed external work:

- read sanitized input files
- use stable public sources named in the prompt
- write report files inside the handoff directory
- propose import steps as commands or review actions

Blocked external work:

- running `khub papers add`, `khub papers embed`, `khub index`, or ingestion commands
- scanning or reading the Obsidian vault
- using raw local-only seed files unless the operator explicitly marks the run as local-only
- mutating DB, indexes, parser artifacts, evidence stores, source spans, vault notes, or product repo files

## Import Manifest Rules

An import manifest is a recommendation, not authorization.

The operator import step must:

1. Validate the handoff manifest schema.
2. Check that all referenced inputs are sanitized or explicitly local-only.
3. Check that proposed `khub` commands are allowlisted for the import task.
4. Run validation commands before any apply command.
5. Record completed and failed steps in the import manifest.

The Hermes public source radar example uses this shape:

```text
KnowledgeOS/artifacts/hermes_public_source_radar/2026-06-01/
```

It includes a report-only prompt, sanitized seed file, generated report, selected
candidate CSV, and `selected_add_candidates_2026-06-01.import-manifest.json`.

## Consequences

- External agents can contribute useful reports without crossing the local data boundary.
- Public-source discovery and classification can happen outside the product repo only from sanitized inputs.
- Product mutations remain local, explicit, and reviewable.
- Future Hermes-like workflows should converge on the schema stub instead of inventing per-agent package rules.

## References

- `docs/schemas/knowledge-hub.external-agent-handoff.v1.json`
- `docs/guides/agent-architecture.md`
- `docs/adr/2026-06-03-four-layer-agent-architecture.md`
- `KnowledgeOS/artifacts/hermes_public_source_radar/2026-06-01/`
