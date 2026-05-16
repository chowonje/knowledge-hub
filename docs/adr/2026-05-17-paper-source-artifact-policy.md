# ADR: Paper Source Artifact Policy

Date: 2026-05-17

## Status

Accepted for the source-artifact corpus policy tranche.

## Context

The paper compare and answer gates now require strict source provenance:
real `sourceContentHash` / `source_content_hash` plus exact
`chars:start-end` offsets. Fallback spans, locator-only anchors,
`memory-unit:` locators, snippets, generated summaries, Korean summaries, and
paraphrases do not satisfy strict evidence.

The AlexNet compare repair made this gate reproducible through
`khub paper repair-source`, but it still depends on an operator-local source
artifact being present in the configured paper store. Without a policy, the
project can drift back into one-off local DB edits, accidental PDF commits, or
headline live-eval numbers that hide missing corpus files.

## Decision

Use a hybrid source-artifact policy.

In git:

- keep a metadata-only corpus manifest at
  `eval/knowledgeos/fixtures/corpus_manifest.json`
- keep tiny deterministic text fixtures when CI needs clean-clone evidence
  coverage; manifest `repo_fixture` entries reference these with a path relative
  to the manifest location
- keep eval templates and scripts
- keep ADR, changelog, project-state, task, review, and worklog records

Outside git:

- keep paper PDFs and full extracted paper text under the configured local
  paper store, normally `papers_dir`
- keep local SQLite databases, backups, private sidecars, local eval cases, and
  generated eval runs outside tracked source

The manifest records artifact ids, source ids, expected filenames, expected
source-content hashes, byte length, provenance URL, license note, corpus tier,
and operator notes. It is not a source store and it is not evidence by itself.

`repair-source` may attach and rebuild from a local artifact only when the
manifest entry is present, the file exists under configured local search paths,
and the observed hash matches the manifest. Missing artifacts and hash
mismatches produce structured diagnostics and must not write source paths,
rebuild derivatives, or promote fallback/locator/snippet/summary evidence.
Durable diagnostics use safe path refs such as `papers_dir/<filename>` and must
not record personal absolute workstation paths.

Live operator evals may declare `corpusRequirements`. Missing or mismatched
local-corpus artifacts skip the affected row with an explicit reason, and the
report must show declared, evaluable, skipped, and coverage metrics next to the
ordinary pass/fail and safety metrics.
`optional_local_corpus` requirements are observation-only and do not block the
case from running when absent.
Required local-corpus skips are not strict-evidence failures for the row, but
the live compare gate defaults to requiring full corpus coverage so an
incomplete corpus cannot produce a green headline by accident.

No default code path may download paper artifacts. Any future acquisition helper
must be explicit, opt-in, hash-verified after download, and separate from
`knowledge_hub.application.*`, eval collectors, repair-source, and CI.

## Consequences

- Clean-clone CI can still test strict evidence mechanics with small fixtures.
- Live corpus gates remain local-first and operator-controlled.
- A `15/15` live compare result is only complete when paired with
  `declared=15`, `evaluable=15`, and full corpus coverage.
- New collaborators can inspect the manifest to know which local files and
  hashes are required for live paper gates.
- Hash drift becomes an explicit corpus maintenance event rather than an
  implicit weakening of evidence gates.

## Non-Goals

- Do not commit paper PDFs in this tranche.
- Do not add automatic download or acquisition behavior.
- Do not add new persistence tables or registry writes.
- Do not expose a new MCP/public schema surface.
- Do not weaken strict evidence, answerability, or no-answer gates.

## Follow-Ups

1. Add an optional, explicit public-paper acquisition helper only after the
   manifest contract is stable.
2. Add a binary/PDF guard if accidental artifact commits recur.
3. Consider a `khub doctor` corpus-health summary after the manifest is used in
   more than one operator gate.
4. Add manifest version/supersession policy for paper revisions when hash drift
   is confirmed.
