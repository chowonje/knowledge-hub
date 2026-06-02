# ADR: Chat Session Storage And Redaction

Date: 2026-06-03

## Status

Accepted for hidden `khub chat`.

## Context

`khub chat --save-session` needs resumable operational evidence without turning assistant conversation text into a new knowledge store. Chat input may contain secrets, P0 content, or material whose classification is unknown at persistence time.

## Decision

- SQLite is the canonical chat session store.
- The JSONL transcript file is a local mirror of the same redacted event payloads, not a second source of truth.
- Session persistence is opt-in. Ordinary chat turns without `--save-session` create no session store.
- User prompts and assistant answers are never stored as raw text. Events store hashes, character counts, route metadata, timestamps, and redaction metadata only.
- Unknown content classification fails closed for raw persistence: the persisted event marks classification as `UNKNOWN`, records `metadata_only_fail_closed`, and stores no raw content.
- Route metadata is allowlisted. Unrecognized metadata keys are dropped before SQLite/JSONL persistence.
- Session payloads returned by the CLI do not expose local session file paths.

## Non-Scope

- No vault, Obsidian, index, ontology, paper, or knowledge DB session storage.
- No public promotion of `khub chat`.
- No conversation-memory retrieval or resume behavior beyond the redacted session event log.

## Consequences

Operators can inspect that a chat session happened and which route/provider policy state applied, but they cannot recover raw prompts, raw answers, API keys, or P0 content from the session store. Future resume features must either continue from redacted metadata or add a separate explicit user-approved transcript contract.
