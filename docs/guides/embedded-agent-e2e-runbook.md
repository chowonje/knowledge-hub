# Embedded Agent E2E Runbook

Use this runbook for a local embedded-agent loop with `khub chat`, Agent Gateway, and the existing evidence runtime. It keeps `chat` hidden/experimental and keeps writes approval-gated.

## Invariants

- Default product path stays `discover -> index -> search/ask -> evidence review`.
- Use `--no-allow-external` unless the operator explicitly approves external calls.
- Use `--no-include-vault` unless the task explicitly needs vault evidence.
- Ordinary chat turns do not write vault, SQLite knowledge DB, vector index, ontology, paper stores, or docs.
- `--save-session` stores redacted metadata only; SQLite is canonical and JSONL is a mirror.
- Repo writes go through `agent writeback-request` plus ops queue approval.

## 1. Read-Only Assist

Assemble a context pack without vault access:

```bash
khub agent context --goal "Summarize the current agent-interface state" --repo-path . --no-include-vault --json
```

Ask against the existing evidence runtime, still blocking external calls:

```bash
khub ask "What is the current agent-interface state?" --source paper --no-allow-external --json
```

Check the hidden/advanced surfaces if a command looks missing:

```bash
khub --help
khub help advanced
```

## 2. Chat Session

Run a local single-turn chat:

```bash
khub chat "Give me the next safe step for the agent interface" --provider ollama --model qwen3:14b --no-allow-external --json
```

Record a redacted metadata-only session only when needed:

```bash
khub chat "Record a short planning turn" --provider ollama --model qwen3:14b --no-allow-external --save-session --json
```

Use `/paper` when the turn needs paper evidence instead of plain provider chat:

```bash
khub chat "/paper What papers support the current local-first agent boundary?" --no-allow-external --json
```

Confirm `chat` is still experimental/hidden:

```bash
khub chat --help
khub --help
```

## 3. Dry-Run Planning

Preview an agent plan without execution or writeback:

```bash
khub agent run --goal "Draft a docs update for the agent-interface runbook" --repo-path . --dry-run --json
```

If the dry-run payload asks for broad access, reduce scope and rerun with `--no-include-vault` or a narrower `--repo-path`.

## 4. Docs Writeback Approval

Create an approval-gated docs writeback request:

```bash
khub agent writeback-request --goal "Update docs for the latest agent-interface tranche" --repo-path . --json
```

Review pending agent actions:

```bash
khub labs ops action-list --scope agent --status pending --json
```

Approve and execute only the reviewed action id:

```bash
khub labs ops action-ack --action-id <id> --actor cli-user --note "reviewed docs-only request" --json
khub labs ops action-execute --action-id <id> --actor cli-user --json
khub labs ops action-receipts --action-id <id> --json
```

## Failure Modes

- `chat` appears in `khub --help`: stop and treat it as an accidental public promotion.
- External provider is blocked: expected unless `--allow-external` is explicitly approved.
- `agent run` suggests writes during `--dry-run`: do not execute; create a writeback request instead.
- `action-execute` fails before ack: expected; ack is the approval gate.
- Session output exposes raw prompt, answer, token, or local path: stop and treat as a redaction regression.
- Any command tries to scan vault without explicit need: rerun with `--no-include-vault`.

## External Agent Pointer

For Hermes or another external worker, do not run this embedded loop remotely. Use the report-only handoff pattern in `docs/guides/agent-architecture.md` and `docs/adr/2026-06-03-external-agent-handoff-v1.md`: sanitized bundle, `mutationRows=0`, no vault scan, local review before import.
