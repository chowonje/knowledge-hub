# Ask Tranche Start Plan

Date: 2026-04-16

## Baseline

- The weekly core-loop baseline is now fixed separately.
- `ops/checkpoints/09-core-loop-stabilization.pathspec` remains the current weekly baseline and should not be widened again unless that smoke regresses.
- The next tranche should therefore optimize for `ask` surface isolation, not for `doctor/status/index/search` repair.

## Objective

- Split the next reviewable tranche for the `ask` path out of the broad `retrieval/ask-v2` bucket.
- Keep the new tranche small enough to review, but broad enough to make `ask` behavior understandable and testable as one unit.

## First-pass scope

Start with the runtime and entrypoint files that directly determine whether `ask` can:

1. choose a route,
2. collect retrieval evidence,
3. produce a grounded answer or explicit degraded envelope,
4. avoid regressing the now-green weekly core-loop baseline.

Candidate first-pass clusters:

- RAG search/answer runtime core
  - `knowledge_hub/ai/rag.py`
  - `knowledge_hub/ai/rag_answer_runtime.py`
  - `knowledge_hub/ai/rag_answer_route_resolver.py`
  - `knowledge_hub/ai/rag_answer_evidence.py`
  - `knowledge_hub/ai/rag_evidence_context.py`
  - `knowledge_hub/ai/rag_search_runtime.py`
  - `knowledge_hub/ai/rag_legacy_runtime.py`
- ask-v2 routing / support
  - `knowledge_hub/ai/ask_v2.py`
  - `knowledge_hub/ai/ask_v2_support.py`
  - `knowledge_hub/ai/ask_v2_verification.py`
  - `knowledge_hub/ai/ask_v2_card_selectors.py`
  - `knowledge_hub/ai/rag_ask_v2_gate.py`
- thin entrypoints / orchestration
  - `knowledge_hub/interfaces/cli/commands/search_cmd.py`
  - `knowledge_hub/application/answer_loop.py`
- focused tests
  - `tests/test_search_cmd.py`
  - `tests/test_rag_search.py`
  - `tests/test_rag_runtime_services.py`
  - `tests/test_retrieval_pipeline_services.py`
  - `tests/test_answer_loop.py`
  - `tests/test_ask_v2*.py`

## Explicitly out of the first ask tranche

- broad docs/eval corpus churn under `docs/experiments/**`, `docs/research/**`, `eval/**`
- foundry / TypeScript bridge work
- agent-memory expansion
- broad curation or ingest work that is not directly needed for `ask`

## Entry criteria

- Preserve the green weekly core-loop baseline:
  - `python scripts/check_release_smoke.py --mode weekly_core_loop --json`
- Add one ask-focused gate before widening further:
  - local-profile `ask --json` should return a grounded answer or explicit degraded envelope
- Keep the first ask tranche reviewable:
  - prefer runtime core + CLI/tests before docs/eval bulk

## Initial manifest

- `ops/checkpoints/10-ask-runtime-first.pathspec` is now the first exact ask manifest.
- The slice is intentionally broader than the original note because `ask_v2` and the runtime wrappers import many untracked modules that do not exist in a clean `HEAD` worktree.
- The current rule is:
  - include ask entrypoints/runtime files,
  - include their first direct changed dependencies that are absent from clean `HEAD`,
  - stop before broad docs/eval/foundry carryover,
  - then widen only from replay evidence.

Likely remaining closure pressure after the first replay:

- deeper answer-orchestrator support files such as claim-adjudication or other answer payload helpers
- deeper domain helpers under `knowledge_hub/domain/ai_papers/**`
- card-builder/materialization paths pulled in by ask-v2 source cards

Current first replay target:

- replay `09-core-loop-stabilization.pathspec` first
- replay `10-ask-runtime-first.pathspec` second
- keep the weekly core-loop smoke green
- then run focused ask-path pytest or the first minimal local `ask --json` smoke

## First replay evidence

Clean replay into `/Users/won/Desktop/allinone/knowledge-hub-ask-runtime-first-v1` has already produced useful closure evidence:

- `weekly_core_loop` regressed at `search`, not at help/doctor/status/index
- first missing module was `knowledge_hub.knowledge.contracts`
- after adding that file, the next missing module was `knowledge_hub.knowledge.ontology_profiles`
- after adding that file, the next missing module became `knowledge_hub.learning.model_router`

Interpretation:

- the current blocker is import-closure breadth, not an ask-runtime logic regression
- the next widening step should prefer direct `knowledge/*` and `learning/*` runtime dependencies over unrelated docs/eval carryover
- a smaller first practical gate may be `CLI ask reaches explicit degraded envelope` rather than `full ask-v2 quality`

## Ask smoke evidence

After widening from the first replay blockers:

- the clean replay regained `weekly_core_loop` smoke `5/5`
- a local `ask --source vault --mode keyword --no-allow-external --json` smoke now returns a grounded `status=ok` JSON answer on the minimal vault fixture
- the remaining ask risk is no longer import closure; it is answer-quality behavior under local-only verification/rewrite defaults, where the current smoke still reports heuristic verification and skipped rewrite warnings

Observed ask-smoke blocker sequence so far:

- `knowledge_hub.domain.vault_knowledge`
- `knowledge_hub.papers.card_v2_builder`
- `knowledge_hub.papers.pymupdf_adapter`
- `knowledge_hub.web.ingest_chunking`
- `knowledge_hub.papers.memory_projection`
- `knowledge_hub.ai.claim_adjudication`
- `knowledge_hub.ai.answer_verification`
- `knowledge_hub.ai.answer_rewrite`
- after widening through those answer-support files, the same smoke returns a grounded vault answer instead of another import failure

## Done condition

- The next tranche has a named file set, not just a bucket label.
- `ask` has a repeatable local smoke path.
- The first ask review can be discussed independently from the already-closed weekly core-loop tranche.
