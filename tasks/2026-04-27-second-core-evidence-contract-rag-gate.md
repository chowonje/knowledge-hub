# Second Core #2: Evidence-Contract RAG Local Performance Gate

Date: 2026-04-27
Branch: `frontier/evidence-contract-rag-gate-20260427`

## Objective

Add a frontier-local performance and quality gate for the Evidence-contract RAG track. The gate should measure the existing `ask` path contract outputs instead of introducing a new RAG runtime.

## Affected Paths

- `eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py`
- `eval/knowledgeos/queries/evidence_contract_perf_gate_cases_v1.json`
- `knowledge_hub/mcp/handlers/search.py`
- `tests/test_evidence_contract_perf_gate.py`
- `tests/test_mcp_search_handler.py`
- `tests/test_mcp_server.py`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`

## Implementation Notes

- Live mode calls the same searcher answer path used by `khub ask --json`, then applies the shared ask payload normalization.
- `--stub-llm` uses a deterministic fixture searcher so contract regression can run quickly without local corpus/provider variance.
- The measured contracts are `evidencePacketContract`, `answerContract`, and `verificationVerdict`.
- The gate records contract validity, citation-grade coverage, abstain correctness, verification pass rate, unsupported-claim rate, conservative-fallback rate, latency p50/p95, and timeout count.
- Failed cases are now categorized as `contract_missing`, `citation_grade`, `abstain_mismatch`, `latency_timeout`, or `provider/corpus_dependency` so live reports separate product contract gaps from local corpus/provider variance.
- The default case set covers paper, vault, web, mixed, and abstain scenarios.
- MCP `ask_knowledge` now passes through `evidencePacketContract`, `answerContract`, and `verificationVerdict` as additive top-level fields in both the normal payload and the policy-classified artifact envelope.
- Live `--run-profile auto` is now thermal-friendly: it selects one representative case per source family by default, while stub mode stays full for deterministic coverage.
- Full live coverage requires `--run-profile full`; source-specific or exact-case runs can use `--source`, `--max-cases`, or repeated `--case-id`.
- Codex/API-backed answer generation is explicit with `--answer-route codex --allow-external`. Use this only for sanitized/public evidence; Ollama/local remains the private fallback.
- Live mode remains local/frontier-only; required CI promotion is intentionally deferred until corpus/provider variance is understood.

## Live Baseline

- A full live run with `--timeout-sec 60` was started but stopped after more than five minutes without stdout so the workstation would not remain blocked on an unbounded provider/corpus path.
- The completed full live baseline used `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --json --run-profile full --timeout-sec 20`.
- Result: `status=failed`, `6/24` passed, `contractValidRate=0.541667`, `citationGradeCoverageRate=0.125`, `abstainCorrectRate=0.75`, `verificationPassRate=0.0`, `unsupportedClaimRate=0.208333`, `conservativeFallbackRate=0.208333`, `timeoutCount=0`, `latencyMs.p50=20123.288`, `latencyMs.p95=85983.314`.
- Failure categories: `contract_missing=11`, `abstain_mismatch=7`, `citation_grade=5`.
- Interpretation: the deterministic contract path is stable, but the live corpus/provider path is not ready for CI promotion; remaining work is contract propagation on legacy ask branches, citation-grade evidence coverage, abstain calibration, and latency control.

## Verification

- `python -m ruff check eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py knowledge_hub/mcp/handlers/search.py tests/test_evidence_contract_perf_gate.py tests/test_mcp_search_handler.py tests/test_mcp_server.py`
- `python -m py_compile eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py`
- `python -m pytest tests/test_evidence_contract_perf_gate.py -q` (`8 passed`)
- `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --stub-llm --json --timeout-sec 10` (`status=ok`, `24/24` cases passed)
- `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --stub-llm --run-profile thermal --json --timeout-sec 10` (`status=ok`, `5/5` source-balanced cases passed)
- `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --json --run-profile full --timeout-sec 20` (`status=failed`, `6/24` cases passed; baseline recorded only, not CI-required)
- `python -m pytest tests/test_answer_contracts_runtime.py tests/test_answer_contract_schemas.py tests/test_answer_quality_gate.py tests/test_retrieval_span_golden.py tests/test_evidence_contract_perf_gate.py -q` (`26 passed`)
- `python -m pytest tests/test_mcp_server.py tests/test_mcp_search_handler.py tests/test_mcp_server_helpers.py -q` (`48 passed`)

## Follow-up

- Make legacy/fast ask branches consistently emit `answerContract` and `verificationVerdict` when `evidencePacketContract` is already present.
- Improve citation-grade coverage for live paper/web/vault cases by ensuring source ids, content hashes, and span locators survive answer synthesis.
- Tune abstain behavior against the fixed case set without weakening conservative fallback semantics.
- Decide later whether a deterministic subset should become a required CI gate.
