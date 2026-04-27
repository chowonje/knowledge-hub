# Second Core #2: Evidence-Contract RAG Local Performance Gate

Date: 2026-04-27
Branch: `frontier/evidence-contract-rag-gate-20260427`

## Objective

Add a frontier-local performance and quality gate for the Evidence-contract RAG track. The gate should measure the existing `ask` path contract outputs instead of introducing a new RAG runtime.

## Affected Paths

- `eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py`
- `eval/knowledgeos/queries/evidence_contract_perf_gate_cases_v1.json`
- `tests/test_evidence_contract_perf_gate.py`
- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`

## Implementation Notes

- Live mode calls the same searcher answer path used by `khub ask --json`, then applies the shared ask payload normalization.
- `--stub-llm` uses a deterministic fixture searcher so contract regression can run quickly without local corpus/provider variance.
- The measured contracts are `evidencePacketContract`, `answerContract`, and `verificationVerdict`.
- The gate records contract validity, citation-grade coverage, abstain correctness, verification pass rate, unsupported-claim rate, conservative-fallback rate, latency p50/p95, and timeout count.
- The default case set covers paper, vault, web, mixed, and abstain scenarios.
- Live mode remains local/frontier-only; required CI promotion is intentionally deferred until corpus/provider variance is understood.

## Verification

- `python -m ruff check eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py tests/test_evidence_contract_perf_gate.py`
- `python -m py_compile eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py`
- `python -m pytest tests/test_evidence_contract_perf_gate.py -q` (`5 passed`)
- `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --stub-llm --json --timeout-sec 10` (`status=ok`, `24/24` cases passed)
- `python -m pytest tests/test_answer_contracts_runtime.py tests/test_answer_contract_schemas.py tests/test_answer_quality_gate.py tests/test_retrieval_span_golden.py -q` (`18 passed`)
- `python -m pytest tests/test_mcp_server.py tests/test_mcp_server_helpers.py -q` (`46 passed`)

## Follow-up

- Promote stable MCP `ask_knowledge` contract fields after the gate shows consistent contract behavior.
- Decide later whether a deterministic subset should become a required CI gate.
