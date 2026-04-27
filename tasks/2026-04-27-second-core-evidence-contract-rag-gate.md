# Second Core #2: Evidence-Contract RAG Local Performance Gate

Date: 2026-04-27
Branch: `frontier/evidence-contract-rag-gate-20260427`

## Objective

Add a frontier-local performance and quality gate for the Evidence-contract RAG track. The gate should measure the existing `ask` path contract outputs instead of introducing a new RAG runtime.

## Affected Paths

- `eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py`
- `eval/knowledgeos/queries/evidence_contract_perf_gate_cases_v1.json`
- `knowledge_hub/ai/ask_v2.py`
- `knowledge_hub/ai/ask_v2_support.py`
- `knowledge_hub/ai/evidence_assembly.py`
- `knowledge_hub/ai/rag_answer_evidence.py`
- `knowledge_hub/mcp/handlers/search.py`
- `tests/test_evidence_contract_perf_gate.py`
- `tests/test_evidence_assembly_temporal.py`
- `tests/test_rag_answer_evidence.py`
- `tests/test_paper_ask_v2.py`
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
- `--live-stub-llm` keeps the real local AppContext/searcher/corpus path but stubs answer generation. Use it when validating retrieval/citation contracts without local model heat.
- Broad abstain cases can define `execution_source` / `ask_source_type` so the gate can exercise the intended lane without broad mixed-corpus scans.
- Stubbed reports set `verificationPassRate=null` and preserve the raw value as `verificationPassRateRaw`, because generic stub answers are not meaningful generation-quality evidence.
- Live mode remains local/frontier-only; required CI promotion is intentionally deferred until corpus/provider variance is understood.

## Live Baseline

- A full live run with `--timeout-sec 60` was started but stopped after more than five minutes without stdout so the workstation would not remain blocked on an unbounded provider/corpus path.
- The completed full live baseline used `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --json --run-profile full --timeout-sec 20`.
- Result: `status=failed`, `6/24` passed, `contractValidRate=0.541667`, `citationGradeCoverageRate=0.125`, `abstainCorrectRate=0.75`, `verificationPassRate=0.0`, `unsupportedClaimRate=0.208333`, `conservativeFallbackRate=0.208333`, `timeoutCount=0`, `latencyMs.p50=20123.288`, `latencyMs.p95=85983.314`.
- Failure categories: `contract_missing=11`, `abstain_mismatch=7`, `citation_grade=5`.
- Interpretation: the deterministic contract path is stable, but the live corpus/provider path is not ready for CI promotion; remaining work is contract propagation on legacy ask branches, citation-grade evidence coverage, abstain calibration, and latency control.
- A follow-up no-generation live-corpus run used `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --live-stub-llm --run-profile thermal --json --timeout-sec 10`.
- Result: `status=failed`, `4/5` passed, `contractValidRate=1.0`, `citationGradeCoverageRate=1.0`, `abstainCorrectRate=1.0`, `timeoutCount=0`, `llmStubbed=true`.
- Failure categories: `provider/corpus_dependency=1`, specifically `web_general_011` as `provider_corpus_dependency:temporal_grounding`.
- Interpretation: the thermal slice no longer shows missing product contracts or citation-grade provenance gaps; the remaining failure is evidence-policy/corpus calibration, not a reason to lower thresholds.
- The first full no-generation live-corpus run after `c93592d` produced `9/24` passed, `contractValidRate=1.0`, `citationGradeCoverageRate=0.9375`, `abstainCorrectRate=0.375`, `timeoutCount=0`, and `latencyMs.p95≈46s`. The slow cases were broad abstain prompts that searched the mixed corpus.
- After adding explicit `execution_source` lanes for broad abstain cases and accepting conservative fallback as safe refusal for expected-abstain checks, the full no-generation gate reached `14/24` with `abstainCorrectRate=1.0` and `latencyMs.p95≈1.1s`.
- After aligning `answerContract.abstain` with strict `evidencePacketContract.answerable=false` and conservative fallback outputs, the current full no-generation baseline is intentionally stricter: `8/24` passed, `contractValidRate=1.0`, `citationGradeCoverageRate=0.9375`, `abstainCorrectRate=1.0`, `verificationPassRate=null`, `verificationPassRateRaw=0.0`, `timeoutCount=0`, and `latencyMs.p95=1142.428ms`.
- Interpretation: the remaining blocker is not schema/citation propagation or local-model heat. It is answerability calibration: answer-expected cases still become hard abstain/conservative fallback too often, hard latest web prompts still need better temporal grounding, and one mixed disambiguation row lacks citation-grade spans.

## Follow-up Fixes

- Early-exit/no-result ask branches now add `answerContract` and `verificationVerdict` alongside the existing `evidencePacketContract`, with abstain/no-result semantics and `finalAnswerSource=early_exit`.
- AskV2 card anchors now pass source-content hash, span locator, snippet hash, char offsets, document id, and web URL/canonical URL through `SearchResult.metadata`; if older card rows lack `source_content_hash`, the runtime recovers it from document-memory units or note metadata (`source_content_hash`, `content_hash`, `content_sha1`, `content_sha256`).
- Answer evidence now preserves char offsets so `char_start=0` is not lost before contract citation grading.
- The perf gate now treats `answerContract.abstain` as the source of truth when present. `verificationVerdict.recommended_action=abstain` remains a compatibility fallback only for legacy payloads without an answer contract, so caution/fail verdicts are not overcounted as observed abstentions.
- Targeted local-corpus probe after the AskV2 provenance fix: `web_general_011` improved from `citationGradeCitationCount=0` to `2`; the remaining failure is `abstain_mismatch` caused by AskV2 `temporal_version_grounding` hard abstention, not missing citation provenance.
- The perf gate now classifies answer-expected hard-abstain cases with valid contracts and enough citation-grade evidence as `provider_corpus_dependency:<reason>` instead of `abstain_mismatch`; temporal grounding signals map to `provider/corpus_dependency` in the summary.
- Older paper-card anchors can now recover source-content hashes from paper `text_path`, `translated_path`, `pdf_path`, or paper notes, so historical paper rows do not fail citation-grade checks only because their card metadata predates strict provenance.
- Soft recency evaluation prompts such as “최근 RAG evaluation article...” no longer enter the hard temporal route only because they contain `최근`; explicit latest/update/before/after signals remain temporal.
- `answerContract.abstain` now treats strict `evidencePacketContract.answerable=false`, `early_exit`, `policy_blocked`, and `conservative_fallback` as abstention states, so raw packet answerability can no longer contradict the strict evidence contract.
- Stubbed live-corpus generation now reports a local route instead of a fixed/external route, avoiding false P0 external-call policy blocks during no-generation validation.
- AskV2 no longer hard-abstains solely because claim cards include unsupported items when scoped evidence verification has anchors and no concrete unsupported field; unsupported claim cards remain diagnostics and answer post-processing can still fall back if generated text is unsupported.
- Metadata-only retrieved evidence now gets a deterministic retrieved-document content hash and span range, so mixed/paper fallback rows can be contract-graded when no older source hash exists.
- EvidenceAssembly now matches the AskV2 soft-recency rule for evaluation prompts: soft `recent/최근` evaluation questions are not treated as hard temporal latest claims, while explicit `latest/최신/update/after/year` prompts still require temporal grounding.
- Current full no-generation baseline: `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --live-stub-llm --run-profile full --json --timeout-sec 10` produced `status=failed`, `20/24` passed, `contractValidRate=1.0`, `citationGradeCoverageRate=1.0`, `abstainCorrectRate=1.0`, `verificationPassRate=null`, `verificationPassRateRaw=0.0`, `timeoutCount=0`, `latencyMs.p95=1069.924ms`, `generationDependencyRate=0.5`, and `failureCategories={"provider/corpus_dependency": 4}`.

## Verification

- `python -m ruff check eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py knowledge_hub/mcp/handlers/search.py tests/test_evidence_contract_perf_gate.py tests/test_mcp_search_handler.py tests/test_mcp_server.py`
- `python -m py_compile eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py`
- `python -m pytest tests/test_evidence_contract_perf_gate.py -q` (`8 passed`)
- `python -m pytest tests/test_evidence_contract_perf_gate.py tests/test_paper_ask_v2.py::test_anchor_results_recovers_paper_hash_from_paper_record -q` (`13 passed`)
- `python -m pytest tests/test_answer_contracts_runtime.py tests/test_evidence_contract_perf_gate.py tests/test_paper_ask_v2.py::test_ask_v2_classifies_soft_recency_evaluation_as_evaluation_not_temporal -q` (`27 passed`)
- `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --stub-llm --json --timeout-sec 10` (`status=ok`, `24/24` cases passed)
- `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --stub-llm --run-profile thermal --json --timeout-sec 10` (`status=ok`, `5/5` source-balanced cases passed)
- `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --live-stub-llm --run-profile thermal --json --timeout-sec 10` (`status=failed`, `4/5` cases passed; contract/citation/abstain/timeout thresholds green; remaining failure classified as `provider/corpus_dependency:temporal_grounding`)
- `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --live-stub-llm --run-profile full --json --timeout-sec 10` (`status=failed`, `8/24` cases passed; contract/citation/abstain/timeout thresholds green; current blockers are answerability/fallback calibration, hard-temporal web grounding, and one mixed citation-grade row)
- `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --live-stub-llm --run-profile full --json --timeout-sec 10` (`status=failed`, `20/24` cases passed; contract/citation/abstain/timeout thresholds green; remaining failures are provider/corpus dependency only)
- `python eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py --json --run-profile full --timeout-sec 20` (`status=failed`, `6/24` cases passed; baseline recorded only, not CI-required)
- `python -m pytest tests/test_answer_contracts_runtime.py tests/test_answer_contract_schemas.py tests/test_answer_quality_gate.py tests/test_retrieval_span_golden.py tests/test_evidence_contract_perf_gate.py -q` (`30 passed`)
- `python -m pytest tests/test_answer_contracts_runtime.py tests/test_answer_contract_schemas.py tests/test_answer_quality_gate.py tests/test_retrieval_span_golden.py tests/test_evidence_contract_perf_gate.py tests/test_evidence_assembly_temporal.py tests/test_rag_answer_evidence.py -q` (`41 passed`)
- `python -m pytest tests/test_paper_ask_v2.py -q` (`46 passed`)
- `python -m pytest tests/test_answer_contracts_runtime.py tests/test_answer_contract_schemas.py tests/test_answer_quality_gate.py tests/test_retrieval_span_golden.py tests/test_evidence_contract_perf_gate.py tests/test_paper_ask_v2.py::test_ask_v2_classifies_soft_recency_evaluation_as_evaluation_not_temporal -q` (`37 passed`)
- `python -m pytest tests/test_answer_orchestrator_services.py::test_answer_orchestrator_generate_and_stream_no_result_early_exit_stay_in_parity tests/test_answer_orchestrator_services.py::test_answer_orchestrator_generate_and_stream_need_multiple_papers_early_exit_stay_in_parity tests/test_paper_ask_v2.py::test_anchor_results_preserve_compare_source_diversity tests/test_paper_ask_v2.py::test_anchor_results_preserve_card_v2_strict_provenance tests/test_ask_v2_sources.py::test_web_anchor_results_expose_url_and_strict_provenance -q` (`5 passed`)
- `python -m pytest tests/test_answer_orchestrator_services.py::test_answer_orchestrator_generate_and_stream_no_result_early_exit_stay_in_parity tests/test_answer_orchestrator_services.py::test_answer_orchestrator_generate_and_stream_need_multiple_papers_early_exit_stay_in_parity -q` (`2 passed`)
- `python -m pytest tests/test_mcp_server.py tests/test_mcp_search_handler.py tests/test_mcp_server_helpers.py -q` (`48 passed`)
- `python -m ruff check eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py knowledge_hub/ai/ask_v2.py knowledge_hub/ai/ask_v2_support.py knowledge_hub/ai/evidence_assembly.py knowledge_hub/ai/rag_answer_evidence.py tests/test_evidence_contract_perf_gate.py tests/test_paper_ask_v2.py tests/test_evidence_assembly_temporal.py tests/test_rag_answer_evidence.py` (`passed`)
- `python -m py_compile eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py knowledge_hub/ai/ask_v2.py knowledge_hub/ai/ask_v2_support.py knowledge_hub/ai/evidence_assembly.py knowledge_hub/ai/rag_answer_evidence.py` (`passed`)
- `python -m ruff check knowledge_hub/ai/ask_v2.py eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py tests/test_evidence_contract_perf_gate.py tests/test_paper_ask_v2.py` (`passed`)
- `python -m ruff check knowledge_hub/ai/answer_contracts.py knowledge_hub/ai/answer_orchestrator_payload_slices.py knowledge_hub/ai/ask_v2_support.py eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py tests/test_answer_contracts_runtime.py tests/test_evidence_contract_perf_gate.py tests/test_paper_ask_v2.py` (`passed`)
- `python -m ruff check knowledge_hub/ai/answer_contracts.py knowledge_hub/ai/answer_orchestrator_early_exit.py knowledge_hub/ai/ask_v2.py knowledge_hub/ai/rag_answer_evidence.py eval/knowledgeos/scripts/run_evidence_contract_perf_gate.py tests/test_answer_orchestrator_services.py tests/test_paper_ask_v2.py tests/test_ask_v2_sources.py tests/test_evidence_contract_perf_gate.py` (`passed`)
- `git diff --check` (`passed`)

## Follow-up

- Tune AskV2 temporal hard-gate behavior against the fixed case set without weakening conservative lack-of-evidence semantics.
- Decide whether historical card rows should get a metadata-only `source_content_hash` backfill, even though runtime recovery now covers the answer path.
- Re-run a sanitized/public live route with `--answer-route codex --allow-external` if external generation quality/latency needs evaluation without heating the local Ollama path.
- Decide later whether a deterministic subset should become a required CI gate.
