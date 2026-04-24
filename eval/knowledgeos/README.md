# KnowledgeOS Eval Assets

이 폴더는 KnowledgeOS 운영 평가 자산의 기준 위치입니다.

구성:

- `queries/`
  - canonical query set과 source/type별 평가 질의
- `templates/`
  - machine eval, human review CSV 템플릿
  - multi-model priority review 템플릿
- `prompts/`
  - GPT/Codex 1차 판정 프롬프트
- `rubrics/`
  - 외부 평가자 공통 rubric
- `scripts/`
  - candidate vs baseline 자동 리포트 스크립트
  - embedding A/B 결과를 machine/human review 시트로 변환하는 스크립트
- `runs/`
  - baseline/candidate machine eval 실행 결과
  - 기본 물리 저장소는 `~/.khub/eval/knowledgeos/runs`이고, repo의 `eval/knowledgeos/runs/`는 호환성을 위해 symlink로 유지할 수 있다
- `review/`
  - 사람이 최종 판정하는 human review 시트

운영 원칙:

- GPT/Codex는 `runs/`의 machine eval CSV에서 `pred_*`만 채운다.
- 사람은 `review/`의 human review CSV에서 `final_*`만 채운다.
- gate는 최종적으로 `review/` 아래 human review 파일을 기준으로 실행한다.
- `report_eval.py`의 Stage A를 먼저 통과한 raw run만 judged 대상으로 올린다. 즉 `retrieval health -> judged CSV -> human final_*` 순서를 고정한다.
- eval 수집은 `scripts/collect_memory_router_eval.py --profile off-control|on-control|candidate-v6`로 세 가지 비교군을 만들 수 있다.
- Stage B 보조 지표는 raw run 단계에서도 `answerable_rate`, `non_substantive_top1_rate`, `temporal_grounded_rate`를 본다.
- `queries/knowledgeos_hard_regression_pack_v0.csv`는 refusal, vault hub noise, web temporal no-result, temporal no-grounding을 고정 추적하는 hard pack이다.

repo 바깥으로 옮기려면:

```bash
python eval/knowledgeos/scripts/externalize_runs.py --json
```

현재 주력 자산:

- canonical query set:
  - `queries/knowledgeos_eval_queries_100_v1.csv`
- embedding promotion starter set:
  - `queries/knowledgeos_embedding_eval_queries_20_v1.csv`
- embedding promotion canonical gate set:
  - `queries/knowledgeos_embedding_eval_queries_30_v1.csv`
- embedding promotion review templates:
  - `templates/knowledgeos_embedding_machine_review_template.csv`
  - `templates/knowledgeos_embedding_human_review_template.csv`
- machine eval prompt:
  - `prompts/knowledgeos_machine_eval_prompt.md`
- multi-model rubric:
  - `rubrics/knowledgeos_multi_model_eval_rubric_v0.md`
- external evaluator runbook:
  - `prompts/knowledgeos_external_eval_runbook_v0.md`
- priority review schema:
  - `prompts/knowledgeos_multi_model_priority_review_v0.md`
- answer-loop system diagnosis prompt:
  - `prompts/answer_loop_system_eval_prompt_v1.md`
- answer-loop system diagnosis rubric:
  - `rubrics/answer_loop_system_diagnosis_rubric_v1.md`
- stage report script:
  - `scripts/report_eval.py`
- embedding review builder:
  - `scripts/build_embedding_review_sheet.py`
- embedding gold/hard-negative seed builder:
  - `scripts/build_embedding_label_seed_sheet.py`
- ask-v2 canonical query set:
  - `queries/knowledgeos_ask_v2_eval_queries_v1.csv`
- paper default-family query set:
  - `queries/paper_default_eval_queries_v1.csv`
- ask-v2 collector:
  - `scripts/collect_ask_v2_eval.py`
- ask-v2 review templates:
  - `templates/knowledgeos_ask_v2_machine_review_template.csv`
  - `templates/knowledgeos_ask_v2_human_review_template.csv`
- user-facing answer query set:
  - `queries/user_answer_eval_queries_v1.csv`
- answer-loop optimize starter subset:
  - `queries/user_answer_optimize_seed_queries_v1.csv`
- user-facing answer collector:
  - `scripts/collect_user_answer_eval.py`
- user-facing answer review builder:
  - `scripts/build_user_answer_review_sheet.py`

source-quality hard gate 운영 규칙:
- `source-quality`는 2026-04-21부터 local daily 운영 루프에서 hard gate로 승격되었다.
- canonical check command:
  - `python eval/knowledgeos/scripts/check_source_quality_hard_gate.py --runs-root eval/knowledgeos/runs --json`
- daily runner에서 enforce:
  - `python scripts/run_daily_source_quality.py --skip-if-local-date-already-covered --enforce-hard-gate --json`
- 통과 기준은 최신 observation report 기준이다:
  - `decision == ready_for_hard_gate_review`
  - blockers 없음
  - `run_count >= required_runs`
  - paper/vault/web `route_correctness == 1.0`
  - vault `stale_citation_rate == 0.0`
  - paper/vault/web `legacy_runtime_rate == 0.0`
  - paper/vault/web `capability_missing_rate == 0.0`
- PR 필수 CI로는 아직 승격하지 않는다. 현재 승격 범위는 persistent local run history를 가진 운영 daily loop이고, remote CI 승격은 run history restore/cache 정책이 따로 필요하다.

source-quality detail observation 운영 규칙:
- 세부 품질 지표는 2026-04-21부터 별도 observation report로 자동화한다.
- canonical command:
  - `python eval/knowledgeos/scripts/report_source_quality_detail_observation.py --runs-root eval/knowledgeos/runs --required-runs 7`
- latest reports:
  - `runs/reports/source_quality_detail_observation_latest.json`
  - `runs/reports/source_quality_detail_observation_latest.md`
- 현재 후보 지표:
  - paper `paper_citation_correctness >= 1.0`
  - vault `vault_abstention_correctness >= 1.0`
  - web `web_recency_violation <= 0.0`
- vault abstention coverage starts from the missing-path row in `queries/vault_default_eval_queries_v1.csv`.
- 승격 조건:
  - base source-quality hard gate가 먼저 통과해야 한다.
  - 각 후보 지표가 `required_runs`만큼 numeric point를 가져야 한다.
  - 모든 후보 지표가 full window 동안 threshold를 통과해야 한다.
- 현재 이 층은 관찰 전용이다. `ready_for_detail_gate_review`가 안정적으로 나오면 별도 change로 hard gate 승격한다.

live retrieval-span eval 운영 규칙:
- 목적은 deterministic CI fixture가 아니라 실제 장기 로컬 DB에서 "이 질문이 기대 source/span을 찾는가"를 operator가 주간/수동으로 확인하는 것이다.
- canonical command:
  - `python eval/knowledgeos/scripts/check_live_retrieval_span_eval.py --cases eval/knowledgeos/queries/live_retrieval_span_eval_cases.local.json --out-json eval/knowledgeos/runs/reports/live_retrieval_span_latest.json --out-md eval/knowledgeos/runs/reports/live_retrieval_span_latest.md --fail-on-insufficient --json`
- `live_retrieval_span_eval_cases.local.json`은 개인 장기 corpus의 source id/path를 담을 수 있으므로 git ignore 대상이다. 시작점은 `templates/live_retrieval_span_eval_cases.template.json`을 복사해서 채운다.
- 이 gate는 live DB와 로컬 index 상태에 의존하므로 required PR CI에는 넣지 않는다. CI는 `fixtures/retrieval_span_golden_cases.json`만 사용한다.
- 통과 기준은 기본적으로 모든 evaluable case가 expected source id를 top-K 안에서 찾고, 지정된 expected text term이 해당 retrieved text와 overlap 되는 것이다. `expected_evidence_role=retrieval_signal_only` case는 signal row가 citation-grade evidence로 오인되지 않는지 확인한다.

embedding promotion 운영 규칙:
- 기본값 승격은 항상 현재 기본 임베딩 대비 pairwise로만 판단한다.
- `gold_doc_ids` / `hard_negative_doc_ids`가 비어 있으면 machine draft는 만들 수 있어도 `recommended=true`는 차단된다.
- `source-hit`은 보조 지표로만 유지하고, 최종 판단은 `gold relevance + hard negative + pairwise winner + 운영 비용`으로 내린다.
- 3자 비교는 `current default vs candidate` 쌍을 주 판정으로 보고, 후보 간 비교는 참고 자료로만 쓴다.
- 현재 운영 승격 대상은 `nomic-embed-text` vs `bge-m3`로 제한한다. `Qwen3-Embedding-0.6B`는 local `pplx-st` 경로의 운영 비용이 높아 연구용 비교군으로만 유지한다.
- latest candidate run:
  - `runs/knowledgeos_machine_eval_v6.csv`
- latest off-control baseline run:
  - `runs/knowledgeos_machine_eval_off_control_v6.csv`
- latest on-control baseline run:
  - `runs/knowledgeos_machine_eval_on_control_v6.csv`
- latest stage reports:
  - `runs/reports/knowledgeos_machine_eval_v6__vs__knowledgeos_machine_eval_off_control_v6.md`
  - `runs/reports/knowledgeos_machine_eval_v6__vs__knowledgeos_machine_eval_on_control_v6.md`
- latest hard-pack reports:
  - `runs/reports/knowledgeos_hard_regression_pack_v6__vs__knowledgeos_hard_regression_pack_off_control_v6.md`
  - `runs/reports/knowledgeos_hard_regression_pack_v6__vs__knowledgeos_hard_regression_pack_on_control_v6.md`


ask-v2 manual gate 운영 규칙:
- 첫 단계는 자동 차단이 아니라 human-reviewed manual gate다.
- canonical source scope는 `paper`, `web`, `vault`, `project`다.
- collector는 current runtime answer payload를 읽되, ask-v2 lazy rebuild를 막는 read-only patch를 적용해 eval 수집 중 SQLite/v2 projection을 변경하지 않는다.
- canonical collection command:
  - `python eval/knowledgeos/scripts/collect_ask_v2_eval.py --out eval/knowledgeos/runs/knowledgeos_ask_v2_eval_v1.csv`
- review flow:
  - collector CSV 생성
  - 필요 시 machine draft는 `label/wrong_source/wrong_era/should_abstain` 또는 별도 `pred_*` 보조열로 작성
  - human review는 `final_label/final_wrong_source/final_wrong_era/final_should_abstain/final_notes`를 최종 판정으로 사용
- gate command:
  - `khub labs eval run --profile ask-v2 --ask-v2-csv eval/knowledgeos/runs/knowledgeos_ask_v2_eval_v1.csv --json`
- v1 priority failure buckets:
  - `project`/`paper` wrong-source
  - temporal wrong-era
  - abstention expected인데 strong answer를 낸 경우
  - weak-evidence without fallback
  - `project` lane이 구조 파일 대신 generic docs/test에 치우치는 경우
- `ask-v2` profile은 manual gate이므로 현재 `warn/pass`만 사용하고 자동 promotion block/CI hard-fail은 하지 않는다.


user-facing answer eval 운영 규칙:
- 목적은 내부 route correctness보다 "사용자가 실제로 읽는 최종 답변"의 품질을 따로 보는 것이다.
- canonical query source는 `queries/user_answer_eval_queries_v1.csv`를 사용한다.
- 기본 수집 명령:
  - `python eval/knowledgeos/scripts/collect_user_answer_eval.py --out eval/knowledgeos/runs/user_answer_eval_api_v1.csv`
- 기본 review 시트 생성:
  - `python eval/knowledgeos/scripts/build_user_answer_review_sheet.py --machine-eval eval/knowledgeos/runs/user_answer_eval_api_v1.csv --out eval/knowledgeos/review/user_answer_eval_api_v1_review.csv`
- collector 기본값은 `allow_external=true`와 `force-api-route=true`다. 즉 답변 생성은 API 경로를 우선 강제한다.
- review는 사람이 아래 축을 직접 채우는 것을 기준으로 둔다.
  - `final_label`
  - `final_groundedness`
  - `final_usefulness`
  - `final_readability`
  - `final_source_accuracy`
  - `final_should_abstain`
  - `final_notes`
- 이 세트는 현재 runtime gate를 자동 차단하기 위한 CI 자산이 아니라, 사용자 체감 답변 품질을 모니터링하는 운영용 review loop다.


answer-loop eval 운영 규칙:
- 목적은 같은 retrieval evidence 위에서 post-retrieval answer backend를 공정 비교하고, judged failure를 Codex patch loop로 다시 밀어보는 것이다.
- production `khub ask` behavior는 바꾸지 않는다. canonical path는 `khub labs eval answer-loop ...`이고, hidden compatibility alias `khub eval answer-loop ...`도 당분간 유지된다.
- internal rerun orchestration도 같은 canonical path를 기준으로 둔다. `collect`는 Python core에서 유지하되, post-collect `judge -> summarize -> autofix` lane은 얇은 executor 경계를 통해 CLI replay나 future satellite executor로 바꿀 수 있게 유지한다.
- 내부 `autofix`에서도 patch 엔진은 별도 executor 경계로 분리되어 있다. 현재는 같은 repo 안에서 Codex patch + targeted verification을 수행하지만, 이 경계 덕분에 이후 external patch runner로 옮길 때 failure-card/brief orchestration을 다시 설계할 필요는 줄어든다.
- frozen packet rule:
  - retrieval/evidence assembly는 질의당 한 번만 수행한다.
  - 그 결과는 `knowledge-hub.answer-eval.packet.v1` packet으로 고정한다.
  - 각 answer backend는 이 frozen packet만 보고 답변한다. v1에서는 backend별 자체 retrieval을 허용하지 않는다.
  - packet collection은 가능한 경우 repo의 domain-aware retrieval pipeline과 query frame을 그대로 재사용한다. 즉 answer-loop packet도 raw search convenience path 대신 paper/web/vault query planning을 따른다.
- v1 backend matrix:
  - `codex_mcp`
  - `openai_gpt5_mini`
  - `ollama_gemma4`
- canonical commands:
  - `khub labs eval answer-loop collect --answer-backend codex_mcp --answer-backend openai_gpt5_mini --answer-backend ollama_gemma4 --json`
  - `khub labs eval answer-loop judge --collect-manifest eval/knowledgeos/runs/answer_loop/latest/answer_loop_collect_manifest.json --json`
  - `khub labs eval answer-loop summarize --judge-manifest eval/knowledgeos/runs/answer_loop/latest/answer_loop_judge_manifest.json --json`
  - `khub labs eval answer-loop autofix --judge-manifest ... --repo-path . --json`
  - `khub labs eval answer-loop run --max-attempts 3 --repo-path . --json`
  - `khub labs eval answer-loop optimize --queries eval/knowledgeos/queries/user_answer_optimize_seed_queries_v1.csv --candidate-count 1 --max-rounds 1 --daily-token-budget-estimate 25000 --judge-budget-ratio 0.10 --generator-model gpt-5.4 --judge-model gpt-5.4 --json`
- optimize starter-set policy:
  - start from `queries/user_answer_optimize_seed_queries_v1.csv` instead of the full user-answer sheet because current Codex transport latency is large enough that the full sheet behaves like a batch job, not an interactive smoke
  - the seed keeps one `paper` row and one `project` row whose frozen packets currently carry real evidence
  - rows that freeze to empty packets should be treated as retrieval backlog, not answer-optimizer targets
- collector output:
  - machine review CSV
  - JSONL packet/result trace for packet + backend debugging
- judge semantics:
  - GPT-family judge 기본값은 `gpt-5`
  - judge는 `pred_*`와 `judge_provider/judge_model`만 채운다
  - `final_*`는 계속 human review 전용이다
  - comparison answer rows는 judge가 query-understanding / retrieval-fit / assembly / overclaim risk를 보되, 출력은 계속 `pred_*`만 유지한다
- comparison answer policy:
  - compare packet은 answer 전에 evidence audit을 하도록 유도한다
  - `target_anchor`, `direct_comparative_evidence`, `background_evidence`만 핵심 차이 본문에 사용한다
  - `task_specific_example`, `weak_indirect_evidence`는 예시나 한계 설명으로만 사용한다
- system diagnosis assets:
  - Codex/GPT 외부 진단용 프롬프트는 `prompts/answer_loop_system_eval_prompt_v1.md`
  - 판단 기준은 `rubrics/answer_loop_system_diagnosis_rubric_v1.md`
  - 첫 비교 실험 요약은 `reports/answer_loop_cnn_vit_codex_eval_2026-04-12.md`
- loop score / stop policy:
  - primary: `pred_label_score` (`good=1.0`, `partial=0.5`, `bad=0.0`)
  - guardrails: groundedness, source accuracy, abstention agreement
  - hard stop: `--max-attempts`
  - early stop: label score `+0.02` 이상 개선이 없고 guardrail 중 하나도 `+0.01` 이상 개선되지 않으면 중단
  - immediate stop: judged rows 없음, dirty-worktree policy block, patch failure, targeted verification failure
- Codex MCP prerequisites:
  - answer generation uses Codex in `read-only`
  - autofix uses Codex in `workspace-write`
  - default transport is `codex exec`, not direct MCP
  - direct MCP remains available with `KHUB_CODEX_TRANSPORT=mcp`
  - env overrides:
    - `KHUB_CODEX_TRANSPORT`
    - `KHUB_CODEX_EXEC_COMMAND`
    - `KHUB_CODEX_EXEC_ARGS`
    - `KHUB_CODEX_EXEC_TIMEOUT_SECONDS`
    - `KHUB_CODEX_MCP_COMMAND`
    - `KHUB_CODEX_MCP_ARGS`
    - `KHUB_CODEX_MCP_ENV`
    - `KHUB_CODEX_MCP_TIMEOUT_SECONDS`
  - current rationale:
    - the local Python MCP client used by `knowledge-hub` can fail on Codex-specific `codex/event` notifications in some environments
    - `codex exec` keeps the same sandbox/approval contract without that protocol compatibility risk
- v1 mutation safety:
  - dirty git worktree는 기본 차단이고 `--allow-dirty`가 있어야 patch 단계로 진입한다
  - auto-commit, auto-push, auto-PR은 하지 않는다
  - loop artifacts는 `eval/knowledgeos/runs/answer_loop/...` 아래에 남기고, 코드 수정은 현재 branch 위에서만 일어난다


paper default-family 운영 규칙:
- 기본 `paper` ask는 `concept_explainer`, `paper_lookup`, `paper_compare`, `paper_discover` 4개 family만 평가 대상으로 본다.
- `queries/paper_default_eval_queries_v1.csv`는 family별 6개씩 총 24개 질의를 담고, 각 row에 `expected_family`, `expected_top1_or_set`, `expected_answer_mode`, `allowed_fallback`를 고정한다.
- 이 세트는 CI hard-fail gate가 아니라 human-reviewed 운영 기준선이다.
- 기본 `paper_discover`는 shortlist-style answer까지만 기대하고, deep synthesis는 `khub labs paper topic-synthesize`로 별도 평가한다.
- collector는 `family_match`, `answer_mode_match`, `representative_match`, `actual_representative_selection_*`, `pred_label`, `pred_reason`까지 machine draft로 채워 concept-explainer/paper-lookup 대표 선택 drift를 먼저 걸러낸다.
- live local generation latency가 eval 수집을 막을 때는 `python eval/knowledgeos/scripts/collect_paper_default_eval.py --stub-llm --out ...`로 route/retrieval/representative-selection diagnostics만 빠르게 수집할 수 있다.
- hard gate와 live gate는 분리해서 본다.
  - hard gate:
    - `python eval/knowledgeos/scripts/collect_paper_default_eval.py --gate-mode stub_hard --out ...`
    - 전체 canonical sheet를 `stub-llm`로 돌리며 `family_match`, `answer_mode_match`, `representative_match`, `no_result`, `runtimeUsed`, `latency_ms`만 gate 기준으로 본다.
  - live gate:
    - `python eval/knowledgeos/scripts/collect_paper_default_eval.py --gate-mode live_smoke --out ...`
    - `CNN을 쉽게 설명해줘`, `AlexNet 논문 요약해줘` 두 개만 실제 generation 경로로 확인한다.
- collector CSV는 additive diagnostics로 `gate_mode`와 `timeout_flag`를 함께 기록한다.
  - planner fallback이 low-confidence 또는 `no_result`에서만 동작하는지
  - answer-generation provider가 unavailable이어도 route diagnostics가 유지되는지


web default-family 운영 규칙:
- 기본 `web` ask는 `reference_explainer`, `temporal_update`, `relation_explainer`, `source_disambiguation` 4개 family를 query-understanding 기준선으로 본다.
- canonical query source는 기존 `queries/knowledgeos_ask_v2_eval_queries_v1.csv`의 `source=web` rows를 그대로 재사용한다.
- collector 명령:
  - hard gate: `python eval/knowledgeos/scripts/collect_web_default_eval.py --gate-mode stub_hard --out ...`
  - live smoke: `python eval/knowledgeos/scripts/collect_web_default_eval.py --gate-mode live_smoke --out ...`
- web hard gate는 현재 retrieval coverage보다 route contract를 먼저 본다.
  - 기준 열: `actual_family`, `family_match`, `actual_answer_mode`, `answer_mode_match`, `no_result`, `actual_runtime_used`, `latency_ms`, `timeout_flag`
  - `observed_at`만 약하게 있는 temporal query는 strong latest answer 대신 guard/abstain으로 판정하는 것이 허용된다.
  - 현재 collector는 nonempty-but-unrelated evidence를 `good`으로 남기지 않는다. query의 technical anchor가 top web sources에 직접 보이지 않으면 `weak_evidence_directness`로 낮춘다.
  - `route_match_no_result`는 여전히 허용되는 중간 상태지만, 이제는 route correctness와 evidence directness를 분리해서 본다.
  - collector는 관련 없는 source title만으로 `good`을 주지 않는다. 현재 web query의 기술 앵커가 top source title/url에 직접 보이지 않으면 machine draft는 `weak_evidence_directness`로 내려간다.
- web collector CSV는 additive diagnostics로 아래를 함께 기록한다.
  - `query_frame_family`
  - `evidence_policy_key`
  - `resolved_source_scope_applied`
  - `temporal_signals_applied`
  - `reference_source_applied`
  - `watchlist_scope_applied`


youtube default-family 운영 규칙:
- 기본 `youtube` ask는 `video_lookup`, `video_explainer`, `section_lookup`, `timestamp_lookup` 4개 family를 query-understanding 기준선으로 본다.
- canonical query source는 `queries/youtube_default_eval_queries_v1.csv`를 사용한다.
- collector 명령:
  - hard gate: `python eval/knowledgeos/scripts/collect_youtube_default_eval.py --gate-mode stub_hard --out ...`
  - live smoke: `python eval/knowledgeos/scripts/collect_youtube_default_eval.py --gate-mode live_smoke --out ...`
- youtube hard gate는 우선 route contract를 본다.
  - 기준 열: `actual_family`, `family_match`, `actual_answer_mode`, `answer_mode_match`, `no_result`, `actual_runtime_used`, `latency_ms`, `timeout_flag`
  - additive diagnostics: `video_scope_applied`, `chapter_scope_applied`
  - 현재 canonical sheet는 로컬 코퍼스에 있는 실제 YouTube note/card에 의존하므로, repo 테스트는 이 local record를 직접 가정하지 않고 fixture/mock 중심으로 유지한다.


vault default-family 운영 규칙:
- 기본 `vault` ask는 `note_lookup`, `vault_explainer`, `vault_compare`, `vault_timeline` 4개 family를 query-understanding 기준선으로 본다.
- canonical query source는 `queries/vault_default_eval_queries_v1.csv`를 사용한다.
- collector 명령:
  - hard gate: `python eval/knowledgeos/scripts/collect_vault_default_eval.py --gate-mode stub_hard --stub-llm --out ...`
  - live smoke: `python eval/knowledgeos/scripts/collect_vault_default_eval.py --gate-mode live_smoke --out ...`
- vault hard gate는 route contract와 answer mode를 먼저 본다.
  - 기준 열: `actual_family`, `family_match`, `actual_answer_mode`, `answer_mode_match`, `no_result`, `actual_runtime_used`, `latency_ms`, `timeout_flag`
  - additive diagnostics: `vault_scope_applied`, `temporal_signals_applied`
  - explicit markdown path / note id scope가 있으면 `note_lookup`으로 고정하고, `최근`/`최신`/`업데이트`형 질문은 `vault_timeline`으로 분리한다.
- 현재 canonical live smoke는 아래 두 질문으로 고정한다.
  - `RAG 검색 품질을 떨어뜨리는 가장 흔한 원인은 무엇인가?`
  - `최근 retrieval pipeline에서 temporal route가 추가된 이유는 무엇인가?`


embedding seed 운영 규칙:
- canonical 30-query CSV는 최종 gold를 직접 저장하지 않고, 먼저 seed review CSV를 만든다.
- seed builder는 `current runtime search + existing pairwise machine review + frequent distractor frequency`를 합쳐 `suggested_gold_doc_ids` / `suggested_hard_negative_doc_ids`를 만든다.
- `expected_primary_source`가 명시된 질의에서 source-match 후보가 없으면 gold를 비우고 hard-negative만 제안한다.
- seed 출력은 review 시작점일 뿐이며, gate는 `final_gold_doc_ids` / `final_hard_negative_doc_ids`가 확정되기 전에는 승격 판단에 사용하지 않는다.
- latest seed review:
  - `review/knowledgeos_embedding_eval_queries_30_v1_seed_review.csv`
