# Knowledge Hub CLI 메모

`knowledge-hub`의 실제 `khub` 커맨드 트리를 기준으로 정리한 메모용 문서입니다.

기준:
- 기본 제품 surface와 `labs` surface를 분리
- 자주 쓰는 커맨드와 전체 인벤토리를 함께 정리
- 실제 옵션은 항상 `--help`로 최종 확인

## 빠른 확인

```bash
khub --help
khub labs --help
khub <command> --help
khub <command> <subcommand> --help
```

예:

```bash
khub add --help
khub paper --help
khub labs crawl --help
khub labs paper --help
```

## 매일 자주 쓰는 커맨드

```bash
khub doctor
khub status
khub add "https://example.com/guide" --topic "rag"
khub add "https://youtu.be/<video-id>" --topic "agents"
khub add "retrieval augmented generation" --type paper -n 3
khub search "주제"
khub ask "질문"
khub agent context "작업 목표" --repo-path .
khub paper list
khub index
```

## 기본 Top-Level Help

```text
khub add
khub agent
khub ask
khub config
khub doctor
khub index
khub init
khub labs
khub paper
khub search
khub status
```

기본 `khub --help`는 representative core loop를 우선 노출합니다. `khub add`가 URL/YouTube/논문 URL/논문 검색의 기본 intake facade입니다. 아래 command들은 여전히 직접 실행 가능하지만 default top-level help에서는 숨겨져 있습니다.

## Direct But Hidden Top-Level

```text
khub crawl
khub discover
khub dinger
khub eval
khub explore
khub health
khub math-memory
khub mcp
khub os
khub paper-memory
khub setup
khub vault
khub vector-compare
khub vector-restore
```

`khub eval`은 hidden compatibility alias이고, canonical eval surface는 `khub labs eval`입니다.

## 기본 Surface

### `khub add`

```text
khub add <source>
```

용도:
- 웹 URL, YouTube URL, 논문 URL, 논문 검색어를 하나의 직관적인 intake 명령으로 처리
- `auto` route가 YouTube, paper URL/arXiv id, 일반 web URL, paper query를 구분
- 기존 `khub crawl ingest`, `khub labs crawl youtube-ingest`, `khub paper import-csv`, `khub discover` 기능을 얇게 감싸며 기존 명령을 제거하지 않음

예:

```bash
khub add "https://example.com/guide" --topic "rag"
khub add "https://youtu.be/<video-id>" --topic "agents"
khub add "https://arxiv.org/abs/2401.00001"
khub add "retrieval augmented generation" --type paper -n 3
khub add "https://example.com/paper.pdf" --type web --topic "paper-notes"
```

### `khub agent`

```text
khub agent context
khub agent run
khub agent writeback-request
```

기본 `khub agent --help`는 gateway-oriented subcommand만 노출합니다.

용도:
- `context`: repo + 지식 문맥 조립
- `run`: gateway-facing agent/foundry bridge 실행 엔벨로프
- `writeback-request`: `Agent Gateway v2`의 approval-gated repo-local writeback lane 진입점. 현재 first-consumer 안전 범위는 **docs-only**이며 (`docs/adr/`, `docs/status/`, `reviews/`, `worklog/`), dry-run plan을 기반으로 pending request를 만들고 advisory `writebackPreview`로 허용된 문서 대상만 예측해 노출한 뒤, `khub labs ops action-ack -> action-execute`로 좁은 execution lane을 탄다. 성공 실행은 agent queue item을 자동 `resolved`로 닫는다.

예:

```bash
khub agent writeback-request "Refactor the RAG fallback flow" --repo-path . --json
khub labs ops action-list --scope agent --json
khub labs ops action-ack --action-id <id> --actor cli-user
khub labs ops action-execute --action-id <id> --actor cli-user --json
```

### `khub labs foundry`

```text
khub labs foundry sync
khub labs foundry discover
khub labs foundry discover-validate
khub labs foundry conflict-list
khub labs foundry conflict-apply
khub labs foundry conflict-reject
```

용도:
- foundry dual-write / connector / operator maintenance 경로
- `agent` 기본 help에서 숨겨야 하는 운영성 command를 canonical하게 모아둔 group
- 구현도 `knowledge_hub.interfaces.cli.commands.foundry_cmd`로 분리되어 gateway module과 ownership을 나눈다
- 기존 `khub agent sync|discover|discover-validate|foundry-conflict-*`는 compatibility alias로만 유지

예:

```bash
khub labs foundry sync --source all --json
khub labs foundry discover --feature daily_coach --json
khub labs foundry discover-validate --input discover.json --json
khub labs foundry conflict-list --json
```

### `khub ask`

```text
khub ask
```

용도:
- grounded answer 생성
- paper source는 기본적으로 4개 family만 사용한다: `concept_explainer`, `paper_lookup`, `paper_compare`, `paper_discover`
- web source도 기본적으로 4개 family만 사용한다: `reference_explainer`, `temporal_update`, `relation_explainer`, `source_disambiguation`
- 기본은 기존 retrieval 경로를 유지하고, `--memory-route-mode compat`는 ask path의 현재 additive memory-prefilter 동작을 켠다
- `--memory-route-mode on`은 retrieval-only strict memory-first v1이고, verifier coupling은 아직 없다
- `--memory-route-mode off`는 현재 practical invariant만 고정한다: memory ranking influence는 없지만 strict `no injection`까지는 아직 아니다
- 기존 `prefilter`는 deprecated alias로 남아 있고 `compat`로 정규화된다
- 기존 `--paper-memory-mode`도 동일하게 `off|compat|on|prefilter`를 받으며, `prefilter`는 deprecated 호환 alias다
- `--answer-route`는 `auto|local|api|codex`를 받는다. `codex`는 main ask runtime에서 `codex_mcp` backend를 강제 요청하는 override이며, `--allow-external`이 꺼져 있거나 Codex runtime readiness가 실패하면 기존 route resolver로 경고와 함께 fallback된다.
- config에서 `routing.llm.tasks.rag_answer.preferred_backend: codex_mcp`를 두면 explicit override가 없을 때도 Codex를 policy-gated preferred backend로 사용할 수 있다. 이 경우에도 retrieval/evidence/policy authority는 기존 Python runtime이 유지한다.
- `khub ask --json`은 이제 `answerRouteRequested`와 별도로 `answerRouteApplied`, `answerProviderApplied`, `answerModelApplied`를 넣어 실제 적용된 route/provider/model을 바로 확인할 수 있다. 텍스트 출력도 같은 정보를 한 줄로 보여준다.
- `khub ask --json`은 `memoryRoute`(요청/effective mode와 alias 여부), `memoryPrefilter`(실제 retrieval 개입 결과), `paperMemoryPrefilter`(paper-source prefilter 결과)를 분리해 노출한다.
- `khub ask`는 명시 옵션이 없으면 configured summarization provider 기준으로 `allowExternal` 기본값을 정하고, MCP `ask_knowledge`는 계속 local-only(`allow_external=false`) contract를 유지한다.

예:

```bash
khub ask "Transformer의 핵심 아이디어는?"
khub ask "RAG implementation의 핵심 tradeoff는?" --json
khub ask "최근 업데이트된 RAG benchmark 차이" --source paper --memory-route-mode compat --json
khub ask "트랜스포머를 대체할 차세대 아키텍처 논문들을 찾아서 정리해줘" --source paper --json
khub ask "최근 벡터 검색 품질 개선 글은 rerank를 어떤 역할로 설명하나?" --source web --json
khub ask "web card v2에서 version grounding이 필요한 이유는 무엇인가?" --source web --json
khub ask "오늘 바뀐 gateway shell-hardening 변경을 정리해줘" --answer-route codex --allow-external --json
```

메모:
- `concept_explainer`: 개념 설명 질문은 개념 자체를 먼저 설명하고, 대표 논문 1편은 예시/전환점으로만 연결한다
- `paper_lookup`: 특정 논문 요약/초록/기여 질문은 단일 논문 scope를 유지한다
- `paper_compare`: 비교 질문은 최소 2편이 잡히지 않으면 비교 불가로 응답한다
- `paper_discover`: `논문들/찾아줘/정리해줘` 같은 주제형 prompt는 shortlist+간단 정리에 머물고 상위 1편으로 바로 좁히지 않는다
- `reference_explainer`: `guide/reference/overview/정의/설명` 계열 web 질문은 stable reference source를 우선한다
- `temporal_update`: `latest/update/changed/최근/업데이트` 계열 web 질문은 version/date/observed grounding이 약하면 강한 최신 답을 피한다
- `relation_explainer`: web 문서/claim/ontology 연결 질문은 claim/section 보조를 쓰되 raw/document-memory 검증을 유지한다
- `source_disambiguation`: reference article vs latest feed 같은 source-class 선택 질문은 한쪽 근거만 있다고 단정하지 않는다
- 깊은 multi-paper synthesis는 기본 `ask`가 아니라 `khub labs paper topic-synthesize ...`가 담당한다
- `--json` payload는 paper에서 `paperFamily`, `queryPlan`, `representativePaper`, `plannerFallback`, `familyRouteDiagnostics`를 유지하고, cross-source 공통으로 `queryFrame`, `evidencePolicy`, `familyRouteDiagnostics`, `retrievalObjectsAvailable`, `retrievalObjectsUsed`, `representativeRole`를 포함한다
- web route diagnostics는 additive로 `temporalSignalsApplied`, `referenceSourceApplied`, `watchlistScopeApplied`를 포함한다
- 내부 구현은 `knowledge_hub.ai`의 범용 engine과 `knowledge_hub.domain.ai_papers`의 paper 전용 규칙으로 분리되며, 기존 `knowledge_hub.ai.paper_query_plan` / `claim_cards` import는 호환 shim으로만 남는다
- web 쪽 기본 query understanding은 `knowledge_hub.domain.web_knowledge`가 맡고, explicit URL/host/title이 있으면 retrieval 전에 hard scope를 건다
- `concept_explainer`의 default route는 아직 legacy hybrid이지만, paper path에서만 bounded fan-out policy를 쓴다
  - expanded queries: `base + 최대 2 planned terms + 최대 1 rescue query`
  - lexical forms: 최대 2개
  - representative scoped extra search: resolved paper id 1개까지만

### `khub labs paper memory-batch`

```text
khub labs paper memory-batch prepare
khub labs paper memory-batch submit
khub labs paper memory-batch status
khub labs paper memory-batch apply
```

용도:
- OpenAI Batch API로 `paper-memory`를 대량 재생성할 때 쓰는 운영 경로
- `prepare`는 로컬에서 compact extraction packet + `requests.jsonl` + `manifest.json`만 생성
- `submit/status/apply`는 `--allow-external`이 필요하고, batch 상태와 결과 파일을 남긴 뒤 SQLite `paper_memory_cards`에 반영

예:

```bash
khub labs paper memory-batch prepare --limit 20 --model gpt-5.4 --json
khub labs paper memory-batch submit --manifest /path/to/manifest.json --allow-external --json
khub labs paper memory-batch status --manifest /path/to/manifest.json --allow-external --json
khub labs paper memory-batch apply --manifest /path/to/manifest.json --allow-external --json
```

### `khub config`

```text
khub config get
khub config list
khub config path
khub config providers
khub config set
```

예:

```bash
khub config list
khub config get embedding.provider
khub config set embedding.provider ollama
khub config providers --models
```

### `khub crawl`

```text
khub crawl collect
khub crawl ingest
khub crawl ko-note-apply
khub crawl ko-note-generate
khub crawl ko-note-status
khub crawl resume
khub crawl run
khub crawl status
```

용도:
- 웹 수집
- staged ko-note 생성/반영

예:

```bash
khub crawl collect --url-file ./urls.txt --topic "ai-trends" --apply
khub crawl ingest --url "https://example.com" --topic "rag" --index
khub crawl status
khub crawl run
```

### `khub dinger`

```text
khub dinger ingest
khub dinger ask
khub dinger capture
khub dinger capture-process
khub dinger file
khub dinger lint
khub dinger recent
```

용도:
- 엔진 이름을 몰라도 쓰는 단순 personal knowledge facade
- `ingest`: paper query, web URL, youtube URL을 목적어 기준으로 ingest
- `ask`: 기존 grounded ask runtime을 유지한 채 단순 surface로 노출
- `capture`: 브라우저/클립퍼가 보낼 수 있는 web capture intake surface. 현재는 runtime-local queue packet을 남기고 `captureId` + queued ack를 반환하며, Obsidian projection write는 하지 않음
- `capture-process`: temp-runtime queue packet을 `normalized -> filed -> linked_to_os|failed`로 소비하는 local processor. queue/runtime sidecar가 운영 read-model이며, broad live vault smoke 대신 temp-runtime / temp-vault 조합으로 검증한다
- 현재 operator-facing capture surface는 `capture status | list | show | cleanup | requeue | retry`다. `capture status`는 기존 read-model + cleanup scan 위에서 aggregate counts와 즉시 취할 수 있는 `requeue|retry|cleanup` 액션 힌트를 보여주고, `capture requeue`는 schema-backed recovery command며, `capture cleanup`은 application cleanup policy를 노출하는 operator command다. 다만 cleanup이 canonical delete를 뜻하는 것은 아니며, runtime cleanup policy 범위에서만 동작한다.
- `file`: 질문 결과나 메모를 managed Dinger page로 filing하고 `Index.md` / `Log.md`를 갱신
- `lint`: staged ko-note + paper completeness 같은 현재 runtime hygiene 점검
- `recent`: 최근 들어온 paper / ko-note run을 목적어 기준으로 보여줌
- envelope-only 확인으로 끝내지 말고 `stage`, `sourceRefs`, `createdAt`, projection path, `traceability` 같은 핵심 typed field까지 함께 본다
- smoke에서는 command별 runtime `status`와 flow lifecycle `stage`를 구분한다. 현재 `dinger file` / `os capture`는 구현상 `status=ok`를 유지하지만, docs-only smoke 계약에서는 각각 `stage=filed` / `stage=linked_to_os`로 본다.

운영 의미:
- orphan는 새 lifecycle `status`가 아니다. `list/show`에서 기존 `queued|processing|filed|linked_to_os|failed`를 유지한 채 `packetPresent=false`, `packetMissing=true`, `orphanedRuntime=true`, `flags`, `warnings`, `operatorAction`으로만 surfaced 된다.
- operator wording contract는 `packet snapshot` 기준으로 고정한다. recoverable orphan는 `exact packet snapshot`, unrecoverable orphan는 `no exact packet snapshot`, cleanup hint는 `requeue from the exact packet snapshot before cleanup`처럼 같은 용어만 사용한다.
- recoverable orphan는 queue packet을 복구하거나 같은 packet을 다시 queue에 두는 쪽이 우선이다. 여기서 `requeue`는 queue packet 복구이지 auto-process가 아니며, 복구 후 실제 재처리는 기존 `capture retry` 또는 `capture-process`가 담당한다.
- unrecoverable orphan는 runtime-local artifact 정리 후보다. 여기서 `cleanup`은 queue/runtime sidecar 같은 temp artifact 정리일 뿐이며, filed Dinger page, OS inbox item, source ref, canonical DB row를 지우는 canonical delete가 아니다.
- 현재 cleanup policy는 dry-run 기본, destructive path에는 explicit confirm 필요, recoverable orphan와 live queue packet은 보호 대상으로 남긴다는 의미로 고정한다. `khub dinger capture cleanup`은 이 runtime cleanup policy를 preview/apply surface로 노출할 뿐, canonical delete를 수행하는 command가 아니다.
- stale claim은 orphan subtype이 아니라 processor safety signal이다. active claim은 중복 실행 방지용이고, stale claim은 retry/processor가 reclaim할 수 있다. 이 경우에도 새 status를 늘리지 않는다.

권장 operator workflow:
1. `khub dinger capture status --json`으로 queued/failed/orphan/stale-claim aggregate counts와 즉시 필요한 `requeue|retry|cleanup` 액션을 먼저 본다.
2. `khub dinger capture list --json`으로 `failed` 또는 `packetPresent=false` 항목을 찾는다.
3. `khub dinger capture show --capture-id <id> --json`으로 packet/runtime artifact, `flags`, `warnings`, `operatorAction`을 확인한다.
4. packet이 없지만 복구 가능하면 `khub dinger capture requeue --capture-id <id> --json` 또는 exact packet snapshot 복구로 canonical queue packet만 되살린다. requeue 자체가 auto-process를 의미하지는 않는다.
5. packet을 복구할 수 없고 runtime artifact만 남았으면 `khub dinger capture cleanup --json`으로 먼저 preview한다. 실제 삭제가 필요하면 `--apply --confirm`으로 runtime cleanup policy를 실행한다.
6. cleanup은 post-close-out operator hygiene 단계이며 canonical delete가 아니다. recoverable orphan와 live queue packet은 cleanup 대상이 아니라 보호 대상으로 남는다.
7. 그 다음 실제 재처리가 필요한 항목만 기존 `khub dinger capture retry --capture-id <id> ...` 또는 `capture-process` 경로로 보낸다. 즉 `retry/process`는 live packet이 있는 항목에만 적용된다.

스모크/검증 메모:
- `dinger capture` raw packet은 입력이다. queue ack와 typed `sourceRefs`만 검증하고 raw capture를 새 canonical store로 승격하지 않는다.
- downstream smoke는 broad live vault run 대신 shared fixture + temp-runtime / temporary-vault 결과를 기준으로 한다. `captureId`, `packetPath` 또는 동등 runtime pointer, filing output pointer, os bridge trace까지만 계약으로 확인한다.
- `khub os capture --json` smoke는 project-scoped inbox item(`item.projectId`, `severity`, `state`, `sourceRefs`, `createdAt/updatedAt`)까지 확인한다.
- `khub os capture --json` smoke는 bridge 이후에도 OS가 inbox/evidence candidate까지만 다룬다는 점을 확인한다. auto-triage, task promotion, decision promotion은 이 smoke 범위가 아니다.
- authority timeout은 현재 `failed` envelope 분류/원인 기록까지만 smoke 대상으로 두고, retry나 기능 수정은 이 tranche 범위 밖으로 둔다.
- approval gate는 raw `capture cleanup --help` 자체를 별도 gate command로 보지 않는다. 대신 `tests/test_dinger_capture_cleanup.py`로 typed runtime cleanup policy를 좁게 검증한다. 즉 cleanup은 의미상 post-close-out operator hygiene 단계이지만, cleanup contract 자체는 targeted verification slice에 포함된다.

예:

```bash
khub dinger ingest --paper "large language model agent" --json
khub dinger ingest --url-file ./urls.txt --topic "rag" --stage-only --json
khub dinger ask "Transformer의 핵심 아이디어는?" --source paper --json
khub dinger capture --source-url "https://example.com/rag" --page-title "RAG Capture" --selection-text "selected text" --client "browser-clipper" --tag "rag" --json
khub dinger capture cleanup --json
khub dinger capture requeue --capture-id cap_123 --json
khub dinger capture-process --packet ~/.khub/dinger_capture_intake/queue/cap_123.json --slug decision-os --json
khub dinger file --from-json ./ask-result.json --json
```

직접 호출 가능하지만 기본 `khub dinger --help`에서는 숨겨진 운영 유틸:

```text
khub dinger capture-http
khub dinger recent
khub dinger lint
```

### `khub os`

```text
khub os capture
khub os project evidence
khub os evidence show
khub os evidence review
khub os inbox list
khub os inbox resolve
khub os inbox triage
khub os task start
khub os task block
khub os task complete
khub os task cancel
khub os decide
khub os next
```

직접 호출 가능하지만 기본 `khub os --help`에서는 숨겨진 low-level record groups:

```text
khub os goal
khub os decision
```

용도:
- review workflow는 `khub dinger capture -> khub dinger capture-process -> khub os capture -> khub os project evidence -> khub os evidence review --action explain -> approve|dismiss`까지를 먼저 분리해서 본다. shipped surface는 candidate review까지이며, 그 다음에만 `khub os inbox triage`, `khub os task ...`, `khub os decide` 같은 promotion command를 사용한다.
- `capture`는 project-scoped inbox item만 추가하고, `--from-dinger-json`은 filed/projection pointer(`relativePath` 또는 vault-relative `filePath`)가 이미 있는 filed-like Dinger 결과만 bridge하는 보조 surface로 둔다. queue-only raw `dinger capture` ack는 여기에 포함하지 않는다.
- `project evidence`는 `projectEvidence.evidenceCandidates`를 나열하는 candidate review list surface다. 기본 텍스트 출력은 candidate마다 `inbox id`, 요약, Dinger page, supporting source refs, `reuse/replay` 힌트, 다음 read-only 액션을 먼저 보여준다. task/decision promotion과 같은 mutation을 하지 않으며, approve/dismiss가 곧바로 finalize를 뜻하는 surface로 문서화하면 안 된다.
- `khub os evidence show`는 같은 candidate를 더 자세히 읽는 read-only detail surface다. `why reused / why replayed`, matched open/resolved item summary, source refs, 다음 review command를 한 번에 보여준다.
- `khub os evidence review`는 그 read model 위에 얹힌 얇은 inspect/disposition surface다. `--action explain`은 단일 candidate를 read-only로 보여주는 explain surface이고, `approve`는 기존 `inbox triage --resolve-only`, `dismiss`는 기존 `inbox resolve`를 재사용한다.
- raw input은 여전히 input-only이며, Dinger filing은 projection-only다. OS handoff는 `SourceRef` + bridge trace를 유지한 inbox/evidence candidate 생성까지만 수행하고, 이 흐름 어디에도 새 canonical store는 추가하지 않는다.
- capture-derived review flow는 `capture -> filed -> OS candidate -> inspect/explain -> approve|dismiss` 순서를 기본으로 본다. 여기서 `capture`는 `khub dinger capture` intake, `filed`는 `khub dinger capture-process` 또는 기존 filed Dinger result, `OS candidate`는 `khub os capture` 뒤 `khub os project evidence`에 보이는 후보, `inspect/explain`은 `khub os evidence review --action explain`에 해당한다.
- `approve`/`dismiss`는 이번 tranche에서 review 결과일 뿐 task/decision finalize 동의어가 아니다. human review 이후에도 실제 promotion 또는 finalization은 여전히 explicit `inbox triage --to-task|--to-decision`, `task complete|cancel`, `decide` 같은 별도 command로 남아야 한다.

예:

```bash
khub os capture --slug "decision-os" --summary "Review bridge follow-up" --paper-id 1706.03762 --json
khub os capture --slug "decision-os" --from-dinger-json ./dinger-file-result.json --json
khub os project evidence --slug "decision-os" --json
khub os evidence show --slug "decision-os" --candidate-id inbox_123 --json
khub os evidence review --slug "decision-os" --candidate-id inbox_123 --action explain --json
khub os evidence review --slug "decision-os" --candidate-id inbox_123 --action approve --json
khub os inbox triage --item-id inbox_123 --to-task --title "Review bridge follow-up" --kind research --json
khub os task start --task-id task_123 --json
khub os decide --slug "decision-os" --kind scope --summary "Keep CLI canonical" --document-scope-id knowledge_os_definition_v1 --json
khub os next --slug "decision-os" --json
```

스모크/검증 메모:
- temp-runtime e2e smoke의 기본 경로는 `capture -> capture-process -> os capture(result)`다. 검증 대상은 queue packet, `runtime/*.normalized.json`, `runtime/*.file-result.json`, `runtime/*.os-capture-result.json`, `runtime/*.state.json`, filed Dinger page pointer, OS inbox/evidence candidate trace까지다.
- `khub os project evidence --json`는 candidate list surface smoke로 취급한다. 여기서는 `projectEvidence.evidenceCandidates`가 derived candidate로 보이는지만 확인하고, 그 candidate를 task/decision으로 승격하는 command와 섞어서 해석하지 않는다.
- `khub os evidence review --action explain --json`는 single-candidate inspect/show smoke로 취급한다. 여기서는 explanation/reason/traceability를 확인하고, approve/dismiss나 이후 promotion 결과와 섞어서 해석하지 않는다.
- queue-only raw `dinger capture` ack JSON은 direct bridge의 first-class 입력이 아니다. raw intake는 `capture-process`를 먼저 거쳐 filed/result shape를 만든 다음 OS로 handoff하는 것을 기본으로 본다.
- human review는 candidate 단계에서만 열린다. smoke가 확인하는 OS payload는 inbox/evidence candidate trace, inspect/explain 결과, explicit triage/promotion entrypoint까지이며, 어떤 approve/dismiss wording도 task/decision auto-finalize contract로 읽히면 안 된다.
- capture operator read-model은 broad live vault가 아니라 runtime sidecar를 본다. `list`는 queue/state snapshot을, `show`는 한 `captureId`의 `currentStatus`, `steps`, `osBridge`, `lastError`를, `retry`는 동일 `captureId`/`packetPath`에 대한 재실행 결과를 읽는 의미로 유지한다.
- `retry`는 non-terminal 또는 `failed` packet에만 의미가 있다. 이미 `linked_to_os`인 packet 재실행은 duplicate/replay 관측용 idempotent no-op이며, 새 Dinger page나 새 OS item을 만들면 안 된다.
- lock/claim은 runtime-local scheduling ownership일 뿐 authority 승격이 아니다. claim success/failure는 temp-runtime packet 처리권만 바꾸고, capture packet이 canonical row로 승격되거나 OS task/decision으로 자동 승격되는 의미를 가져서는 안 된다.
- duplicate/replay policy는 note-first dedupe를 따른다. filed Dinger page가 있으면 vault note marker를 우선 dedupe key로 사용하고, 열린(open) 동일 inbox item은 `linkAction=reused_existing`로 재사용한다. resolved/triaged match는 `replay` metadata만 남기고 새 inbox item을 만든다.
- `duplicateSourceRefsSkipped`는 source-ref collapse 관측치다. duplicate/replay 판단과는 분리해서 해석하며, source ref dedupe가 canonical store 추가를 정당화하지 않는다.
- capture-derived lifecycle smoke vocabulary is fixed as `captured|normalized|filed|linked_to_os|failed`.
- raw capture는 input이고 Dinger filing은 projection-only이며 OS bridge는 inbox/evidence candidate까지만 허용한다. 이 흐름에서 새 canonical store는 추가하지 않는다.
- candidate review와 task/decision promotion은 다른 단계다. 현재 slice에서는 `approve`/`dismiss`를 task/decision finalize shorthand로 다루지 않으며, human review 이후에도 실제 promotion은 `inbox triage` / `task` / `decide` command로 명시적으로 수행해야 한다.
- authority timeout은 이 tranche에서 원인 분리/분류까지만 다루며, smoke에서는 `classification_only`로 기록하고 기능 복구나 retry semantics는 검증 범위에 넣지 않는다.
- authority timeout은 원인 분리/분류용 진단으로만 취급한다. timeout 관찰이 새 workflow state나 새 canonical store를 암시하면 안 된다.
- 알려진 timeout 이슈는 authority classification envelope에만 남긴다. 이번 tranche는 timeout 원인 구분과 fixture/schema/test 최소 보강이 범위이며, timeout recovery, automatic retry, live vault smoke 확대는 비목표다.
- narrow gate note: 현재 실제 help surface에는 `khub os project evidence`와 `khub os evidence show|review`가 있지만, frozen approval gate는 여전히 capture/operator/bridge/authority subset만 본다. 이 split candidate-review surface는 semantics가 candidate review에 머물고 targeted verification slice가 별도로 합의되기 전까지 gate에 넣지 않는다.

### `khub discover`

```text
khub discover
```

용도:
- 논문 검색 -> 다운로드 -> 요약/번역 -> 인덱싱 -> 옵시디언 연결
- `--judge`로 optional paper discovery filter 사용 가능
- 기본 intake는 `khub add "topic" --type paper -n 3`이고, `discover`는 세부 discovery 옵션이 필요할 때 직접 호출하는 compatibility surface

예:

```bash
khub add "large language model agent" --type paper -n 3
khub discover "large language model agent" -n 3
khub discover "retrieval agent" -n 5 --judge
khub discover "scientific taste" -n 5 --judge --json
```

### `khub explore`

```text
khub explore author
khub explore author-papers
khub explore batch
khub explore citations
khub explore network
khub explore paper
khub explore references
```

용도:
- 개별 논문/저자/인용 네트워크 탐색

예:

```bash
khub explore paper 1706.03762
khub explore citations 1706.03762
khub explore author "Yoshua Bengio"
```

### `khub health`

```text
khub health
```

용도:
- 시스템/환경 헬스 체크

### `khub index`

```text
khub index
```

용도:
- papers / vault / vector corpus 인덱싱

예:

```bash
khub index
khub index --all
khub index --vault-all --vault-clear
khub index --vault-all --vault-clear --json
```

### `khub init`

```text
khub init
```

### `khub labs paper`

```text
khub labs paper lanes-backfill
khub labs paper lanes-review
khub labs paper lanes-sync-hubs
khub labs paper topic-synthesize
```

용도:
- paper lane/operator 워크플로
- 주제형 다논문 shortlist + synthesis
- `lanes-backfill`은 이제 AI lane 후보로 보이는 논문만 채운다. non-AI 또는 현재 6-lane taxonomy 바깥 논문은 `primary_lane`을 비운 채 남긴다.
- `lanes-backfill --force`는 `seeded` 행만 다시 판정하고, `reviewed` / `locked` lane은 덮어쓰지 않는다.

예:

```bash
khub labs paper topic-synthesize "트랜스포머를 대체할 차세대 아키텍처 논문들을 찾아 정리해줘" --json
khub labs paper topic-synthesize "state space model papers beyond transformers" --top-k 6 --candidate-limit 14 --json
```

용도:
- 초기 설정
- 고급/custom 설정 surface

### `khub doctor`

```text
khub doctor
khub doctor --json
```

용도:
- 사용자용 환경 진단 요약
- `--json`은 machine-readable public contract `knowledge-hub.doctor.result.v1`를 반환하며 `status`, `checks[]`, `nextActions[]`, `warnings[]`를 포함한다
- local-first 진단에서 허용되는 상태 집합은 `ok|blocked|degraded|needs_setup`다
- local-first profile에서 Ollama가 꺼져 있으면 `blocked/degraded`를 그대로 유지한 채 원인과 다음 명령을 보여준다
- typical recovery order: `ollama serve` -> `ollama pull <each configured model>` -> `python -m knowledge_hub.interfaces.cli.main doctor`
- vector corpus가 아직 `needs_setup`이면 현재 안내된 fix command를 따라 `khub add "AI agent" --type paper -n 1`로 최소 corpus를 만든다

### `khub mcp`

```text
khub mcp
```

용도:
- MCP 서버 실행

### `khub paper`

```text
khub paper add
khub paper board-export
khub paper build-concepts
khub paper download
khub paper embed
khub paper embed-all
khub paper evidence
khub paper feedback
khub paper info
khub paper import-csv
khub paper list
khub paper memory
khub paper normalize-concepts
khub paper related
khub paper resummary-vault
khub paper review
khub paper review-card
khub paper review-card-apply
khub paper review-card-apply-batch
khub paper review-card-plan
khub paper review-card-export
khub paper summary
khub paper summarize
khub paper summarize-all
khub paper sync-keywords
khub paper translate
khub paper translate-all
```

용도:
- 개별 논문 관리
- 번역/요약/임베딩
- 사용자용 읽기 surface (`summary`, `evidence`, `memory`, `related`)
- Obsidian `KnowledgeOS Papers`용 read-only board payload export (`board-export`)
- keyword/concept writeback
- judge calibration feedback 기록
- 빈약한 board/memory 카드 수동 품질 피드백 기록

예:

```bash
khub paper list
khub paper board-export --json
khub paper info 2401.12345
khub paper summary --paper-id 2401.12345
khub paper evidence --paper-id 2401.12345
khub paper memory --paper-id 2401.12345
khub paper related --paper-id 2401.12345
khub paper download 2401.12345
khub paper summarize 2401.12345
khub paper translate 2401.12345
khub paper embed 2401.12345
khub paper import-csv --csv ./ai_papers_curated.csv --min-priority 5 --limit 10
khub paper feedback 2401.12345 --label keep --reason "내 주제와 강하게 맞음"
khub paper review-card 2401.12345 --issue empty_method --issue empty_evidence --note "보드 카드가 너무 얕음"
khub paper review-card-plan 2401.12345
khub paper review-card-apply 2401.12345 --allow-external --provider openai --model gpt-5-nano
khub paper review-card-apply-batch --issue empty_method --json
khub paper review-card-export --issue empty_method --output ./paper_ids.txt
```

### `khub paper-memory`

```text
khub paper-memory build
khub paper-memory rebuild
khub paper-memory search
khub paper-memory show
```

용도:
- paper memory card 생성/조회/검색

예:

```bash
khub paper-memory search --query "RAG"
khub paper-memory show 2401.12345
```

## Labs Surface

### `khub labs memory`

```text
khub labs memory build
khub labs memory show
khub labs memory search
```

용도:
- note / paper / web source를 `Document -> MemoryUnit -> RetrievalChunk` 관점의 문서 메모리로 빌드
- 문서 summary와 section/block 단위 메모리 유닛을 조회
- additive `semanticUnits` payload로 `Document / Element / MemoryCard` 계약을 inspectable하게 노출
- 기본 `search`/`ask`와 별개로 summary-first 메모리 검색 실험
- 수동 평가는 `khub labs eval prepare-document-memory`로 CSV 템플릿을 생성한 뒤 `khub labs eval run`으로 gate를 본다

예:

```bash
khub labs memory build --note-id "vault:Projects/RAG.md" --json
khub labs memory build --paper-id 2603.13017 --json
khub labs memory build --canonical-url "https://example.com/rag" --json
khub labs memory show --document-id "paper:2603.13017" --json
khub labs memory search --query "retrieval evidence" --json
```

### `khub labs eval`

```text
khub labs eval prepare-document-memory
khub labs eval prepare-claim-synthesis
khub labs eval prepare-paper-summary
khub labs eval run
khub labs eval sectioncards
khub labs eval answer-loop collect
khub labs eval answer-loop judge
khub labs eval answer-loop summarize
khub labs eval answer-loop autofix
khub labs eval answer-loop run
khub labs eval answer-loop optimize
```

용도:
- retrieval / document-memory / paper-memory 평가와 user-answer answer-loop를 한 곳에서 실행
- `memory-router-v1` 프로필은 기존 retrieval-core + document-memory + paper-memory non-regression 위에 memory-first delta를 같이 본다
- optional claim-synthesis 수동 평가도 같은 eval surface에서 템플릿/게이트로 다룸
- `answer-loop`는 retrieval을 한 번만 freeze한 packet으로 고정하고, 여러 answer backend를 같은 evidence에서 비교한다
- `judge`는 `pred_*`만 채우고, `final_*`는 human review 전용으로 남긴다
- `run`은 `collect -> judge -> summarize -> autofix`를 최대 시도 수까지 반복하고 dirty worktree는 기본 차단한다
- `optimize`는 retrieval을 한 번 freeze한 뒤 Codex-only answer revision + Codex-only judge를 비파괴적으로 반복하고, judge estimated-token 사용량을 일일 예산 대비 제한된 비율로 묶은 채 review pack만 남긴다
- supporting capability 승격 판단을 같은 프레임으로 읽기 위한 내부 운영 surface
- core runtime 동작을 바꾸지 않고 `pass|warn|fail` 상태만 제공

`khub eval`은 기존 스크립트/메모 호환을 위한 hidden compatibility alias만 남기고, canonical path는 `khub labs eval`로 고정한다.

gate 의미:
- `pass`: 현재 프로파일이 정의된 기준을 통과
- `warn`: 치명적 실패는 아니지만 승격 근거로는 약함
- `fail`: 현재 상태로는 승격/신뢰 판단을 내리면 안 됨

예:

```bash
khub labs eval prepare-document-memory --db data/knowledge.db --json
khub labs eval prepare-claim-synthesis --db data/knowledge.db --paper-id 2501.00001 --paper-id 2501.00004 --json
khub labs eval prepare-paper-summary --db data/knowledge.db --paper-id 2501.00001 --json
khub labs eval run --profile memory-promotion --db data/knowledge.db --document-memory-csv docs/experiments/document_memory_eval_template.csv --json
khub labs eval run --profile memory-promotion --db data/knowledge.db --document-memory-csv docs/experiments/document_memory_eval_template.csv --claim-synthesis-csv docs/experiments/claim_synthesis_eval_template.csv --json
khub labs eval run --profile retrieval-core --retrieval-csv docs/eval_precision_template.csv --json
khub labs eval run --profile memory-router-v1 --db data/knowledge.db --retrieval-csv docs/eval_precision_template.csv --document-memory-csv docs/experiments/document_memory_eval_template.csv --paper-memory-cases tests/fixtures/paper_memory_eval/cases.json --memory-router-csv docs/experiments/memory_router_candidate.csv --memory-router-baseline-csv docs/experiments/memory_router_baseline.csv --json
khub labs eval answer-loop collect --answer-backend openai_gpt5_mini --json
khub labs eval answer-loop judge --collect-manifest eval/knowledgeos/runs/answer_loop/latest/answer_loop_collect_manifest.json --json
khub labs eval answer-loop run --answer-backend codex_mcp --answer-backend openai_gpt5_mini --answer-backend ollama_gemma4 --max-attempts 3 --repo-path . --json
khub labs eval answer-loop optimize --queries eval/knowledgeos/queries/user_answer_eval_queries_v1.csv --daily-token-budget-estimate 120000 --judge-budget-ratio 0.10 --generator-model gpt-5.4 --judge-model gpt-5.4 --json
```

### `khub search`

```text
khub search
```

용도:
- grounded retrieval 검색

예:

```bash
khub search "attention mechanism"
khub search "강화 학습" --json
```

### `khub setup`

```text
khub setup
```

용도:
- 환경/설정 보조
- `--profile local|hybrid|custom`
- `--quick`는 `local --non-interactive` 별칭
- `setup`은 설정을 저장하지만 local runtime을 자동으로 시작하지는 않는다
- local/hybrid에서 Ollama runtime이 응답하지 않으면 setup 출력에 `blocked/degraded`가 아직 정상이라는 안내와 다음 명령이 같이 나온다

### `khub status`

```text
khub status
```

용도:
- 설정, provider, corpus, runtime diagnostics 확인
- high-signal human-readable markers는 `Knowledge Hub v`, `Retrieval Runtime`, `vector corpus`다

### `khub vault`

```text
khub vault cluster-materialize
khub vault cluster-revert
khub vault organize
khub vault organize-ai
khub vault topology-build
```

용도:
- vault 정리
- cluster/topology materialization

예:

```bash
khub vault topology-build
khub vault cluster-materialize
```

## Labs Top-Level

```text
khub labs ask-graph
khub labs belief
khub labs claims
khub labs crawl
khub labs decision
khub labs eval
khub labs feature
khub labs graph
khub labs learn
khub labs ontology
khub labs ops
khub labs outcome
khub labs transform
```

## Labs Surface

### `khub labs ask-graph`

```text
khub labs ask-graph
```

용도:
- multi-step retrieval/answer planner

### `khub labs belief`

```text
khub labs belief list
khub labs belief review
khub labs belief show
khub labs belief upsert
```

### `khub labs claims`

```text
khub labs claims extract-paper
khub labs claims extract-web
khub labs claims normalize
khub labs claims compare
khub labs claims synthesize
khub labs claims pending
khub labs claims pending list
```

### `khub labs crawl`

```text
khub labs crawl benchmark
khub labs crawl domain-policy
khub labs crawl domain-policy approve
khub labs crawl domain-policy list
khub labs crawl domain-policy reject
khub labs crawl ko-note-enrich
khub labs crawl ko-note-enrich-status
khub labs crawl ko-note-reject
khub labs crawl ko-note-remediate
khub labs crawl ko-note-report
khub labs crawl ko-note-review-approve
khub labs crawl ko-note-review-list
khub labs crawl ko-note-review-reject
khub labs crawl metadata-audit
khub labs crawl pending
khub labs crawl pending apply
khub labs crawl pending list
khub labs crawl pending reject
khub labs crawl reindex-approved
```

주의:
- `continuous-sync`, `latest-build`, `reference-build`, `reference-sync`는 현재 `khub crawl ...` 기본 surface에 있습니다.

### `khub labs decision`

```text
khub labs decision create
khub labs decision list
khub labs decision review
```

### `khub labs feature`

```text
khub labs feature snapshot
khub labs feature top
```

### `khub labs graph`

```text
khub labs graph concept
khub labs graph paper
khub labs graph path
khub labs graph stats
```

### `khub labs learn`

```text
khub labs learn analyze-gaps
khub labs learn assess-template
khub labs learn grade
khub labs learn graph-build
khub labs learn graph-pending
khub labs learn graph-pending apply
khub labs learn graph-pending list
khub labs learn graph-pending reject
khub labs learn map
khub labs learn next
khub labs learn path-generate
khub labs learn quiz-generate
khub labs learn quiz-grade
khub labs learn reinforce
khub labs learn review-writeback
khub labs learn run
khub labs learn suggest-patch
```

### `khub labs ontology`

```text
khub labs ontology merge-apply
khub labs ontology merge-list
khub labs ontology merge-reject
khub labs ontology pending-apply
khub labs ontology pending-list
khub labs ontology pending-reject
khub labs ontology predicate-approve
khub labs ontology predicate-list
khub labs ontology profile
khub labs ontology profile activate
khub labs ontology profile export
khub labs ontology profile import
khub labs ontology profile list
khub labs ontology profile show
khub labs ontology proposal
khub labs ontology proposal apply
khub labs ontology proposal list
khub labs ontology proposal reject
khub labs ontology proposal submit
```

### `khub labs ops`

```text
khub labs ops action-ack
khub labs ops action-execute
khub labs ops action-list
khub labs ops action-receipts
khub labs ops action-resolve
khub labs ops rag-report
khub labs ops report-run
```

### `khub labs outcome`

```text
khub labs outcome record
khub labs outcome show
```

### `khub labs transform`

```text
khub labs transform list
khub labs transform preview
khub labs transform run
```

## 추천 사용 루틴

### 1. 검색/답변

```bash
khub status
khub search "주제"
khub ask "질문"
```

### 2. 코딩/작업 문맥

```bash
khub agent context "작업 목표" --repo-path .
```

### 3. 논문 수집

```bash
khub add "주제" --type paper -n 5
khub paper list
```

### 4. 인덱싱

```bash
khub index
khub index --vault-all --vault-clear
```

### 5. judge calibration

```bash
khub paper feedback <paper_id> --label keep --reason "..."
khub paper feedback <paper_id> --label skip --reason "..."
```

### 6. card quality triage

```bash
khub paper review-card <paper_id> --issue empty_method --issue empty_evidence --note "..."
khub paper review-card <paper_id> --issue likely_semantic_mismatch --note "제목과 카드 내용이 안 맞음"
khub paper review-card-plan <paper_id>
khub paper review-card-apply <paper_id> --dry-run
khub paper review-card-apply <paper_id> --allow-external --provider openai --model gpt-5-nano
khub paper review-card-apply-batch --paper-id-file ./paper_ids.txt --dry-run
khub paper review-card-apply-batch --issue empty_method --allow-external --provider openai --model gpt-5-nano
khub paper review-card-export --issue empty_method --output ./paper_ids.txt
```

메모:
- `review-card`와 `review-card-plan`은 현재 artifact snapshot과 누적 issue를 바탕으로 `rebuild_structured_summary`, `rebuild_paper_memory`, `refresh_concept_links`, `repair_source_content` 같은 remediation plan을 계산한다.
- `review-card-apply`는 위 plan 중 안전한 auto action만 실행한다. 현재는 `rebuild_structured_summary`와 per-paper `paper-memory` 재빌드를 실행하고, `refresh_concept_links`는 같은 run의 memory rebuild로 충족되면 skip한다.
- `review-card-apply-batch`는 `--paper-id`, `--paper-id-file`, `--issue` selector로 여러 paper를 모아 같은 safe auto remediation을 반복 실행한다. selector를 하나도 주지 않으면 전체 로그를 무턱대고 돌리지 않고 바로 실패한다.
- `review-card-apply`는 source repair가 필요한 plan(`likely_semantic_mismatch`, `latex_*`)은 자동 실행하지 않고 blocked로 남긴다.
- `review-card-export`가 만드는 newline-delimited `paper_ids.txt`는 `scripts/paper_memory_audit.py --paper-id-file ...` 와 `scripts/rebuild_memory_stores.py --paper-id-file ...` 에 바로 넣을 수 있다.

## 메모

- 기본 기능은 `khub ...`
- 비핵심/실험 기능은 `khub labs ...`
- JSON 출력이 너무 길면 평소에는 `--json` 없이 사용
- 상세 옵션은 항상 `--help`가 최종 기준
