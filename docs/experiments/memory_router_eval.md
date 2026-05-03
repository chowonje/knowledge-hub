# Memory Router Eval v1

`memory-router-v1`는 `ask` 경로의 additive memory-first prefilter가 baseline 대비 실제로 나아지는지 보는 수동 평가 루프입니다.

평가 목적:
- 기존 retrieval-core, document-memory, paper-memory non-regression 확인
- `memory_route_mode=prefilter`가 `top1 good-hit rate`를 실제로 올리는지 확인
- temporal query에서 `wrong-era hit`를 줄이는지 확인

## Query Set

- query set: `docs/experiments/memory_router_eval_queries_v1.txt`
- source-aware query set: `docs/experiments/memory_router_eval_queries_v1.csv`
- temporal paper/web seed set: `docs/experiments/memory_router_eval_queries_temporal_paper_web_v1.csv`
- canonical 100-query set: `eval/knowledgeos/queries/knowledgeos_eval_queries_100_v1.csv`
- machine-eval template: `eval/knowledgeos/templates/knowledgeos_machine_eval_template.csv`
- human-review template: `eval/knowledgeos/templates/knowledgeos_human_review_template.csv`
- GPT/Codex prompt: `eval/knowledgeos/prompts/knowledgeos_machine_eval_prompt.md`
- baseline sheet: `docs/experiments/memory_router_baseline.csv`
- candidate sheet: `docs/experiments/memory_router_candidate.csv`

추가 메모:
- `memory_router_eval_queries_temporal_paper_web_v1.csv`는 paper/web temporal reasoning을 더 강하게 보는 seed set입니다.
- 이 세트는 바로 operator 수집에 쓸 수 있지만, 실제 코퍼스 coverage를 보고 일부 질의는 교체/축소하는 전제를 둡니다.
- `knowledgeos_eval_queries_100_v1.csv`는 운영용 canonical query set입니다. 여기서 `source=all`은 cross-source bucket을 뜻합니다.
- 운영용 KnowledgeOS 평가 자산은 `eval/knowledgeos/README.md` 아래에 정리합니다. `docs/experiments`는 레거시/실험용 seed와 메모를 유지하는 공간으로 둡니다.

## Review Workflow

권장 운영 흐름:

1. canonical query set에서 machine-eval sheet 생성

```bash
python scripts/collect_memory_router_eval.py \
  --queries eval/knowledgeos/queries/knowledgeos_eval_queries_100_v1.csv \
  --out eval/knowledgeos/runs/knowledgeos_machine_eval.csv \
  --memory-route-mode prefilter
```

2. machine-eval sheet에서 human-review sheet 생성

```bash
python scripts/build_human_review_sheet.py \
  --machine-eval eval/knowledgeos/runs/knowledgeos_machine_eval.csv \
  --out eval/knowledgeos/review/knowledgeos_human_review.csv
```

3. GPT/Codex는 `pred_*`를 채우고, 사람은 `final_*`를 확정한다.

4. gate는 human-review sheet를 직접 읽게 한다.

```bash
khub labs eval run \
  --profile memory-router-v1 \
  --db data/knowledge.db \
  --retrieval-csv docs/eval_precision_template.csv \
  --document-memory-csv docs/experiments/document_memory_eval_template.csv \
  --paper-memory-cases tests/fixtures/paper_memory_eval/cases.json \
  --memory-router-csv eval/knowledgeos/review/knowledgeos_human_review.csv \
  --memory-router-baseline-csv eval/knowledgeos/review/knowledgeos_human_review_baseline.csv \
  --memory-router-label-col final_label \
  --memory-router-wrong-era-col final_wrong_era \
  --json
```

추천 운영 방식:
- baseline: `memory_route_mode=off`
- candidate: `memory_route_mode=prefilter`
- 둘 다 같은 query set, 같은 source policy, 같은 corpus snapshot에서 수집
- 가능하면 `.csv` query set을 써서 질의별 `source=paper|vault|web`를 고정한다. unscoped 질의는 현재 구조에서 `prefilter`가 `source_not_supported`로 빠질 수 있다.

## Labeling

각 query는 현재 템플릿에서 `rank=1`만 기본으로 둡니다.

- `label`
  - `good`: 질문 핵심을 직접 다룸
  - `partial`: 관련은 있지만 질문 핵심이 비껴감
  - `bad`: 사실상 다른 주제
- `no_result`
  - `1`: 결과가 없었거나 평가할 top1이 없음
  - `0`: top1이 존재
- `temporal_query`
  - `1`: 최신성, 업데이트, before/after, changed since 같은 시간성 질문
  - `0`: 일반 설명/구현 질문
- `wrong_era`
  - `1`: temporal query인데 top1이 명백히 오래된 버전, 이전 시대, stale explanation에 묶임
  - `0`: temporal miss가 아님

주의:
- `wrong_era`는 `temporal_query=1`일 때만 채웁니다.
- `no_result=1`이면 `label`은 비워도 됩니다.

## Gate 실행

```bash
khub labs eval run \
  --profile memory-router-v1 \
  --db data/knowledge.db \
  --retrieval-csv docs/eval_precision_template.csv \
  --document-memory-csv docs/experiments/document_memory_eval_template.csv \
  --paper-memory-cases tests/fixtures/paper_memory_eval/cases.json \
  --memory-router-csv docs/experiments/memory_router_candidate.csv \
  --memory-router-baseline-csv docs/experiments/memory_router_baseline.csv \
  --json
```

현재 gate 기준:
- `top1GoodRate` baseline 대비 `+5pp` 이상
- `noResultRate` 악화 `+2pp` 이내
- temporal query의 `wrongEraHitRate` `50%` 이상 감소

## Interpretation

- `pass`
  - 기존 supporting eval이 유지되고 memory-first delta도 의미 있음
- `warn`
  - 정보는 있지만 temporal set이 부족하거나 baseline이 약해서 승격 근거가 충분치 않음
- `fail`
  - 기본 on 승격 금지
