# Document Memory Evaluation Loop

Last updated: 2026-03-19

`khub labs memory`는 아직 실험 계층이라, 코어 `search/ask`에 연결하기 전에 별도 평가 루프가 필요합니다.

## 목적

- `MemoryUnit` 기반 summary-first retrieval이 실제로 해석 가능한 결과를 내는지 본다
- `contextHeader`, `documentThesis`, `retrievalSignals`가 사람이 판정할 때 도움이 되는지 본다
- 코어 연결 여부를 작은 수동 평가셋으로 먼저 판단한다

## 기본 절차

1. 대상 문서들에 대해 `khub labs memory build ...`를 실행한다.
2. 질의셋으로 결과를 뽑는다.
3. CSV에 `label`과 `notes`를 수동으로 채운다.
4. 반복해서 자주 실패하는 query/type만 다음 구현 대상으로 올린다.

## 쿼리 파일

기본 질의셋:
- `docs/research/document-memory-eval-queries-v1.txt`

## 실행

```bash
python scripts/eval_document_memory.py \
  --db data/knowledge.db \
  --queries docs/research/document-memory-eval-queries-v1.txt \
  --out docs/experiments/document_memory_eval_template.csv \
  --top-k 3
```

## 출력

CSV에는 다음이 들어간다.

- `query`
- `rank`
- `document_id`
- `document_title`
- `source_type`
- `matched_unit_title`
- `matched_unit_type`
- `matched_summary`
- `matched_segment_anchor`
- `matched_segment_titles`
- `matched_segment_text`
- `document_thesis`
- `related_unit_titles`
- `strategy`
- `title_match`
- `source_type_boost`
- `generic_title_penalty`
- `placeholder_penalty`
- `label`
- `notes`

`label`은 예를 들어 `good`, `partial`, `bad`처럼 간단히 채우면 된다.

## 해석 기준

- `matched_unit_type`가 질문 의도와 맞는가
- `matched_summary`와 `document_thesis`가 왜 이 문서가 선택됐는지 설명해 주는가
- `matched_segment_titles`와 `matched_segment_text`가 한 개 chunk보다 더 자연스러운 읽기 단위를 주는가
- `related_unit_titles`가 같은 문서 안에서 다음 탐색 행동을 돕는가
- section hierarchy가 결과 해석에 실제로 도움이 되는가
- `title_match / source_type_boost / generic_title_penalty / placeholder_penalty`가 랭킹 해석에 도움이 되는가

## 비목표

- 아직 코어 `khub search` / `khub ask`와 직접 성능 비교하지 않는다
- 대규모 자동 메트릭을 먼저 만들지 않는다
- parser 교체 실험은 여기서 하지 않는다

## First seed-run note

첫 실데이터 seed run은 다음 파일로 남긴다.

- query set: `docs/experiments/document_memory_eval_queries_seed_v1.txt`
- output: `docs/experiments/document_memory_eval_template.csv`

초기 관찰:

- heading이 잘 잡힌 vault note는 section-level 결과가 비교적 해석 가능하다
- paper 쪽은 아직 `pending_summary` 또는 메타데이터 중심 요약 유닛이 쉽게 상위에 올라온다
- 다음 개선은 parser 교체보다 먼저 `metadata-heavy summary`를 labs memory retrieval에서 약하게 만드는 것이다

이후 보강:

- `matchedSegment`가 추가되었으므로, 이제는 `matched_summary`뿐 아니라 stitched segment가 실제로 더 읽기 좋은지 함께 판정해야 한다
