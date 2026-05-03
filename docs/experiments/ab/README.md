# A/B Experiments

`knowledge-hub`의 코어를 흐리지 않으면서 외부 프로젝트의 좋은 패턴만 흡수하기 위한 비교 실험 모음입니다.

원칙:
- 코드 복사보다 기능/패턴 비교를 우선한다.
- 코어 지표 개선이 확인된 경우에만 흡수한다.
- raw 실행 결과는 `runs/ab/` 아래에 두고 커밋하지 않는다.
- 기본 물리 저장소는 `~/.khub/runs/ab`이고, repo의 `runs/ab/`는 호환성을 위해 symlink로 유지할 수 있다.
- 문서, 질문셋, 평가 기준만 repo에 남긴다.

## 1차 비교군

- `Khoj`: 개인 지식베이스 검색 품질 비교
- `PaperQA`: 논문 질의응답 품질 비교
- `Open Notebook`: bounded notebook/workbench workflow 비교
- `GraphRAG`: graph/ontology 기반 retrieval boost 비교

## 추가 비교군

- `Chroma Context-1 Data Gen`: synthetic multi-hop retrieval/eval task generation 비교

## 공통 평가 기준

- `hit@3`
- `precision@5`
- `top1_relevant`
- `answer_good`
- `citation_good`
- `duplicate_bad`
- `latency_ms`
- `operator_notes`

## 공통 질문셋

고정 질문셋은 `docs/experiments/ab/query_set_v1.md`를 사용합니다.

## 평가표

수동 기록은 `docs/experiments/ab/eval_sheet_template.csv`를 사용합니다.

## 상태 추적

현재 실험 진행 상태와 blocker는 `docs/experiments/ab/status.md`에 누적합니다.

## 결과 저장 규칙

- 문서:
  - `docs/experiments/ab/*.md`
- 실행 보조 스크립트:
  - `scripts/ab/*.py`
  - 현재 추가된 runner: `run_paperqa_ab.py`, `run_khoj_ab.py`, `run_embedding_model_ab.py`
  - embedding A/B의 기본 로컬 기준선은 `ollama/nomic-embed-text:latest`, 실험군은 `ollama/bge-m3:latest`를 우선 비교한다.
  - `pplx-st` 기반 `BAAI/bge-m3`는 대안 비교군으로 남기되, product 기본값 승격 판단은 먼저 Ollama 경로 결과로 내린다.
- raw 결과:
  - `runs/ab/khoj/`
  - `runs/ab/paperqa/`
  - `runs/ab/open_notebook/`
  - `runs/ab/graphrag/`
  - `runs/ab/embedding_models/`

repo 바깥으로 옮기려면:

```bash
python scripts/ab/externalize_ab_artifacts.py --json
```

예:

```text
runs/ab/khoj/knowledge_hub_q01.json
runs/ab/khoj/khoj_q01.json
runs/ab/paperqa/knowledge_hub_q01.json
runs/ab/paperqa/paperqa_q01.json
```

## 흡수 판단 규칙

- `core`에 반영:
  - retrieval/answer 품질이 분명히 좋아지고 운영 복잡도가 과하지 않을 때
- `supporting`으로 반영:
  - 특정 workflow에는 좋지만 코어 전체에 필수는 아닐 때
- `labs`에만 유지:
  - 흥미롭지만 사용 빈도나 ROI가 낮을 때
- `반영 안 함`:
  - 품질 개선이 작거나 현재 아키텍처와 충돌이 클 때

## 1차 실행 순서

1. `Khoj`
2. `PaperQA`
3. `Open Notebook`
4. `GraphRAG`
