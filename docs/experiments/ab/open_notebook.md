# Open Notebook Comparison

## 목적

bounded notebook workspace 측면에서 `knowledge-hub`의 notebook bridge/workbench와 `Open Notebook`을 비교합니다.

## 비교 대상

- topic bundle 품질
- multi-source synthesis
- source scope control
- workbench usability

## 사용 질문셋

- `q11` ~ `q15`

## 우리 쪽 기록 예시

```bash
khub notebook topic-preview "attention residuals"
khub notebook topic-sync "retrieval safety evaluation"
```

## 볼 것

- source leakage 여부
- notebook session usefulness
- topic bundle completeness
- local bounded workbench 필요성

## 흡수 후보

- notebook-style workflow
- bundle selection UX
- bounded multi-doc synthesis 패턴

## 반영 위치

- `notebook_bridge`
- `notebook_workbench`

## 결론

- repo clone과 docs review는 완료했다.
- 현재 repo snapshot은 `runs/ab/open_notebook/repo`에 보관했다.
- 빠른 시작 경로는 `docker compose` 전제인데, 이 머신에는 `docker compose`와 `docker-compose`가 모두 없어 실제 runtime A/B는 아직 막혀 있다.
- 문서상 유의미한 패턴은 이미 보인다:
  - `Notebook / Sources / Notes`의 3계층 컨테이너 모델
  - `Ask`와 `Chat`을 분리한 UX
  - `full / summary / excluded` 식의 명시적 context control
- 따라서 지금 단계의 가치 있는 흡수 후보는 UI 전체가 아니라 `bounded source scope`와 `notebook mental model` 쪽이다.
