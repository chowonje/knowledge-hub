# Khoj Comparison

## 목적

개인 지식베이스 검색 품질에서 `knowledge-hub`와 `Khoj`를 비교합니다.

## 비교 대상

- vault/note 검색 감각
- query interpretation
- top result relevance
- 중복 노출 제어

## 사용 질문셋

- `q01` ~ `q05`

## 우리 쪽 기록 예시

```bash
khub search "강화 학습" --json > runs/ab/khoj/knowledge_hub_q01.json
khub search "Transformer architecture" --json > runs/ab/khoj/knowledge_hub_q02.json
```

## 볼 것

- `hit@3`
- `precision@5`
- top-1 만족도
- 같은 문서 청크가 상위를 독점하는지

## 흡수 후보

- query expansion
- vault ranking heuristics
- related note suggestion UX

## 반영 위치

- `search`
- `retrieval_fit`
- `context_pack`

## 결론

- 실행 확인 완료
- `knowledge-hub` 기준 baseline 결과는 `runs/ab/khoj/knowledge_hub_q01.json` ~ `knowledge_hub_q05.json`에 저장했다.
- `khoj[local]`는 짧은 경로 가상환경(`/tmp/khoj-ab/.venv`)에서는 설치와 self-host 기동이 모두 가능했다.
- 이 머신에서 실제로 통과한 실행 조건은 아래와 같다.

```bash
source /tmp/khoj-ab/.venv/bin/activate
export USE_EMBEDDED_DB=true
export PGSERVER_DATA_DIR=/tmp/khoj-ab/pgdata
export KHOJ_TELEMETRY_DISABLE=true
export KHOJ_ADMIN_EMAIL=ab@example.local
export KHOJ_ADMIN_PASSWORD=khoj-ab-pass
khoj --anonymous-mode --non-interactive --host 127.0.0.1 --port 42111
```

- deep path 가상환경(`runs/ab/khoj/.venv`)에서는 embedded Postgres Unix socket path가 너무 길어 실패했고, short-path workaround가 필요했다.
- `--anonymous-mode`만으로는 충분하지 않았고, `--non-interactive` 경로에서는 admin env (`KHOJ_ADMIN_EMAIL`, `KHOJ_ADMIN_PASSWORD`)가 필요했다.
- 최종적으로 `http://127.0.0.1:42111/`에서 `GET / -> 200`, `HEAD / -> 405`를 확인해 runtime 자체는 검증했다.
- `scripts/ab/run_khoj_ab.py`를 추가해 baseline 상위 결과에서 최소 공통 corpus를 뽑고, `computer` source를 비운 뒤, `/api/content`로 markdown을 업로드하고 `q01~q05` 결과를 저장할 수 있게 했다.
- 첫 실행 결과는 아래에 남아 있다.
  - manifest: `runs/ab/khoj/khoj_corpus_manifest.json`
  - upload audit: `runs/ab/khoj/khoj_upload_response.json`
  - search outputs: `runs/ab/khoj/khoj_q01.json` ~ `runs/ab/khoj/khoj_q05.json`
- 첫 minimal-corpus run의 현재 관찰:
  - `q02 Transformer architecture`, `q04 agent retrieval`, `q05 safety evaluation`은 비교적 그럴듯한 top hit를 냈다.
  - `q01 강화 학습`은 top hit가 산만했고, `q03 RAG implementation`은 거의 맞지 않았다.
  - 즉 현재 Khoj 비교의 의미 있는 흡수 후보는 설치나 sync mechanics보다 `query interpretation`과 `ranking` behavior다.
