# PaperQA Comparison

## 목적

논문 중심 질문응답에서 `knowledge-hub`와 `PaperQA`를 비교합니다.

## 비교 대상

- citation fidelity
- paper-grounded answer 품질
- wrong-paper drift 여부
- multi-paper confusion 여부

## 사용 질문셋

- `q06` ~ `q10`

## 우리 쪽 기록 예시

```bash
khub ask "이 논문의 핵심 기여는?" --source paper --json > runs/ab/paperqa/knowledge_hub_q06.json
```

## 실행 경로

- `PaperQA` 실행 runner:
  - `python scripts/ab/run_paperqa_ab.py --source <path> --citation <label> --output-dir runs/ab/paperqa --index-path runs/ab/paperqa/index_note_eval`
- 현재 1차 비교 source:
  - `AI/AI_Papers/Papers/Evaluating Collective Behaviour of Hundreds of LLM Agents.md`
- 현재 1차 비교 model stack:
  - `qwen3:14b`
  - `nomic-embed-text:latest`
  - `OPENAI_BASE_URL=http://localhost:11434/v1` (Ollama OpenAI-compatible)

## 볼 것

- `answer_good`
- `citation_good`
- unsupported claim 비율
- 특정 paper scope 유지 여부

## 흡수 후보

- citation assembly
- paper-specific retrieval narrowing
- answer structuring

## 반영 위치

- `paper`
- `ask`
- `paper-memory`

## 결론

- 2026-03-19 기준 `PaperQA + Ollama(OpenAI-compatible)` 로컬 실행은 성공했다.
- `q06` ~ `q10` 결과 파일:
  - `runs/ab/paperqa/paperqa_q06.json`
  - `runs/ab/paperqa/paperqa_q07.json`
  - `runs/ab/paperqa/paperqa_q08.json`
  - `runs/ab/paperqa/paperqa_q09.json`
  - `runs/ab/paperqa/paperqa_q10.json`
- 같은 시점의 `knowledge-hub ask` baseline은 대부분 `ReadTimeout`으로 실패해서 공정 비교가 아직 성립하지 않는다.
- 현재까지의 관찰:
  - `PaperQA`는 single-paper note source에서도 `q06` 기여 요약은 비교적 깔끔하게 답했다.
  - 하지만 `q07` 한계, `q10` 후속 연구처럼 source에 직접 쓰여 있지 않은 항목도 적극적으로 추론하는 경향이 있어 unsupported inference 위험이 있다.
  - 따라서 1차 흡수 후보는 `citation assembly`, `paper-scoped answer formatting`, `evidence budget discipline`이며, 답변 공격성 자체를 그대로 따라가는 것은 위험하다.
