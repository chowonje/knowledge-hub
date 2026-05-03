## 목적

첨부된 CSV는 KnowledgeOS가 생성한 `paper-memory` 카드와 해당 논문의 원문 발췌를 함께 담고 있다.
각 row마다 GPT Pro가 먼저 `source_excerpt`만 보고 독립적으로 핵심 내용을 요약한 뒤, 기존 `paper-memory` 카드와 비교 평가해줘.

## 입력 파일

- `knowledgeos_paper_memory_vs_gptpro_sample_12_v1.csv`

각 row 주요 컬럼:

- `paper_id`, `title`, `year`, `field`
- `paper_core`, `method_core`, `evidence_core`, `limitations`
- `source_excerpt`
- `quality_flag`, `issue_score`, `latex_core`, `text_starts_latex`, `generic_limitation`

## 평가 규칙

1. 반드시 `source_excerpt` 기준으로 먼저 독립 판단한다.
2. `paper-memory` 카드가 원문보다 과장하거나, 없는 내용을 넣었으면 감점한다.
3. 너무 일반적이거나 빈약해서 핵심을 못 담으면 감점한다.
4. `method_core`와 `evidence_core`는 특히 엄격하게 본다.
5. 원문 발췌만으로 limitation이 명확하지 않다면, 카드가 신중하게 표현했는지 본다.
6. `latex_core`, `text_starts_latex`가 `1`이어도, 카드가 실제로 원문 의미를 잘 복구했으면 그 점은 인정한다.

## 라벨 정의

- `good`
  - 핵심 주장, 방법, 근거가 원문 발췌와 대체로 일치하고, 정보 밀도도 충분함
- `partial`
  - 일부는 맞지만 빠진 축이 있거나, 너무 일반적이거나, 다소 과장/불명확함
- `bad`
  - 핵심이 틀리거나, 엉뚱한 내용을 넣었거나, 너무 빈약해서 실사용 가치가 낮음

## 출력 형식

CSV만 반환해줘. 헤더는 정확히 아래를 사용:

```csv
paper_id,title,overall_label,core_fidelity,method_fidelity,evidence_fidelity,limitations_fidelity,coverage,overclaim_risk,short_reason,gpt_pro_core_summary,gpt_pro_method_summary,gpt_pro_evidence_summary,gpt_pro_limitations_summary
```

필드 설명:

- `overall_label`: `good|partial|bad`
- `core_fidelity`: `high|medium|low`
- `method_fidelity`: `high|medium|low`
- `evidence_fidelity`: `high|medium|low`
- `limitations_fidelity`: `high|medium|low`
- `coverage`: `high|medium|low`
- `overclaim_risk`: `high|medium|low`
- `short_reason`: 한 문장
- `gpt_pro_*_summary`: 각 항목당 1~2문장

## 중요

- 설명문, 표, 마크다운 블록 없이 최종 CSV만 반환해줘.
- 각 row는 하나의 논문만 평가한다.
- `paper-memory` 카드가 부정확해도, 네 독립 요약은 최대한 원문 발췌에 충실하게 작성해줘.
