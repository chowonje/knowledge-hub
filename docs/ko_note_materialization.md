# 한국어 노트 Materialization

`crawl run` 이후 별도 후처리로 한국어 지식 노트를 생성합니다. 원본/raw/normalized/indexed 계약은 그대로 유지하고, Obsidian 반영은 staging 후 apply로 분리합니다.

## 기본 흐름

```bash
khub crawl ko-note-generate --latest-job --max-source-notes 20 --max-concept-notes 10
khub crawl ko-note-status --run-id <RUN_ID>
khub crawl ko-note-apply --run-id <RUN_ID>
```

기본 product 흐름은 staging 후 Obsidian에서 직접 내용을 확인한 다음 `ko-note-apply`로 반영하는 방식입니다. 승인 기반 워크플로를 유지하려면 `khub crawl ko-note-apply --run-id <RUN_ID> --only-approved`를 사용합니다.

고급 inspection/remediation/manual enrichment는 labs surface로 이동했습니다.

```bash
khub labs crawl ko-note-review-list --run-id <RUN_ID> --quality-flag all
khub labs crawl ko-note-remediate --run-id <RUN_ID> --quality-flag all --strategy section
khub labs crawl ko-note-review-approve --item-id <ITEM_ID>
khub labs crawl ko-note-enrich --run-id <RUN_ID>
```

## 생성 규칙

- source note:
  - indexed 레코드 중 상위 품질 문서만 선택
  - 한국어 제목/요약과 핵심 근거 단락을 staging에 생성
- concept note:
  - 동일 crawl job 안에서 다수 문서가 지지하는 concept만 선택
  - 여러 문서의 근거를 합쳐 통합 노트 생성

## 경로

- staging:
  - `LearningHub/ai/ko_notes/YYYY/MM/DD/<run_id>/sources`
  - `LearningHub/ai/ko_notes/YYYY/MM/DD/<run_id>/concepts`
- final:
  - `AI/AI_Papers/Web_Sources`
  - `AI/AI_Papers/Concepts`

## 병합 정책

- source note:
  - 기존 파일이 `khub_managed: true`면 덮어씀
  - 수동 파일이면 `__run_<shortid>` suffix로 분기 저장
- concept note:
  - 기존 파일이 `khub_managed: true`면 전체 갱신
  - 수동 파일이면 본문은 보존하고 `KHUB` 관리 섹션만 marker 기반 upsert

## Advanced review loop

- 각 ko note item은 `payload_json["quality"]`와 별도로 `payload_json["review"]`를 가집니다.
- `review.queue=true`인 item은 `khub labs crawl ko-note-review-list`에서 우선 확인합니다.
- `khub labs crawl ko-note-remediate --run-id ...`는 기본적으로 review queue에 남아 있는 staged item의 flagged section만 다시 보강하고, `payload_json["quality"]`와 `payload_json["review"]`를 다시 계산합니다.
- `--strategy section`이 기본값이며, `--strategy full`은 전체 re-enrich fallback입니다.
- remediation은 item을 계속 `staged`에 남겨 두고 `review.remediation.attemptCount`, `lastAttemptStatus`, `lastImproved`를 기록합니다.
- section remediation은 target section 전체를 무조건 덮어쓰지 않고, target field 내부의 weak/placeholder line만 교체하고 strong line은 그대로 보존합니다.
- remediation 메타데이터에는 `strategy`, `targetSections`, `patchedSections`, `preservedSectionsCount`, `lastPatchedLineCount`, `lastPreservedLineCount`, `recommendedStrategy`가 함께 기록됩니다.
- `khub labs crawl ko-note-review-approve --item-id ...`는 staged item을 `approved`로 올리고 review decision을 기록합니다.
- `khub labs crawl ko-note-review-reject --item-id ...`는 staged item을 `rejected`로 전환하고 review decision을 기록합니다.
- concept note는 quality가 낮아도 review로 `approved`되면 apply에서 override됩니다.
- source note는 여전히 soft gate이며, review 승인 여부와 관계없이 apply 시 warning이 함께 기록될 수 있습니다.

## 현재 기본값

- local-first 한국어화
- 외부 API fallback은 `--allow-external`일 때만 사용
- 기술 용어는 `~/.khub/ko_glossary.yaml` 우선, 없으면 내장 glossary 사용
