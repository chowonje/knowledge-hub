# Scheduled Ops Report Automation

사용 목적:
- 기존 `ko-note-report`와 `rag-report`를 한 번에 실행
- 로컬 JSON snapshot을 canonical artifact로 저장
- Obsidian 운영 요약 노트를 갱신

수동 실행:
- `khub labs ops report-run --json`
- `khub labs ops report-run --run-id <ko_note_run_id> --recent-runs 20 --days 7 --limit 100`

기본 동작:
- `--run-id`를 생략하면 최신 `ko_note_run`을 사용합니다.
- snapshot은 `~/.khub/ops-reports/<timestamp>/` 아래에 저장됩니다.
- 기본 노트 경로는 `<vault>/LearningHub/ops/Knowledge Hub Ops Report.md` 입니다.
- note writeback은 managed block만 갱신하고, 블록 밖 사용자 내용은 보존합니다.
- `recommendedActions`는 별도 ops action queue에 적재되며 `pending -> acked -> resolved`로 추적됩니다.

생성 artifact:
- `ops-report.json`
- `ko-note-report.json`
- `rag-report.json`

운영 queue:
- `khub ops-action-list --json`
- `khub ops-action-ack --action-id <id>`
- `khub ops-action-execute --action-id <id> --json`
- `khub ops-action-receipts --action-id <id> --json`
- `khub ops-action-resolve --action-id <id>`

safe execution 규칙:
- v1은 allowlisted action만 실행합니다.
- 허용 예:
  - `rag-report`
  - `crawl ko-note-review-list`
  - `crawl ko-note-remediate --strategy section`
  - `config list`
- 비허용 예:
  - `ko-note-apply`
  - approval/reject 명령
  - `--strategy full`
  - arbitrary shell 실행

receipt:
- 실행 시도마다 별도 receipt가 남습니다.
- CLI는 동기 실행 후 최종 receipt를 바로 반환합니다.
- MCP는 기존 `mcp_job` 인프라를 재사용하고, queued 시점 `started` receipt와 완료 후 최종 상태를 연결합니다.
- report-generated recommended-action queue items are not auto-`resolved` by execution. (The separate `Agent Gateway v2` agent writeback lane may auto-resolve its own `agent_repo_writeback_request` items after a successful approved execution.)

실행 스크립트:
- `scripts/run_ops_report.sh`

launchd:
- workstation-specific launchd plists are local-only and are not tracked in this public repository.
- point a local plist at `scripts/run_ops_report.sh` if scheduled local runs are needed.

기본 스케줄:
- 매일 `09:30` 로컬 시간

로그:
- `~/.khub/logs/ops_report.log`
- `~/.khub/logs/ops_report.err.log`
- `~/.khub/logs/launchd.ops_report.out.log`
- `~/.khub/logs/launchd.ops_report.err.log`

Codex app automation 메모:
- repo 코드가 scheduler business logic를 모두 소유합니다.
- Codex automation은 `khub labs ops report-run --json`만 호출하면 됩니다.
- 권장 prompt 요약:
  - latest ko-note run 기준으로 combined ops report를 실행
  - JSON snapshot path와 Obsidian ops note path를 확인
  - warning/critical alert와 상위 recommended action만 inbox에 요약

주의:
- 이 자동화는 report-only 입니다.
- remediation/apply/approval/answer rewrite를 자동 실행하지 않습니다.
