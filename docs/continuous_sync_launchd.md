# Continuous Sync Launchd

사용 목적:
- `continuous_sources` 최신 글을 매일 수집
- T9에 raw/normalized/indexed 저장
- ko-note 생성 후 Obsidian Vault 반영

실행 스크립트:
- `scripts/run_continuous_sync.sh`

launchd 템플릿:
- `ops/launchd/com.won.knowledge-hub.continuous-sync.plist`

실행 표면:
- 사용자-facing 기준으로는 `khub crawl continuous-sync`
- launchd 스크립트는 현재 내부 launcher로 `python -m knowledge_hub.interfaces.cli.main crawl continuous-sync`를 호출

기본 스케줄:
- 매일 `09:10` 로컬 시간

로그:
- `~/.khub/logs/continuous_sync.log`
- `~/.khub/logs/continuous_sync.err.log`
- `~/.khub/logs/launchd.continuous_sync.out.log`
- `~/.khub/logs/launchd.continuous_sync.err.log`

주의:
- `/Volumes/T9`가 마운트되지 않으면 실행을 건너뜁니다.
- 동일 스크립트가 이미 실행 중이면 lock으로 중복 실행을 막습니다.
