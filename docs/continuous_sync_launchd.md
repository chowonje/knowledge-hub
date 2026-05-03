# Continuous Sync Launchd

사용 목적:
- `continuous_sources` 최신 글을 매일 수집
- T9에 raw/normalized/indexed 저장
- ko-note 생성 후 Obsidian Vault 반영

실행 스크립트:
- `scripts/run_continuous_sync.sh`

launchd 템플릿:
- `ops/launchd/com.example.knowledge-hub.continuous-sync.plist`

실행 표면:
- 사용자-facing 기준으로는 `khub crawl continuous-sync`
- launchd 스크립트는 현재 내부 launcher로 `python -m knowledge_hub.interfaces.cli.main crawl continuous-sync`를 호출
- 공유 브랜치의 plist는 예시 템플릿입니다. 설치할 때 repo 경로, 로그 경로, `KHUB_CONTINUOUS_SYNC_ROOT`를 로컬 값으로 치환하세요.

기본 스케줄:
- 매일 `09:10` 로컬 시간

로그:
- `~/.khub/logs/continuous_sync.log`
- `~/.khub/logs/continuous_sync.err.log`
- `~/.khub/logs/launchd.continuous_sync.out.log`
- `~/.khub/logs/launchd.continuous_sync.err.log`

주의:
- `KHUB_CONTINUOUS_SYNC_ROOT`가 비어 있거나 마운트되지 않으면 실행을 건너뜁니다.
- 동일 스크립트가 이미 실행 중이면 lock으로 중복 실행을 막습니다.
- `.env` 로드는 기본 비활성화입니다. 필요하면 `KHUB_CONTINUOUS_SYNC_LOAD_ENV=1`로 명시하세요.
- `--apply`, `--allow-external`은 기본 비활성화입니다. 필요하면 각각 `KHUB_CONTINUOUS_SYNC_APPLY=1`, `KHUB_CONTINUOUS_SYNC_ALLOW_EXTERNAL=1`로 명시하세요.
