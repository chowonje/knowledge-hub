# Daily Source Quality Automation

사용 목적:
- 로컬에서 매일 `source-quality` battery를 다시 실행
- trend / readiness / observation 최신 보고서를 갱신
- detail-quality observation 최신 보고서를 갱신
- 최신 observation verdict가 hard gate 기준을 통과하지 못하면 자동 실행을 실패시킴
- 필요한 경우 결과를 `docs/status/` 와 `worklog/`에 docs-only consumer로 기록

수동 실행:
- `python scripts/run_daily_source_quality.py --json`
- `python scripts/run_daily_source_quality.py --enforce-hard-gate --json`
- `python eval/knowledgeos/scripts/check_source_quality_hard_gate.py --runs-root eval/knowledgeos/runs --json`
- `python eval/knowledgeos/scripts/report_source_quality_detail_observation.py --runs-root eval/knowledgeos/runs --required-runs 7`
- `python scripts/run_daily_source_quality.py --writeback --json`
- `python scripts/run_daily_source_quality.py --writeback --apply-writeback --include-workspace --actor won --json`
- `WRITEBACK=1 APPLY_WRITEBACK=1 scripts/run_daily_source_quality.sh`

기본 동작:
- runner는 CI nightly와 같은 순서로 `run_source_quality_battery -> report_source_quality_trend -> report_legacy_runtime_readiness -> report_source_quality_observation -> report_source_quality_detail_observation`를 실행합니다.
- `--enforce-hard-gate`를 주면 observation report 생성 후 `check_source_quality_hard_gate.py`를 실행하고, 통과하지 못하면 non-zero로 종료합니다.
- 출력 summary는 최소한 아래 항목을 다시 보여줍니다.
  - `decision`
  - `blockers`
  - `paper/vault/web route_correctness`
  - `vault stale_citation_rate`
- `--writeback`를 주면 기존 `scripts/run_agent_docs_writeback_loop.py`를 재사용합니다.
- `--apply-writeback`를 같이 주면 preview가 아니라 실제 `ack -> execute`까지 수행합니다.
- launchd wrapper 기본값은 observation/report refresh만 자동 실행합니다. docs/status + worklog writeback은 `WRITEBACK=1`과 `APPLY_WRITEBACK=1`을 명시적으로 켰을 때만 자동 실행합니다.
- repo-side와 installed launchd wrapper는 기본적으로 hard gate를 enforce합니다. 임시 진단용으로만 `ENFORCE_HARD_GATE=0`을 사용할 수 있습니다.
- repo-side와 installed launchd wrapper는 기본적으로 `--skip-if-local-date-already-covered`를 넘깁니다. 그래서 같은 로컬 날짜에 이미 한 번 실행했다면 이후 호출은 skip되고, 같은 날 수동 run과 예약 run이 둘 다 카운트되는 일을 막습니다.

hard gate 기준:
- `decision == ready_for_hard_gate_review`
- `blockers == []`
- `run_count >= required_runs`
- `paper/vault/web route_correctness >= 1.0`
- `vault stale_citation_rate <= 0.0`
- `paper/vault/web legacy_runtime_rate == 0.0`
- `paper/vault/web capability_missing_rate == 0.0`

detail observation 기준:
- 아직 hard gate가 아니라 승격 후보 관찰층입니다.
- 최신 base source-quality observation이 `ready_for_hard_gate_review`여야 합니다.
- paper `paper_citation_correctness >= 1.0`
- vault `vault_abstention_correctness >= 1.0`
- web `web_recency_violation <= 0.0`
- 각 지표는 `required_runs`만큼 numeric point가 있어야 합니다. 값이 `None`이면 승격 준비가 아니라 coverage 보강 대상으로 봅니다.

writeback goal:
- 기본 goal은 최신 observation summary를 바탕으로 자동 생성됩니다.
- 결과적으로 consumer prompt 안에 `decision`, `blockers`, `paper/vault/web route_correctness`, `vault stale_citation_rate`가 같이 들어갑니다.

실행 스크립트:
- `scripts/run_daily_source_quality.py`
- `scripts/run_daily_source_quality.sh`
- `eval/knowledgeos/scripts/check_source_quality_hard_gate.py`
- `eval/knowledgeos/scripts/report_source_quality_detail_observation.py`
- installed launchd helper: `~/.khub/bin/run_daily_source_quality_launchd.sh`

launchd:
- workstation-specific launchd plists are local-only and are not tracked in this public repository.
- point a local plist at `~/.khub/bin/run_daily_source_quality_launchd.sh` if scheduled local runs are needed.

기본 스케줄:
- 매일 `10:00` 로컬 시간

로그:
- `~/.khub/logs/daily_source_quality.log`
- `~/.khub/logs/daily_source_quality.err.log`
- `~/.khub/logs/launchd.daily_source_quality.out.log`
- `~/.khub/logs/launchd.daily_source_quality.err.log`

안전 규칙:
- source-quality는 이제 로컬 daily 운영 루프에서 hard gate로 승격되었습니다. 관찰 리포트는 계속 생성하지만, gate 기준을 통과하지 못하면 자동 실행은 실패합니다.
- PR 필수 CI로는 아직 승격하지 않습니다. 현재 승격 범위는 persistent local run history를 가진 운영 daily loop입니다.
- docs writeback은 여전히 기존 docs-only consumer 범위(`docs/status/`, `worklog/`)에 묶입니다.
- predicted target이 그 범위를 벗어나면 writeback은 실패 폐쇄됩니다.
- launchd wrapper는 lock으로 중복 실행을 막습니다.
