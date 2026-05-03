# Daily Eval Center Automation

사용 목적:
- `Eval Center v0`를 하루 한 번 read-only로 스냅샷
- 최신 source-quality / answer-loop / query inventory / gaps / recommendations 상태를 한 번에 저장
- 실행기와 요약기를 섞지 않고, 기존 평가 artifact 위에 얇은 상태판만 추가

핵심 원칙:
- `Eval Center`는 실행 파이프라인이 아니라 read-only summary surface입니다.
- 그래서 daily automation도 `source-quality battery`나 `answer-loop`를 직접 돌리지 않습니다.
- 기본 동작은 최신 artifact를 읽고 JSON/Markdown snapshot만 남기는 것입니다.

수동 실행:
- `python scripts/run_daily_eval_center.py --json`
- `python scripts/run_daily_eval_center.py --skip-if-local-date-already-covered --local-timezone Asia/Seoul --json`
- `scripts/run_daily_eval_center.sh`

기본 동작:
- runner는 `knowledge_hub.application.eval_center.build_eval_center_summary(...)`를 직접 호출합니다.
- 실행 결과를 아래 두 군데에 저장합니다.
  - dated snapshot: `~/.khub/eval/knowledgeos/runs/eval_center_snapshot_<timestamp>/`
  - latest alias: `~/.khub/eval/knowledgeos/runs/reports/eval_center_latest.json`
- 같은 payload의 Markdown 요약도 함께 저장합니다.
  - `eval_center_summary.md`
  - `reports/eval_center_latest.md`
- `--skip-if-local-date-already-covered`를 주면 같은 로컬 날짜에는 중복 snapshot을 만들지 않습니다.
- duplicate-day skip은 기존 latest JSON이 Eval Center schema validation을 통과할 때만 신뢰합니다. 깨진 latest artifact는 skip 근거로 쓰지 않고 새 snapshot으로 self-heal합니다.
- `--wait-for-today-source-quality-seconds <seconds>`를 주면 today's source-quality observation이 fresh가 될 때까지 read-only로 기다린 뒤 snapshot을 만듭니다. 이 옵션은 source-quality를 실행하지 않고 latest observation/report만 polling하므로 pipeline 통합은 하지 않습니다.
- launchd는 installed helper `~/.khub/bin/run_daily_eval_center_launchd.sh`를 호출합니다. 현재 macOS launchd 환경에서는 repo-side shell wrapper를 helper에서 다시 여는 방식이 실패할 수 있어, installed helper가 source-quality helper처럼 Python runner를 직접 호출합니다.
- JSON payload에는 `operatorBrief`가 포함됩니다. 이 필드는 source-quality, answer-loop, query inventory, eval maturity를 파트별로 나눠 `ran`, `problem`, `nextAction`, `findings`를 제공합니다.
- Markdown 최신본 `reports/eval_center_latest.md`는 같은 `operatorBrief`를 사람이 바로 볼 수 있는 daily brief로 렌더링합니다.
- artifact-backed path summaries include `modifiedAt`, so the daily brief can expose stale/latest ambiguity instead of making old answer-loop artifacts look current.

출력 요약:
- `status`
- `warningCount`
- `priority`
- `sourceQualityBaseDecision`
- `sourceQualityDetailDecision`
- `answerLoopStatus`
- `answerLoopRowCount`
- `answerLoopPredLabelScore`
- `gapIds`
- `operatorBrief.sections`
- `operatorBrief.findings`

주의:
- 이 자동화는 hard gate가 아닙니다.
- `warn`이나 `missing` 상태도 snapshot으로 저장합니다.
- 저장 전 Eval Center schema validation을 통과해야 합니다.
- 실패를 발견해도 자동으로 `answer-loop`, `Failure Bank`, `EvalCase`, `autofix`를 실행하지 않습니다.
- Failure Bank 동기화는 별도 명시 명령으로만 수행합니다.
  - `khub labs eval failure-bank sync --runs-root ~/.khub/eval/knowledgeos/runs --json`
  - `khub labs eval failure-bank list --json`

실행 스크립트:
- `scripts/run_daily_eval_center.py`
- `scripts/run_daily_eval_center.sh`
- installed launchd helper: `~/.khub/bin/run_daily_eval_center_launchd.sh`

launchd 템플릿:
- `ops/launchd/com.won.knowledge-hub.daily-eval-center.plist`

기본 스케줄:
- 매일 `10:05` 로컬 시간
- repo-side wrapper와 installed launchd helper는 기본적으로 source-quality freshness를 최대 `1800`초 기다립니다. source-quality가 늦게 끝나도 Eval Center가 stale 상태판을 먼저 고정하지 않게 하기 위한 race guard입니다.

로그:
- `~/.khub/logs/daily_eval_center.log`
- `~/.khub/logs/daily_eval_center.err.log`
- `~/.khub/logs/launchd.daily_eval_center.out.log`
- `~/.khub/logs/launchd.daily_eval_center.err.log`

추천 운영 방식:
- source-quality: 계속 기존 daily hard gate로 운영
- answer-loop: 필요할 때만 별도 실행
- eval center: 매일 read-only snapshot으로 현재 상태를 기록

즉:
- pipeline 통합: 하지 않음
- daily 상태판 통합: 수행

잔여 운영 리스크:
- installed helper와 repo-side wrapper는 같은 Python runner를 호출하지만 shell wrapper 로직은 중복됩니다. wrapper 동작을 바꾸는 경우 installed helper도 같이 갱신해야 합니다.
