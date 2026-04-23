# Repository Layout

`knowledge-hub`의 실제 실행 진입점은 `khub` CLI와 `khub-mcp`입니다. 루트 정리는 이 워크플로를 기준으로 소스, 운영 데이터, 문서, 임시 산출물을 분리하는 방향으로 유지합니다.

Canonical map surface:
- `docs/maps/README.md`
- `docs/maps/canonical-ownership-map.md`
- `docs/maps/agent-execution-map.md`
- `docs/maps/data-policy-flow-map.md`

## 핵심 실행 흐름

1. `knowledge_hub/interfaces/cli/main.py`
   `khub` CLI의 canonical 진입점입니다.
2. `knowledge_hub/interfaces/mcp/server.py`
   `khub-mcp`의 canonical MCP 진입점입니다.
3. `foundry-core/src/personal-foundry/`
   Plan/Act/Verify/Writeback 런타임과 정책 경계를 담당합니다.

## 캐노니컬 패키지 소유권

- `knowledge_hub/interfaces/`
  CLI/MCP 진입점과 명령 등록 같은 human/tool entry surface입니다.
- `knowledge_hub/application/`
  런타임 조립, task context, Foundry bridge 같은 application orchestration입니다.
- `knowledge_hub/infrastructure/`
  config, persistence, provider lookup 같은 runtime-facing infrastructure surface입니다.
- `knowledge_hub/core/`
  로컬 store/model/validator/chunking 같은 low-level primitive입니다.
- `knowledge_hub/cli/`, `knowledge_hub/mcp_server.py`
  외부 호환성을 위한 shim입니다. 내부 코드의 직접 import 대상이 아닙니다.

## 루트 디렉터리 규칙

- `knowledge_hub/`
  Python 제품 코드와 CLI/MCP 표면입니다.
- `foundry-core/`
  TypeScript 기반 개인 Foundry 런타임입니다.
- `tests/`
  Python 테스트입니다.
- `docs/`
  제품 문서, 운영 가이드, 구조 설명입니다.
- `scripts/`
  일회성 자동화와 유지보수 스크립트입니다.
- `ops/`
  launchd 같은 운영 배포 자산입니다.

## 운영 데이터

아래 경로는 생성물이지만 현재 워크플로에서 직접 참조되므로 루트에 유지합니다.

- `data/`
  SQLite, Chroma, raw web, curated watchlist 등 런타임 데이터입니다.
- `logs/`
  실행 로그입니다.
- `.khub/`
  로컬 런타임 상태와 보조 아티팩트입니다.
- `config.yaml`
  로컬 개발 설정입니다.

## 정리 원칙

- 루트에 남는 문서는 진입점 성격의 문서만 둡니다.
- CLI/상태/요약 문서는 `docs/` 하위 기능 폴더로 이동합니다.
- 로고 같은 비코드 자산은 `docs/assets/`에 둡니다.
- 캐시, 테스트 산출물, 임시 URL 목록, 재시도 로그, 잘못 생성된 경로 복사본은 루트에 두지 않습니다.
- subprocess나 운영 스크립트가 Python CLI를 직접 호출해야 하면 `knowledge_hub.interfaces.cli.main`을 기준으로 맞춥니다.
