# Agent Instructions

## 실행 환경

- `spark-cluster`와 `spark cluster`는 모두 `spark1`, `spark2` 두 노드로 구성된 동일한 클러스터를 의미합니다. 이 표현이 나오면 아래 토폴로지와 실행 위치를 전제로 작업합니다.
- `controller`는 개발 및 실행 조율을 담당합니다. 실제 LLM 워크로드는 Spark 노드인 `spark1`, `spark2`에서 실행합니다.
- 각 노드에는 `ssh spark@spark{i}` 형식으로 접속합니다. `{i}`는 `1` 또는 `2`로 바꾸며, 실제 명령은 `ssh spark@spark1` 또는 `ssh spark@spark2`입니다.
- Spark 노드에 SSH로 접속해 명령을 실행하거나, controller에서 원격 실행 또는 분산 실행 명령을 직접 내려도 됩니다. 어느 방식이든 실제 LLM 연산은 Spark 노드에서 수행하도록 설정합니다.
- 현재 코드의 multi-node 지원 여부와 필요한 실행 환경은 별도로 확인하고, 계획된 실행 구성과 실제 검증 결과를 구분합니다.

## NFS 공유 디렉토리

- controller의 `/home/daegyu/shared` 디렉토리는 각 Spark 노드에서 NFS client를 통해 `/home/spark/shared`에 마운트하여 사용합니다.
- controller의 `/home/daegyu/shared/<relative-path>`와 각 Spark 노드의 `/home/spark/shared/<relative-path>`는 같은 공유 파일을 가리킵니다.
- 명령과 설정에 공유 파일 경로를 넣을 때는 실제로 해당 경로를 사용하는 노드의 마운트 경로를 사용합니다.

## Git 실행 위치

- branch 생성·전환, 상태·diff·log 확인, stage, commit, fetch, pull과 push를 포함한 모든 Git 명령은 controller에서만 실행합니다.
- `spark1`, `spark2`에서는 Git 명령을 실행하지 않습니다. 두 노드는 NFS를 통해 controller와 같은 worktree를 보므로 별도 Git 작업이 필요하지 않습니다.
- Spark 노드는 실제 workload와 해당 노드에서 필요한 실행 검증에만 사용합니다.

## 검증과 실행 증거

- 기본 저장소 검사는 `python -m pytest -q`, `python -m compileall -q backends experiments observability tests run_summary.py`와 `backends/*/scripts/*.sh`, `observability/scripts/*.sh`, `setups/spark/*.sh`의 `bash -n`입니다.
- 정적 검사, CPU 테스트, dry-run과 실제 GPU 실행을 구분하고 한 단계의 성공을 다른 단계의 증거로 사용하지 않습니다.
- 설정 allowlist를 바꾸면 `docs/architecture.md`와 `docs/experiments.md`의 설정 계약도 함께 확인합니다.
- `setups/spark/local.json`, 모델 가중치, 실행 산출물과 과거 run log는 커밋하지 않습니다.
