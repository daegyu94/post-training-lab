# Verification

정적 검사, CPU 테스트, dry-run, 실제 GPU 실행은 서로 다른 증거입니다.
한 단계의 성공이 다른 단계의 성공을 의미하지 않습니다.

## Repository Checks

저장소 루트에서 CPU 테스트를 실행합니다.
최소 테스트 의존성은 pytest와 datasets이며 전체 GPU 학습 stack은 이 CPU 테스트의 필수 조건이 아닙니다.
새 테스트 환경이 필요하면 다음 명령으로 준비합니다.
명령은 패키지를 다운로드하고 별도 venv를 생성합니다.

```bash
python -m venv .venv-check
. .venv-check/bin/activate
python -m pip install 'pytest>=8,<10' 'datasets==5.0.0'
python -m pytest -q
python -m compileall -q backends experiments observability tests run_summary.py
for script in backends/*/scripts/*.sh observability/scripts/*.sh; do
  bash -n "$script" || exit 1
done
```

테스트는 mock과 작은 입력을 사용하는 회귀 검사이며 실제 GPU·NCCL·fresh installation 성공을 증명하지 않습니다.
실제 실행 전에는 [Getting Started의 실행 계획 검토](getting-started.md#check-the-execution-plan)로 설정을 점검할 수 있습니다.

## Judge a Run

| 확인 대상 | 성공 조건 | 증명하지 않는 것 |
| --- | --- | --- |
| Dry-run | 설정 검증·계획 생성·exit 0 | 원격 파일, GPU, SSH 준비 |
| Controller 실행 | manifest의 `passed`, 모든 rank exit 0 | 학습 품질 |
| 학습 | 예상 optimizer step, finite loss·gradient, 저장물 | 장기 수렴·다른 모델의 메모리 적합성 |
| Adapter reload | 별도 process에서 읽기·평가 완료 | optimizer·scheduler resume |
| Checkpoint resume | 기대 iteration과 상태 load 후 추가 step | uninterrupted run과 전체 수치 동등성 |
| Async 저장 | 필요한 shard와 pending save finalization | fsync·장애 후 durability |

백엔드 summary는 원격 출력 경로에 있고 controller에 자동 복사되지 않습니다.
모든 rank 로그와 모델·데이터 revision, 실제 선택한 입력, topology, seed, 환경 버전을 함께 확인합니다.
Sharded backend의 optimizer-step 증거를 전체 parameter checksum 검증으로 읽지 않습니다.
ZeRO-3 과거 summary의 parameter count 0은 placeholder 계측 문제이며 모델 크기 0이 아닙니다.

## Historical Evidence

| 기록 | 범위 |
| --- | --- |
| [TRL Spark](verification/trl-20260908/README.md) | 모델·백엔드별 성공, OOM·NVML·state-dict 실패 |
| [Megatron 통합 전 해설](verification/megatron-20260908.md) | 30B LoRA와 작은 모델의 제한된 checkpoint 비교 |
| [구조 통합·공통 runner](verification/integration-20260908/README.md) | 실행 commit, 환경, 수정 이력과 raw evidence |
| [반복 측정](verification/integration-20260908/benchmarks/README.md) | 실제 축소 실행안, 실패·생략과 측정 해석 |
| [통합 전 보존 위치](verification/migration/preserved-branches.json) | 과거 브랜치·worktree 보존 정보 |

기록의 원래 host·경로·수치는 당시 provenance이며 재사용 가능한 현재 설정 예제가 아닙니다.
대형 가중치·checkpoint와 일부 통합 전 로그는 Git에 포함되지 않습니다.
이번 문서 정리에서 과거 raw manifest·log·measurement의 내용은 변경하지 않았습니다.

## Documentation Review

이번 정리의 구조 변경과 실제 검사 범위는 [문서 검토 기록](verification/documentation-20260908.md)에 남깁니다.

## Known Implementation Limits

- Controller commit이 `unknown`이면 runner가 원격 commit 일치 검사를 생략합니다.
- Megatron selective recompute는 `recompute_num_layers=1`을 설정해 Bridge의 `None` 요구와 충돌합니다.
- Megatron Spark requirements의 ModelOpt stable pin과 Bridge의 rc 의존성이 달라 fresh-install 전체 성공이 검증되지 않았습니다.
- 서비스 변환 manifest는 두 Spark validator의 고정 데이터 형식과 다릅니다.
- Sharded export·resume와 parameter 계측은 백엔드별 제한이 있으므로 학습 성공과 별도 검증해야 합니다.

문서의 명령을 맞추기 위해 위 구현을 변경하지 않았습니다.
실패를 지원 불가능으로 일반화하거나 임의의 revision·버전으로 우회하지 않습니다.
