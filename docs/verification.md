# Verification

검증은 정적 검사, CPU unit test, configuration dry-run과 실제 GPU/distributed 실행을 서로 다른 증거로 취급합니다.
한 단계의 성공이 다른 단계의 성공을 의미하지 않습니다.

## Repository-level checks

Repository root에서 CPU test와 Python compile을 실행합니다.

```bash
python -m pytest -q
python -m compileall -q backends experiments observability
```

Shell script syntax는 다음처럼 검사합니다.

```bash
for script in backends/*/scripts/*.sh observability/scripts/*.sh; do
  bash -n "$script" || exit 1
done
```

## 실행 판정

Dry-run 성공은 setup과 experiment validation 및 plan 생성만 증명합니다.
실제 run은 controller manifest와 모든 rank log를 확인해야 합니다.
Training run은 expected step 수, finite loss/gradient, 정상 process 종료와 expected checkpoint 또는 adapter를 확인합니다.
Adapter reload와 checkpoint resume은 각각 별도 process와 별도 stage의 증거로 확인합니다.

## 결과 해석의 한계

짧은 smoke는 model quality, long-run convergence와 일반적인 throughput을 증명하지 않습니다.
Checkpoint save timing은 crash durability나 fsync 보장을 증명하지 않습니다.
Synthetic observability trace는 실제 LLM trace를 대신하지 않습니다.
특정 model과 topology에서 통과한 설정을 다른 model에 일반화하지 않습니다.

## Historical evidence

통합 실행에서 생성한 raw manifest, log, summary와 measurement는 [verification archive](verification/integration-20260908/)에 보관합니다.
Archive는 당시 commit과 hardware 조건에 묶인 기록입니다.
새로운 결과를 추가할 때는 실행 commit, model·dataset revision, topology, seed, command와 실패 여부를 함께 기록합니다.
