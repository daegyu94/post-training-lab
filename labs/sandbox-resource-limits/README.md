# Lab: Sandbox Resource Limits (cgroup v2)

`systemd-run --user --scope`로 process 단위 cgroup v2 CPU·memory 상한을 확인합니다.
단일 사용자 환경에서 별도 container runtime 없이 검증합니다.
**CPU quota는 학습에서 동작했지만 이 DGX Spark의 CUDA 할당에는 memory limit이 적용되지 않았습니다.**

## Prerequisites

systemd 255+, cgroup v2의 `cpu`·`memory` controller 활성화(`cat /sys/fs/cgroup/cgroup.controllers`로 확인), root 권한 불필요.
학습 예제는 준비된 `experiments/trl/single-node-smoke.json` 환경(Qwen2.5-0.5B, DDP, 1노드)을 사용합니다.
아래 학습·CUDA 예제에는 GPU가 필요하며 CPU quota 자체는 CPU workload로도 확인할 수 있습니다.

## Run

### CPU quota

Spark 노드의 `backends/trl`에서 `MAX_STEPS=30`·`STAGE=train`과 아래 설명한 환경변수를 export한 뒤 실행합니다.
`OUTPUT_DIR`·`OBSERVATORY_RUN_ID`는 실행마다 새 이름을 씁니다.

```bash
# baseline: 제한 없음
OUTPUT_DIR=/mnt/post-training/trl/cpu-quota-baseline OBSERVATORY_RUN_ID=cpu-quota-baseline \
  ./scripts/run_spark_cluster.sh

# 4 core / 1 core / 0.5 core
for quota in 400 100 50; do
  systemd-run --user --scope -p "CPUQuota=${quota}%" -- env \
    OUTPUT_DIR="/mnt/post-training/trl/cpu-quota-$quota" OBSERVATORY_RUN_ID="cpu-quota-$quota" \
    ./scripts/run_spark_cluster.sh
done
```

나머지 `MODEL_ID`, `MODEL_REVISION`, `DATASET_ID`, `DATASET_REVISION`, `DATA_DIR`, `MODEL_DIR`, `DISTRIBUTED_BACKEND=ddp`, `FINETUNING_MODE=lora`, `OPTIMIZER=adamw`, `NNODES=1`, `NPROC_PER_NODE=1`, `NODE_RANK=0`, `MASTER_ADDR=127.0.0.1`, `PYTHON`은 `experiments/trl/single-node-smoke.json`과 `setups/spark/local.json`의 값을 그대로 씁니다.

### Memory limit

Host·CUDA 할당을 비교하며 swap으로 초과분이 흡수되지 않도록 `MemorySwapMax=0`을 함께 지정합니다.
CUDA 예제의 `python3`는 준비된 Torch 가상환경을 사용합니다.

```bash
# host 메모리: 300MB 제한에서 500MB 할당 → 실패해야 함 (강제됨)
systemd-run --user --scope -p MemoryMax=300M -p MemorySwapMax=0 -- \
  python3 -c 'bytearray(500*1024*1024)'

# CUDA 메모리: 1GB 제한에서 2GB 할당 → 이 클러스터에서는 성공함 (강제 안 됨)
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- \
  python3 -c 'import torch; torch.zeros(2_000_000_000, dtype=torch.uint8, device="cuda"); torch.cuda.synchronize(); print("allocated ok")'
```

## Results

2026-09-09에 spark1 단독(DDP, Qwen2.5-0.5B, 30 step)에서 측정했습니다.

| CPUQuota | train_seconds | steps/s | 결과 |
| --- | ---: | ---: | --- |
| 없음 (baseline) | 4.27 | 7.82 | 통과 |
| 400% (4 core) | 4.26 | 7.85 | 통과, baseline과 유의미한 차이 없음 |
| 100% (1 core) | 4.20 | 7.90 | 통과, baseline과 유의미한 차이 없음 |
| 50% (0.5 core) | 8.28 | 4.00 | 통과, steps/s 약 절반 — quota 축소에 비례해 저하 |

네 실행 모두 `validation.training_result_verified: true`로 finite loss·optimizer step을 확인했고, 0.5 core에서 처리율이 약 절반으로 줄었습니다.

300MB 상한의 500MB host 할당은 실패(exit 255), 1GB 상한의 2GB CUDA 할당은 `allocated ok`로 성공했습니다.
관측된 차이의 커널·드라이버 원인은 확인하지 않았습니다.

## Cleanup

실행 종료와 로그 보존을 확인한 뒤 해당 run 출력만 삭제합니다.

```bash
for run in cpu-quota-baseline cpu-quota-400 cpu-quota-100 cpu-quota-50; do
  rm -rf "/mnt/post-training/trl/$run"
done
```

`systemd-run --scope`는 명령이 끝나면 scope도 정리되므로 별도 정리가 필요 없습니다.

## Limitations

**CPU quota**는 spark1 단일 노드 0.5B smoke에서만 검증했습니다.
30B NVMe DeepSpeed 같은 CPU-heavier workload(AIO thread, tokenization)에서는 임계점이 더 높은 quota에서 나타날 수 있으며 그 값은 아직 측정하지 않았습니다.

DGX Spark GB10은 CPU·GPU가 물리 memory를 공유하며, 위 실험의 `memory.max`는 host 할당에만 적용됐습니다.
따라서 이 환경에서 학습 memory ceiling·controlled OOM 비교는 하지 않습니다.
CUDA 할당의 cgroup 계측 지원을 확인하거나 discrete GPU 노드에서 별도 검증해야 합니다.
