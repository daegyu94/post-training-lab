# Lab: Sandbox Resource Limits (cgroup v2)

`systemd-run --user --scope`로 학습 process를 cgroup v2 resource control 아래 실행해 CPU와 memory 상한이 실제로 강제되는지 확인합니다.
Docker나 다른 container runtime은 쓰지 않습니다 — 2노드 단일 사용자 환경이라 이미지 기반 격리보다 process 단위 resource fencing이 더 가볍고 검증하기 쉽습니다.

**CPU quota는 실제 학습 실행으로 검증했고, memory limit은 DGX Spark의 unified memory 때문에 이 클러스터에서 유효하지 않다는 것을 확인해 한계로 기록합니다.**

## Prerequisites

systemd 255+, cgroup v2의 `cpu`·`memory` controller 활성화(`cat /sys/fs/cgroup/cgroup.controllers`로 확인), root 권한 불필요.
학습 예제는 `experiments/trl/single-node-smoke.json`(Qwen2.5-0.5B, DDP, 1노드)을 그대로 쓰므로 추가 준비물이 없고 전체 실행은 노드 하나에서 1분 이내입니다.
GPU는 memory-limit 확인에만 필요하며 CPU quota 검증은 GPU 없이도 재현됩니다.

## Run

### CPU quota

각 실행은 `MAX_STEPS=30`으로 `run_spark_cluster.sh`의 `STAGE=train`을 서로 다른 `CPUQuota`로 감쌉니다.
`OUTPUT_DIR`와 `OBSERVATORY_RUN_ID`는 실행마다 새 이름을 씁니다.

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

아래 두 명령이 "plain host 메모리는 강제되지만 CUDA 메모리는 강제되지 않는다"는 핵심 관찰을 재현합니다.
`MemorySwapMax=0`이 없으면 초과분이 swap으로 흡수돼 강제가 관찰되지 않으므로 반드시 함께 지정합니다.

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

네 실행 모두 `summary-train.json`의 `validation.training_result_verified: true`로 finite loss와 optimizer step을 확인했습니다 — **throttling이 correctness를 깨지 않고 속도만 낮춘다**는 뜻입니다.
이 workload는 1 core까지는 CPU가 병목이 아니고 0.5 core에서 처음 CPU-bound 구간에 들어갑니다.

Memory limit 명령도 실제로 실행했습니다: 300MB 제한 + 500MB 할당은 즉시 실패(exit code 255), 1GB 제한 + 2GB CUDA 할당은 `allocated ok`를 출력하며 성공했습니다.
재현 가능한 관찰이며 추측이 아닙니다 — 다만 원인(NVIDIA UMA driver가 CUDA 할당을 cgroup memory controller가 추적하지 않는 경로로 잡는지)까지는 커널·드라이버 소스 없이 확정할 수 없습니다.

## Cleanup

```bash
for run in cpu-quota-baseline cpu-quota-400 cpu-quota-100 cpu-quota-50; do
  rm -rf "/mnt/post-training/trl/$run"
done
```

`systemd-run --scope`는 명령이 끝나면 scope도 정리되므로 별도 정리가 필요 없습니다.

## Limitations

**CPU quota**는 spark1 단일 노드 0.5B smoke에서만 검증했습니다.
30B NVMe DeepSpeed 같은 CPU-heavier workload(AIO thread, tokenization)에서는 임계점이 더 높은 quota에서 나타날 수 있으며 그 값은 아직 측정하지 않았습니다.

**Memory limit은 이 클러스터의 알려진 한계입니다.**
DGX Spark GB10은 CPU와 GPU가 같은 물리 memory pool을 쓰는 unified memory architecture이고, cgroup v2 `memory.max`는 plain host allocation만 정확히 강제합니다.
따라서 실제 학습 workload에 대한 memory ceiling 탐색이나 controlled OOM 비교는 이 하드웨어에서 실행하지 않습니다.
풀려면 (a) NVIDIA UMA driver가 CUDA 메모리를 cgroup에 노출하는 mechanism이 있는지 확인하거나 (b) discrete GPU 노드에서 다시 검증해야 합니다.
