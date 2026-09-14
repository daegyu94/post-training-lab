# Lab: 30B NVMe Data Movement

두 Spark 노드에서 TRL의 DeepSpeed ZeRO-3 parameter·optimizer offload와 Megatron의 로컬 NVMe async checkpoint I/O를 관찰합니다.
**Megatron은 runtime state offload가 아닙니다.**
두 backend는 JSONL을 로컬 NVMe에 순차 전처리하고 disk-backed Arrow cache에서 batch를 읽습니다.

## Prerequisites

두 노드 모두 `/`가 로컬 NVMe ext4인지 확인하고 `spark` 사용자가 backend별 디렉터리에 쓸 수 있어야 합니다.
각 노드의 [공통 준비 스크립트](../../setups/spark/README.md#prepare-each-spark-node)로 `libaio-dev`·Python headers·32GiB memlock을 준비합니다.
NVMe 경로가 없다면 관리자가 한 번 생성합니다.

```bash
sudo install -d -o spark -g spark -m 700 \
  /mnt/post-training/trl /mnt/post-training/megatron
```

DeepSpeed JIT는 TRL `requirements-spark.txt`의 `ninja`를 사용합니다.
설치 후 controller에서 경로와 async I/O build를 확인합니다.

```bash
for host in spark1 spark2; do
  ssh "spark@$host" \
    'findmnt -T /mnt/post-training/trl; test -w /mnt/post-training/trl; test -w /mnt/post-training/megatron'
  ssh "spark@$host" \
    '$HOME/.local/ptl/venvs/trl/bin/python -c "from deepspeed.ops.op_builder import AsyncIOBuilder; AsyncIOBuilder().load(verbose=True)"'
done
```

설정 후 기존 SSH 연결을 끊고 다시 접속해 `ulimit -l`이 `33554432` 이상인지 확인합니다.

두 노드에 `Qwen/Qwen3-30B-A3B` revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` snapshot과 No Robots revision `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b` 데이터가 필요하며, `output_root`는 backend별 로컬 NVMe 경로를 씁니다.

```json
"output_root": {
  "trl": "/mnt/post-training/trl",
  "megatron": "/mnt/post-training/megatron"
}
```

## Run

controller의 저장소 루트에서 두 계획을 먼저 확인합니다.

```bash
python experiments/run.py --backend trl \
  --setup setups/spark/local.json \
  --experiment experiments/trl/nvme-offload-30b.json \
  --output results/trl-nvme-30b

python experiments/run.py --backend megatron \
  --setup setups/spark/local.json \
  --experiment experiments/megatron/qwen3-30b-lora.json \
  --output results/megatron-nvme-30b
```

각 rank의 output이 `/mnt/post-training/<backend>/...`인지 확인한 뒤 같은 명령에 `--execute`를 붙여 한 번에 하나씩 실행합니다.

<a id="expected-results-and-verification"></a>

## Expected Results and Verification

**TRL 성공 조건**은 두 rank의 정상 종료, 1 optimizer step, `/mnt/post-training/trl`의 DeepSpeed NVMe read/write 발생입니다.
Full SFT 전용이며 LoRA 조합은 검증에서 거부됩니다.
과거 실패를 reentrant checkpointing 탓으로 설명한 기록은 현재 `use_reentrant=False` 코드와 맞지 않으므로 원인으로 단정하지 않습니다.

`train` 직후 같은 process에서 평가하면 ZeRO-3의 학습 trace와 backward 없는 평가 경로가 어긋나 NVMe swap buffer가 소진됩니다.
`buffer_count` 증가로는 해결되지 않아 **별도 `STAGE=tuned` process**를 사용합니다.
새 엔진이 `trainer.save_model()`의 native ZeRO checkpoint를 `deepspeed_load_checkpoint`로 rank별 복원하므로 전체 모델의 FP32 집계는 필요 없습니다.
`tuned`는 `train`과 같은 topology를 써야 하며 `STAGE=all`은 이를 재사용합니다.

**Megatron 성공 조건**은 두 rank의 정상 종료, 1 optimizer step, async save finalization과 각 노드의 checkpoint shard 생성입니다.
실행 중 `iostat -dx 1 nvme0n1`로 장치 I/O를 관찰합니다.
Arrow dataset은 memory map과 page cache를 쓰므로 같은 batch를 다시 읽을 때 물리 NVMe I/O가 생략될 수 있습니다.

<a id="why-nvme-offload-is-necessary"></a>

## Why NVMe Offload Is Necessary Here

2026-09-09 Qwen3-30B-A3B full SFT(ZeRO-3, world size 2)에서 각 노드의 `/mnt/post-training/trl/zero_stage_3`는 **약 256GiB**, 물리 RAM은 **119GiB**였습니다.
이 실행은 `OPTIMIZER=sgd`로 설정됐지만 DeepSpeed가 offload 시 `DeepSpeedCPUAdam`으로 교체하므로 실제 swap 내용은 Adam 모멘트(`exp_avg`·`exp_avg_sq`)였고, 256GiB라는 크기도 그 때문입니다([정정 근거](../../docs/experiments/checkpoint-io.md#memory-footprint)).
이 footprint는 RAM의 두 배 이상으로 `offload_param`·`offload_optimizer`의 `device: nvme` 선택을 뒷받침합니다.
단, `cpu`·`none` 대조 실행의 OOM을 관찰한 것은 아니며 다른 optimizer·병렬 구성으로 일반화하지 않습니다.

### 규모가 커지면: local offload, remote checkpoint

Local NVMe·NFS의 throughput·contention 비교는 보류했으며 아래는 **설계 방향**입니다.

- **Runtime offload**: 매 step 접근하고 rank 간 공유가 필요 없어 node-local을 우선합니다.
- **Checkpoint**: 노드 장애 시 접근성과 용량 확장을 위해 remote shared filesystem을 고려합니다. 현재 용량은 노드당 3.7TB, 30B 학습 사용량은 합계 1TB 이내입니다.
- pNFS·3FS 등 filesystem 선택과 성능 검증은 범위 밖입니다.

<a id="training-data-storage"></a>

### 학습 데이터 저장: 일반 원칙과 이 PoC

대규모 환경에서는 object storage 정본 → parallel/distributed filesystem 읽기 경로 → node-local NVMe prefetch cache로 계층을 나눌 수 있습니다.
공통 원칙은 **한 번 생성한 데이터를 checksum으로 검증해 배포**하는 것입니다.

이 2노드 PoC에는 별도 storage cluster가 없어 controller NFS의 부하를 줄이도록 모델·데이터를 node-local에 둡니다.
데이터는 한 번 준비해 복사하며([이유](../../docs/datasets.md#public-data)), 32KB 수준의 소형 입력이라 전체 복제가 가능합니다.
규모가 커져 전체 복제가 어려워지면 정본·공유 읽기 경로·부분 prefetch를 분리합니다.

## Cleanup

실행이 끝나고 보존할 로그·요약을 옮긴 뒤 run 디렉터리를 삭제합니다.
DeepSpeed가 남긴 `zero_stage_3` swap 파일은 실행 process가 없는지 확인한 뒤 정리합니다 — 학습이 끝나면 필요 없는 임시 데이터이며 노드당 수백 GB를 차지합니다.

## Limitations

DGX Spark의 CPU와 GPU는 unified memory를 쓰므로 일반적인 discrete GPU의 CPU offload와 같은 의미로 해석하지 않습니다.
Megatron Core의 native training offload 대상은 CPU memory이고 NVMe state offload는 지원하지 않으므로 이 실습의 Megatron 경로는 checkpoint I/O까지만 다룹니다.
Node-local Megatron checkpoint의 cross-node reload와 장애 복구는 완료 조건이 아닙니다.
