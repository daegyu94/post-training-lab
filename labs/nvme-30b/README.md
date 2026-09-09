# 30B NVMe Data Movement

## Status and Objective

Status: `implemented, host preparation required`

이 실습은 두 Spark 노드의 로컬 NVMe가 30B post-training 과정에 미치는 영향을 관찰합니다.
TRL은 DeepSpeed ZeRO-3 parameter·optimizer state offload를 실행하고, Megatron은 async distributed checkpoint를 로컬 NVMe에 기록합니다.
Megatron 경로는 학습 state offload가 아니라 checkpoint I/O라는 차이를 결과 해석에서 유지합니다.
두 backend 모두 canonical JSONL을 로컬 NVMe에 순차 전처리하고 disk-backed Arrow cache에서 batch 단위로 읽습니다.

## Prerequisites

두 노드 모두 `/`가 로컬 NVMe의 ext4 파일시스템인지 확인하고, `spark` 사용자가 backend별 디렉터리에 쓸 수 있어야 합니다.
Controller에서 확인한 2026-09-09 상태에서는 각 노드의 `/dev/nvme0n1p2`가 `/`에 마운트되어 있었지만 디렉터리와 `libaio-dev`가 없었습니다.
같은 날 두 노드에서 실제 Qwen3-30B tokenizer와 No Robots 입력을 순차 전처리했고, TRL과 Megatron dataset이 모두 Hugging Face `MemoryMappedTable`로 열리는 것까지 확인했습니다.

각 노드에서 관리자가 한 번 실행합니다.

```bash
sudo install -d -o spark -g spark -m 700 \
  /mnt/post-training/trl /mnt/post-training/megatron
sudo apt-get install libaio-dev python3-dev
```

TRL Python 환경에는 `requirements-spark.txt`에 고정된 `ninja`도 필요합니다.
Launcher는 `PYTHON`이 가리키는 환경의 `bin`을 `PATH`에 추가하므로 DeepSpeed JIT가 같은 환경의 `ninja`를 사용합니다.

그다음 controller에서 경로와 DeepSpeed async I/O를 확인합니다.

```bash
for host in spark1 spark2; do
  ssh "spark@$host" \
    'findmnt -T /mnt/post-training/trl; test -w /mnt/post-training/trl; test -w /mnt/post-training/megatron'
  ssh "spark@$host" \
    '/home/spark/ptl-envs/trl/bin/python -c "from deepspeed.ops.op_builder import AsyncIOBuilder; AsyncIOBuilder().load(verbose=True)"'
done
```

Setup의 두 노드에는 `Qwen/Qwen3-30B-A3B` revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` snapshot과 No Robots revision `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b` 데이터가 필요합니다.
`output_root`는 backend별 로컬 NVMe 경로를 사용합니다.

```json
"output_root": {
  "trl": "/mnt/post-training/trl",
  "megatron": "/mnt/post-training/megatron"
}
```

## Run

먼저 controller의 저장소 루트에서 두 계획을 확인합니다.

```bash
python experiments/run.py \
  --backend trl \
  --setup setups/spark/local.json \
  --experiment experiments/trl/nvme-offload-30b.json \
  --output results/trl-nvme-30b

python experiments/run.py \
  --backend megatron \
  --setup setups/spark/local.json \
  --experiment experiments/megatron/nvme-checkpoint-30b.json \
  --output results/megatron-nvme-30b
```

Dry-run의 각 rank에서 TRL output이 `/mnt/post-training/trl/trl-nvme-30b`, Megatron output이 `/mnt/post-training/megatron/megatron-nvme-30b`인지 확인합니다.
같은 명령 끝에 `--execute`를 붙여 한 번에 하나씩 실행합니다.

## Expected Results and Verification

TRL 성공 조건은 두 rank의 정상 종료, 1 optimizer step과 `/mnt/post-training/trl`의 DeepSpeed NVMe read/write 발생입니다.
이 preset은 전체 parameter와 optimizer state를 NVMe training-time memory tier로 사용하는 full SFT입니다.
현재 고정된 TRL·DeepSpeed 조합에서 LoRA와 ZeRO-3 NVMe parameter offload를 함께 쓰면 reentrant gradient checkpointing이 여러 MoE layer의 swap buffer를 계속 점유하므로 지원하지 않습니다.
각 Spark 노드에서 `spark` 사용자의 memlock soft/hard limit을 32GiB 이상으로 설정해야 parameter buffer 약 12.3GB와 optimizer tile 약 4.6GB를 함께 고정할 수 있습니다.

```bash
printf 'spark soft memlock 33554432\nspark hard memlock 33554432\n' \
  | sudo tee /etc/security/limits.d/90-post-training-lab.conf
```

설정 후 기존 SSH 연결을 끊고 다시 접속한 다음 `ulimit -l`이 `33554432` 이상인지 확인합니다.
전처리 JSONL과 Hugging Face Arrow cache도 run output 아래에 남으며 dataset 전체를 Python list로 적재하지 않습니다.

Megatron 성공 조건은 두 rank의 정상 종료, 1 optimizer step, async save finalization과 각 노드의 `/mnt/post-training/megatron/megatron-nvme-30b/checkpoints` shard 생성입니다.
Controller manifest와 rank 로그를 함께 확인하고 실행 중 `iostat -dx 1 nvme0n1`로 장치 I/O를 관찰합니다.
Arrow dataset은 memory map과 운영체제 page cache를 사용하므로 동일 batch를 다시 읽을 때 물리 NVMe I/O가 생략될 수 있습니다.

## Cleanup

실행이 끝나고 보존할 로그·요약을 옮긴 뒤 run 디렉터리만 삭제합니다.
DeepSpeed가 남긴 임시 swap 파일은 실행 process가 없는지 확인한 뒤 정리합니다.

## Limitations and Next Steps

DGX Spark의 CPU와 GPU는 unified memory를 사용하므로 일반적인 discrete GPU의 CPU offload와 같은 의미로 해석하지 않습니다.
Megatron Core의 native training offload 대상은 CPU memory이며 NVMe state offload는 지원하지 않으므로 이 실습은 checkpoint I/O까지만 다룹니다.
Node-local Megatron checkpoint의 cross-node reload와 장애 복구는 현재 완료 조건이 아닙니다.
