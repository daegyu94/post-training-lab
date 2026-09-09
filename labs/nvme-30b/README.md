# 30B NVMe Data Movement

## Status and Objective

Status: `implemented, host preparation required`

이 실습은 두 Spark 노드의 로컬 NVMe가 30B post-training 과정에 미치는 영향을 관찰합니다.
TRL은 DeepSpeed ZeRO-3 parameter·optimizer state offload를 실행하고, Megatron은 async distributed checkpoint를 로컬 NVMe에 기록합니다.
Megatron 경로는 학습 state offload가 아니라 checkpoint I/O라는 차이를 결과 해석에서 유지합니다.
두 backend 모두 canonical JSONL을 로컬 NVMe에 순차 전처리하고 disk-backed Arrow cache에서 batch 단위로 읽습니다.

## Prerequisites

두 노드 모두 `/`가 로컬 NVMe의 ext4 파일시스템인지 확인하고, `spark` 사용자가 backend별 디렉터리에 쓸 수 있어야 합니다.
같은 날 두 노드에서 실제 Qwen3-30B tokenizer와 No Robots 입력을 순차 전처리했고, TRL과 Megatron dataset이 모두 Hugging Face `MemoryMappedTable`로 열리는 것까지 확인했습니다.

각 노드에서 [공통 준비 스크립트](../../setups/spark/README.md#prepare-each-spark-node)를 먼저 실행합니다.
이 스크립트가 `libaio-dev`, Python headers와 32GiB memlock 설정을 준비합니다.
NVMe 경로가 없다면 관리자가 한 번 생성합니다.

```bash
sudo install -d -o spark -g spark -m 700 \
  /mnt/post-training/trl /mnt/post-training/megatron
```

TRL Python 환경에는 `requirements-spark.txt`에 고정된 `ninja`도 필요합니다.
Launcher는 `PYTHON`이 가리키는 환경의 `bin`을 `PATH`에 추가하므로 DeepSpeed JIT가 같은 환경의 `ninja`를 사용합니다.

그다음 controller에서 경로와 DeepSpeed async I/O를 확인합니다.

```bash
for host in spark1 spark2; do
  ssh "spark@$host" \
    'findmnt -T /mnt/post-training/trl; test -w /mnt/post-training/trl; test -w /mnt/post-training/megatron'
  ssh "spark@$host" \
    '$HOME/.local/ptl/venvs/trl/bin/python -c "from deepspeed.ops.op_builder import AsyncIOBuilder; AsyncIOBuilder().load(verbose=True)"'
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
  --experiment experiments/megatron/qwen3-30b-lora.json \
  --output results/megatron-nvme-30b
```

Dry-run의 각 rank에서 TRL output이 `/mnt/post-training/trl/trl-nvme-30b`, Megatron output이 `/mnt/post-training/megatron/megatron-nvme-30b`인지 확인합니다.
같은 명령 끝에 `--execute`를 붙여 한 번에 하나씩 실행합니다.

## Expected Results and Verification

TRL 성공 조건은 두 rank의 정상 종료, 1 optimizer step과 `/mnt/post-training/trl`의 DeepSpeed NVMe read/write 발생입니다.
이 preset은 전체 parameter와 optimizer state를 NVMe training-time memory tier로 사용하는 full SFT입니다.
현재 고정된 TRL·DeepSpeed 조합에서 LoRA와 ZeRO-3 NVMe parameter offload를 함께 쓰는 것은 과거 관측된 실패를 근거로 권장하지 않습니다.
다만 그 원인으로 적어뒀던 "reentrant gradient checkpointing이 swap buffer를 점유한다"는 설명은, 현재 `spark_train.py`가 gradient checkpointing을 항상 `use_reentrant=False`로만 켜도록 고정돼 있어 더 이상 코드와 맞지 않습니다(2026-09-09 기준). 이 조합은 이번 세션에서 다시 실행해 재확인하지 않았으므로, 제한 자체는 유지하되 원인 설명은 stale일 수 있다는 점만 표시해둡니다.
각 Spark 노드에서 `spark` 사용자의 memlock soft/hard limit을 32GiB 이상으로 설정해야 parameter buffer 약 12.3GB와 optimizer tile 약 4.6GB를 함께 고정할 수 있습니다.

`train` stage는 학습 직후 같은 프로세스에서 평가를 실행하지 않습니다.
DeepSpeed ZeRO-3의 parameter coordinator는 학습 forward+backward에서 기록한 실행 trace를 기준으로 NVMe swap buffer 반납 시점을 정하는데, backward가 없는 평가 forward는 이 trace와 어긋나 buffer가 반납되지 않고 소진됩니다(`buffer_count`를 늘려도 소진 시점만 미뤄질 뿐 해결되지 않음).
대신 평가는 `STAGE=tuned`를 별도 프로세스로 실행해 수행합니다: 새 프로세스는 학습 trace가 없는 새 ZeRO-3 엔진을 만들고, `trainer.save_model()`이 남긴 native DeepSpeed ZeRO checkpoint(`output_dir/model/global_step*`)를 `deepspeed_load_checkpoint`로 그 엔진에 복원한 뒤 평가합니다.
이 복원은 rank별 partition을 그대로 불러오는 저메모리 경로만 사용하며, 30B 전체를 하나의 프로세스에 fp32로 모으는 변환(zero_to_fp32류)은 쓰지 않습니다.
`tuned`는 그 checkpoint를 만든 `train` run과 동일한 node/GPU topology(`NNODES`/`NPROC_PER_NODE`)로 실행해야 하며, `experiments/run.py`가 `STAGE=all`로 두 stage를 실행하면 같은 환경변수를 그대로 재사용하므로 이 조건이 자동으로 맞습니다.

설정 후 기존 SSH 연결을 끊고 다시 접속한 다음 `ulimit -l`이 `33554432` 이상인지 확인합니다.
전처리 JSONL과 Hugging Face Arrow cache도 run output 아래에 남으며 dataset 전체를 Python list로 적재하지 않습니다.

Megatron 성공 조건은 두 rank의 정상 종료, 1 optimizer step, async save finalization과 각 노드의 `/mnt/post-training/megatron/megatron-nvme-30b/checkpoints` shard 생성입니다.
Controller manifest와 rank 로그를 함께 확인하고 실행 중 `iostat -dx 1 nvme0n1`로 장치 I/O를 관찰합니다.
Arrow dataset은 memory map과 운영체제 page cache를 사용하므로 동일 batch를 다시 읽을 때 물리 NVMe I/O가 생략될 수 있습니다.

## Why NVMe Offload Is Necessary Here

2026-09-09 실행(`trl-nvme-30b-tuned-eval-r3`)에서 두 노드 모두 `/mnt/post-training/trl/zero_stage_3`가 **약 256GiB**까지 자랐고, 같은 시점 각 노드의 물리 RAM은 `free -h` 기준 **119GiB**였습니다.
즉 이 구성(Qwen3-30B-A3B full SFT, ZeRO-3, SGD optimizer, 이 setup의 2노드 world size)에서 parameter·optimizer state footprint는 노드 RAM보다 2배 이상 큽니다.
이 구성에서는 `offload_param`/`offload_optimizer`의 `device: nvme`가 있어야 학습이 성립하며, NVMe offload는 처리량 최적화가 아니라 이 footprint를 RAM만으로 담을 수 없다는 사실 자체가 근거입니다.

이 결론은 측정된 footprint와 노드 RAM 용량을 비교한 것이지, `device: cpu`나 `device: none`으로 같은 설정을 실제로 돌려 OOM을 관찰한 결과가 아닙니다.
즉 "간접적이지만 정량적인 근거"이며, "실패를 직접 재현한 근거"는 아닙니다. 그 control 실행은 아직 하지 않았고, 필요하면 별도로 계획합니다.
또한 이 수치는 이 모델 크기·optimizer·병렬 구성에 한정되며, 다른 모델 크기나 optimizer(AdamW 등)로 일반화하지 않습니다.

## Cleanup

실행이 끝나고 보존할 로그·요약을 옮긴 뒤 run 디렉터리만 삭제합니다.
DeepSpeed가 남긴 임시 swap 파일은 실행 process가 없는지 확인한 뒤 정리합니다.

## Limitations and Next Steps

DGX Spark의 CPU와 GPU는 unified memory를 사용하므로 일반적인 discrete GPU의 CPU offload와 같은 의미로 해석하지 않습니다.
Megatron Core의 native training offload 대상은 CPU memory이며 NVMe state offload는 지원하지 않으므로 이 실습은 checkpoint I/O까지만 다룹니다.
Node-local Megatron checkpoint의 cross-node reload와 장애 복구는 현재 완료 조건이 아닙니다.
