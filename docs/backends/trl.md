# TRL Backend

TRL backend의 검증된 기본 경로는 `spark1`·`spark2`에서 노드당 process 하나를 쓰는 DDP LoRA SFT입니다.

모델의 native chat template으로 prompt와 마지막 assistant completion을 분리해 SFT loss를 계산합니다.
학습 단계는 `base` 평가 → `train` 저장 → 별도 process의 `tuned` 재로딩 평가입니다.
LoRA는 adapter를 `output_dir/adapter`에, full SFT는 모델을 `output_dir/model`에 저장합니다.
DPO와 RL trainer는 구현되어 있지 않습니다.

Canonical JSONL을 rank별 output으로 순차 전처리하고 같은 저장장치에 Arrow cache를 둡니다.
Trainer는 memory-mapped dataset에서 batch를 읽으며 page cache 때문에 매번 물리 storage read가 발생하지는 않습니다.

## Prepare the Spark Environment

각 노드에 ARM64·CUDA에 맞는 독립 Python 환경이 필요합니다.
`requirements-spark.txt`가 검증 기준인 Torch 2.10.0, Transformers 5.12.1, TRL 1.12.0, Accelerate 1.14.0을 고정합니다.
**version pin만으로는 CUDA build가 고정되지 않으므로** cu130 인덱스를 명시해 Torch를 먼저 설치합니다.

먼저 각 노드에서 [공통 준비 스크립트](../../setups/spark/README.md#prepare-each-spark-node)를 실행한 뒤 node-local 가상환경에 설치합니다.
첫 설치는 Torch와 CUDA runtime wheel을 합쳐 수 GB를 다운로드합니다.

```bash
cd "/path/to/shared/post-training-lab/backends/trl"
. "$HOME/.local/ptl/venvs/trl/bin/activate"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.10.0
python -m pip install -r requirements-spark.txt
python -c 'import torch, transformers, trl, accelerate; assert torch.__version__ == "2.10.0+cu130" and torch.version.cuda == "13.0" and torch.cuda.is_available(); print(transformers.__version__, trl.__version__, accelerate.__version__); print((torch.ones(1, device="cuda") + 1).item())'
```

Import·CUDA 확인 후 NCCL 통신, 모델 적합성, 학습은 두 노드 smoke로 검증합니다.
의존성 충돌로 version을 바꾸면 별도 환경으로 기록합니다.

## Run the Standard Workflow

공통 runner가 controller에서 SSH로 각 rank를 시작합니다.
준비·실행은 [Getting Started](../getting-started.md)를 따릅니다.
한 노드 smoke가 필요하면 `experiments/trl/single-node-smoke.json`을 사용합니다.

직접 `backends/trl/scripts/run_spark_cluster.sh`를 쓰면 **모든 참여 노드에서** launcher를 시작합니다.
Model·dataset·output과 rendezvous 변수(`MASTER_ADDR`·`MASTER_PORT`·`NNODES`·`NPROC_PER_NODE`)는 맞추고 `NODE_RANK`만 달리 지정합니다.
Checkout 검사·process 관리·controller manifest가 없으므로 runner 미지원 topology 검증에만 사용합니다.

## Configure Training

Model·dataset revision은 40-hex SHA여야 하고 데이터 디렉터리에는 `training.jsonl`, `validation.jsonl`, `manifest.json`이 있어야 합니다.
30B 대상 모델은 model directory 이름도 요청한 snapshot revision과 일치해야 합니다.

| 설정 | 값 | 동작 |
| --- | --- | --- |
| `FINETUNING_MODE` | `lora`, `full` | LoRA adapter 또는 전체 parameter 학습 |
| `DISTRIBUTED_BACKEND` | `ddp`, `fsdp2`, `deepspeed` | Trainer의 분산 backend 선택 |
| `STAGE` | `base`, `train`, `tuned`, `all` | `all`은 base → train → tuned 순서 |
| `OPTIMIZER` | `adamw`, `sgd` | SFT optimizer |
| `MAX_STEPS`, `MAX_LENGTH` | 양의 정수 | 학습 step과 입력 최대 token 길이 |
| `TRAIN_SAMPLES`, `EVAL_SAMPLES` | 양의 정수 | 사용할 prepared row 수 제한 |
| `DEEPSPEED_CONFIG` | ZeRO-2 또는 ZeRO-3 JSON | DeepSpeed 실행에서 필수 |

DDP는 rank마다 모델을 복제하며 LoRA도 base weight 메모리는 필요합니다.
Runner는 노드당 process 하나만 지원하므로 노드 내 multi-GPU는 별도 topology·launcher 검증이 필요합니다.

## Choose a Distributed Backend

| backend | `base` | `train` | `tuned` | 설정 파일 | 비고 |
| --- | :---: | :---: | :---: | --- | --- |
| `ddp` | ✅ | ✅ | ✅ | 불필요 | 기본 선택 |
| `fsdp2` | ✅ | ✅ | ❌ | 불필요 | sharded export·tuned reload·optimizer resume 미검증 |
| `deepspeed` | ✅ | ✅ | ✅ | **필수** | [ZeRO-2](../../backends/trl/configs/deepspeed-zero2.json) 또는 [ZeRO-3](../../backends/trl/configs/deepspeed-zero3.json) |

`tuned`는 새 process에서 저장물을 재로딩합니다.

- **DDP**: 각 rank가 로컬 adapter를 저장하므로 node-local `output_root`에서도 재로딩합니다.
- **FSDP2**: `SHARDED_STATE_DICT`를 rank별로 저장합니다. Node-local 출력에서는 다른 rank의 shard를 볼 수 없어 `tuned`를 지원하지 않습니다.
- **DeepSpeed**: LoRA는 adapter를, full SFT는 native ZeRO checkpoint를 새 엔진에 rank-local로 복원해 평가합니다.

### DeepSpeed NVMe offload profile

`deepspeed-zero3-nvme.json`은 `/mnt/post-training/trl`로 parameter와 optimizer state를 offload하는 **30B 전용** 설정이며 일반 ZeRO-3와 제약이 다릅니다.

- **full fine-tuning 전용** — LoRA와 함께 쓰면 검증에서 거부됩니다. LoRA에는 DDP나 parameter offload 없는 DeepSpeed profile을 씁니다.
- `train` 단계 **안에서의 평가만** 건너뜁니다. 평가는 `tuned` 단계를 별도 process로 실행합니다 ([이유와 재로딩 방식](../../labs/nvme-30b/README.md#expected-results-and-verification)).
- 두 노드에 쓰기 가능한 로컬 디렉터리, DeepSpeed async I/O build, 32GiB 이상의 memlock 한도가 필요합니다.
- offload 경로가 설정 파일에 고정되어 있어 다른 mount를 쓰려면 사본의 `nvme_path` 두 곳을 바꿔야 합니다.

## Verify a Run

Controller `manifest.json`의 `status`가 `passed`이고 모든 rank가 정상 종료했는지 확인한 뒤, Spark output의 `summary-{base,train,tuned}.json`과 `logs/rank-<rank>-<stage>.log`를 봅니다.

`summary-train.json`에서 finite training/evaluation loss, 기대한 `actual_optimizer_steps`, 저장물을 확인합니다.
DDP summary는 sample한 trainable parameter의 변화도 기록합니다.
FSDP2·DeepSpeed summary의 `parameter_update_evidence`가 `optimizer steps only`이면 parameter 변화 자체를 검증한 결과로 읽을 수 없습니다.

0.5B smoke 성공을 30B full SFT의 메모리 적합성이나 장기 수렴 증거로 해석하지 않습니다.
판정 기준은 [Getting Started](../getting-started.md#6-verify-the-result), 30B 실측값은 [Experiments](../experiments.md#30b-gpu-results)를 따릅니다.
