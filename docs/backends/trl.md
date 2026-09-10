# TRL on NVIDIA DGX Spark

TRL backend는 NVIDIA DGX Spark의 `spark1`, `spark2`에서 supervised fine-tuning(SFT)을 실행합니다.
Controller는 실행을 조율하고 기록을 남기며, 모델 로딩과 GPU 학습 process는 Spark 노드에서만 실행합니다.
현재 공통 runner의 검증된 기본 경로는 두 노드에서 node당 process 하나를 사용하는 DDP LoRA smoke입니다.

## What This Backend Runs

이 backend는 모델의 native chat template으로 prompt와 마지막 assistant completion을 분리해 SFT loss를 계산합니다.
학습 단계는 `base` 평가, `train` 저장, 별도 process의 `tuned` 재로딩 평가로 구성됩니다.
LoRA 학습은 adapter를 `output_dir/adapter`에 저장하고 full SFT는 모델을 `output_dir/model`에 저장합니다.
DPO와 RL trainer는 구현되어 있지 않습니다.

Canonical JSONL은 순차 전처리하여 rank별 output 아래에 기록하고 Hugging Face Arrow cache도 같은 저장장치에 둡니다.
Trainer는 전체 split을 Python list로 올리지 않고 memory-mapped dataset에서 batch를 읽습니다.
운영체제 page cache가 데이터를 재사용할 수 있으므로 모든 batch가 반드시 물리 storage read를 발생시킨다는 의미는 아닙니다.

## Run the Standard Spark Workflow

처음 실행할 때는 공통 runner를 사용합니다.
Runner가 controller에서 `ssh spark@spark1`, `ssh spark@spark2`로 접속해 각 노드의 launcher를 시작하므로, 직접 SSH 세션에서 각 rank를 관리할 필요가 없습니다.

1. [Spark cluster setup](../../setups/spark/README.md)의 controller, SSH, NCCL rendezvous, NFS 경로를 확인합니다.
2. [Getting Started](../getting-started.md)의 순서에 따라 두 Spark 노드에 Python 환경, 동일 모델 snapshot, prepared data와 쓰기 가능한 output root를 준비합니다.
3. `setups/spark/local.json`에 각 노드에서 보이는 `checkout`, `model_dirs`, `data_dir`, `output_root` 절대 경로를 입력합니다.
4. `experiments/trl/smoke.json`으로 실행 계획을 확인한 뒤 `--execute`를 붙여 DDP LoRA smoke를 시작합니다.
5. Controller manifest, rank log, Spark output의 stage summary를 함께 확인합니다.

공통 runner의 실행 명령과 setup 필드 설명은 [Getting Started](../getting-started.md)에 한 곳에서 관리합니다.
한 노드 smoke가 필요하면 `experiments/trl/single-node-smoke.json`을 사용합니다.

## Prepare the Spark Environment

각 Spark 노드는 ARM64와 CUDA에 맞는 독립 Python 환경이 필요합니다.
`requirements-spark.txt`는 이 backend의 검증 기준인 Torch 2.10.0, Transformers 5.12.1, TRL 1.12.0, Accelerate 1.14.0을 고정합니다.
과거 실행은 Torch `2.10.0+cu130`을 사용했으므로 version pin만으로 같은 CUDA build가 선택된다고 가정할 수 없습니다.

먼저 각 노드에서 [공통 준비 스크립트](../../setups/spark/README.md#prepare-each-spark-node)를 실행합니다.
그다음 node-local TRL 가상환경에 Python package를 설치합니다.

```bash
cd "/path/to/shared/post-training-lab/backends/trl"
. "$HOME/.local/ptl/venvs/trl/bin/activate"
python -m pip install -r requirements-spark.txt
python -c 'import torch, transformers, trl, accelerate; print(torch.__version__, torch.cuda.is_available()); print(transformers.__version__, trl.__version__, accelerate.__version__)'
```

이 확인은 package import와 CUDA 가용성만 검사합니다.
NCCL 통신, 모델 적합성, 실제 학습 성공은 두 Spark 노드의 smoke 실행으로 별도로 확인해야 합니다.
의존성 충돌이 발생하면 실제 설치 결과를 기록하고 임의의 다른 version을 같은 검증 환경으로 취급하지 않습니다.

## Configure Training

공통 runner는 experiment preset의 환경변수를 각 node launcher에 전달합니다.
Model과 dataset revision은 immutable 40-hex SHA여야 하며, 데이터 디렉터리에는 `training.jsonl`, `validation.jsonl`, `manifest.json`이 있어야 합니다.
30B 대상 모델은 model directory 이름도 요청한 snapshot revision과 일치해야 합니다.

| 설정 | 값 | 동작 |
| --- | --- | --- |
| `FINETUNING_MODE` | `lora`, `full` | LoRA adapter 또는 전체 parameter를 학습합니다. |
| `DISTRIBUTED_BACKEND` | `ddp`, `fsdp2`, `deepspeed` | Trainer의 분산 backend를 선택합니다. |
| `STAGE` | `base`, `train`, `tuned`, `all` | `all`은 base → train → tuned 순서로 실행합니다. |
| `OPTIMIZER` | `adamw`, `sgd` | SFT optimizer를 선택합니다. |
| `MAX_STEPS`, `MAX_LENGTH` | 양의 정수 | 학습 step과 입력 최대 token 길이를 설정합니다. |
| `TRAIN_SAMPLES`, `EVAL_SAMPLES` | 양의 정수 | smoke에서 사용할 prepared row 수를 제한합니다. |
| `DEEPSPEED_CONFIG` | ZeRO-2 또는 ZeRO-3 JSON | DeepSpeed 실행에서 필수입니다. |

DDP는 각 rank에 모델 복제본을 유지합니다.
노드 수를 늘려도 모델 가중치가 자동으로 분할되지는 않으며 LoRA도 base weight의 메모리를 없애지 않습니다.
현재 공통 runner는 node당 process 하나를 지원하므로 node 내 multi-GPU 확장은 별도 topology와 launcher 검증이 필요합니다.

## Use the Launcher Directly

직접 `backends/trl/scripts/run_spark_cluster.sh`를 실행할 때는 모든 참여 Spark 노드에서 launcher를 시작해야 합니다.
각 노드는 같은 model, dataset, output 설정을 사용하고 `NODE_RANK`만 다르게 지정합니다.
`MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NPROC_PER_NODE`도 모든 node에서 같은 rendezvous 구성으로 지정해야 합니다.
NFS를 사용한다면 Spark 노드 기준인 `/home/spark/shared/...` 경로를 사용합니다.

직접 실행은 공통 runner의 원격 checkout 검사, process lifecycle 관리, controller manifest 생성을 제공하지 않습니다.
일반적인 실험은 runner를 사용하고, launcher 직접 실행은 runner가 제공하지 않는 topology를 검증할 때만 사용합니다.

## Choose a Distributed Backend

| backend | `base` | `train` | `tuned` | 설정 파일 | 비고 |
| --- | :---: | :---: | :---: | --- | --- |
| `ddp` | ✅ | ✅ | ✅ | 불필요 | 기본 선택 |
| `fsdp2` | ✅ | ✅ | ❌ | 불필요 | sharded export·tuned reload·optimizer resume 미검증 |
| `deepspeed` | ✅ | ✅ | ✅ | **필수** | [ZeRO-2](../../backends/trl/configs/deepspeed-zero2.json) 또는 [ZeRO-3](../../backends/trl/configs/deepspeed-zero3.json) |

`tuned` 단계는 새 process에서 저장물을 다시 읽으므로, 저장과 재로딩을 함께 확인할 수 있습니다.

- **DDP** — 데이터만 rank별로 나누고 모델과 LoRA adapter는 각 rank에 복제합니다. `output_root`가 [node-local 경로](../../labs/nvme-30b/README.md#training-data-storage-general-principle-vs-this-poc)여도 `train`이 각 rank에 로컬 adapter를 남기므로 `tuned`가 바로 재로딩합니다.
- **DeepSpeed** — LoRA는 기존 adapter reload 경로를 그대로 쓰고, full fine-tuning은 별도 process가 native ZeRO checkpoint를 새 엔진에 rank-local로 불러와 평가합니다.

### DeepSpeed NVMe offload profile

`deepspeed-zero3-nvme.json`은 `/mnt/post-training/trl`로 parameter와 optimizer state를 offload하는 **30B 실습 전용** 설정이며, 일반 ZeRO-3와 제약이 다릅니다.

- **full fine-tuning 전용** — LoRA와 함께 쓰면 검증 단계에서 거부됩니다. LoRA에는 DDP 또는 NVMe parameter offload가 없는 DeepSpeed profile을 씁니다.
- `train` 단계 **안에서의 평가만** 건너뜁니다. 평가 자체는 `tuned` 단계로 수행합니다 ([이유와 재로딩 방식](../../labs/nvme-30b/README.md#expected-results-and-verification)).
- 두 노드에 쓰기 가능한 로컬 디렉터리와 DeepSpeed async I/O build가 필요합니다.
- offload 경로는 설정 파일에 고정되어 있어, 다른 mount를 쓰려면 설정 사본의 `nvme_path` 두 곳을 바꿔야 합니다.

모델과 backend별 실행 결과와 실패 원인은 [Verification](../verification.md)에 요약합니다.
0.5B smoke 성공을 30B full SFT의 메모리 적합성이나 장기 수렴 증거로 해석해서는 안 됩니다.

## Verify a Run

Controller의 `manifest.json`에서 최종 `status`가 `passed`이고 모든 rank가 정상 종료했는지 확인합니다.
Spark output의 `summary-base.json`, `summary-train.json`, `summary-tuned.json`과 `logs/rank-<rank>-<stage>.log`도 함께 확인합니다.

`summary-train.json`에서는 finite training loss와 evaluation loss, 기대한 `actual_optimizer_steps`, 저장물을 확인합니다.
DDP summary는 sample한 trainable parameter의 변화도 기록합니다.
FSDP2와 DeepSpeed summary의 `parameter_update_evidence`가 `optimizer steps only`이면 parameter 변화 자체를 검증한 결과로 읽을 수 없습니다.

결과 판정 기준과 공통 검사 명령은 [Verification](../verification.md)을 따릅니다.
