# Megatron Backend

Hugging Face snapshot을 Bridge로 연결해 Megatron Core 분산 SFT를 실행합니다.
Launcher 기본값은 LoRA, 소형 smoke preset은 full SFT입니다.

Canonical JSONL은 rank별 output으로 순차 변환하고 같은 위치에 Arrow cache를 둡니다.
DataLoader는 전체 row를 RAM에 모으지 않고 disk-backed dataset에서 batch를 읽습니다.

## Parallelism Concepts

| 방식 | 나누는 대상 | 이 저장소의 범위 |
| --- | --- | --- |
| DP | 데이터 배치 | 작은 dense 모델의 기본 비교 |
| TP | 레이어의 텐서 연산 | TP=2와 sequence parallel 비교 |
| PP | 레이어 구간 | topology 검증과 논리적 배치 |
| CP | 한 샘플의 token 구간 | Spark 학습은 CP=1 |
| EP | MoE expert | MoE 모델의 expert 분산 |

`world_size = TP*PP*DP`이며 `world_size`는 `TP*PP`와 `PP*EP`로 나누어져야 하고, `GLOBAL_BATCH_SIZE`는 `MICRO_BATCH_SIZE*DP`로 나누어져야 합니다.
노드 메모리가 자동으로 하나의 pool이 되는 것은 아닙니다.

## Spark Environment

검증된 환경은 Torch `2.10.0+cu130`, Bridge `0.6.0`, Core `0.19.0`, Transformer Engine `2.18.0`입니다.
Transformer Engine은 ARM64에 사전 빌드 wheel이 없어 각 노드에서 source build하며 optional NCCL EP를 `NVTE_WITH_NCCL_EP=0`으로 제외합니다.
이미 설치된 wheel에 환경변수만 바꿔도 extension이 추가되지 않습니다.

**`megatron-bridge==0.6.0`은 `--no-deps`로 설치합니다.**
Base package가 요구하는 `fast-hadamard-transform`은 해당 PyPI sdist에 `csrc/`가 없고 ARM64 wheel도 없어 설치가 막힙니다.
Qwen/GLM SFT·LoRA·checkpoint에 필요한 의존성만 [requirements-spark.txt](../../backends/megatron/requirements-spark.txt)로 설치합니다.
고정된 `nvidia-modelopt==0.46.0`은 Core의 `nvidia-modelopt[torch]>=0.44` 요구를 충족합니다.

각 노드에서 [공통 준비 스크립트](../../setups/spark/README.md#prepare-each-spark-node) 실행 후 아래 Python package를 설치합니다.
Native helper build에 필요하면 `PYTHON_HEADERS`에 development header 경로를 지정하고, wheel은 checkout 밖에 보관합니다.

```bash
cd "/path/to/shared/post-training-lab/backends/megatron"
. "$HOME/.local/ptl/venvs/megatron/bin/activate"
export PYTHON="$HOME/.local/ptl/venvs/megatron/bin/python"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.10.0
python -m pip install --no-deps megatron-bridge==0.6.0
python -m pip install -r requirements-spark.txt
source scripts/spark_runtime_env.sh
install -d "$HOME/.local/ptl/wheels"
python -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir "$HOME/.local/ptl/wheels" transformer-engine-torch==2.18.0
python -m pip install transformer-engine==2.18.0 transformer-engine-cu13==2.18.0 \
  "$HOME"/.local/ptl/wheels/transformer_engine_torch-2.18.0-*.whl
python -c 'import torch, megatron.bridge, megatron.core, transformer_engine; from megatron.bridge import AutoBridge; from megatron.bridge.peft.lora import LoRA; assert torch.__version__ == "2.10.0+cu130" and torch.version.cuda == "13.0" and torch.cuda.is_available(); print((torch.ones(1, device="cuda") + 1).item())'
```

`spark_runtime_env.sh`는 venv의 userspace library 경로와 extension suffix를 설정할 뿐 driver나 system package를 설치하지 않습니다.
`python -m pip check`는 위에서 건너뛴 package 9개를 누락으로 보고하며 이는 알려진 결과입니다.
Import 성공은 CUDA kernel·NCCL·학습 성공과 구분합니다.

## Spark SFT

모델·데이터·setup 준비는 [Getting Started](../getting-started.md)와 [Datasets](../datasets.md)를 따르고, 공통 runner에서 `--backend megatron`과 `experiments/megatron/smoke.json`을 사용합니다.
이 smoke preset은 작은 dense 모델, TP=1·PP=1·EP=1, micro 1·global 2, full SFT 2 step이며 launcher 기본값인 30B MoE·EP=2·LoRA와 다릅니다.

분산 preset은 `STAGE=train`으로 학습과 node-local checkpoint 저장만 실행합니다.
Backend의 restore 기능은 남아 있지만 shared checkpoint storage가 없는 현재 실험 범위에서는 사용하지 않습니다.
직접 `run_spark_cluster.sh`를 쓰면 각 노드에서 따로 시작해야 하며 위 runtime helper도 source해야 합니다.

`TRANSFORMER_IMPL=auto`는 Qwen에 local, GLM에 Transformer Engine 경로를 선택합니다.
`SEQUENCE_PARALLEL=true`에는 TP>=2와 `TRANSFORMER_IMPL=transformer_engine`이 필요하며, 비교할 때는 off/on 두 variant가 같은 Transformer Engine 경로를 써야 합니다.

## Local Checkpoint

`CHECKPOINT_PLACEMENT=local`은 각 rank의 `<OUTPUT_DIR>/checkpoints`에 shard를 저장합니다.
Sync/async 비교는 `CHECKPOINT_MODE`로 선택하고 save 호출과 blocking finalization을 따로 계측합니다.
Async checkpoint는 train에서 `torch_dist`와 persistent worker를 사용합니다.
종료 전 pending save finalization과 모든 shard를 확인하되, save 완료를 장애 후 durability 보장으로 해석하지 않습니다.

30B preset의 output·전처리 JSONL·Arrow cache·로그는 `/mnt/post-training/megatron` 로컬 NVMe에 기록합니다.
이는 실행 중 NVMe state offload가 아니라 dataset cache와 checkpoint I/O의 배치입니다.

분산 restore·resume은 모든 rank shard가 보이는 shared checkpoint storage가 준비된 뒤 검증할 TODO입니다.
현재 local checkpoint 실험은 cross-node 복원, topology 변경과 장애 복구를 증명하지 않습니다.

## CPU Offload and Known Limits

Megatron Core의 native training offload는 **CPU memory만** 지원합니다(optimizer state·계산 또는 activation 이동).
NVMe state offload는 없고 이 저장소의 Bridge wrapper는 CPU offload도 노출하지 않습니다.
`RECOMPUTE`는 activation 재계산입니다.

Qwen LoRA의 full/selective recompute는 `experiments/megatron/recompute-30b.json`으로 비교합니다.
현재 launcher는 평가 loss를 출력하는 마지막 global rank의 로그를 읽고 마지막 노드에서 `summary.json`을 작성합니다.
실험 판정은 [Experiments](../experiments.md#판정-기준)를 따릅니다.
