# Megatron Backend

이 백엔드는 Hugging Face snapshot을 Megatron Bridge provider와 설정으로 연결하고 Megatron Core 기반 분산 SFT를 실행합니다.
기본 launcher는 LoRA를 사용하며 제공된 소형 smoke preset은 full SFT를 명시합니다.
설정·실행·측정과 checkpoint 재개를 구분해서 사용합니다.

Canonical JSONL은 전체 row를 RAM에 모으지 않고 rank별 output으로 순차 변환합니다.
Bridge의 Hugging Face dataset cache도 같은 output 아래에 두며 DataLoader는 disk-backed Arrow dataset에서 batch를 읽습니다.

## Parallelism Concepts

| 방식 | 나누는 대상 | 이 저장소의 범위 |
| --- | --- | --- |
| DP | 데이터 배치 | 작은 dense 모델의 기본 비교 |
| TP | 레이어의 텐서 연산 | TP=2와 sequence parallel 비교 |
| PP | 레이어 구간 | topology 검증과 논리적 배치 |
| CP | 한 샘플의 token 구간 | Spark 학습은 CP=1 |
| EP | MoE expert | MoE 모델의 expert 분산 |

Spark에서 `world_size=TP*PP*DP`이며 `world_size`는 `TP*PP`와 `PP*EP`로 나누어져야 합니다.
`GLOBAL_BATCH_SIZE`는 `MICRO_BATCH_SIZE*DP`로 나누어져야 합니다.
노드 메모리가 자동으로 하나의 pool이 되는 것은 아닙니다.

## Spark Environment

각 노드에 ARM64·CUDA 호환 Python, CUDA Torch, Megatron Bridge·Core와 Transformer Engine을 준비합니다.
[requirements-spark.txt](../../backends/megatron/requirements-spark.txt)는 Torch 2.10.0의 cu130 build를 먼저 설치하도록 명시합니다.
일반 `scripts/setup.sh`는 이 환경을 완성하는 설치기가 아닙니다.

과거 환경은 Torch `2.10.0+cu130`, Bridge `0.6.0`, Core `0.19.0`, Transformer Engine `2.18.0`을 사용했습니다.
Transformer Engine은 ARM64에서 source build했고 optional NCCL EP를 `NVTE_WITH_NCCL_EP=0`으로 제외했습니다.
이미 설치된 wheel에 환경변수를 바꿔도 extension이 추가되지 않습니다.

현재 requirements의 ModelOpt `0.46.0`은 Bridge가 선언하는 `0.46.0rc1`과 다릅니다.
과거 실행은 기존 환경을 복제해 검증한 것이며 새 환경의 전체 dependency resolution 성공을 보장하지 않습니다.
따라서 이 저장소만으로 검증된 단일 fresh-install 명령을 제공할 수 없습니다.
충돌을 임의로 무시하는 설치 명령 대신 실제 의존성·build 결과를 별도 기록해야 합니다.

먼저 각 노드에서 [공통 준비 스크립트](../../setups/spark/README.md#prepare-each-spark-node)를 실행해 system package와 node-local Megatron 가상환경을 준비합니다.
이 스크립트는 아래 Python package를 대신 설치하지 않으므로, 현재 검증된 환경을 옮기거나 실제 build 결과를 기록하며 의존성을 설치해야 합니다.
준비된 환경을 검사할 때는 각 노드의 `backends/megatron`에서 실행합니다.
`PYTHON_HEADERS`는 native helper build에 필요한 경우에만 해당 환경의 Python development header 경로로 지정합니다.

```bash
cd "/path/to/shared/post-training-lab/backends/megatron"
. "$HOME/.local/ptl/venvs/megatron/bin/activate"
export PYTHON="$HOME/.local/ptl/venvs/megatron/bin/python"
source scripts/spark_runtime_env.sh
python -c 'import torch, megatron.bridge, megatron.core, transformer_engine; print(torch.__version__, torch.cuda.is_available())'
```

Helper는 venv의 userspace library 경로와 extension suffix를 설정하며 driver나 system package를 설치하지 않습니다.
Import 성공은 CUDA kernel·NCCL·학습 성공과 구분합니다.

## Spark SFT

모델·No Robots 데이터·setup 준비는 [Getting Started](../getting-started.md)와 [Datasets](../datasets.md)를 따릅니다.
공통 runner에서 `--backend megatron`과 `experiments/megatron/smoke.json`을 사용합니다.
이 preset은 작은 dense 모델, TP=1·PP=1·EP=1, micro batch 1·global batch 2, full SFT 2 step입니다.
Launcher 기본값인 30B MoE·EP=2·LoRA와 혼동하지 않습니다.

`STAGE=all`은 base 평가 → train 저장 → tuned 재로딩 평가 순서입니다.
`RESUME_AFTER_TRAIN=true`이면 train과 tuned 사이에 resume stage를 추가합니다.
공통 runner는 SSH로 모든 노드를 시작하지만 직접 `run_spark_cluster.sh`를 사용하면 각 노드에서 별도로 시작해야 합니다.
직접 실행 전에는 위 runtime helper도 source해야 합니다.

`TRANSFORMER_IMPL=auto`는 Qwen에 local, GLM에 Transformer Engine 경로를 선택합니다.
`SEQUENCE_PARALLEL=true`에는 TP>=2와 `TRANSFORMER_IMPL=transformer_engine`이 필요합니다.
비교하려면 off/on 두 variant 모두 같은 Transformer Engine 경로를 사용합니다.

## CPU Offload

Megatron Core가 native training offload 대상으로 지원하는 저장 계층은 CPU memory입니다.
Optimizer state와 계산을 CPU로 옮기는 optimizer CPU offload와 activation을 비동기로 CPU에 옮기는 activation offload가 있지만, NVMe를 parameter·optimizer·activation의 실행 중 저장 계층으로 사용하는 기능은 native하게 지원하지 않습니다.
현재 이 저장소의 Bridge wrapper는 CPU offload 옵션도 아직 노출하지 않으며 `RECOMPUTE`는 activation offload가 아니라 activation 재계산입니다.

## Checkpoint and Resume

| 목적 | 설정·진입점 | 검증할 것 |
| --- | --- | --- |
| 저장물 평가 | `tuned` stage | 같은 held-out 입력으로 재로딩 |
| 같은 topology 학습 재개 | `experiments/megatron/resume-smoke.json` | iteration 2 load 후 3 실행, optimizer·scheduler load |
| Sync/async 저장 비교 | `CHECKPOINT_MODE=sync` 또는 `async` | save와 blocking finalization 별도 계측 |
| Topology 변경 | `RESUME_TP`, `RESUME_PP`, `RESUME_EP` | 저장 format과 state 복원 범위 |

Async checkpoint는 train/resume stage의 `torch_dist`와 persistent worker를 사용합니다.
종료 전에 pending save의 finalization과 필요한 모든 shard를 확인합니다.
파일이 보이거나 save 호출이 끝났다는 사실만으로 장애 후 durability를 주장하지 않습니다.

30B preset의 output·전처리 JSONL·Arrow cache·로그는 backend별 `output_root`인 `/mnt/post-training/megatron`의 로컬 NVMe에 기록합니다.
공통 runner는 `torch_dist` checkpoint만 두 노드가 함께 보는 `<checkout>/artifacts/checkpoints/<run-id>`에 기록해 다음 `tuned` 또는 `resume` process가 모든 shard와 metadata를 읽게 합니다.
직접 launcher를 실행할 때 `OUTPUT_DIR`가 node-local이면 `CHECKPOINT_DIR`을 두 노드에서 같은 NFS 경로로 지정해야 하며, 지정하지 않으면 `<output>/checkpoints`를 사용합니다.
이는 parameter·optimizer·activation의 실행 중 NVMe offload가 아니라 dataset cache와 checkpoint I/O 배치입니다.

TP/PP를 바꾸는 optimizer 재분할은 기본 `dp_reshardable` format으로 해결되지 않습니다.
`DIST_CKPT_OPTIM_FULLY_RESHARDABLE=true`인 별도 source checkpoint와 optimizer 저장·로드 조건이 필요합니다.
`FULLY_PARALLEL_SAVE=true`만으로 이 format이 활성화되지는 않습니다.
현재 검증 범위와 남은 제한은 [Verification](../verification.md)을 확인합니다.

## Measurements and Known Limits

반복 비교와 output schema는 [Experiments](../experiments.md)에서 관리합니다.
현재 selective recompute는 `recompute_num_layers=1` 설정 때문에 Bridge 검증에서 실패합니다.
이는 메모리 부족이나 하드웨어의 원천적 미지원 증거가 아니며, 문서 작업에서 구현을 변경하지 않았습니다.

Stage metadata와 rank별 로그를 모두 확인합니다.
현재 launcher는 평가 loss를 출력하는 마지막 global rank의 로그를 읽고 마지막 노드에서 `summary.json`을 작성합니다.
Rank 0만 summary를 쓴다는 이전 설명은 현재 코드와 맞지 않습니다.
재로딩 성공·전체 상태의 수치 동등성·장애 복구는 서로 다른 검증이며 [Verification](../verification.md)의 기준을 따릅니다.
