# Megatron Core Lab

이 브랜치는 작은 GPT와 mock token을 사용해 Megatron Core의 model-parallel process group과 tensor/data parallel 동작을 직접 확인하는 독립 실습입니다. 큰 model이나 실제 corpus가 없어도 single GPU와 single-host multi-GPU에서 core API를 실행할 수 있고, multi-node topology는 GPU cluster 없이 검증할 수 있습니다.

## 먼저 알아둘 점: Megatron Core와 Megatron-LM

이 저장소는 **Megatron Core**를 사용하는 입문 실습입니다. Megatron Core는 GPT layer, TP/DP process group, 병렬 linear layer처럼 다른 학습 프로그램에서 재사용할 수 있는 라이브러리입니다. 이 실습의 `megatron_lab/train.py`도 `megatron.core.models.gpt.GPTModel`과 `megatron.core.parallel_state`만 직접 가져옵니다.

**Megatron-LM**은 Megatron Core 위에 만든 더 큰 학습 애플리케이션입니다. 실제 데이터셋 읽기, 수많은 학습 옵션, checkpoint, 재시작, 대규모 launcher recipe와 `pretrain_gpt.py` 같은 실행 진입점까지 포함합니다.

| 지금 이 실습 | Megatron-LM 전체 학습 |
| --- | --- |
| 병렬화가 모델과 GPU에 어떤 일을 하는지 확인 | 대규모 데이터셋으로 모델을 오래 학습 |
| 작은 mock batch와 2-layer GPT | 데이터 전처리, 설정 파일, checkpoint와 scheduler 필요 |
| `megatron-core` 패키지만 설치 | Megatron-LM checkout/환경과 training recipe 필요 |

따라서 Megatron-LM 설명이 적었던 것은 누락이 아니라 의도된 범위입니다. 처음부터 전체 training stack을 올리면 병렬화 자체보다 데이터와 환경 설정이 더 큰 장벽이 됩니다. 이 실습으로 TP/DP를 이해한 뒤 [Megatron-LM training examples](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/training-examples.md)의 `pretrain_gpt.py --mock-data`로 넘어가는 순서가 좋습니다.

## 처음 실행하는 순서

1. 저장소 최상위에서 `nvidia-smi`를 실행해 CUDA GPU가 보이는지 확인합니다. GPU가 없으면 topology simulation만 실행할 수 있습니다.
2. `./scripts/setup.sh`를 실행합니다. 프로젝트 안의 `.venv`에 PyTorch, Megatron Core, pytest가 설치되고 이후 스크립트는 이 환경을 자동으로 사용합니다.
3. GPU 없이 rank 배치를 먼저 확인하려면 `./scripts/simulate_multi_node.sh --nodes 1 --gpus-per-node 2 --tp 2 --pp 1`을 실행합니다. `world_size: 2`, `data_parallel: 1`, 두 rank의 `tp: 0`/`tp: 1`이 보이면 같은 모델의 tensor shard 둘을 뜻합니다. 실제 출력은 [topology console](docs/reproduced-result.md#topology-console-gpu-불필요)에 있습니다.
4. 다음으로 `./scripts/run_single_gpu.sh --steps 20`을 실행합니다. `step=1 loss=...`부터 `step=20 loss=...`와 JSON summary가 출력되면 성공입니다. 정확히 같은 수치보다 마지막 loss가 처음보다 전반적으로 낮은지를 봅니다. GPU 0이 바쁘고 GPU 1이 비어 있다면 `CUDA_VISIBLE_DEVICES=1 ./scripts/run_single_gpu.sh --steps 20`을 사용합니다.
5. 마지막으로 두 GPU가 모두 유휴일 때 DP와 TP 명령을 각각 실행합니다. DP는 모델 사본 둘의 gradient를 평균내고, TP는 한 model layer의 연산을 둘로 나눠 collective 통신합니다.

Transformer Engine과 Apex는 이 입문 실습에 설치하지 않습니다. 따라서 `Apex is not installed. Falling back to Torch Norm` 경고는 오류가 아니라 PyTorch 기본 LayerNorm을 사용한다는 뜻입니다.

## Exercise Matrix

| Exercise | Required hardware | What it verifies | This node |
| --- | --- | --- | --- |
| Single GPU | NVIDIA GPU 1장 | GPTModel forward/backward, optimizer, local PyTorch layer spec | 실제 실행 |
| Data parallel | NVIDIA GPU 2장 | replicated model, 서로 다른 mock batch, DP gradient all-reduce | 실제 실행 |
| Tensor parallel | NVIDIA GPU 2장 | attention/MLP weight shard와 TP collective | 실제 실행 |
| Pipeline parallel | GPU 2장 이상 | layer stage와 pipeline schedule | topology dry run |
| Context parallel | GPU 2장 이상과 긴 sequence | sequence shard와 CP communication | topology dry run |
| Multi-node 3D parallel | node 2대 이상 | TP·PP·DP rank 배치와 launcher 설정 | simulation |

실제 node에는 RTX PRO 4000 Blackwell 24GiB GPU가 2장 있지만 GPU 사이에 NVLink가 없고 PCIe host bridge 경로를 사용합니다. 따라서 이 브랜치의 multi-GPU 결과는 기능 검증에는 유효하지만 NVLink/NVSwitch cluster의 성능을 대표하지 않습니다.

## Setup

Megatron Core 공식 요구사항인 Python 3.10 이상, PyTorch 2.6 이상과 CUDA GPU가 필요합니다.

```bash
./scripts/setup.sh
```

이 실습은 Transformer Engine이나 Apex 없이 `transformer_impl="local"`을 사용합니다. fused kernel과 FP8을 학습하려면 NVIDIA NGC PyTorch container 또는 Megatron Core의 `training,dev` extra가 필요하며, 별도 CUDA extension build는 이 실습 범위에 포함하지 않습니다.

## Single GPU

```bash
./scripts/run_single_gpu.sh --steps 20
```

두 layer, hidden size 256, sequence length 64의 GPT가 synthetic next-token rule을 학습합니다. 결과는 `results/single-gpu/summary.json`에 저장됩니다.

## Data Parallel

```bash
./scripts/run_two_gpu_dp.sh --steps 20
```

각 DP rank는 다른 seed의 mock batch를 만들고, optimizer step 전에 Megatron data-parallel group에서 gradient를 합산합니다. 두 rank의 model은 같은 update를 적용합니다.

## Tensor Parallel

```bash
./scripts/run_two_gpu_tp.sh --steps 20
```

TP=2에서는 Megatron Core가 attention head, MLP와 vocabulary projection을 두 rank에 나누고 collective communication을 수행합니다. `hidden-size`, `num-attention-heads`, `vocab-size`는 TP 크기로 나누어져야 합니다.

## Multi-Node Simulation

다음 명령은 2 node x 8 GPU에서 TP=2, PP=2를 가정하고 DP=4를 계산한 뒤 16개 global rank의 node, local rank와 parallel coordinate를 출력합니다.

```bash
./scripts/simulate_multi_node.sh
```

잘못된 parallel product는 실행 전에 실패합니다.

```bash
NODES=2 GPUS_PER_NODE=8 TP=4 PP=8 ./scripts/simulate_multi_node.sh
```

실제 multi-node 실행에서는 topology 검증 후 scheduler 환경에 맞게 `torchrun --nnodes`, `--node-rank`, `--master-addr`, `--master-port`를 설정해야 합니다. 실습 script는 존재하지 않는 host나 port를 임의로 만들지 않습니다.

## Results

이 node에서 직접 실행한 single-GPU, DP=2, TP=2 결과와 해석은 [reproduced result](docs/reproduced-result.md)에 기록합니다.

## What to Try Next

- sequence length를 늘려 activation memory 변화를 비교합니다.
- `--hidden-size`와 `--num-layers`를 늘려 TP가 필요한 지점을 찾습니다.
- NGC container에서 Transformer Engine local spec, BF16과 FP8을 비교합니다.
- official `pretrain_gpt.py --mock-data`로 full Megatron-LM training stack을 확인합니다.
- distributed checkpoint를 TP=1로 저장하고 TP=2로 resharding해 불러옵니다.

## References

- [Megatron Core installation](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html)
- [Megatron Core first training run](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)
- [Parallelism strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [NVIDIA Megatron-LM training examples](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/training-examples.md)
