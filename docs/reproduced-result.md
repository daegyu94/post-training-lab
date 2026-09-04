# Reproduced Result

이 문서는 이 브랜치에서 직접 실행한 Megatron Core 결과를 기록합니다. 생성된 `summary.json`은 Git에서 제외하고 재현 가능한 command와 핵심 수치만 남깁니다.

## Environment

| Item | Value |
| --- | --- |
| Date | 2026-09-04 |
| GPU | NVIDIA RTX PRO 4000 Blackwell 24GiB 2장 |
| GPU-to-GPU topology | PCIe `NODE`, NVLink 없음 |
| Driver | 580.173.02 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| Megatron Core | 0.15.3 |
| CUDA runtime | 12.8 |

Transformer Engine과 Apex는 설치하지 않았으며, 세 run 모두 `transformer_impl="local"`, BF16 parameter, PyTorch LayerNorm을 사용했습니다. 시작 시 출력되는 local implementation fallback warning은 이 구성에서 예상된 동작입니다.

## Model Configuration

| Item | Value |
| --- | --- |
| Layers | 2 |
| Hidden size | 256 |
| Attention heads | 4 |
| FFN hidden size | 1024 |
| Vocabulary | 1024 |
| Sequence length | 64 |
| Micro batch per DP rank | 4 |
| Optimizer | AdamW |
| Learning rate | `3e-3` |
| Steps | 20 |
| Seed | 1234 |

각 step의 mock input token `x`에 대한 label은 `(x + 1) % vocab_size`입니다. DP rank는 서로 다른 input을 사용하고 TP rank는 같은 input을 사용하므로, loss가 내려가면 forward/backward와 해당 parallel group의 collective를 거친 optimizer update가 작동했다는 뜻입니다.

## Results

| Mode | TP | DP | First loss | Last loss | Change | Steps/s | Global samples/s | Peak allocated/rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single GPU | 1 | 1 | 6.9636 | 5.4001 | -22.45% | 54.50 | 218.00 | 0.0332GiB |
| Two-GPU DP | 1 | 2 | 6.9648 | 3.6476 | -47.63% | 45.84 | 366.73 | 0.0332GiB |
| Two-GPU TP | 2 | 1 | 6.9884 | 5.6107 | -19.71% | 54.76 | 219.03 | 0.0252GiB |

single-GPU와 TP run은 global batch 4이고 DP run은 global batch 8이므로 loss 감소율을 parallel strategy의 품질 차이로 직접 비교할 수 없습니다. DP는 step/s가 15.9% 낮아졌지만 두 배의 sample을 처리해 global sample throughput이 68.2% 증가했습니다.

TP=2의 rank당 allocated memory는 TP=1보다 약 24.1% 작았습니다. 전체 GPU reserved memory나 NCCL buffer가 아니라 PyTorch peak allocated 값이고 model이 매우 작으므로, 이 비율을 큰 model의 memory scaling으로 일반화하면 안 됩니다.

TP=2 처리량이 single GPU와 거의 같은 결과는 작은 matrix 연산에서 PCIe collective overhead가 shard 계산 감소를 상쇄했음을 보여줍니다. TP는 이 예제의 속도를 높이기보다 Megatron의 실제 sharding과 collective path를 확인하는 실습입니다.

## Commands

```bash
./scripts/run_single_gpu.sh --steps 20
./scripts/run_two_gpu_dp.sh --steps 20
./scripts/run_two_gpu_tp.sh --steps 20
```

세 명령은 모두 exit code 0으로 끝났고 각 run의 모든 rank가 summary 수집과 process group 정리를 완료했습니다.

## Multi-Node Simulation

```bash
./scripts/simulate_multi_node.sh
```

기본 설정인 2 node x 8 GPU, TP=2, PP=2, CP=1, EP=1에서 world size 16과 DP=4를 계산했습니다. 16개 rank 모두 `tp-cp-ep-dp-pp` 순서의 coordinate, node와 local rank를 받았습니다.

불가능한 설정도 실패하는지 test로 확인했습니다. 예를 들어 world size 2에서 TP=2, PP=2는 `world_size=2 is not divisible by TP*PP*CP*EP=4` 오류를 내므로 실제 launcher가 뜨기 전에 잘못된 parallel product를 발견할 수 있습니다.

simulation은 rank arithmetic과 configuration validation만 다룹니다. 실제 multi-node NCCL 연결, rendezvous, network bandwidth, failure handling과 pipeline schedule 실행을 증명하지 않으므로 cluster에서는 별도로 검증해야 합니다.

## Actual Console Capture (2026-09-04)

다음 출력은 현재 worktree의 `megatron_lab.train`을 PyTorch 2.11.0+cu128, Megatron Core 0.15.3에서 실제 실행해 얻었습니다. GPU 0에는 다른 작업이 15 GiB 이상을 사용 중이어서 유휴 GPU 1을 선택했습니다. `Apex is not installed. Falling back to Torch Norm` warning은 `transformer_impl="local"`에서 PyTorch LayerNorm을 사용한다는 예상된 안내이며 실패가 아닙니다.

### Single-GPU Console

현재 worktree의 `megatron_lab.train`에 `--tp 1 --steps 20`을 주어 실행했습니다. 기록 과정에서는 이미 설치된 호환 환경의 `torchrun`과 임시 output directory를 사용했으며, README의 스크립트는 같은 module과 학습 인수를 프로젝트 `.venv`에서 실행합니다. 아래 JSON은 출력 전체 중 결과를 판단하는 필드만 남긴 것이며, step 출력은 원문 그대로입니다.

```console
$ CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc-per-node=1 -m megatron_lab.train --tp 1 --steps 20 --output-dir /tmp/megatron-lab-single
... UserWarning: Apex is not installed. Falling back to Torch Norm
step=1 loss=6.963608
step=2 loss=6.766295
step=3 loss=6.723458
step=4 loss=6.715652
step=5 loss=6.660501
step=6 loss=6.749352
step=7 loss=6.483191
step=8 loss=6.480265
step=9 loss=6.299361
step=10 loss=6.305313
step=11 loss=6.130777
step=12 loss=5.945565
step=13 loss=6.042095
step=14 loss=5.812787
step=15 loss=5.872142
step=16 loss=5.578861
step=17 loss=5.477566
step=18 loss=5.561527
step=19 loss=5.338500
step=20 loss=5.400084
{
  "topology": {
    "world_size": 1,
    "tensor_parallel": 1,
    "data_parallel": 1,
    "pipeline_parallel": 1
  },
  "metrics": {
    "first_loss": 6.9636077880859375,
    "last_loss": 5.400084495544434,
    "loss_change_percent": -22.452776493480027,
    "steps_per_second": 66.28884079508123
  },
  "ranks": [
    {
      "rank": 0,
      "local_rank": 0,
      "peak_allocated_memory_gib": 0.03322649002075195
    }
  ]
}
```

처음 loss `6.963608`보다 마지막 loss `5.400084`가 22.45% 낮으므로 forward, backward, optimizer step이 수행됐음을 알 수 있습니다. 중간 step이 조금 오르내리는 것은 매번 새 mock batch를 만들기 때문입니다. `world_size`, TP, DP가 모두 1이고 rank도 하나이므로 이 출력은 병렬 통신이 없는 기준선입니다.

### Topology Console (GPU 불필요)

다음 명령도 실제 실행했습니다. GPU 두 장을 TP=2로 쓴다고 가정하되, GPU를 점유하거나 NCCL 통신을 만들지는 않습니다.

```console
$ ./scripts/simulate_multi_node.sh --nodes 1 --gpus-per-node 2 --tp 2 --pp 1
{
  "nodes": 1,
  "gpus_per_node": 2,
  "tensor_parallel": 2,
  "pipeline_parallel": 1,
  "context_parallel": 1,
  "expert_parallel": 1,
  "data_parallel": 1,
  "world_size": 2,
  "order": "tp-cp-ep-dp-pp",
  "ranks": [
    {"tp": 0, "cp": 0, "ep": 0, "dp": 0, "pp": 0, "rank": 0, "node": 0, "local_rank": 0},
    {"tp": 1, "cp": 0, "ep": 0, "dp": 0, "pp": 0, "rank": 1, "node": 0, "local_rank": 1}
  ]
}
```

`data_parallel: 1`은 model 사본이 하나라는 뜻이고, rank 0/1의 `tp`만 0/1로 다릅니다. 즉 두 GPU가 같은 model의 서로 다른 tensor 조각을 맡습니다. NCCL 통신과 TP model 실행까지 검증하려면 두 GPU가 유휴 상태에서 `./scripts/run_two_gpu_tp.sh --steps 20`을 실행해야 합니다.

### Test Console

```console
$ python -m pytest -q
...                                                                      [100%]
3 passed in 0.01s
```

이 테스트는 유효한 topology의 DP 계산과, 불가능한 parallel product가 `ValueError`로 거부되는지를 확인합니다.
