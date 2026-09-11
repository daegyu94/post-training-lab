# LoRA Trainable-Ratio Checkpoint I/O 실험

## 목적

[30B Local NVMe Checkpoint와 Memory 실험](checkpoint-memory-30b-experiment.md) Part A는 LoRA rank 8(기본값, 전체 model-parallel shard의 약 0.03%)에서만 checkpoint I/O를 측정했습니다.
이 실험은 같은 측정 파이프라인(Part A/B)을 재사용해 **LoRA trainable parameter 비율이 커지면 checkpoint 크기와 I/O 특성이 어떻게 달라지는지**를 봅니다.

## LoRA rank 선택

Megatron은 학습 시작 시 rank-local model-parallel shard 기준 trainable parameter 수와 비율을 직접 로그로 남깁니다 (`Trainable parameters`, `Trainable percentage`).
`lora_dim=8`(기본값) 실측값을 기준으로 삼았습니다.

```text
Trainable parameters (lora_dim=8): 5,111,808
Rank-local shard parameters (TP=1, PP=1, EP=2): 16,041,719,808
Trainable percentage: 5,111,808 / 16,041,719,808 = 0.0319%
```

LoRA는 `target_modules=["linear_qkv", "linear_proj"]`에 rank당 `r × (in_dim + out_dim)`만큼 파라미터를 더하므로 trainable parameter 수는 `lora_dim`에 정확히 비례합니다.
`5,111,808 / 8 = 638,976`이 rank 1당 trainable parameter 수이며, 목표 비율 `R`에 대해 `lora_dim = round(R × 16,041,719,808 / 638,976)`으로 역산합니다.

| ID | 목표 비율 | `LORA_DIM` | 역산 비율 |
| --- | ---: | ---: | ---: |
| `RATIO-0.1PCT` | 0.1% | 25 | 0.0996% |
| `RATIO-0.5PCT` | 0.5% | 126 | 0.5019% |
| `RATIO-1PCT` | 1% | 251 | 0.9998% |

비율은 global 30.52B parameter가 아니라 **Megatron이 실제로 로그에 남기는 rank-local shard 기준**입니다 — 이 저장소의 다른 곳에서 "trainable percentage"를 볼 때와 같은 정의를 씁니다.

## 조건

Part A와 동일한 topology·데이터·계측을 재사용하고 `LORA_DIM`만 바꿉니다.

```text
FINETUNING_MODE=lora
TP=1
PP=1
EP=2
DP=2
STAGE=train
MAX_STEPS=8
SAVE_INTERVAL=2
SAVE_OPTIMIZER=true
MAX_LENGTH=2048
PAD_TO_MAX_LENGTH=true
MEASURE_TIMING=true
CHECKPOINT_MODE=sync
CHECKPOINT_PLACEMENT=local
```

Sync만 사용합니다 — sync/async 비교는 Part A가 이미 답했고, 여기서는 checkpoint 크기 축만 바꿉니다.
Cohort는 checkpoint phase가 쓰는 `ultrachat-qwen3-30b-2048-v1`을 그대로 재사용합니다(LoRA rank는 데이터 요구사항에 영향을 주지 않음).

각 조건은 pilot 1회 + 측정 3회입니다(총 3개 조건 × 4 run = 12 run).
측정값·집계 방법·유효성 기준은 [checkpoint-memory-30b-experiment.md](checkpoint-memory-30b-experiment.md)의 Part A/B와 동일한 파이프라인(`checkpoint_io_probe.py`, `resource_sampler.py`)을 그대로 사용하므로 여기서 반복하지 않습니다.

## 실행

진입점은 기존 `experiments/checkpoint_memory_30b.py`이며 `--plan`만 이 실험 전용 plan으로 바꿉니다.

```bash
python experiments/checkpoint_memory_30b.py \
  --setup setups/spark/local-checkpoint-memory-30b.json \
  --phase checkpoint \
  --plan experiments/megatron/lora-ratio-checkpoint-io.json \
  --output results/lora-ratio-checkpoint-io-dry \
  --repeats 3

python experiments/checkpoint_memory_30b.py \
  --setup setups/spark/local-checkpoint-memory-30b.json \
  --phase checkpoint \
  --plan experiments/megatron/lora-ratio-checkpoint-io.json \
  --output results/lora-ratio-checkpoint-io \
  --repeats 3 \
  --execute
```

`checkpoint_summary()`의 variant 키는 plan의 cell이 정의한 이름(`ratio-0.1pct`, `ratio-0.5pct`, `ratio-1pct`)을 그대로 따르며, 각 variant의 `logical_checkpoint_bytes_median`으로 checkpoint 크기가 LoRA rank에 비례하는지 확인합니다.

## 유효성 기준

[checkpoint-memory-30b-experiment.md의 유효성 기준](checkpoint-memory-30b-experiment.md#유효성-기준)을 그대로 적용합니다.
추가로, 세 조건의 `logical_checkpoint_bytes_median`이 `LORA_DIM`에 거의 선형으로 비례하는지 확인합니다 — 벗어나면 optimizer state 계산이나 target module 구성이 예상과 다르다는 신호입니다.
