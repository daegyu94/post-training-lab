# Step 1: SFT Workflow Baseline

## Objectives

Step 1은 LLM supervised fine-tuning(SFT)의 전체 경로가 실제로 동작하는지 확인하는 단계입니다. 작은 sample과 짧은 run을 사용하더라도 dataset 준비, tokenization, training, checkpoint 저장, 새 프로세스 재로딩, held-out evaluation을 모두 통과해야 합니다.

## Workflow

```text
Dataset → preprocessing/tokenization → SFT training → checkpoint or adapter save
        → fresh-process reload → held-out evaluation
```

학습 중 출력되는 batch loss는 서로 다른 batch에서 계산되므로 항상 감소할 필요가 없습니다. 최종 확인은 학습에 사용하지 않은 같은 held-out subset에서 base와 tuned loss를 비교하는 방식으로 합니다.

## TRL Baseline

TRL workflow는 4-bit QLoRA를 사용해 제한된 GPU memory에서도 SFT 경로를 빠르게 확인합니다.

```bash
./scripts/trl/setup.sh
CUDA_VISIBLE_DEVICES=0 ./scripts/trl/run_experiment.sh
```

기본 virtual environment는 `.venv-trl`입니다. dataset 경로와 결과 경로는 각각 `DATASET_DIR`, `DATASET_PARQUET_DIR`, `OUTPUT_DIR`로 바꿀 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> DATASET_PARQUET_DIR=<dataset-parquet-dir> OUTPUT_DIR=<output-dir> \
  ./scripts/trl/run_experiment.sh
```

실행이 끝난 뒤에는 `summary.json`에서 `quality.tuned_eval_loss < quality.base_eval_loss`인지, `artifacts.adapter_dir`에 adapter가 생성됐는지 확인합니다. 실제 실행 기록은 [TRL Run Record](trl-run-record.md)에 남겨 두었습니다.

## Megatron-LM Workflow

Megatron Bridge workflow는 Megatron-LM 계열의 model·data·checkpoint 경로를 검증하기 위한 실행입니다. 현재 script는 한 GPU에서 동작하도록 제한되어 있으므로, large-scale 성능 측정이나 multi-node 검증 결과를 의미하지 않습니다.

```bash
./scripts/megatron/setup.sh
./scripts/megatron/prepare_data.sh
./scripts/megatron/download_model.sh
CUDA_VISIBLE_DEVICES=0 ./scripts/megatron/run_experiment.sh
```

기본 virtual environment는 `.venv-megatron`입니다. BF16 Qwen2.5-14B checkpoint를 사용하므로 GPU memory가 부족하면 preflight가 실행 전에 중단합니다. 이때는 더 작은 model로 recipe와 memory requirement를 함께 조정해야 하며, 단순히 GPU 수를 노출하는 것만으로 multi-GPU 실행이 되지는 않습니다.

## Validation

두 workflow 모두 `results/` 아래에 log와 `summary.json`을 만듭니다. 다음 항목을 확인하세요.

두 `summary.json`은 모두 `schema_version: 1`과 `configuration`, `environment`, `quality`, `performance`, `artifacts`, `validation` section을 사용합니다. TRL은 측정한 training performance를 기록하고, 현재 Megatron workflow는 측정값을 추정하지 않고 빈 `performance` object를 기록합니다.

| 항목 | 확인 기준 |
| --- | --- |
| held-out loss | tuned loss가 base loss보다 낮은지 |
| perplexity | tuned perplexity가 base perplexity보다 낮은지 |
| training log | `nan` 또는 무한대가 없는지 |
| 저장 산출물 | adapter 또는 checkpoint가 생성되고 재로딩되는지 |

짧은 subset에서의 loss 감소는 구현된 학습 경로가 동작했다는 신호일 뿐, 일반적인 instruction-following 품질 향상이나 benchmark 성능을 보장하지 않습니다.

GPU 실행 전에 framework별 CPU unit test를 독립 environment에서 확인할 수 있습니다.

```bash
.venv-trl/bin/python -m pytest -q tests/trl
.venv-megatron/bin/python -m pytest -q tests/megatron
```
