# TRL Backend

TRL backend는 SFT, LoRA/QLoRA, adapter 저장·재로딩과 Spark 분산 실행을 제공합니다.
DPO와 RL trainer는 이 repository에 포함되어 있지 않습니다.

## Single-GPU QLoRA

backends/trl/scripts/setup.sh는 .venv와 UltraChat parquet을 준비하지만 model snapshot을 자동으로 내려받지는 않습니다.
실행 host에서 Qwen model을 사용할 수 있게 준비하고 필요하면 DATASET_DIR와 DATASET_PARQUET_DIR를 지정합니다.

```bash
cd backends/trl
./scripts/setup.sh
./scripts/run_experiment.sh
```

기본 output은 results/qwen2.5-14b-qlora입니다.
summary.json에는 base/tuned evaluation, generation, train time과 peak memory가 기록되고 adapter는 adapter/에 저장됩니다.
추가 option은 trl_lab.train --help에 정의된 --model, --dataset-jsonl-dir, --max-steps 등을 launcher 뒤에 전달할 수 있습니다.

Adapter를 별도 process에서 확인할 때는 다음 진입점을 사용합니다.

```bash
.venv/bin/python -m trl_lab.infer results/qwen2.5-14b-qlora/adapter \
  --model Qwen/Qwen2.5-14B-Instruct \
  --local-files-only
```

## Spark SFT

공통 runner로 experiments/trl/smoke.json 또는 사용자가 만든 experiment를 실행합니다.
실제 stage launcher는 base, train, tuned를 별도 process로 실행합니다.
summary-<stage>.json과 rank log를 output에서 확인합니다.

FINETUNING_MODE는 lora 또는 full입니다.
DISTRIBUTED_BACKEND는 ddp, fsdp2, deepspeed 중 하나입니다.
FSDP2와 DeepSpeed는 현재 base 또는 train stage만 허용되며 sharded export/reload를 일반 지원으로 간주하지 않습니다.
DeepSpeed를 선택하면 DEEPSPEED_CONFIG가 필요합니다.

## 확인 기준

Train output에서 loss와 gradient가 finite인지 확인합니다.
각 rank의 process가 정상 종료하고 expected summary와 adapter/checkpoint가 생성됐는지 확인합니다.
짧은 smoke는 품질 향상이나 30B 모델의 메모리 적합성을 증명하지 않습니다.
세부적인 검증 판정은 [Verification](../verification.md)으로 통일합니다.

## 테스트

Repository root에서 CPU unit test를 실행합니다.

```bash
python -m pytest -q
```
