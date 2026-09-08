# TRL Post-Training Lab

TRL backend는 SFT, LoRA/QLoRA, adapter 저장·재로딩과 Spark 분산 실행을 제공합니다.
DPO와 RL trainer는 이 repository에 포함되어 있지 않습니다.

실제 실행 방법과 제한은 [TRL backend guide](../../docs/backends/trl.md)에 모읍니다.
공통 환경과 runner 사용법은 [Getting Started](../../docs/getting-started.md)에서 확인합니다.

```text
trl_lab/train.py          single-GPU QLoRA entry point
trl_lab/infer.py          adapter reload entry point
trl_lab/spark_train.py    Spark training entry point
trl_lab/spark_config.py   model and dataset validation
scripts/                  setup, data preparation and launchers
configs/                  DeepSpeed configuration files
```

Repository root에서 테스트를 실행합니다.

```bash
python -m pytest -q
```
