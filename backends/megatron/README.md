# Megatron Bridge Post-Training Lab

Megatron backend는 Megatron Bridge recipe를 사용한 Spark 분산 SFT와 checkpoint workflow를 제공합니다.
주요 학습 대상은 full parameter SFT이며 LoRA와 parallelism feature도 설정으로 다룹니다.

실제 실행 방법과 제한은 [Megatron backend guide](../../docs/backends/megatron.md)에 모읍니다.
공통 runner와 setup은 [Getting Started](../../docs/getting-started.md)에서 확인합니다.

```text
megatron_lab/config.py          model provider configuration
megatron_lab/sft.py             base/train/resume/tuned entry point
megatron_lab/cluster.py         topology validation
megatron_lab/feature_lab.py     feature measurement harness
megatron_lab/parallelism.py     CPU-only rank layout simulation
scripts/                        setup, data and launchers
```

```bash
python -m pytest -q
```
