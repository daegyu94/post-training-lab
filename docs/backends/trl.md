# TRL Backend

TRL은 SFT와 LoRA/QLoRA, adapter 저장·재로딩을 실습하는 백엔드입니다.
단일 GPU QLoRA와 Spark native BF16 실행은 환경·진입점·출력 형식이 다릅니다.
DPO와 RL trainer는 구현되어 있지 않습니다.

## Spark Environment

각 Spark 노드에 ARM64·CUDA 호환 Python 환경을 준비합니다.
[requirements-spark.txt](../../backends/trl/requirements-spark.txt)는 Torch 2.10.0, Transformers 5.12.1, TRL 1.12.0, Accelerate 1.14.0 등을 지정합니다.
과거 실행은 Torch `2.10.0+cu130`을 사용했으며 버전 pin만으로 같은 CUDA build가 선택되지는 않습니다.
일반 `scripts/setup.sh`는 이 Spark 설치 절차를 대신하지 않습니다.

아래는 이미 준비한 Spark 환경에 의존성을 설치하는 단계입니다.
노드의 `backends/trl`에서 실행하며 패키지를 다운로드하고 선택한 환경을 변경합니다.
CUDA Torch와 해당 architecture의 wheel 지원을 먼저 확인합니다.

```bash
export PYTHON='<spark-python>'
"$PYTHON" -m pip install -r requirements-spark.txt
"$PYTHON" -c 'import torch, transformers, trl, accelerate; print(torch.__version__, torch.cuda.is_available()); print(transformers.__version__, trl.__version__, accelerate.__version__)'
```

CUDA 가용성 출력만으로 kernel·NCCL 통신이나 모델 적합성을 확인한 것은 아닙니다.
노드별 kernel 검사 후 작은 실제 smoke로 분산 실행을 검증합니다.
설치 오류를 만났다면 의존성 충돌을 기록하고 임의의 다른 버전으로 통과시킨 결과를 같은 환경으로 취급하지 않습니다.

## Spark SFT

[Getting Started](../getting-started.md)의 setup·입력 준비와 runner를 사용합니다.
공통 runner는 SSH로 양 노드 launcher를 시작하지만 `scripts/run_spark_cluster.sh` 자체는 다른 노드에 접속하지 않습니다.
직접 launcher를 사용할 경우 모든 참여 노드에서 같은 설정과 서로 다른 `NODE_RANK`로 시작해야 합니다.

| 설정 | 지원 범위 |
| --- | --- |
| `FINETUNING_MODE=lora` 또는 `full` | adapter 또는 전체 parameter 학습 |
| `DISTRIBUTED_BACKEND=ddp` | `base`, `train`, `tuned`, `all` |
| `DISTRIBUTED_BACKEND=fsdp2` | `base` 또는 `train`만 허용 |
| `DISTRIBUTED_BACKEND=deepspeed` | `base` 또는 `train`; `DEEPSPEED_CONFIG` 필수 |
| `OPTIMIZER=adamw` 또는 `sgd` | 학습 모드와 함께 명시해 비교 |

`STAGE=all`은 base 평가 → 학습·저장 → 별도 process의 tuned 평가 순서입니다.
FSDP2와 DeepSpeed는 일반적인 sharded export·tuned reload·optimizer resume 지원을 의미하지 않습니다.
예제 ZeRO 설정은 [ZeRO-2](../../backends/trl/configs/deepspeed-zero2.json)와 [ZeRO-3](../../backends/trl/configs/deepspeed-zero3.json)에 있습니다.

DDP는 모델 복제본을 rank마다 유지하므로 노드 수를 늘려도 모델 가중치 자체가 자동 분할되지 않습니다.
LoRA도 base weight를 없애지 않습니다.
0.5B 성공을 30B full SFT의 메모리 적합성으로 일반화하지 않습니다.
모델·백엔드별 실패 원인은 [과거 TRL 검증 기록](../verification/trl-20260908/README.md)에 보존합니다.

## Single-GPU QLoRA

이 경로는 Qwen용 template과 NF4 QLoRA를 사용하며 CUDA와 BF16 지원이 필요합니다.
`backends/trl`에서 아래 setup을 실행하면 `.venv`에 일반 requirements를 설치하고 UltraChat parquet을 내려받습니다.
모델은 내려받지 않으며 학습 wrapper는 offline 모드이므로 실행 전에 모델 snapshot을 준비해야 합니다.

```bash
cd backends/trl
bash scripts/setup.sh
```

모델 준비 후 같은 디렉터리에서 `bash scripts/run_experiment.sh`를 실행합니다.
기본 모델은 `Qwen/Qwen2.5-14B-Instruct`, 출력은 `results/qwen2.5-14b-qlora`입니다.
GPU 작업을 시작하므로 가용 메모리와 출력 경로를 먼저 확인합니다.
Wrapper의 기본 학습량은 128 train / 16 eval, 20 step, 최대 길이 512, gradient accumulation 8입니다.
추가 인자는 `trl_lab.train`에 전달됩니다.

서비스 JSONL을 사용한다면 [데이터 가이드](../datasets.md#reviewed-service-traces)로 파일을 만든 뒤 모듈을 직접 호출합니다.
Wrapper는 JSONL을 지정해도 parquet 디렉터리를 사전 검사하므로 아래 직접 진입점이 더 명확합니다.
다음 자리표시자는 실제 snapshot과 데이터 디렉터리로 바꿉니다.

```bash
.venv/bin/python -m trl_lab.train \
  --model '<qwen-model-snapshot>' \
  --dataset-jsonl-dir '<prepared-jsonl-directory>' \
  --train-samples 2 --eval-samples 1 --max-steps 1 \
  --output-dir results/service-qlora-smoke \
  --local-files-only
```

출력의 `summary.json`에는 설정, base/tuned 평가, 생성 예제, 학습 시간과 메모리가 기록됩니다.
Spark의 `summary-<stage>.json`과 동일 schema가 아니므로 같은 key를 가정하지 않습니다.
저장한 adapter를 별도 process에서 확인하는 명령은 다음과 같습니다.
학습에 쓴 base snapshot을 그대로 지정합니다.

```bash
.venv/bin/python -m trl_lab.infer results/service-qlora-smoke/adapter \
  --model '<qwen-model-snapshot>' \
  --local-files-only
```

## Verification

Rank 0의 stage summary와 모든 rank 로그에서 정상 종료, finite loss, 실제 optimizer step과 저장물을 확인합니다.
Sharded backend의 `parameter_update_evidence`가 `optimizer steps only`이면 전체 parameter 변화 검증으로 읽지 않습니다.
Adapter reload는 optimizer·scheduler 상태를 복원하는 학습 resume와 다릅니다.
공통 검사 명령과 판정은 [Verification](../verification.md)을 따릅니다.
