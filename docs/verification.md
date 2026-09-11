# Verification

정적 검사, CPU 테스트, dry-run, 실제 GPU 실행은 서로 다른 증거입니다.
한 단계의 성공이 다른 단계의 성공을 의미하지 않습니다.

## Repository Checks

저장소 루트에서 CPU 테스트를 실행합니다.
최소 테스트 의존성은 `requirements-dev.txt`에서 관리하며 전체 GPU 학습 stack은 이 CPU 테스트의 필수 조건이 아닙니다.
새 테스트 환경이 필요하면 다음 명령으로 준비합니다.
명령은 패키지를 다운로드하고 별도 venv를 생성합니다.

```bash
python -m venv .venv-check
. .venv-check/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q backends datasets_lab experiments observability tests
for script in backends/*/scripts/*.sh observability/scripts/*.sh setups/spark/*.sh scripts/*.sh; do
  bash -n "$script" || exit 1
done
```

`.venv-check`는 Git에서 제외되므로 같은 checkout의 원격 실행 상태를 오염시키지 않습니다.

테스트는 mock과 작은 입력을 사용하는 회귀 검사이며 실제 GPU·NCCL·fresh installation 성공을 증명하지 않습니다.
실제 실행 전에는 [Getting Started의 실행 계획 검토](getting-started.md#check-the-execution-plan)로 설정을 점검할 수 있습니다.

## Judge a Run

| 확인 대상 | 성공 조건 | 증명하지 않는 것 |
| --- | --- | --- |
| Dry-run | 설정 검증·계획 생성·exit 0 | 원격 파일, GPU, SSH 준비 |
| Controller 실행 | manifest의 `passed`, 모든 rank exit 0 | 학습 품질 |
| 학습 | 예상 optimizer step, finite loss·gradient, 저장물 | 장기 수렴·다른 모델의 메모리 적합성 |
| Adapter reload | 별도 process에서 읽기·평가 완료 | optimizer·scheduler resume |
| Checkpoint resume | 기대 iteration과 상태 load 후 추가 step | uninterrupted run과 전체 수치 동등성 |
| Async 저장 | 필요한 shard와 pending save finalization | fsync·장애 후 durability |

백엔드 summary는 원격 출력 경로에 있고 controller에 자동 복사되지 않습니다.
모든 rank 로그와 모델·데이터 revision, 실제 선택한 입력, topology, seed, 환경 버전을 함께 확인합니다.
Sharded backend의 optimizer-step 증거를 전체 parameter checksum 검증으로 읽지 않습니다.
ZeRO-3 과거 summary의 parameter count 0은 placeholder 계측 문제이며 모델 크기 0이 아닙니다.

## 30B GPU Results

2026-09-09에 commit `54a5fe0b69d168a86746d7de576d5dcae61c36b9`의 깨끗한 checkout으로 `spark1`과 `spark2`에서 GPU당 process 하나를 실행했습니다.
모든 결과는 1-step 실행 가능성만 보여주며 장기 안정성이나 학습 품질을 뜻하지 않습니다.

Megatron은 `TP=1`, `PP=1`, `EP=2`, `DP=2`, BF16, sequence length 2048, global batch 2와 attention LoRA를 사용했습니다.
Qwen3-30B-A3B와 GLM-4.7-Flash 모두 1 optimizer step, 평가, async `torch_dist` checkpoint와 양 rank exit 0을 확인했습니다.
같은 조건은 `experiments/megatron/qwen3-30b-lora.json`과 `experiments/megatron/glm-4.7-flash-30b-lora.json`으로 다시 실행할 수 있습니다.

| Model | Train result | Peak allocated |
| --- | --- | --- |
| Qwen3-30B-A3B | loss `4.110986`, grad norm `15.480`, step `7.28 s` | `34.759 GiB` |
| GLM-4.7-Flash | loss `2.497179`, grad norm `7.883`, step `6.37 s` | `34.671 GiB` |

TRL은 Qwen3-30B-A3B, BF16, sequence length 2048, global batch 2와 attention LoRA를 사용했습니다.
DDP와 FSDP2 모두 1 optimizer step, finite train/eval loss, checkpoint와 양 rank exit 0을 확인했습니다.

| Backend | Train / eval loss | Peak allocated / reserved | Update evidence |
| --- | --- | --- | --- |
| DDP | `3.156563` / `2.587339` | `58.825 / 58.971 GiB` | sampled parameter delta가 0이 아님 |
| FSDP2 | `3.156250` / `2.566406` | `32.147 / 34.188 GiB` | optimizer step과 sharded checkpoint |

## NVMe and Full SFT

두 Spark 노드의 `/mnt/post-training`은 로컬 NVMe root filesystem에 있고 backend별 디렉터리에 `spark` 쓰기 권한이 있습니다.
Megatron Qwen3-30B-A3B와 GLM-4.7-Flash LoRA는 로컬 NVMe에 데이터 cache와 로그를 기록하고, NFS의 공통 `torch_dist` checkpoint에서 iteration 1을 새 process로 재로딩해 `STAGE=all`을 완료했습니다.
Qwen은 40,000/8,000건에서 base loss `1.402232`와 tuned loss `1.342737`, GLM은 160,000/30,000건에서 `2.269922`와 `2.012836`을 기록했습니다.

Megatron은 NVMe를 native training state offload 대상으로 지원하지 않으므로 이 결과는 dataset cache와 checkpoint I/O 검증입니다.
TRL의 ZeRO-3 NVMe offload는 별도의 DeepSpeed AIO extension build와 실제 1-step 완료를 성공 조건으로 사용합니다.

30B full SFT의 기존 DDP와 FSDP2 실패는 dataset 전체 적재가 원인이 아닙니다.
DDP는 parameter와 gradient가 unified memory 한도에 근접하고, 설치된 Accelerate의 FSDP2 준비 과정은 sharding 전에 trainable BF16 parameter를 FP32로 올립니다.
남은 후보였던 TRL의 DeepSpeed ZeRO-3 NVMe parameter·optimizer offload는 2노드에서 1 optimizer step(train loss 약 13.21), 별도 `tuned` process에서의 평가(eval loss 약 11.96)와 복구 가능한 native ZeRO checkpoint까지 확인했습니다.
같은 실행에서 두 노드 모두 `/mnt/post-training/trl/zero_stage_3` swap footprint가 약 256GiB로, 각 노드 물리 RAM 119GiB보다 컸습니다. NVMe offload가 없으면 이 구성은 노드 RAM만으로 담을 수 없다는 근거이며, 자세한 수치와 한계는 [30B NVMe 실습](../labs/nvme-30b/README.md#why-nvme-offload-is-necessary-here)을 따릅니다.

## Known Implementation Limits

- Controller commit이 `unknown`이면 runner가 원격 commit 일치 검사를 생략합니다.
- Megatron `requirements-spark.txt`는 `megatron-bridge==0.6.0`을 `--no-deps`로 설치하고 sub-dependency는 별도로 고정합니다. base package METADATA가 무조건 요구하는 `fast-hadamard-transform==1.1.0`(PyPI sdist에 `csrc/` 소스가 없고 ARM64 wheel도 없어 설치 불가능)을 `--no-deps`로 건너뛰기 위한 것으로, `[recipes]` extra와는 무관합니다. 2026-09-11에 fresh venv(`torch==2.10.0+cu130` + 두 단계 설치)에서 `megatron.bridge`, `megatron.core`, `AutoBridge`, `megatron.bridge.peft.lora.LoRA` import를 확인했습니다. 자세한 내용은 [Megatron 문서](backends/megatron.md#spark-environment)를 따릅니다.
- 서비스 변환 manifest는 두 Spark validator의 고정 데이터 형식과 다릅니다.
- Sharded export·resume와 parameter 계측은 백엔드별 제한이 있으므로 학습 성공과 별도 검증해야 합니다.
- Megatron 30B full-parameter(non-LoRA) SFT는 아직 **실측으로는** 검증되지 않았습니다. `experiments/megatron/qwen3-30b-full.json`(검증된 `qwen3-30b-lora.json`과 `FINETUNING_MODE`만 다름)으로 시도했으나, 로컬에 준비된 `no_robots` 5000행 중 한 행이 검증된 LoRA preset과 같은 `MAX_LENGTH=2048`을 넘어 `cluster_data.py`의 "never truncates" 정책에 막혔습니다 — 이 경계값은 memory 비교의 기준이라 늘리지 않았습니다.
  [메모리 추정기](experiments.md#estimate-memory-before-running-experimentsestimate_memorypy)는 이 구성을 per-rank 149.7 GiB(파라미터 29.9 + gradient 29.9 + Adam optimizer 89.6)로 예측해 노드당 119 GiB 예산을 **약 31 GiB 초과**한다고 봅니다. 같은 구성에서 `--optimizer sgd`는 90.0 GiB로 예산 안에 들어옵니다. 둘 다 추정이며 실행으로 확인한 값이 아닙니다.

실패를 지원 불가능으로 일반화하거나 임의의 revision·버전으로 우회하지 않습니다.
