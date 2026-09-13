# Execution Feedback

이 실험은 사내 framework 없이 공개·합성 Python 함수 문제에서 `generation → Docker evaluation → feedback dataset → additional training → comparison`을 한 번 재현합니다.
Megatron SFT 경로는 변경하지 않으며 execution feedback은 TRL 경로에서만 실험합니다.

성능 향상 자체가 완료 조건은 아닙니다.
같은 입력과 설정으로 전체 cycle을 끝내고 데이터·checkpoint·평가 결과를 남기는 것이 완료 조건입니다.

## Experiment Matrix

| Variant | Training sequence | Question |
| --- | --- | --- |
| A | SFT only | 시작 checkpoint의 실행 성공률은 얼마인가? |
| B | SFT → execution-filtered SFT | 통과 답변을 더 학습하면 개선되는가? |
| C | SFT → DPO | 같은 문제의 성공·실패 답변 선호 학습이 효과적인가? |
| D | SFT → execution-filtered SFT → DPO | filtered SFT 뒤에도 같은 DPO pair가 추가 효과를 내는가? |

A/B/C/D는 같은 최초 SFT checkpoint에서 파생됩니다.
B와 D는 같은 filtered-SFT checkpoint를 사용하고 C와 D는 최초 SFT 모델이 만든 동일한 DPO 파일을 사용합니다.
D를 위해 후보를 다시 생성하지 않습니다.
D는 DPO pair가 하나 이상일 때만 실행합니다.

## Data Contract

`execution_feedback.prepare`는 다음 필드를 가진 `tasks.jsonl`과 split별 JSONL을 만듭니다.

| Field | Meaning |
| --- | --- |
| `task_id` | 문제의 안정적인 ID |
| `prompt` | 문제 설명과 함수 요구사항 |
| `reference_solution` | 최초 SFT용 정답 코드 |
| `test_setup` | test 전에 한 번 실행할 선택적 setup 코드 |
| `tests` | 독립적으로 채점하는 Python assertion 목록 |
| `split` | `train`, `validation`, `test` |
| `source` | 합성 workload 또는 공개 dataset ID |
| `template_group` | 같은 template 변형의 split 누수 검사 단위 |

합성 workload는 pipeline smoke와 correctness 확인용입니다.
모델 성능 결론에는 공개 dataset 평가를 사용하고 합성 결과와 구분합니다.

합성 데이터를 준비합니다.

```bash
python -m execution_feedback.prepare \
  --source synthetic \
  --output-dir /path/to/shared/execution-feedback/data
```

MBPP를 사용하려면 이용 조건을 먼저 확인하고 Hugging Face dataset의 불변 commit SHA를 지정합니다.
원본의 `train`, `validation`, `test` split을 유지합니다.

```bash
python -m execution_feedback.prepare \
  --source mbpp \
  --revision '<40-character-dataset-commit>' \
  --output-dir /path/to/shared/execution-feedback/data
```

`initial_sft_train.jsonl`과 `initial_sft_validation.jsonl`은 최초 SFT 입력입니다.
다음처럼 최초 SFT checkpoint를 만든 뒤 아래 cycle의 공통 시작점으로 전달할 수 있습니다.

```bash
python -m execution_feedback.train \
  --mode sft --model-dir /path/to/base-model \
  --train-file /path/to/data/initial_sft_train.jsonl \
  --eval-file /path/to/data/initial_sft_validation.jsonl \
  --output-dir /path/to/checkpoints/initial-sft
```

기존 TRL SFT workflow로 동일 입력을 학습해도 됩니다.

## Execution Evaluation

모델 응답의 가장 큰 Python code fence를 코드로 사용하며 fence가 없으면 전체 응답을 코드로 사용합니다.
평가기는 syntax compile과 assertion별 실행을 수행하고 다음 상태를 기록합니다.

| Status | Meaning |
| --- | --- |
| `pass` | syntax와 모든 test 통과 |
| `fail` | 추출·syntax·runtime·test 실패 |
| `timeout` | 제한 시간 초과 |
| `infra_error` | Docker 실행 자체를 시작하거나 완료할 수 없음 |

`score`는 통과한 assertion 비율입니다.
`infra_error`와 `timeout`은 DPO rejected에 넣지 않습니다.

Docker가 기본 engine입니다.
각 후보는 별도 임시 디렉터리와 container에서 실행되며 network·capability를 제거하고 read-only root filesystem, CPU·memory·PID·시간 제한을 적용합니다.
Dataset과 checkpoint는 container에 mount하지 않습니다.

```bash
python -m execution_feedback.evaluate \
  --tasks /path/to/data/tasks.jsonl \
  --candidates /path/to/train-candidates.jsonl \
  --output /path/to/train-evaluations.jsonl \
  --engine docker --image python:3.12-slim \
  --workers 4 --cpus 1 --memory 256m --pids-limit 64 --timeout-seconds 10
```

`timeout`으로 분류된 후보의 container는 강제로 제거됩니다.
`subprocess.run(timeout=...)`은 `docker run` client 프로세스만 죽이고 그 client가 daemon에 띄운 container는 그대로 남아 CPU·memory 자원을 계속 점유합니다(`--rm`은 container가 스스로 끝났을 때만 정리하며 client 연결 종료로는 정리되지 않습니다).
각 실행에 고유한 `--name`을 부여하고 timeout 시 `docker rm --force`로 정리해 이를 막습니다.
정리는 best-effort이며 실패해도 timeout 판정 자체는 그대로 유지됩니다.

`--engine local`은 저장소의 신뢰할 수 있는 smoke test에만 사용합니다.
모델 출력이나 공개 dataset 코드를 local engine으로 실행하지 않습니다.

한 호스트에서 같은 후보 묶음으로 `1 / 2 / 4 / 8` workers를 각각 측정해 처리량, 전체 시간과 timeout 비율을 비교한 뒤 기본값을 정합니다.
다중 호스트 worker와 pNFS/3FS 비교는 이 PoC 범위에 포함하지 않습니다.

## Feedback Dataset

후보는 최초 SFT checkpoint로 train 문제에서 한 번만 생성합니다.

```bash
python -m execution_feedback.generate \
  --model-dir /path/to/sft-checkpoint \
  --tasks /path/to/data/tasks.jsonl --split train \
  --num-candidates 8 --seed 42 --temperature 0.8 --top-p 0.95 \
  --output /path/to/train-candidates.jsonl

python -m execution_feedback.feedback \
  --tasks /path/to/data/tasks.jsonl \
  --evaluations /path/to/train-evaluations.jsonl \
  --output-dir /path/to/feedback \
  --max-sft-per-task 1 --max-pairs-per-task 1
```

`filtered_sft.jsonl`은 pass 코드만 포함합니다.
`dpo.jsonl`은 같은 문제와 같은 후보 pool의 pass를 `chosen`, fail을 `rejected`로 사용합니다.
Validation과 test 평가가 입력되면 feedback 생성은 실패합니다.
문제별 상한은 쉬운 문제가 데이터 대부분을 차지하는 것을 막습니다.

## Additional Training

`execution_feedback.train`은 TRL의 `SFTTrainer`와 `DPOTrainer`를 사용합니다.
단일 GPU에서는 `python`, 분산 실행에서는 준비된 환경의 `torchrun` 또는 `accelerate launch` 뒤에 `-m execution_feedback.train ...`을 붙입니다.

```bash
python -m execution_feedback.train \
  --mode sft --model-dir /path/to/sft-checkpoint \
  --train-file /path/to/feedback/filtered_sft.jsonl \
  --eval-file /path/to/data/initial_sft_validation.jsonl \
  --output-dir /path/to/checkpoints/B

python -m execution_feedback.train \
  --mode dpo --model-dir /path/to/sft-checkpoint \
  --train-file /path/to/feedback/dpo.jsonl \
  --output-dir /path/to/checkpoints/C --beta 0.1

python -m execution_feedback.train \
  --mode dpo --model-dir /path/to/checkpoints/B/model \
  --train-file /path/to/feedback/dpo.jsonl \
  --output-dir /path/to/checkpoints/D --beta 0.1
```

D는 C와 정확히 같은 `dpo.jsonl`과 hyperparameter를 사용합니다.
D의 총 학습 token과 시간이 더 크므로 성공률만으로 효율이 더 좋다고 판단하지 않습니다.

`--lora-r`를 생략한 full fine-tuning DPO(위 C·D)는 `precompute_ref_log_probs=True`를 자동으로 켭니다.
DPOTrainer는 `ref_model=None`이고 PEFT가 아니면 기본적으로 모델 전체를 한 번 더 로드해 reference로 상주시키는데, 이는 30B에서 resident weight를 조용히 두 배로 만듭니다.
이 옵션은 그 대신 학습 시작 전 현재 모델로 reference logprob을 한 번만 계산해 캐싱하고 두 번째 사본을 만들지 않습니다.
LoRA checkpoint(새로 만든 adapter거나 이미 있는 adapter)는 TRL이 adapter를 끄는 것만으로 reference를 저렴하게 얻으므로 이 옵션이 필요 없습니다.

## End-to-End Cycle

순차 실행 wrapper는 generation, Docker evaluation, B/C/D 추가 학습, 고정 test 평가와 비교를 연결합니다.
GPU 연산은 Spark 노드에서 실행하고 dataset·checkpoint·결과는 NFS, Docker 임시는 local NVMe에 둡니다.

```bash
cd /home/spark/shared/post-training-lab
SFT_CHECKPOINT=/path/to/common-sft-checkpoint \
WORK_DIR=/home/spark/shared/execution-feedback/run-001 \
DOCKER_WORKERS=4 \
bash scripts/run_execution_feedback_cycle.sh
```

분산 trainer launcher는 환경에 맞게 바꿉니다.

```bash
TRAIN_LAUNCH='torchrun --nproc-per-node=1' \
SFT_CHECKPOINT=/path/to/common-sft-checkpoint \
bash scripts/run_execution_feedback_cycle.sh
```

Wrapper는 GPU generation, Docker evaluation과 training을 순차 실행해 자원 경쟁을 피합니다.
DPO pair가 없으면 C와 D를 건너뛰고 A/B만 비교합니다.

## Final Comparison

각 variant는 같은 test task, seed, decoding 설정으로 후보 하나만 생성합니다.
주요 지표는 모든 test를 통과한 문제의 비율인 strict pass@1입니다.

```bash
python -m execution_feedback.compare \
  --variant A=/path/to/test-evaluations-A.jsonl \
  --variant B=/path/to/test-evaluations-B.jsonl \
  --variant C=/path/to/test-evaluations-C.jsonl \
  --variant D=/path/to/test-evaluations-D.jsonl \
  --training-summary B=/path/to/checkpoints/B/execution_feedback_training.json \
  --training-summary C=/path/to/checkpoints/C/execution_feedback_training.json \
  --training-summary D=/path/to/checkpoints/D/execution_feedback_training.json \
  --training-summary D=/path/to/checkpoints/B/execution_feedback_training.json \
  --output /path/to/comparison.json
```

비교기는 test 외 split, `candidate_id != 0`, 중복 task와 variant 사이 task 집합 차이를 오류로 처리합니다.
`comparison.json`에는 pass@1, 상태별 개수, Docker 평가 시간과 추가 학습 시간·관측된 입력 token 수가 남습니다.
후보 생성 manifest와 training summary에는 모델 경로, generation 설정, 학습 설정과 시간이 남습니다.

## CPU Smoke Validation

합성 정답과 의도적으로 실패하는 답변으로 데이터 준비, 평가와 feedback 생성을 확인합니다.

```bash
ENGINE=local bash scripts/run_execution_feedback_smoke.sh /tmp/execution-feedback-smoke
```

`ENGINE=local`은 격리 검증이 아닙니다.
실제 후보를 사용하기 전 `ENGINE=docker`로 다시 실행해 Docker image와 제한 설정을 검증합니다.
