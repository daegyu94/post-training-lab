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

synthetic과 MBPP(`google-research-datasets/mbpp`, `sanitized` config, cc-by-4.0) 둘 다 실제로 `prepare` → 실제 Docker 평가까지 검증했습니다.
MBPP `sanitized` config의 문제 설명 필드는 `prompt`입니다(`text`가 아님) — 다른 config나 향후 dataset 개정에서 필드명이 다시 바뀔 수 있으니, 새 revision으로 바꿀 때는 `datasets.load_dataset(...).column_names`로 실제 스키마를 먼저 확인합니다.

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

Reasoning 모델(Qwen3 계열 등)은 코드보다 먼저 `<think>...</think>` 블록을 출력하므로 `enable_thinking=False`로 미리 닫아 생성을 요청합니다.
이걸 하지 않으면 짧은 `--max-new-tokens`에서 사고 과정만으로 예산이 소진되어 코드가 전혀 나오지 않을 수 있습니다(Qwen3-30B-A3B, `--max-new-tokens 64`에서 12/12 후보가 잘린 `<think>` 텍스트만 반환한 사례로 확인).

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
단일 process 실행(`WORLD_SIZE=1`)에서는 모델을 `device_map="auto"`로 불러 GPU에 shard 단위로 직접 올립니다.
분산 실행에서는 각 rank가 이미 자기 device를 알고 있으므로 이 옵션을 켜지 않습니다(켜면 보이는 GPU 전체에 잘못 분산됩니다).
`--mode sft`는 `loss_type="nll"`을 명시적으로 지정합니다 — 기본값 `chunked_nll`은 model forward를 patch하는데, `device_map="auto"`가 메모리 부족으로 일부 layer를 CPU offload하면 그 layer의 forward가 `functools.partial`로 감싸져 patch가 `'functools.partial' object has no attribute '__func__'`로 깨집니다.

`--lora-r`는 PEFT의 `target_modules="all-linear"`로 적용됩니다.
PEFT의 architecture 자동 매핑은 이 repo가 다루는 MoE 구조(Qwen3-30B-A3B의 fused-expert parameter, GLM의 MLA attention)를 모르기 때문에, target_modules를 지정하지 않으면 `--lora-r`가 `No target_modules passed but also no target_parameters found`로 즉시 실패합니다.

Qwen3-30B-A3B(LoRA, 48 layer 전체 all-linear)로 이 세 단계 모두 실제 GPU에서 검증했습니다.
`--gradient-checkpointing` 없이는 이 하드웨어(GB10, 통합 메모리 119GiB)에서 backward 시점에 CUDA OOM이 발생했습니다 — LoRA라도 48 layer 활성화를 전부 들고 있으면 여유가 없습니다.
30B 스케일에서는 `--gradient-checkpointing`을 기본으로 켜는 것을 권장합니다.

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
GPU 연산은 Spark 노드에서 실행하고 Docker 임시는 local NVMe에 둡니다.
`WORK_DIR`(dataset·checkpoint·결과)은 30B 규모에서는 각 노드의 local NVMe(`/mnt/post-training/execution-feedback/...`, 다른 backend와 같은 관례)를 권장합니다.
NFS(`/home/spark/shared/...`)는 controller에서 바로 확인하기 편하지만 30B checkpoint(adapter만 써도 수 GB)가 매 단계 네트워크를 타므로, 여러 checkpoint를 오가는 A/B/C/D 전체 cycle에서는 local이 더 안전합니다.
비교·공유가 필요한 최종 산출물(`comparison.json`, 작은 manifest)만 다 끝난 뒤 NFS로 복사합니다.

```bash
cd /home/spark/shared/post-training-lab
SFT_CHECKPOINT=/path/to/common-sft-checkpoint \
WORK_DIR=/mnt/post-training/execution-feedback/run-001 \
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

다음 환경변수로 규모를 조정합니다(기본값은 30B 전체 규모, 필요하면 축소).

| 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `NUM_TRAIN_CANDIDATES` | 8 | train task당 생성할 candidate 수 |
| `MAX_NEW_TOKENS` | 512 | candidate 생성 최대 token 수(train·test 공통) |
| `TRAIN_TEMPERATURE` / `TRAIN_TOP_P` | 0.8 / 0.95 | train candidate 생성 sampling 설정 |
| `MAX_STEPS` | 64 | B/C/D 공통 학습 step 수 |
| `GRADIENT_CHECKPOINTING` | 0 | `1`이면 B/C/D 학습에 `--gradient-checkpointing` 적용 |
| `SAVE_CHECKPOINT` | 0 | `1`이면 B/C/D 학습에 `--save-checkpoint` 적용(중간 재개용 Trainer checkpoint 저장) |
| `LORA_R` | 0 | 0이면 `SFT_CHECKPOINT`를 adapter로 간주하고 이어서 학습 |

`--save-checkpoint`(기본 off)는 HF Trainer의 자체 중간 checkpoint(전체 optimizer state 포함)를 저장할지 정합니다.
이 checkpoint는 `--resume-from-checkpoint`로 재개할 때만 쓰이고, `trainer.save_model()`이 저장하는 실제 결과물(`output_dir/model`)과 별개로 중복 저장됩니다.
실측으로 LoRA adapter 4GB에 optimizer state만 8GB가 추가로 붙어 checkpoint당 2배 이상 커졌습니다 — 재개 계획이 없다면 기본값(off)을 유지합니다.

쉬운 task와 충분히 학습된 checkpoint를 쓰면 sampling만으로는 fail이 전혀 안 나올 수 있습니다.
실제로 Qwen3-30B-A3B(LoRA)로 4개 synthetic train task를 시도했을 때 `temperature=1.4, top_p=1.0`, candidate 6개까지도 24/24 전부 pass했습니다.
이 경우 DPO pair를 얻으려면 같은 task에 대해 `--max-new-tokens`를 의도적으로 줄인 별도 생성을 추가해 진짜 truncation fail을 섞는 방법이 있습니다 — 조작된 label이 아니라 짧은 예산에서 나온 실제 모델 출력입니다.

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


## Verification Evidence

Spark 실행 기록은 commit 메시지에서 확인할 수 있습니다.

- `f7b9f0b`: spark1의 실제 Docker에서 단일 timeout과 workers=4 평가 후 container 잔존 여부를 확인했고, network·filesystem·PID 제한을 검사했습니다.
- `8efc8e0b`: 설치된 TRL 1.12.0 source에서 full DPO의 reference precompute 경로를 확인했습니다. GPU 학습 완료나 메모리 실측 결과를 의미하지 않습니다.

이후 spark1에서 Qwen3-30B-A3B(실제 base checkpoint, LoRA r=16 all-linear)로 A/B/C/D 전체 cycle을 처음부터 끝까지(초기 SFT → train candidate 생성 → 실제 Docker 평가 → feedback → B/C/D 학습 → test candidate 생성·평가 → compare)를 실제로 실행해 완료했습니다.
그 과정에서 이 문서 위쪽에 기록된 네 가지 실제 버그(`target_modules` 누락, 단일 process `device_map`, `chunked_nll`/CPU-offload 충돌, reasoning 모델의 `<think>` 예산 소진)를 GPU 실행 중 발견하고 고쳤습니다.
DPO(C, D)는 loss가 0.70→0.04~0.09로, `rewards/accuracies`가 1.0으로, `rewards/margins`가 계속 증가하는 정상적인 학습 곡선을 보였습니다.
Test task가 2개뿐이고 synthetic task 자체가 이 모델에는 쉬워서 A/B/C/D 모두 pass@1=1.0으로 나와 성공률 차이는 관측되지 않았습니다 — 이는 이 실행의 task 난이도·표본 크기 한계이며 pipeline 결함이 아닙니다.
실험 비교(성공률 차이 관측 포함)에는 더 크거나 어려운 test set과 각 variant의 generation manifest, training summary, test evaluation이 필요합니다.

MBPP(`sanitized` config)도 `prepare`와 실제 Docker 평가로 검증했습니다.
`adapt_mbpp`가 존재하지 않는 `text` 필드를 읽어 모든 MBPP row에서 `KeyError`로 즉시 실패하는 버그가 있었습니다(실제 필드명은 `prompt`) — 이 경로는 네트워크와 실제 dataset이 필요해 CPU tier 테스트가 전혀 커버하지 못했습니다.
고친 뒤 실제 MBPP revision(`4bb6404fdc6cacfda99d4ac4205087b89d32030c`)으로 4개 task를 준비하고 참조 정답을 실제 Docker에서 평가해 4/4 pass를 확인했습니다.

평가 timeout의 stdout/stderr는 UTF-8 문자열로 변환하고 마지막 4,000자만 저장합니다.
출력이 있는 후보가 timeout되어도 JSONL 저장을 계속할 수 있습니다.
채점 결과의 test 개수와 pass 개수를 검사하며, 모든 task test가 통과한 경우에만 pass로 판정합니다.
이 검사는 runner 출력의 형식 검사이며, 같은 process에서 실행되는 코드에 대한 부정행위 방지 장치는 아닙니다.

Trainer는 입력 token 계측을 명시적으로 활성화합니다.
이 변경 이전 summary의 0 또는 누락된 token 수를 실제 처리량 0으로 해석하지 않습니다.
