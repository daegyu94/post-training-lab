# Execution Feedback

이 실험은 사내 framework 없이 공개 MBPP Python 함수 문제에서 `generation → Docker evaluation → feedback dataset → additional training → comparison`을 한 번 재현합니다.
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
| `source` | 공개 dataset ID |
| `template_group` | 같은 template 변형의 split 누수 검사 단위 |

UltraChat과 No Robots는 실행 test가 없는 대화 SFT 데이터이므로 execution feedback의 채점 입력으로 쓸 수 없습니다.
이 branch의 execution feedback 준비·학습·평가는 test가 포함된 MBPP와 spark1의 단일 GPU만 사용합니다.

MBPP 이용 조건을 확인한 뒤 데이터를 준비합니다.
기본 revision은 실제 검증한 불변 commit SHA로 고정되어 있으며 필요할 때만 다른 40자리 SHA를 지정합니다.
원본의 `train`, `validation`, `test` split을 유지합니다.

```bash
python -m execution_feedback.prepare \
  --output-dir /path/to/shared/execution-feedback/data
```

MBPP(`google-research-datasets/mbpp`, `sanitized` config, cc-by-4.0)는 실제로 `prepare` → Docker 평가까지 검증했습니다.
MBPP `sanitized` config의 문제 설명 필드는 `prompt`입니다(`text`가 아님) — 다른 config나 향후 dataset 개정에서 필드명이 다시 바뀔 수 있으니, 새 revision으로 바꿀 때는 `datasets.load_dataset(...).column_names`로 실제 스키마를 먼저 확인합니다.

`initial_sft_train.jsonl`과 `initial_sft_validation.jsonl`은 최초 SFT 입력입니다.
다음처럼 spark1에서 최초 SFT checkpoint를 만든 뒤 아래 cycle의 공통 시작점으로 전달합니다.

```bash
python -m execution_feedback.train \
  --mode sft --model-dir /path/to/base-model \
  --train-file /mnt/post-training/execution-feedback/data/initial_sft_train.jsonl \
  --eval-file /mnt/post-training/execution-feedback/data/initial_sft_validation.jsonl \
  --output-dir /mnt/post-training/execution-feedback/initial-sft
```

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
기본 실행은 같은 sampling stream에서 정상 token budget 후보 8개와 32-token 후보 1개를 생성합니다.
짧은 후보도 실제 모델 출력이며, 정상 후보가 pass한 문제에서 truncation fail을 만들어 DPO pair가 비는 일을 줄입니다.

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
이 branch에서는 spark1의 단일 process만 지원하며 분산 환경으로 실행하면 즉시 실패합니다.
모델은 `device_map="auto"`로 불러 GPU에 shard 단위로 직접 올립니다.
`--mode sft`는 `loss_type="nll"`을 명시적으로 지정합니다 — 기본값 `chunked_nll`은 model forward를 patch하는데, `device_map="auto"`가 메모리 부족으로 일부 layer를 CPU offload하면 그 layer의 forward가 `functools.partial`로 감싸져 patch가 `'functools.partial' object has no attribute '__func__'`로 깨집니다.

`--lora-r`는 attention projection(GQA의 `q/k/v/o_proj`, GLM MLA의 `q_a/q_b/kv_a_with_mqa/kv_b_proj`)과 MoE router(`gate`)를 명시적 `target_modules` 목록으로 적용합니다.
PEFT의 architecture 자동 매핑은 이 repo가 다루는 MoE 구조(Qwen3-30B-A3B의 fused-expert parameter, GLM의 MLA attention)를 모르기 때문에, target_modules를 지정하지 않으면 `--lora-r`가 `No target_modules passed but also no target_parameters found`로 즉시 실패합니다.
`target_modules="all-linear"`는 이 에러는 피하지만 MoE expert의 fused parameter(`experts.gate_up_proj`/`down_proj`)까지 LoRA 대상에 포함시킵니다 — `device_map="auto"`가 그중 일부를 CPU/meta device로 offload하면 backward에서 `GroupedMmBackward0 returned an invalid gradient ... expected device meta but got cuda:0`로 깨집니다(Qwen3-30B-A3B에서 실제로 재현).
그래서 expert parameter는 아예 건드리지 않는 명시적 목록을 씁니다.
`up_proj`/`gate_proj`/`down_proj`는 여기 넣지 않습니다 — 이 이름들이 이 MoE 구조의 fused expert parameter 이름(`gate_up_proj`, `down_proj`)과 suffix가 겹쳐서 같은 문제를 다시 끌어들이기 때문입니다(dense MLP만 있는 모델이라면 안전하게 추가할 수 있습니다).

Qwen3-30B-A3B(LoRA, attention+router target_modules)로 이 세 단계 모두 실제 GPU에서 검증했습니다.
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

순차 실행 wrapper는 MBPP 준비, generation, Docker evaluation, B/C/D 추가 학습, 고정 test 평가와 비교를 연결합니다.
GPU 연산은 spark1에서 실행하며 dataset·checkpoint·결과는 NFS를 거치지 않도록 local NVMe의 `WORK_DIR`에 둡니다.

```bash
cd /home/spark/shared/post-training-lab
PYTHON=/home/spark/.local/ptl/venvs/trl/bin/python \
SFT_CHECKPOINT=/mnt/post-training/execution-feedback/initial-sft/model \
WORK_DIR=/mnt/post-training/execution-feedback/run-001 \
DOCKER_WORKERS=4 \
bash scripts/run_execution_feedback_cycle.sh
```

Wrapper는 GPU generation, Docker evaluation과 training을 순차 실행해 자원 경쟁을 피합니다.
DPO pair가 없으면 C와 D를 건너뛰고 A/B만 비교합니다.

다음 환경변수로 규모를 조정합니다(기본값은 30B 전체 규모, 필요하면 축소).

| 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `NUM_TRAIN_CANDIDATES` | 8 | train task당 생성할 candidate 수 |
| `NUM_TRUNCATED_CANDIDATES` | 1 | train task당 짧은 token budget 후보 수 |
| `TRUNCATED_MAX_NEW_TOKENS` | 32 | 짧은 후보의 생성 token 상한 |
| `LIMIT_PER_SPLIT` | 미지정 | 검증용으로 각 MBPP split의 앞 N개만 사용 |
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
이 경우 DPO pair를 얻으려면 같은 task에 대해 `--max-new-tokens`를 의도적으로 줄인 별도 생성을 추가해 진짜 truncation fail을 섞는 방법이 있습니다 — 조작된 label이 아니라 짧은 예산에서 나온 실제 모델 출력입니다.

## Final Comparison

각 variant는 같은 test task, seed, decoding 설정으로 후보 하나만 생성합니다.
주요 지표는 모든 test를 통과한 문제의 비율인 strict pass@1입니다.
각 variant를 생성할 때 같은 MBPP validation reference에 대한 NLL도 한 번 측정해 `validation_nll`로 기록합니다.
이 값은 A–D에 공통인 보조 지표지만 execution pass@1을 대신하지 않습니다.
SFT와 DPO의 training loss는 목적함수가 달라 서로 비교하지 않고, 각 stage 내부의 `loss_history`가 내려가는지만 확인합니다.

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

### GPU Cycle Runs

spark1에서 Qwen3-30B-A3B(실제 base checkpoint, LoRA r=16)로 전체 cycle(초기 SFT → train candidate 생성 → 실제 Docker 평가 → feedback → B/C/D 학습 → test candidate 생성·평가 → compare)을 세 번 실행했습니다.

| Run | `WORK_DIR` | 규모 | 결과 |
| ---: | --- | --- | --- |
| 1 | NFS (`/home/spark/shared/execution-feedback/run-30b-poc`) | synthetic 축소 test set | 완료. 아래 버그 5건 발견 |
| 2 | local NVMe (`/mnt/post-training/execution-feedback/run-local-verify`) | synthetic 축소 test set | 5건 수정 후 완료 |
| 3 | local NVMe | MBPP 원본 split 전체 | 완료. [아래 결과](#full-scale-mbpp-run-single-process-validation-nll-loss-history) |

**Run 1에서 발견한 실제 버그**

| # | 증상 | 원인 | 수정 |
| ---: | --- | --- | --- |
| 1 | `No target_modules passed but also no target_parameters found` | PEFT 자동 매핑이 이 repo의 MoE 구조를 모름 | 명시적 `target_modules` 지정 |
| 2 | 분산 실행 시 깨짐 | `device_map="auto"`는 단일 process 전제 | `world_size != 1`이면 즉시 실패 |
| 3 | `'functools.partial' object has no attribute '__func__'` | 기본 `chunked_nll`이 CPU-offload된 layer의 forward patch에 실패 | `loss_type="nll"` 명시 |
| 4 | 짧은 `--max-new-tokens`에서 코드가 전혀 안 나옴 | reasoning 모델이 `<think>` 블록으로 예산 소진 | `enable_thinking=False` |
| 5 | `GroupedMmBackward0 returned an invalid gradient ... expected device meta but got cuda:0` | `target_modules="all-linear"`가 MoE expert의 fused parameter까지 포함 | attention+router 명시 목록으로 축소 |

Run 1은 checkpoint I/O가 네트워크를 타는 문제도 드러냈습니다 — 그래서 위 "End-to-End Cycle"이 local NVMe `WORK_DIR`를 권장합니다.

**Run 2 관측**

- DPO(C, D): loss 0.70 → 0.15~0.25, `rewards/accuracies` 1.0, `rewards/margins` 지속 증가 — 정상 학습 곡선.
- 버그 5 수정의 부수 효과: LoRA target이 줄어 adapter 크기 4GB → 60MB. trainable parameter가 줄어 step당 속도도 크게 향상.
- A/B/C/D 모두 test pass@1=1.0. **synthetic test task가 2개뿐이고 이 모델에 쉬워서 차이가 안 보이는 것이며 pipeline 결함이 아닙니다.** 성공률 차이를 관측하려면 더 크거나 어려운 test set이 필요합니다.

**MBPP 경로 검증**

`adapt_mbpp`가 존재하지 않는 `text` 필드를 읽어 모든 MBPP row에서 `KeyError`로 즉시 실패했습니다(실제 필드명은 `prompt`).
이 경로는 네트워크와 실제 dataset이 필요해 CPU tier 테스트가 전혀 커버하지 못했습니다.
고친 뒤 실제 MBPP revision(`4bb6404fdc6cacfda99d4ac4205087b89d32030c`)으로 4개 task를 준비하고 참조 정답을 실제 Docker에서 평가해 4/4 pass를 확인했습니다.

평가 timeout의 stdout/stderr는 UTF-8 문자열로 변환하고 마지막 4,000자만 저장합니다.
출력이 있는 후보가 timeout되어도 JSONL 저장을 계속할 수 있습니다.
채점 결과의 test 개수와 pass 개수를 검사하며, 모든 task test가 통과한 경우에만 pass로 판정합니다.
이 검사는 runner 출력의 형식 검사이며, 같은 process에서 실행되는 코드에 대한 부정행위 방지 장치는 아닙니다.

Trainer는 입력 token 계측을 명시적으로 활성화합니다.
이 변경 이전 summary의 0 또는 누락된 token 수를 실제 처리량 0으로 해석하지 않습니다.

### Full-scale MBPP run (single process, validation NLL, loss history)

이전 두 번의 실행은 `--limit-per-split`로 줄인 소규모 test set(4~6개)에서 A/B/C/D를 비교했습니다.
이후 synthetic 경로를 제거하고 MBPP `sanitized` 원본 split을 그대로(`train=120`, `validation=43`, `test=257`) 사용해 spark1에서 세 번째 전체 cycle을 실행했습니다 — `LORA_R=16`, `GRADIENT_CHECKPOINTING=1`, `WORLD_SIZE=1` 강제(`train.py`가 `world_size != 1`이면 즉시 실패), `WORK_DIR`는 local NVMe.

`generate.py`에 `--eval-file`/`--eval-output`을 추가해 각 variant test 생성 직후 같은 checkpoint로 `initial_sft_validation.jsonl`에 대한 NLL을 한 번 더 측정하고(`validation_nll`), `train.py`는 `trainer.state.log_history`를 `loss_history`로 저장합니다. `compare.py`는 `--evaluation-summary`로 이 값을 받아 `comparison.json`에 병합합니다.

257개 전체 test task 기준 결과:

| Variant | Training sequence | pass@1 | validation NLL |
| --- | --- | --- | --- |
| A | SFT only | 0.066 (17/257) | 2.746 |
| B | SFT → filtered SFT | 0.062 (16/257) | 1.800 |
| C | SFT → DPO | **0.090 (23/257)** | 2.903 |
| D | SFT → filtered SFT → DPO | 0.070 (18/257) | 2.266 |

**두 지표가 서로 반대 방향입니다.** B는 validation NLL을 가장 크게 낮췄지만(2.746 → 1.800) pass@1은 A보다 낮고, C는 NLL이 가장 높은데 pass@1이 가장 높습니다.
NLL은 reference 코드와의 token 일치도이고 pass@1은 실행 성공 여부이므로, 두 값이 같은 방향으로 움직일 이유가 없습니다 — **NLL을 execution 성능의 대리 지표로 쓰지 않습니다.**

차이 크기의 제약: variant 간 pass@1 격차는 최대 7개 task(16 → 23/257)입니다.
반복 실행과 신뢰구간이 없으므로 이 순위를 방법 간 우열로 확정하지 않습니다.

A 초기 SFT loss는 3.5→2.1로 하락했고 NaN/Inf는 없었습니다.
B(filtered SFT)는 첫 5-step 평균 약 1.6에서 마지막 5-step 평균 약 1.0으로 하락했습니다.
C·D DPO는 loss가 0.70→0.06~0.11로, `rewards/accuracies`가 0→1.0으로, `rewards/margins`가 0 근처에서 2.2~3.6까지 계속 증가하는 곡선을 보였습니다 — 이번 스케일에서도 정상적인 DPO 수렴입니다.
Swap 사용은 관측되지 않았습니다(119GiB 중 최대 사용 약 65GiB, free 41GiB 유지).

이 실행 도중 한 번 실패했습니다: 최초 SFT checkpoint(A)가 이미 LoRA adapter인데 cycle의 B/C 학습에도 `LORA_R=16`을 그대로 넘겨 `--lora-r cannot create a second adapter on an adapter checkpoint`로 죽었습니다.
`run_execution_feedback_cycle.sh`의 `LORA_R`는 "새 adapter를 만들지(>0), 기존 adapter를 이어 학습할지(0)"를 뜻하므로, `SFT_CHECKPOINT` 자체가 이미 adapter면 cycle 실행 시 `LORA_R=0`이어야 합니다.
이미 끝난 generation(1080 candidates)·Docker 평가·feedback 데이터(파일로 저장됨)는 재사용하고 B 학습부터 올바른 값으로 재시작해 완료했습니다 — 코드 버그가 아니라 잘못된 실행 인자였습니다.

### Candidate generation batching

`generate.py`는 원래 task당 candidate마다 별도로 `model.generate()`를 호출했습니다(정상 8개 + truncated 1개 = 9번).
같은 task의 candidate는 같은 prompt를 공유하므로, 같은 `max_new_tokens` budget끼리 `num_return_sequences`로 묶어 한 번에 생성하도록 바꿨습니다(greedy/`do_sample=False` 경로는 `num_return_sequences`가 실질적으로 batch되지 않아 입력을 직접 `repeat_interleave`).
30B checkpoint 하나를 한 번만 로드한 상태에서 3개 task × 8 candidate × 128 token으로 직접 측정한 결과, 순차 76.8초 → batched 44.9초로 **1.71배** 단축을 확인했습니다.
Autoregressive decode가 메모리 대역폭 bound라는 이론상 기대(거의 free한 batch 확장)보다는 낮은 배수인데, MoE routing 오버헤드·KV cache 증가·sequence별 조기 EOS 종료 편차가 원인으로 보입니다.
