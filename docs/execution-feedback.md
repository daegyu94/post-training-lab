# Execution Feedback

공개 MBPP Python 함수 문제에서 `generation → Docker evaluation → feedback dataset → additional training → comparison` cycle을 사내 framework 없이 한 번 재현합니다.

| 항목 | 내용 |
| --- | --- |
| 대상 경로 | TRL만. Megatron SFT 경로는 건드리지 않음 |
| 실행 위치 | spark1 단일 GPU, 단일 process |
| 데이터 | MBPP `sanitized` (실행 test 포함) |
| 완료 조건 | 같은 입력·설정으로 cycle 완주 + 데이터·checkpoint·평가 결과 산출 |
| 완료 조건 아님 | pass@1 향상 |

## Pipeline

| # | 단계 | 모듈 | 입력 | 출력 |
| ---: | --- | --- | --- | --- |
| 1 | 데이터 준비 | `execution_feedback.prepare` | MBPP revision | `tasks.jsonl`, `initial_sft_{train,validation}.jsonl` |
| 2 | 후보 생성 | `execution_feedback.generate` | SFT checkpoint + `tasks.jsonl` | `train-candidates.jsonl` |
| 3 | 실행 채점 | `execution_feedback.evaluate` | candidates | `train-evaluations.jsonl` |
| 4 | feedback 데이터 | `execution_feedback.feedback` | evaluations | `filtered_sft.jsonl`, `dpo.jsonl` |
| 5 | 추가 학습 | `execution_feedback.train` | feedback 데이터 | `checkpoints/{B,C,D}/model` |
| 6 | 비교 | `execution_feedback.compare` | test evaluations | `comparison.json` |

전체를 순차 실행하는 wrapper는 [End-to-End Cycle](#end-to-end-cycle)입니다.

## Experiment Matrix

| Variant | Training sequence | 검증 질문 |
| --- | --- | --- |
| A | SFT only | 시작 checkpoint의 실행 성공률은 얼마인가? |
| B | SFT → execution-filtered SFT | 통과 답변을 더 학습하면 개선되는가? |
| C | SFT → DPO | 성공·실패 답변 선호 학습이 효과적인가? |
| D | SFT → execution-filtered SFT → DPO | filtered SFT 뒤에도 같은 DPO pair가 추가 효과를 내는가? |

파생 규칙:

- 네 variant 모두 **같은 최초 SFT checkpoint**에서 출발합니다.
- D는 B의 checkpoint를 이어받습니다. C와 D는 **최초 SFT 모델이 만든 동일한 `dpo.jsonl`**을 씁니다 — D용 후보를 다시 생성하지 않습니다.
- D는 DPO pair가 1개 이상일 때만 실행합니다.
- D의 총 학습 token·시간이 C보다 크므로 **성공률만으로 효율 비교를 하지 않습니다**.

## Data Contract

`execution_feedback.prepare`가 만드는 `tasks.jsonl` 필드:

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

데이터 제약:

- **UltraChat·No Robots는 쓸 수 없습니다** — 실행 test가 없는 대화 SFT 데이터라 채점 입력이 안 됩니다.
- MBPP 원본의 `train`/`validation`/`test` split을 그대로 유지합니다.
- `--revision` 기본값은 실제 검증한 불변 commit SHA(`4bb6404fdc6cacfda99d4ac4205087b89d32030c`)입니다. 다른 40자리 SHA는 필요할 때만 지정합니다.
- MBPP 라이선스: `google-research-datasets/mbpp`, `sanitized` config, cc-by-4.0. 이용 조건 확인 후 준비합니다.

```bash
python -m execution_feedback.prepare \
  --output-dir /path/to/shared/execution-feedback/data
```

> **함정 — dataset 스키마**: MBPP `sanitized`의 문제 설명 필드는 `prompt`입니다(`text` 아님).
> 다른 config나 향후 개정에서 또 바뀔 수 있으니, revision을 올릴 때는 `datasets.load_dataset(...).column_names`로 실제 스키마를 먼저 확인합니다.

최초 SFT checkpoint는 `initial_sft_train.jsonl` / `initial_sft_validation.jsonl`로 만들고, 이후 cycle의 공통 시작점으로 넘깁니다.

```bash
python -m execution_feedback.train \
  --mode sft --model-dir /path/to/base-model \
  --train-file /mnt/post-training/execution-feedback/data/initial_sft_train.jsonl \
  --eval-file /mnt/post-training/execution-feedback/data/initial_sft_validation.jsonl \
  --output-dir /mnt/post-training/execution-feedback/initial-sft
```

## Execution Evaluation

코드 추출: 응답에서 **가장 큰 Python code fence**를 코드로 씁니다. fence가 없으면 전체 응답을 코드로 취급합니다.
채점: syntax compile + assertion별 실행.

| Status | 의미 |
| --- | --- |
| `pass` | syntax와 모든 test 통과 |
| `fail` | 추출·syntax·runtime·test 실패 |
| `timeout` | 제한 시간 초과 |
| `infra_error` | Docker 실행 자체를 시작·완료할 수 없음 |

- `score` = 통과한 assertion 비율.
- `infra_error`와 `timeout`은 **DPO rejected에 넣지 않습니다** — 모델 품질이 아니라 환경 문제이므로.

격리 설정(Docker가 기본 engine, 후보마다 별도 임시 디렉터리 + container):

| 설정 | 기본값 | 비고 |
| --- | --- | --- |
| `--image` | `python:3.12-slim` | |
| `--timeout-seconds` | 10 | |
| `--cpus` | 1.0 | |
| `--memory` | `256m` | |
| `--pids-limit` | 64 | |
| `--workers` | 1 | 호스트에서 `1 / 2 / 4 / 8`을 측정해 정함 |
| network·capability | 제거 | root filesystem은 read-only |
| dataset·checkpoint mount | 없음 | container에 올리지 않음 |

```bash
python -m execution_feedback.evaluate \
  --tasks /path/to/data/tasks.jsonl \
  --candidates /path/to/train-candidates.jsonl \
  --output /path/to/train-evaluations.jsonl \
  --engine docker --image python:3.12-slim \
  --workers 4 --cpus 1 --memory 256m --pids-limit 64 --timeout-seconds 10
```

> **함정 — timeout 시 container 누수**: `subprocess.run(timeout=...)`은 `docker run` **client 프로세스만** 죽입니다.
> 그 client가 daemon에 띄운 container는 남아서 CPU·memory를 계속 점유합니다(`--rm`은 container가 스스로 끝났을 때만 정리하며, client 연결 종료로는 정리되지 않음).
> 대응: 실행마다 고유 `--name`을 주고 timeout 시 `docker rm --force`. 정리는 best-effort이며 실패해도 timeout 판정은 유지됩니다.

`--engine local` 사용 범위:

- 허용: 저장소의 신뢰할 수 있는 smoke test.
- 금지: 모델 출력, 공개 dataset 코드. 격리가 없습니다.

범위 밖: 다중 호스트 worker, pNFS/3FS 비교.

## Feedback Dataset

후보 생성 규칙:

- 최초 SFT checkpoint로 **train split에서 한 번만** 생성합니다.
- 기본 구성: 같은 sampling stream에서 정상 token budget 후보 8개 + 32-token 후보 1개.
- 짧은 후보도 **실제 모델 출력**입니다. 정상 후보가 pass한 문제에서 truncation fail을 만들어 DPO pair가 비는 것을 줄입니다.

> **함정 — reasoning 모델의 `<think>` 예산 소진**: Qwen3 계열은 코드보다 먼저 `<think>...</think>`를 출력합니다.
> `enable_thinking=False`로 미리 닫지 않으면 짧은 `--max-new-tokens`에서 사고 과정만으로 예산이 끝나 코드가 아예 안 나옵니다.
> 실측: Qwen3-30B-A3B, `--max-new-tokens 64`에서 12/12 후보가 잘린 `<think>` 텍스트만 반환.

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

| 출력 | 내용 |
| --- | --- |
| `filtered_sft.jsonl` | pass 코드만 |
| `dpo.jsonl` | 같은 문제·같은 후보 pool에서 pass를 `chosen`, fail을 `rejected` |

- validation·test 평가가 입력되면 feedback 생성은 **실패합니다**(train 전용).
- `--max-sft-per-task` / `--max-pairs-per-task`(기본 1)는 쉬운 문제가 데이터 대부분을 차지하는 것을 막습니다.

## Additional Training

`execution_feedback.train`은 TRL `SFTTrainer` / `DPOTrainer`를 씁니다.

| 제약 | 내용 |
| --- | --- |
| 병렬성 | spark1 단일 process만. `world_size != 1`이면 즉시 실패 |
| 모델 적재 | `device_map="auto"` — GPU에 shard 단위로 직접 올림 |
| LoRA target | attention projection + MoE router만(아래 함정 참고) |
| 메모리 | 30B에서는 `--gradient-checkpointing` 기본 권장 |

`--gradient-checkpointing` 없이는 이 하드웨어(GB10, 통합 메모리 119GiB)에서 backward 시점에 CUDA OOM이 발생했습니다 — LoRA라도 48 layer 활성화를 전부 들고 있으면 여유가 없습니다.

### 반드시 필요한 세 설정

`device_map="auto"`가 일부 layer를 CPU/meta로 offload하면서 생기는 문제들입니다. 세 가지 모두 실제 GPU 실행에서 재현했습니다.

| 설정 | 안 하면 나는 에러 | 원인 |
| --- | --- | --- |
| `--mode sft`에 `loss_type="nll"` 명시 | `'functools.partial' object has no attribute '__func__'` | 기본값 `chunked_nll`이 model forward를 patch하는데, CPU offload된 layer의 forward는 `functools.partial`로 감싸져 patch가 깨짐 |
| `target_modules` 명시 | `No target_modules passed but also no target_parameters found` | PEFT 자동 매핑이 이 repo의 MoE 구조(Qwen3-30B-A3B fused-expert parameter, GLM MLA attention)를 모름 |
| full FT DPO에 `precompute_ref_log_probs=True` | (에러 없이) resident weight 2배 | `DPOTrainer`는 `ref_model=None` + PEFT 아님이면 모델 전체를 한 번 더 로드해 reference로 상주시킴 |

`target_modules` 목록은 attention projection(GQA `q/k/v/o_proj`, GLM MLA `q_a/q_b/kv_a_with_mqa/kv_b_proj`)과 MoE router(`gate`)입니다. 선택 근거:

- `target_modules="all-linear"`는 위 에러는 피하지만 MoE expert의 fused parameter(`experts.gate_up_proj`/`down_proj`)까지 포함합니다 → 그중 일부가 CPU/meta로 offload되면 backward에서 `GroupedMmBackward0 returned an invalid gradient ... expected device meta but got cuda:0`(Qwen3-30B-A3B에서 재현).
- 그래서 **expert parameter를 아예 건드리지 않는** 명시적 목록을 씁니다.
- `up_proj`/`gate_proj`/`down_proj`는 넣지 않습니다 — fused expert parameter 이름(`gate_up_proj`, `down_proj`)과 suffix가 겹쳐 같은 문제를 다시 끌어옵니다. dense MLP만 있는 모델이라면 안전하게 추가할 수 있습니다.

`precompute_ref_log_probs=True`는 `--lora-r` 없는 full fine-tuning DPO(아래 C·D)에서 자동으로 켜집니다. 학습 시작 전 현재 모델로 reference logprob을 한 번 계산해 캐싱하므로 두 번째 사본이 없습니다.
LoRA checkpoint(새 adapter든 기존 adapter든)는 TRL이 adapter를 끄는 것만으로 reference를 얻으므로 이 옵션이 필요 없습니다.

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

D는 C와 정확히 같은 `dpo.jsonl`과 hyperparameter를 씁니다.

## End-to-End Cycle

Wrapper가 준비 → generation → Docker evaluation → B/C/D 학습 → test 평가 → 비교를 **순차** 실행합니다(GPU와 Docker가 자원 경쟁하지 않도록).

```bash
cd /home/spark/shared/post-training-lab
PYTHON=/home/spark/.local/ptl/venvs/trl/bin/python \
SFT_CHECKPOINT=/mnt/post-training/execution-feedback/initial-sft/model \
WORK_DIR=/mnt/post-training/execution-feedback/run-001 \
DOCKER_WORKERS=4 \
bash scripts/run_execution_feedback_cycle.sh
```

`WORK_DIR`는 local NVMe에 둡니다 — dataset·checkpoint·결과가 NFS를 거치지 않게.

| 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `SFT_CHECKPOINT` | (필수) | 네 variant 공통 시작 checkpoint |
| `WORK_DIR` | `output/execution-feedback-cycle` | 모든 산출물 위치. local NVMe 권장 |
| `PYTHON` | `python` | TRL venv의 interpreter |
| `DATA_REVISION` | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | MBPP revision |
| `LIMIT_PER_SPLIT` | 미지정 | 각 split 앞 N개만 사용(검증용 축소) |
| `NUM_TRAIN_CANDIDATES` | 8 | train task당 후보 수 |
| `NUM_TRUNCATED_CANDIDATES` | 1 | train task당 짧은 budget 후보 수 |
| `TRUNCATED_MAX_NEW_TOKENS` | 32 | 짧은 후보의 token 상한 |
| `MAX_NEW_TOKENS` | 512 | 후보 생성 token 상한(train·test 공통) |
| `TRAIN_TEMPERATURE` / `TRAIN_TOP_P` | 0.8 / 0.95 | train 후보 sampling |
| `DOCKER_WORKERS` | 1 | evaluate 병렬도 |
| `MAX_STEPS` | 64 | B/C/D 공통 학습 step |
| `LORA_R` | 0 | `>0`이면 새 adapter 생성, `0`이면 `SFT_CHECKPOINT`를 adapter로 보고 이어서 학습 |
| `GRADIENT_CHECKPOINTING` | 0 | `1`이면 `--gradient-checkpointing` |
| `SAVE_CHECKPOINT` | 0 | `1`이면 `--save-checkpoint` |

wrapper가 고정하는 값(환경변수로 못 바꿈):

- 학습: `--per-device-batch-size 1 --gradient-accumulation-steps 4 --seed 42`
- test 생성: `--num-candidates 1 --seed 42 --temperature 0`(greedy)
- `LORA_R`은 B와 C에만 전달됩니다. **D는 항상 B의 adapter를 이어서 학습**하므로 `--lora-r`를 받지 않습니다.

중단·분기 조건:

| 조건 | 동작 |
| --- | --- |
| `filtered_sft.jsonl`이 빈 경우 | `exit 2` — 통과 후보가 없어 추가 학습 불가 |
| `dpo.jsonl`이 빈 경우 | C·D를 건너뛰고 A/B만 비교 |

> **함정 — `LORA_R` 값 혼동**: `SFT_CHECKPOINT` 자체가 이미 adapter인데 `LORA_R>0`을 주면 `--lora-r cannot create a second adapter on an adapter checkpoint`로 죽습니다. 이 경우 `LORA_R=0`입니다.

> **함정 — `SAVE_CHECKPOINT`의 저장 비용**: 이 옵션은 HF Trainer의 중간 checkpoint(전체 optimizer state 포함)를 켭니다.
> `--resume-from-checkpoint` 재개에만 쓰이고, 실제 결과물(`output_dir/model`, `trainer.save_model()`)과 **별개로 중복 저장**됩니다.
> 실측: LoRA adapter 4GB + optimizer state 8GB → checkpoint당 2배 이상. 재개 계획이 없으면 기본값 off 유지.

DPO pair가 안 생기는 경우: 쉬운 task와 충분히 학습된 checkpoint를 쓰면 sampling만으로 fail이 안 나옵니다. 이때 `--max-new-tokens`를 의도적으로 줄인 별도 생성을 섞어 **진짜 truncation fail**을 만듭니다 — 조작된 label이 아니라 짧은 예산에서 나온 실제 출력입니다.

## Final Comparison

| 지표 | 정의 | 용도 |
| --- | --- | --- |
| strict pass@1 | 후보 1개로 **모든 test 통과**한 문제의 비율 | 주요 지표 |
| `validation_nll` | 같은 MBPP validation reference에 대한 NLL | 보조 지표. execution pass@1을 대신하지 않음 |
| `loss_history` | 각 stage 내부 loss 추이 | 하락 여부만 확인 |

- 모든 variant가 **같은 test task, 같은 seed, 같은 decoding 설정**으로 후보 하나만 생성합니다.
- SFT loss와 DPO loss는 목적함수가 달라 **서로 비교하지 않습니다**.

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

`--training-summary D=`가 두 번 나오는 것은 오타가 아닙니다 — `--training-summary`는 stage를 누적하므로, D의 총 학습 비용은 **B stage + D stage**입니다.

비교기가 **오류로 처리**하는 입력:

- test 외 split이 섞인 경우
- `candidate_id != 0`
- 같은 task의 중복 결과
- variant 간 test task 집합 불일치

`comparison.json` 내용: pass@1, 상태별 개수, Docker 평가 시간, 추가 학습 시간·관측 입력 token 수, `validation_nll`.
후보 생성 manifest와 training summary에는 모델 경로, generation 설정, 학습 설정과 시간이 남습니다.

## CPU Smoke Validation

합성 정답 + 의도적 실패 답변으로 데이터 준비·평가·feedback 생성을 확인합니다.

```bash
ENGINE=local bash scripts/run_execution_feedback_smoke.sh /tmp/execution-feedback-smoke
```

`ENGINE=local`은 **격리 검증이 아닙니다**. 실제 후보를 쓰기 전 `ENGINE=docker`로 다시 실행해 image와 제한 설정을 검증합니다.

## Verification Evidence

Commit 단위 기록:

| Commit | 확인한 것 | 확인하지 않은 것 |
| --- | --- | --- |
| `f7b9f0b` | spark1 실제 Docker에서 단일 timeout·workers=4 평가 후 container 잔존 여부, network·filesystem·PID 제한 | — |
| `8efc8e0b` | 설치된 TRL 1.12.0 source의 full DPO reference precompute 경로 | GPU 학습 완료, 메모리 실측 |

### GPU Cycle Runs

spark1에서 Qwen3-30B-A3B(실제 base checkpoint, LoRA r=16)로 전체 cycle을 세 번 실행했습니다(초기 SFT → train 후보 생성 → 실제 Docker 평가 → feedback → B/C/D 학습 → test 생성·평가 → compare).

| Run | 규모 | 결과 |
| ---: | --- | --- |
| 1 | synthetic 축소 test set | 완료. 아래 버그 5건 발견 |
| 2 | synthetic 축소 test set | 5건 수정 후 완료 |
| 3 | MBPP 원본 split 전체 | 완료. [아래 결과](#full-scale-mbpp-run-single-process-validation-nll-loss-history) |

**Run 1에서 발견한 실제 버그**

| # | 증상 | 원인 | 수정 |
| ---: | --- | --- | --- |
| 1 | `No target_modules passed but also no target_parameters found` | PEFT 자동 매핑이 이 repo의 MoE 구조를 모름 | 명시적 `target_modules` 지정 |
| 2 | 분산 실행 시 깨짐 | `device_map="auto"`는 단일 process 전제 | `world_size != 1`이면 즉시 실패 |
| 3 | `'functools.partial' object has no attribute '__func__'` | 기본 `chunked_nll`이 CPU-offload된 layer의 forward patch에 실패 | `loss_type="nll"` 명시 |
| 4 | 짧은 `--max-new-tokens`에서 코드가 전혀 안 나옴 | reasoning 모델이 `<think>` 블록으로 예산 소진 | `enable_thinking=False` |
| 5 | `GroupedMmBackward0 returned an invalid gradient ... expected device meta but got cuda:0` | `target_modules="all-linear"`가 MoE expert의 fused parameter까지 포함 | attention+router 명시 목록으로 축소 |

**Run 2 관측**

- DPO(C, D): loss 0.70 → 0.15~0.25, `rewards/accuracies` 1.0, `rewards/margins` 지속 증가 — 정상 학습 곡선.
- 버그 5 수정의 부수 효과: LoRA target이 줄어 adapter 크기 4GB → 60MB, trainable parameter 감소로 step당 속도도 크게 향상.
- A/B/C/D 모두 test pass@1=1.0. **synthetic test task가 2개뿐이고 이 모델에 쉬워서 차이가 안 보이는 것이며 pipeline 결함이 아닙니다.** 차이를 관측하려면 더 크거나 어려운 test set이 필요합니다.

**MBPP 경로 검증**

- 버그: `adapt_mbpp`가 없는 `text` 필드를 읽어 모든 row에서 `KeyError`(실제 필드명 `prompt`). 네트워크와 실제 dataset이 필요한 경로라 CPU tier 테스트가 전혀 커버하지 못했습니다.
- 수정 후: 실제 revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c`으로 4개 task를 준비하고 참조 정답을 실제 Docker에서 평가해 **4/4 pass** 확인.

**평가기 견고성**

- timeout의 stdout/stderr는 UTF-8로 변환해 마지막 4,000자만 저장 — 출력이 있는 후보가 timeout돼도 JSONL 저장이 계속됩니다.
- 채점 결과의 test 개수와 pass 개수를 검사하고, 모든 test 통과일 때만 `pass`로 판정합니다. 이는 **runner 출력의 형식 검사**이며 같은 process에서 실행되는 코드에 대한 부정행위 방지 장치는 아닙니다.
- Trainer는 입력 token 계측을 명시적으로 활성화합니다. 이 변경 **이전** summary의 0·누락 token 수를 처리량 0으로 해석하지 않습니다.

### Full-scale MBPP run (single process, validation NLL, loss history)

앞의 두 실행은 `--limit-per-split`로 줄인 test set(4~6개)이었습니다. 세 번째는 synthetic 경로를 제거하고 MBPP `sanitized` 원본 split을 그대로 사용했습니다.

| 항목 | 값 |
| --- | --- |
| Split | `train=120`, `validation=43`, `test=257` |
| 설정 | `LORA_R=16`, `GRADIENT_CHECKPOINTING=1`, `WORLD_SIZE=1` 강제 |
| `WORK_DIR` | local NVMe |

이 실행을 위해 추가한 계측:

- `generate.py`의 `--eval-file`/`--eval-output`: 각 variant test 생성 직후 같은 checkpoint로 `initial_sft_validation.jsonl`에 대한 NLL을 측정 → `validation_nll`.
- `train.py`: `trainer.state.log_history`를 `loss_history`로 저장.
- `compare.py`: `--evaluation-summary`로 받아 `comparison.json`에 병합.

**결과 (test 257개 기준)**

| Variant | Training sequence | pass@1 | validation NLL |
| --- | --- | --- | --- |
| A | SFT only | 0.066 (17/257) | 2.746 |
| B | SFT → filtered SFT | 0.062 (16/257) | 1.800 |
| C | SFT → DPO | **0.090 (23/257)** | 2.903 |
| D | SFT → filtered SFT → DPO | 0.070 (18/257) | 2.266 |

**두 지표가 반대 방향입니다.** B는 NLL을 가장 크게 낮췄지만(2.746 → 1.800) pass@1은 A보다 낮고, C는 NLL이 가장 높은데 pass@1이 가장 높습니다.
NLL은 reference 코드와의 token 일치도, pass@1은 실행 성공 여부이므로 같은 방향으로 움직일 이유가 없습니다 — **NLL을 execution 성능의 대리 지표로 쓰지 않습니다.**

차이 크기의 제약: variant 간 pass@1 격차는 최대 7개 task(16 → 23/257)입니다. 반복 실행과 신뢰구간이 없으므로 이 순위를 방법 간 우열로 확정하지 않습니다.

**학습 곡선·자원**

| 항목 | 관측 |
| --- | --- |
| A 초기 SFT | loss 3.5 → 2.1, NaN/Inf 없음 |
| B filtered SFT | 첫 5-step 평균 약 1.6 → 마지막 5-step 평균 약 1.0 |
| C·D DPO | loss 0.70 → 0.06~0.11, `rewards/accuracies` 0 → 1.0, `rewards/margins` 0 근처 → 2.2~3.6 |
| 메모리 | swap 미사용(119GiB 중 최대 약 65GiB, free 41GiB 유지) |

**실행 중 실패 1회** — 코드 버그가 아니라 잘못된 인자였습니다.

- 증상: 최초 SFT checkpoint(A)가 이미 LoRA adapter인데 B/C 학습에도 `LORA_R=16`을 넘겨 `--lora-r cannot create a second adapter on an adapter checkpoint`로 종료.
- 원인: `LORA_R`은 "새 adapter를 만들지(`>0`) / 기존 adapter를 이어 학습할지(`0`)"를 뜻하므로, `SFT_CHECKPOINT`가 adapter면 `0`이어야 합니다.
- 복구: 이미 끝난 generation(1080 candidates)·Docker 평가·feedback 데이터는 파일로 남아 있어 재사용하고, B 학습부터 올바른 값으로 재시작해 완료.

### Candidate generation batching

- 이전: task당 candidate마다 `model.generate()`를 따로 호출(정상 8개 + truncated 1개 = 9번).
- 변경: 같은 task의 candidate는 prompt가 같으므로 같은 `max_new_tokens` budget끼리 `num_return_sequences`로 묶어 한 번에 생성. greedy(`do_sample=False`) 경로는 `num_return_sequences`가 실질적으로 batch되지 않아 입력을 직접 `repeat_interleave`합니다.
- 측정: 30B checkpoint를 한 번만 로드한 상태에서 3 task × 8 candidate × 128 token — 순차 76.8초 → batched 44.9초, **1.71배** 단축.
- 이론상 기대(autoregressive decode는 메모리 대역폭 bound이므로 batch 확장이 거의 free)보다 낮은 배수입니다. MoE routing 오버헤드, KV cache 증가, sequence별 조기 EOS 종료 편차가 원인으로 보입니다.
