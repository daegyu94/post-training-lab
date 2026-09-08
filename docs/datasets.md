# Datasets

이 repository의 학습 입력은 대화형 JSONL입니다.
공개 dataset 변환기는 pinned source revision과 seed를 manifest에 기록합니다.
서비스 trace 변환기는 승인된 기록만 학습·검증·test 파일로 분리합니다.

## Canonical output

| 파일 | 용도 |
| --- | --- |
| training.jsonl | SFT 학습 |
| validation.jsonl | 학습 중 검증 |
| test.jsonl | 최종 평가용 입력과 reference answer |
| manifest.json | source hash, revision, seed, count와 exclusion |

학습 example은 messages 배열을 사용합니다.
각 message의 role은 system, user, assistant 중 하나여야 하고 마지막 message는 assistant여야 합니다.
Test example은 마지막 assistant 응답을 reference_answer로 분리합니다.

```json
{"messages":[{"role":"user","content":"질문"},{"role":"assistant","content":"검수된 응답"}]}
```

## TRL 공개 dataset

TRL은 ultrachat, no_robots, self_oss, xlam preset을 제공합니다.
원본 dataset의 license와 access 조건은 사용자가 원본 페이지에서 확인해야 합니다.

```bash
cd backends/trl
./scripts/prepare_public_data.sh \
  --preset no_robots \
  --revision e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b \
  --output-dir data/public/no_robots \
  --train-count 32 \
  --eval-count 8 \
  --seed 42
```

xlam은 gated dataset일 수 있으며 script가 이용 약관에 대신 동의하지 않습니다.
변환이 끝나면 output directory의 manifest.json에서 revision과 count를 확인합니다.

## Megatron 공개 dataset

Spark launcher가 요구하는 pinned UltraChat 입력은 prepare_spark_data.sh로 준비합니다.
이 wrapper는 DATASET_REVISION을 읽어 prepare_data.py의 --dataset-revision으로 전달합니다.

```bash
cd backends/megatron
DATASET_REVISION=8049631c405ae6576f93f445c6b8166f76f5505a \
TRAIN_SAMPLES=32 \
EVAL_SAMPLES=8 \
DATA_DIR=data/public/ultrachat \
./scripts/prepare_spark_data.sh
```

이 명령은 training.jsonl, validation.jsonl과 manifest.json을 생성합니다.
Megatron의 prepare_public_data.sh는 별도의 viewer 기반 subset 경로이며 Spark pinned 입력에는 사용하지 않습니다.

## 승인된 service trace

두 backend의 prepare_service_data.sh는 TRACE_FILE의 JSONL을 변환합니다.
입력에는 trace_id, session_id, messages와 승인 상태를 포함하는 review가 필요합니다.
민감정보 제거 여부가 true가 아니거나 승인되지 않은 기록은 제외됩니다.

```bash
cd backends/trl
TRACE_FILE=/path/to/reviewed-traces.jsonl \
DATA_DIR=data/service-sft \
./scripts/prepare_service_data.sh
```

기본 입력은 examples/service-traces.jsonl이고 기본 output은 data/service-sft입니다.
split을 생략하면 session_id와 seed로 분할하며 같은 session이 여러 split으로 나뉘면 중단합니다.

## 학습 전에 확인할 것

Manifest의 source hash, revision, split별 count와 exclusion을 확인합니다.
Train·validation·test 사이의 session과 의미상 중복을 별도로 검사합니다.
선택한 tokenizer로 chat template과 assistant loss mask를 확인합니다.
Test file은 학습 입력에 넣지 않습니다.
DPO의 chosen/rejected 입력과 tool-call message는 현재 SFT 변환기의 지원 범위가 아닙니다.
