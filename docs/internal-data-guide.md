# 사내 LLM 서비스 데이터로 SFT 데이터셋 만들기

이 문서는 사내 LLM 서비스의 trace와 benchmark를 TRL `SFTTrainer`가 읽을 수 있는 학습·검증 데이터로 바꾸고, 학습에 노출되지 않는 test set과 정답을 별도로 관리하는 기준을 설명합니다. 저장소의 예제는 synthetic trace만 사용하며 실제 사내 데이터는 포함하지 않습니다.

## 먼저 구분해야 할 데이터

| 구분 | 모델에 제공하는 내용 | 사용 목적 | 정답 노출 |
| --- | --- | --- | --- |
| Train | prompt와 검수된 assistant 응답 | gradient update | 학습에 노출 |
| Validation | prompt와 검수된 assistant 응답 | held-out loss와 설정 비교 | 학습에는 미사용, 반복 실험에는 노출 |
| Test | prompt, 별도 보관한 reference answer와 grader | 최종 품질 평가 | 학습·튜닝 과정에 미노출 |

SFT의 target은 서비스가 과거에 생성한 응답 그 자체가 아니라, 같은 요청에 모델이 앞으로 생성하기를 원하는 응답이어야 합니다. 사용자 반응이 좋았다는 이유만으로 운영 응답을 정답으로 간주하지 말고, 전문가 검수·수정이나 실행 기반 검증을 거쳐야 합니다.

## 데이터 원천과 정답을 만드는 방법

| 원천 | 학습 예제로 만드는 방법 | 적합한 정답 또는 검증 방식 |
| --- | --- | --- |
| 운영 trace | 대표 요청을 sampling하고 민감 정보를 제거한 뒤 응답을 전문가가 수정·승인 | 승인된 최종 응답, rubric 기반 평가 |
| 실패 trace | 실패 원인을 분류하고 동일 prompt에 대한 올바른 응답을 새로 작성 | 수정된 응답, 재현 가능한 regression check |
| 사내 benchmark | 업무별 task와 입력을 고정하고 담당자가 reference를 작성 | exact match, required fields, rubric |
| code·tool task | 실행 환경과 입력을 고정하고 성공 조건을 정의 | unit test, exit code, schema validator |
| 사내 문서 | 사용 권한과 최신성을 확인한 뒤 실제 질문·응답 예제로 재구성 | 근거 문서와 전문가 승인 |
| synthetic data | 부족한 유형을 생성하되 실제 업무 분포를 반영하고 사람이 최종 검수 | 전문가 승인 또는 실행 검증 |

운영 모델의 원본 출력, thumbs-up만 있는 응답, 검수 상태를 알 수 없는 대화, secret·개인정보·고객 데이터가 남은 trace는 학습 target으로 사용하지 않습니다. 사용자 요청과 정답의 라이선스·보존 기간·학습 이용 동의도 데이터 반출 전에 확인해야 합니다.

## 권장 수집 파이프라인

1. 서비스 logging 계층에서 `trace_id`, `session_id`, model version, prompt와 response, latency, feedback을 원본 보존 영역으로 수집합니다.
2. 허용된 task와 기간만 선택하고 secret, credential, 개인정보, 고객 식별자, 내부 host 정보를 제거합니다.
3. 중복·거의 동일한 prompt와 자동 재시도를 제거하고, task·언어·난이도·성공/실패 유형별 분포를 기록합니다.
4. 전문가가 assistant target을 수정·승인하거나 code·tool task를 실행해 정답을 검증합니다.
5. 같은 사용자, session, 문서, task template이 서로 다른 split에 들어가지 않도록 먼저 group을 만든 뒤 train/validation/test로 나눕니다. 실제 배포 성능을 보려면 최근 기간을 test로 남기는 time-based split도 함께 사용합니다.
6. test의 reference와 grader는 학습자가 접근할 수 없는 별도 저장소에 고정하고, 학습 데이터와의 exact·semantic 중복을 검사합니다.
7. dataset version, 원본 query 기준, 필터 수, 제외 사유, reviewer, source hash, split seed를 data card와 manifest에 기록합니다.

## 이 브랜치의 입력 형식

변환 전 trace는 다음 최소 schema를 사용합니다.

```json
{
  "trace_id": "trace-001",
  "session_id": "session-001",
  "split": "train",
  "messages": [
    {"role": "user", "content": "요청"},
    {"role": "assistant", "content": "검수된 목표 응답"}
  ],
  "review": {
    "status": "approved",
    "label_source": "expert",
    "sensitive_data_removed": true
  }
}
```

`split`을 생략하면 converter가 `session_id`의 stable hash와 seed로 split을 결정합니다. 운영 데이터에서는 split manifest를 먼저 승인한 뒤 `split`을 명시하는 방식을 권장합니다. 같은 session을 여러 split에 지정하면 converter가 중단됩니다.

현재 Qwen assistant-only template과 실습 converter는 `system`, `user`, `assistant` role만 허용합니다. `tool` message와 구조화된 `tool_calls`가 있는 agent trace는 원본을 보존하되, 대상 모델의 chat template과 loss mask가 해당 구조를 지원하는지 검증한 전용 adapter를 만든 후 포함해야 합니다. 이를 일반 문자열로 임의 변환하면 serving 때의 형식과 학습 형식이 달라질 수 있습니다.

## TRL 형식으로 변환하는 실습

```bash
./scripts/setup.sh
./scripts/prepare_service_data.sh
```

기본 입력은 `examples/service-traces.jsonl`, 출력은 `data/service-sft`입니다. 예제에는 train 2개, validation 1개, test 1개와 승인되지 않아 제외되는 trace 1개가 들어 있습니다.

```text
data/service-sft/
├── training.jsonl
├── validation.jsonl
├── test.jsonl
└── manifest.json
```

`training.jsonl`과 `validation.jsonl`은 TRL conversational format인 `messages`를 포함합니다. `SFTTrainer`는 이 브랜치의 `assistant_only_loss=True` 설정과 Qwen chat template을 사용해 assistant token에만 loss를 계산합니다.

`test.jsonl`은 마지막 assistant 응답을 prompt에서 제거하고 `reference_answer`로 분리합니다. 이 파일은 `SFTTrainer`에 전달하지 않습니다. `grader`는 `exact_match`, `required_phrases`, schema validator, executable test, human rubric처럼 task에 맞는 평가기로 연결하기 위한 metadata입니다.

실제 trace와 출력 위치는 환경 변수로 바꿀 수 있습니다.

```bash
TRACE_FILE=<reviewed-traces.jsonl> DATA_DIR=<prepared-data-dir> \
  ./scripts/prepare_service_data.sh
```

변환 결과를 이 브랜치의 학습 경로에서 사용하려면 다음처럼 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m sft_lab.train \
  --dataset-jsonl-dir data/service-sft \
  --local-files-only \
  --train-samples 2 \
  --eval-samples 1 \
  --max-steps 1 \
  --output-dir results/service-data-practice
```

`manifest.json`에서 입력 파일 hash, split별 개수, 제외 사유를 확인합니다. 실제 데이터 pipeline에서는 승인 데이터가 갑자기 줄거나 특정 제외 사유가 늘면 build를 실패시키는 기준을 추가해야 합니다.

## 운영 전 확인 항목

- target 응답이 원본 model output이 아니라 승인된 목표 동작인가
- 개인정보, credential, 고객 데이터, 내부 주소가 제거되었는가
- 동일 session·문서·task template이 split을 넘나들지 않는가
- test 정답과 grader가 학습·hyperparameter tuning에서 격리되었는가
- serving tokenizer와 training tokenizer의 chat template이 같은가
- 잘린 token 비율과 assistant loss mask를 sample별로 검사했는가
- task 분포, 언어, 길이, 안전성, 최신성이 실제 서비스 목표와 맞는가
- dataset version과 model version으로 결과를 다시 추적할 수 있는가

## 참고 자료

- [TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
