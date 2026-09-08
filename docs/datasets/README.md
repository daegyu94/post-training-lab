# SFT 데이터 준비 가이드

SFT는 질문과 **모델이 답해야 할 좋은 응답**을 짝지어 학습하는 방법입니다.
이 가이드는 공개 데이터나 사내 서비스 기록을 학습용 파일로 바꾸고, TRL 또는 Megatron에 연결하는 과정을 설명합니다.

- 공개 데이터를 써보려면 [공개 데이터 준비](#public-data)부터 시작하세요.
- 사내 서비스 기록을 쓰려면 [사내 데이터 준비](#internal-data)를 따라가세요.
- 파일을 만들었다면 [학습 연결](#training)과 [학습 전 확인](#checks)을 확인하세요.

## 먼저 알아둘 파일과 형식

| 파일 | 용도 |
| --- | --- |
| `training.jsonl` | 모델이 정답 응답을 배우는 학습 데이터 |
| `validation.jsonl` | 학습 중 loss를 확인하고 설정을 비교하는 검증 데이터 |
| `test.jsonl` | 최종 품질 평가용 데이터. 사내 데이터 변환 시 생성하며 학습에 넣지 않음 |
| `manifest.json` | 원본 버전, 변환 조건, 개수와 파일 해시를 기록한 명세서 |

JSONL은 **한 줄에 예제 하나를 JSON으로 저장한 파일**입니다.
학습·검증 파일의 핵심은 대화를 담는 `messages`입니다.
다음은 한 예제를 읽기 쉽게 펼친 모습이며, 실제 JSONL에서는 한 줄로 저장합니다.

```json
{
  "messages": [
    {"role": "user", "content": "회의 내용을 세 줄로 요약해 줘."},
    {"role": "assistant", "content": "검수된 요약 응답"}
  ]
}
```

`user`는 요청, `assistant`는 학습할 목표 응답, 선택 항목인 `system`은 모델에 주는 지시입니다.
변환 결과는 비어 있지 않은 문자열로 대화를 표현하며 마지막 메시지가 `assistant`여야 합니다.
공개 데이터에는 원본 예제 식별자(`prompt_id`)와 출처 정보(`provenance`)도 붙습니다.

Validation은 설정 선택에 반복해서 사용하므로 최종 평가용 test와 구분합니다.
Test의 정답과 평가 기준은 학습·튜닝 과정에서 사용하지 않도록 따로 관리합니다.

<a id="public-data"></a>

## 공개 데이터 준비

### 1. 목적에 맞는 데이터 선택

먼저 일반 대화, 코드 작성, JSON 응답 중 학습하려는 작업을 고릅니다.
`preset`은 선택한 원본 데이터의 서로 다른 필드 이름을 공통 대화 형식으로 바꾸는 변환 규칙입니다.
현재 다음 네 가지를 지원합니다.

| Preset | 담긴 내용 | 원본 데이터 | 기록된 라이선스 |
| --- | --- | --- | --- |
| `ultrachat` | 일반적인 질문과 여러 차례의 대화 | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT |
| `no_robots` | 사람이 작성한 지시와 응답 | [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) | CC BY-NC 4.0, 비상업 조건 |
| `self_oss` | 코드 작성 요청과 응답 | [bigcode/self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) | ODC-By |
| `xlam` | 도구 목록을 보고 호출할 함수와 인자를 JSON으로 답하는 예제 | [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | CC BY 4.0 |

사용 전 원본 페이지의 이용 조건을 확인하세요.
XLAM은 접근 승인을 받은 Hugging Face 계정이 필요할 수 있으며, 스크립트가 이용 약관에 대신 동의하지 않습니다.

변환 시 UltraChat과 No Robots는 기존 `messages`를 사용합니다.
Self-OSS는 `instruction`을 요청으로, `response`를 응답으로 사용합니다(`prompt` 필드는 데이터 생성용 지시이므로 사용하지 않음).

XLAM은 도구 목록을 `system`, 질문을 `user`, 정답 JSON을 `assistant`에 넣습니다.
따라서 XLAM 변환은 **JSON 응답을 학습하는 용도**이며 실제 도구 실행이나 agent 학습을 구현하지 않습니다.
잘못된 도구·정답 JSON은 변환 오류로 처리합니다.

### 2. 작은 파일로 변환

TRL과 Megatron의 변환 명령은 같습니다.
사용할 backend의 환경을 먼저 준비한 뒤 해당 디렉터리에서 실행하세요([환경 준비](../getting-started.md)).

```bash
# 저장소 루트에서 둘 중 사용할 backend로 이동
cd backends/trl
# Megatron을 쓰면 대신 cd backends/megatron

./scripts/prepare_public_data.sh \
  --preset no_robots \
  --revision e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b \
  --output-dir data/public/no_robots \
  --train-count 32 \
  --eval-count 8 \
  --seed 42
```

이 예시는 학습 32개와 검증 8개를 준비하는 작은 실행입니다.
`revision`은 같은 원본 버전을 다시 가져오기 위한 값이고, `seed`는 데이터 선택·분할을 재현하기 위한 값입니다.
다른 데이터를 준비하려면 `preset`, `revision`, 출력 디렉터리를 함께 바꾸세요.
아래는 기존 준비 예시에서 사용하는 고정 버전입니다.

| Preset | Revision |
| --- | --- |
| `ultrachat` | `8049631c405ae6576f93f445c6b8166f76f5505a` |
| `no_robots` | `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b` |
| `self_oss` | `356bb069eee815daa6e23e9a282eeefe1490ad44` |
| `xlam` | `26d14ebfe18b1f7b524bd39b404b50af5dc97866` |

변환기는 원본을 정해진 범위까지만 읽고, 요청 내용이 중복되는 예제를 제거한 뒤 학습·검증으로 나눕니다.
UltraChat의 `train_sft`, 나머지 데이터의 `train`에서 가져오며, XLAM은 `dataset` 구성을 사용합니다.
No Robots의 공식 test는 가져오지 않고 train의 일부를 validation으로 남깁니다.
Branch나 tag를 revision으로 주면 실제 commit SHA로 해석해 기록합니다.
이렇게 기록한 버전을 사용하면 원본 데이터가 나중에 바뀌어도 같은 입력을 다시 준비할 수 있습니다.

출력 디렉터리의 `manifest.json`에서 원본 ID·버전, 변환 규칙, seed, 읽은 범위, 생성 개수와 파일 해시를 확인하세요.
파일 생성 성공은 학습 성공을 의미하지 않습니다.
기존 No Robots·Self-OSS의 8개 학습/2개 검증 변환 확인도 데이터 준비 단계의 결과입니다.

<a id="internal-data"></a>

## 사내 데이터 준비

### 1. 학습할 응답을 검수

서비스 기록에 남은 모델 응답을 그대로 정답으로 쓰지 마세요.
목표는 과거 응답의 복제가 아니라 **앞으로 모델이 해야 할 응답**을 학습시키는 것입니다.
긍정적인 사용자 반응만으로는 정답이 검증되지 않습니다.

| 원본 | 준비 방법 |
| --- | --- |
| 일반 서비스 대화 | 대표 요청을 고르고 전문가가 응답을 수정·승인 |
| 실패한 대화 | 같은 요청에 올바른 응답을 새로 작성 |
| 사내 문서·업무 benchmark | 사용 가능한 최신 자료로 질문과 정답 작성 |
| 코드·도구 작업 | 테스트나 schema 검사로 응답의 성공 여부 확인 |
| 생성형 모델로 만든 예제 | 실제 업무와 맞는지 검수한 뒤 승인 |

학습 이용 권한을 확인하고 개인정보, 비밀 키, 고객 식별자와 내부 주소를 제거합니다.
중복 요청과 재시도를 정리하고, 언어·업무·난이도 분포도 확인합니다.

같은 세션·문서·업무 템플릿의 비슷한 예제가 train과 test에 동시에 들어가면 평가가 부풀려질 수 있습니다.
먼저 관련 예제를 묶고 train/validation/test를 나누세요.
변환기는 같은 세션의 분할 충돌을 검사하지만, 문서나 의미상 중복까지 자동으로 찾아주지는 않습니다.

### 2. 승인된 기록을 JSONL로 저장

다음 예제를 한 줄로 저장하는 형식입니다.
저장소에는 실제 사내 데이터 대신 가상 예제만 들어 있습니다.

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

`split`에는 `train`, `validation`, `test` 중 하나를 지정합니다.
생략하면 `session_id`와 seed로 분할하며, 같은 세션을 서로 다른 split으로 지정하면 중단됩니다.
실제 데이터는 분할을 미리 검토하고 명시하는 편이 좋습니다.

현재 변환기는 `system`, `user`, `assistant` 메시지만 지원합니다.
도구 호출 기록은 아래 [지원하지 않는 입력](#현재-지원하지-않는-입력)을 확인하세요.

### 3. 변환 결과 확인

환경을 준비한 `backends/trl` 또는 `backends/megatron`에서 실행합니다.

```bash
# 저장소의 가상 예제로 동작 확인
./scripts/prepare_service_data.sh

# 검수한 실제 파일 사용: 경로는 실행 환경에 맞게 변경
TRACE_FILE=/path/to/reviewed-traces.jsonl \
DATA_DIR=data/service-sft \
  ./scripts/prepare_service_data.sh
```

기본 입력은 `examples/service-traces.jsonl`, 출력은 `data/service-sft`입니다.
가상 예제에서는 학습 2개, 검증 1개, test 1개가 생성되고 미승인 기록 1개가 제외됩니다.
`manifest.json`에서 입력 해시, 분할별 개수와 제외 사유를 확인합니다.
실제 데이터에는 수집 기준·검수 이력·데이터 버전도 남기세요.

`test.jsonl`은 마지막 assistant 응답을 대화에서 빼고 `reference_answer`로 분리합니다.
`grader`는 평가 방법을 연결하기 위한 정보이며 자동으로 평가기를 실행하지는 않습니다.
생성된 test 정답은 최종 평가용으로 별도 보관하세요.

<a id="training"></a>

## TRL·Megatron 학습에 연결

변환은 두 backend에서 같은 명령을 쓰지만 학습에 넘기는 방법과 검사 조건은 다릅니다.

| 실행 경로 | 파일 연결 방법 | 확인할 조건 |
| --- | --- | --- |
| TRL 직접 실행 | `--dataset-jsonl-dir`에 출력 디렉터리 지정 | 모델의 chat template과 assistant 응답에 적용되는 loss 확인 |
| TRL Spark | `DATA_DIR`로 지정 | [TRL Spark 가이드](../backends/trl/spark-cluster.md)의 실행·데이터 검사 조건 확인 |
| Megatron 학습 코드 | `--train-data`에 `training.jsonl`, `--eval-data`에 `validation.jsonl` 지정 | JSON loader와 마지막 assistant 응답의 completion 전처리 확인 |
| Megatron Spark | `DATA_DIR`, `DATASET_ID`, `DATASET_REVISION`을 함께 지정 | [Megatron Spark 가이드](../backends/megatron/spark-cluster.md)의 manifest·데이터 버전 검사 통과 필요 |

**Megatron Spark에는 사내 데이터 출력 디렉터리만 지정해서 바로 연결할 수 없습니다.**
현재 사내 데이터 변환기의 manifest가 Spark 실행기가 요구하는 고정 버전 형식과 다릅니다.
승인된 데이터의 불변 버전과 호환 manifest를 준비하고 completion 전처리도 검증해야 합니다.

TRL의 assistant-only loss와 Megatron의 completion loss는 목표 응답 토큰에 학습 손실을 계산하기 위한 설정입니다.
선택한 모델의 tokenizer가 실제로 그 응답을 올바르게 구분하는지는 별도 확인이 필요합니다.
Test 파일은 학습 입력에 포함하지 않습니다.

Spark에서 실행할 때 파일 경로는 해당 Spark 노드에서 보이는 경로를 사용합니다.
모델·노드·학습 옵션은 위 backend 실행 가이드를 따르세요.

<a id="checks"></a>

## 학습 전 확인

- `manifest.json`의 원본 버전, 생성 개수와 제외 사유가 예상과 맞는지 확인합니다.
- 학습·검증·test 사이에 같은 요청이나 유사한 세션·문서가 섞이지 않았는지 확인합니다.
- 실제 학습 tokenizer로 몇 개를 읽어 대화가 올바르게 표시되는지 확인합니다.
- 길이 제한 때문에 요청이나 정답이 얼마나 잘리는지, 응답 끝 토큰(EOS)이 유지되는지 확인합니다.
- Loss를 계산할 토큰 표시(loss mask)가 목표 응답을 포함하는지 확인합니다.
- 작은 학습 실행으로 데이터 로딩과 loss 계산을 확인한 뒤 크기를 늘립니다.

데이터 변환 성공, GPU 학습 성공, 모델 품질 향상은 각각 다른 검증입니다.
서로 다른 데이터의 validation loss만 비교해 데이터 품질 순위를 매기지 마세요.
품질은 공통 test와 업무에 맞는 정답 검사·코드 테스트·사람의 평가로 비교해야 합니다.

## 현재 지원하지 않는 입력

| 입력 | 필요한 작업 |
| --- | --- |
| 원시 텍스트 또는 임의의 `prompt`/`completion` 파일 | 대화 형식으로 바꾸는 변환 규칙 추가 |
| `chosen`/`rejected` 선호도 쌍 | SFT와 별도의 학습 경로 필요 |
| `tool_calls`, `tool` 메시지를 포함한 agent 기록 | 구조를 보존하는 전용 변환기와 chat template·loss 검사 필요 |
| SWE-agent 실행 기록 | 현재는 참고 자료이며 변환기 미구현 |

도구 호출 기록을 임의로 문자열로 바꾸면 실제 서비스에서 사용하는 형식과 달라질 수 있습니다.
XLAM의 JSON 응답 학습이 이러한 구조화된 agent 기록 전체를 지원한다는 뜻은 아닙니다.
데이터가 커지면 먼저 길이·잘림 비율을 측정하고, 버전별 파일 분할과 캐시·packing을 검토하세요.
