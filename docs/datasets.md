# Datasets

SFT는 "사용자 요청"과 "모델이 배워야 할 목표 응답"을 짝지어 학습합니다.
이 문서는 원본 데이터를 TRL과 Megatron이 읽을 수 있는 대화형 JSONL로 바꾸고, 그 결과를 학습에 연결하는 방법을 설명합니다.

준비 경로는 두 가지이며 원본 데이터의 출처에 따라 나뉩니다.

| 경로 | 원본 | 특징 |
| --- | --- | --- |
| [Public Data](#public-data) | Hugging Face Hub의 공개 데이터셋 | preset과 고정 revision으로 누구나 같은 결과를 재현 |
| [Internal Data](#internal-data) | 자체 서비스에서 쌓인 사내 대화 로그 | 검수와 민감정보 제거를 거친 예제만 사용 |

어느 경로든 작업 순서는 같습니다.

1. 목적에 맞는 원본 데이터를 선택하고 학습 이용 조건을 확인합니다.
2. 학습·검증·평가 파일로 변환한 뒤 내용과 `manifest.json`을 검사합니다.
3. 선택한 backend에 연결하고 작은 학습으로 데이터 로딩과 loss 계산을 확인합니다.

## Output Formats

두 경로 모두 아래 파일을 만들지만 담기는 내용이 다릅니다.

| 파일 | 공개 데이터 | 검수한 사내 대화 |
| --- | --- | --- |
| `training.jsonl` | `messages`, `prompt_id`, `provenance` | 모델이 배울 요청과 목표 응답 |
| `validation.jsonl` | 원본 학습 split의 결정적 holdout | 세션 단위 검증 대화 |
| `test.jsonl` | 생성하지 않음 | 응답을 `reference_answer`로 분리한 평가 입력 |
| `manifest.json` | 원본 ID·revision, seed, 선택 ID, 개수, 출력 해시 | 입력 해시, seed, 개수와 제외 사유 |

공개 데이터 변환기는 원본 test split을 읽지 않으므로 `test.jsonl`을 만들지 않습니다.
두 manifest 형식은 서로 다르므로 교환해서 사용할 수 없습니다.

<a id="public-data"></a>

## Public Data

공개 데이터는 고정된 버전과 변환 조건을 기록해 언제든 같은 파일을 다시 만들 수 있게 합니다.
변환 규칙과 고정 revision은 두 백엔드가 공유하는 `datasets_lab/public_data.py` 한 곳이 정의하며, 진입점은 저장소 루트의 `scripts/prepare_public_data.sh`입니다.

| Preset | 원본 ID | 변환 대상 |
| --- | --- | --- |
| `ultrachat` | `HuggingFaceH4/ultrachat_200k` | 다중 대화 |
| `no_robots` | `HuggingFaceH4/no_robots` | 지시와 응답 |
| `self_oss` | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | `instruction`, `response` |
| `xlam` | `Salesforce/xlam-function-calling-60k` | 도구와 호출 결과의 JSON 문자열 |

원본 이용 권한과 라이선스는 사용 전에 확인합니다.
`xlam`은 코드에서 gated 데이터로 표시하며 스크립트가 접근 조건에 대신 동의하지 않습니다.
SWE-agent 기록은 참고 항목일 뿐 CLI preset이 아닙니다.

### 변환 실행

[백엔드 환경](getting-started.md#prepare-the-nodes)을 준비한 뒤 해당 노드에서 실행합니다.
다음은 TRL 예시이며, Megatron은 첫 줄만 `cd backends/megatron`으로 바꿉니다.
`<backend-python>`은 의존성이 설치된 Python 절대 경로로 바꿉니다.
명령은 Hub에 접속하고 출력 파일을 교체할 수 있으므로 새 출력 디렉터리를 사용합니다.

```bash
PYTHON='<backend-python>' bash scripts/prepare_public_data.sh \
  --preset no_robots \
  --revision e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b \
  --output-dir data/public/no_robots \
  --train-count 8 --eval-count 2 --seed 42
```

성공하면 마지막 출력에 `train=8 validation=2 overlap=0`이 표시됩니다.
`manifest.json`의 `dataset`, `dataset_revision`, `train_count`, `eval_count`, `files`를 확인합니다.
> **이 명령은 한 노드에서만 실행합니다.**
> 나머지 노드에는 같은 명령을 다시 실행하지 말고, 생성된 JSONL 두 개와 manifest를 그대로 복사합니다.

각 노드가 Hub에서 독립적으로 다시 만들면 안 되는 이유는 두 가지입니다.

- **내용이 갈릴 수 있습니다.** venv 간 라이브러리 버전 차이로 같은 seed에서도 결과가 미묘하게 달라지는데, `validate_dataset_manifest()`는 `dataset_id`·`revision`만 비교하므로 이런 차이를 잡지 못합니다.
- **다운로드가 실패할 수 있습니다.** `huggingface_hub`/`datasets`의 재시도 로직이 환경에 따라 실패합니다 (실제 관측: `RuntimeError: Cannot send a request, as the client has been closed.`).

이미 검증된 결과물을 복사하는 쪽이 더 안전하고 빠릅니다.

### 옵션과 주의점

`--revision`을 생략하면 preset의 고정 revision을 사용합니다.
`--max-scan`은 기본 10000이며 이 범위에서 요청한 개수를 채우지 못하면 실패합니다.
중복 ID와 동일 입력 대화는 제외하지만 의미상 유사한 대화까지 검사하지는 않습니다.
원본 소스에 형식이 깨진 행(예: 빈 메시지 content)이 섞여 있으면 그 행만 건너뛰고 stderr에 이유를 남깁니다 — 전체 변환을 중단하지 않습니다. 이 저장소에서 실제로 관측: `no_robots` 9,500행 중 1개, `ultrachat` 207,865행 중 31개.
제공된 smoke는 위 No Robots revision을 요구하므로 UltraChat으로 바꾸려면 실험의 데이터 ID와 revision도 함께 변경해야 합니다.

Megatron의 `prepare_spark_data.sh`는 별도의 UltraChat 전용 이전 경로입니다.
Viewer 기반 경로는 `megatron_lab.prepare_data`에서 revision을 생략한 경우이며 `prepare_public_data.sh`의 동작이 아닙니다.

Spark 노드에서 `RuntimeError: Cannot send a request, as the client has been closed.`가 나면 실제 원인은 대부분 그 앞에 찍히는 `[SSL: CERTIFICATE_VERIFY_FAILED] ... unable to get local issuer certificate`입니다 — venv에 설치된 `certifi`의 CA 번들이 이 시스템의 실제 root CA와 안 맞을 때 발생하며, `huggingface_hub`가 이 SSL 실패를 재시도하다 client를 닫고 못 살리는 게 뒤에 나오는 오류입니다. `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`를 지정해 시스템 CA 번들을 쓰게 하면 해결됩니다.

<a id="internal-data"></a>

## Internal Data

여기서 말하는 "검수한 내부 대화"는 공개 데이터셋이 아니라 **사내 서비스에서 쌓인 자체 대화 로그**를 학습 데이터로 만드는 경우입니다.
즉 Hub에서 내려받는 대신, 우리 서비스의 실제 요청과 응답 기록을 원본으로 삼습니다.
공개 데이터 경로가 "고정 revision 재현"에 초점을 둔다면, 이 경로는 "무엇을 학습해도 되는지 사람이 판단하고 남기는" 것에 초점을 둡니다.

시작하기 전에 다음 세 가지를 먼저 정리합니다.

- **이용 권한**: 해당 대화를 학습에 사용해도 되는지 확인합니다.
- **민감정보 제거**: 개인정보, 비밀 키, 내부 주소 등을 제거한 예제만 변환합니다.
- **응답 검수**: 과거 모델 응답을 그대로 복사하지 말고, 앞으로 모델이 해야 할 응답으로 고쳐 검수합니다.

서비스 로그는 그 자체로 정답이 아니라 학습 예제를 만들기 위한 원본이라는 점이 핵심입니다.

### 입력 형식

다음은 검수를 마친 입력 한 줄을 읽기 쉽게 펼친 예시입니다.

```json
{
  "trace_id": "trace-001",
  "session_id": "session-001",
  "split": "train",
  "messages": [
    {"role": "user", "content": "요청"},
    {"role": "assistant", "content": "검수된 응답"}
  ],
  "review": {
    "status": "approved",
    "label_source": "expert",
    "sensitive_data_removed": true
  }
}
```

규칙은 다음과 같습니다.

- `label_source`는 `expert`, `human_reviewed`, `verified_execution` 중 하나여야 합니다.
- `split`을 생략하면 세션과 seed로 분할하며, 같은 세션에 서로 다른 split을 지정하면 실패합니다.
- 메시지는 비어 있지 않은 `system`, `user`, `assistant` 대화이고 마지막 메시지는 `assistant`여야 합니다.
- 중복 `trace_id`는 오류이며 같은 입력 대화의 반복은 제외됩니다.

### 변환 실행

저장소에 포함된 예제 입력으로 동작을 먼저 확인할 수 있습니다.
다음 예제는 controller의 저장소 루트에서 새 임시 디렉터리에 실행합니다.

```bash
trace_output="$(mktemp -d)"
python -m datasets_lab.prepare_service_data \
  --input datasets_lab/examples/service-traces.jsonl \
  --output-dir "$trace_output"
python -m json.tool "$trace_output/manifest.json"
```

예상 결과는 학습 2개, 검증 1개, test 1개와 미승인 기록 제외 1개입니다.
`test.jsonl`의 `grader`는 평가 정보를 담지만 평가기를 자동 실행하지 않습니다.
`prepare_service_data.sh` 래퍼는 기본적으로 `.venv/bin/python`을 사용하며 `PYTHON`으로 다른 interpreter를 지정할 수 있습니다.

## Connect to Training

| 실행 경로 | 입력 연결 | 제한 |
| --- | --- | --- |
| TRL Spark | setup의 `nodes[].data_dir` | 공개 데이터 형식 manifest와 `prompt_id` 필요 |
| Megatron Spark | setup의 `nodes[].data_dir` | 고정 데이터 ID·revision과 native completion 전처리 필요 |

사내 대화 변환 결과는 두 Spark 백엔드의 manifest 검증을 그대로 통과하지 못합니다.
임의 revision을 채워 우회하지 말고, 불변 버전과 호환 manifest를 만드는 구현이 별도로 필요하다는 제한으로 취급합니다.

Spark 전처리는 모델의 native template으로 마지막 assistant 이전 prompt와 마지막 응답·EOS를 분리합니다.
TRL은 tokenizer 경계 병합을 Trainer와 같은 방식으로 처리하고 길이 초과 행을 stderr에 기록한 뒤 제외하며, Megatron은 길이 초과 시 중단합니다.
Test 입력, `chosen`/`rejected` 선호도 쌍, 구조화된 `tool_calls`는 현재 SFT 입력 경로의 대체물이 아닙니다.

변환 후에도 작은 학습으로 로딩과 loss를 확인해야 하며, 데이터 간 validation loss만으로 품질 순위를 정하지 않습니다.
