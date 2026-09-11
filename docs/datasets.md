# Datasets

SFT는 "사용자 요청"과 "모델이 배워야 할 목표 응답"을 짝지어 학습합니다.
이 문서는 원본 데이터를 두 백엔드가 읽는 대화형 JSONL로 바꾸고 학습에 연결하는 방법을 다룹니다.

준비 경로는 원본 출처에 따라 둘로 나뉩니다.

| 경로 | 원본 | 초점 |
| --- | --- | --- |
| [Public Data](#public-data) | Hugging Face Hub 공개 데이터셋 | preset과 고정 revision으로 같은 파일을 재현 |
| [Internal Data](#internal-data) | 사내 서비스의 대화 로그 | 무엇을 학습해도 되는지 사람이 판단하고 남김 |

어느 경로든 순서는 같습니다: 원본 선택과 이용 조건 확인 → 변환 후 내용·manifest 검사 → 작은 학습으로 로딩과 loss 확인.

## Output Formats

| 파일 | 공개 데이터 | 검수한 사내 대화 |
| --- | --- | --- |
| `training.jsonl` | `messages`, `prompt_id`, `provenance` | 모델이 배울 요청과 목표 응답 |
| `validation.jsonl` | 원본 학습 split의 결정적 holdout | 세션 단위 검증 대화 |
| `test.jsonl` | 생성하지 않음 | 응답을 `reference_answer`로 분리한 평가 입력 |
| `manifest.json` | 원본 ID·revision, seed, 선택 ID, 개수, 출력 해시 | 입력 해시, seed, 개수와 제외 사유 |

공개 데이터 변환기는 원본 test split을 읽지 않으므로 `test.jsonl`을 만들지 않습니다.
두 manifest 형식은 서로 다르므로 교환해서 쓸 수 없습니다.

<a id="public-data"></a>

## Public Data

변환 규칙과 고정 revision은 두 백엔드가 공유하는 `datasets_lab/public_data.py` 한 곳이 정의하고, 진입점은 저장소 루트의 `scripts/prepare_public_data.sh`입니다.

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

[백엔드 환경](getting-started.md#2-prepare-the-nodes)을 준비한 뒤 **저장소 루트에서** 실행합니다.
`<backend-python>`은 의존성이 설치된 Python 절대 경로이며, `--output-dir`에는 그 노드의 **절대 경로**를 지정합니다.

```bash
cd "/path/to/shared/post-training-lab"
PYTHON='<backend-python>' bash scripts/prepare_public_data.sh \
  --preset no_robots \
  --revision e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b \
  --output-dir /path/to/node-local/data/no_robots \
  --train-count 8 --eval-count 2 --seed 42
```

성공하면 마지막 줄에 `train=8 validation=2 overlap=0`이 표시됩니다.
`manifest.json`의 `dataset`, `dataset_revision`, `train_count`, `eval_count`, `files`를 확인합니다.

> **이 명령은 한 노드에서만 실행합니다.**
> 나머지 노드에는 다시 실행하지 말고 생성된 JSONL 두 개와 manifest를 그대로 복사합니다.

노드별로 다시 만들면 안 되는 이유는 두 가지입니다.

- **내용이 갈릴 수 있습니다.** venv 간 라이브러리 버전 차이로 같은 seed에서도 결과가 미묘하게 달라지는데, `validate_dataset_manifest()`는 `dataset_id`·`revision`만 비교하므로 이런 drift를 잡지 못합니다.
- **다운로드가 실패할 수 있습니다.** `huggingface_hub`/`datasets`의 재시도 로직이 환경에 따라 실패합니다. 실제 관측된 에러는 `RuntimeError: Cannot send a request, as the client has been closed.`이며, 실제 원인은 그 직전의 `[SSL: CERTIFICATE_VERIFY_FAILED] ... unable to get local issuer certificate`입니다 — venv의 `certifi` 번들이 시스템 root CA와 맞지 않을 때 발생하고, `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`로 시스템 번들을 쓰게 하면 해결됩니다.

이미 검증된 결과물을 복사하는 쪽이 더 안전하고 빠릅니다.
이 "한 번 만들고 검증해서 배포한다"는 원칙이 대규모 환경에서 어떻게 바뀌는지는 [30B NVMe 실습](../labs/nvme-30b/README.md#training-data-storage)을 따릅니다.

### 옵션과 주의점

- `--revision`을 생략하면 preset의 고정 revision을 사용합니다.
- `--max-scan`은 기본 10000이며 이 범위에서 요청한 개수를 못 채우면 실패합니다.
- 중복 ID와 동일 입력 대화는 제외하지만 의미상 유사한 대화까지 검사하지는 않습니다.
- 형식이 깨진 행(예: 빈 메시지 content)은 그 행만 건너뛰고 stderr에 이유를 남깁니다. 실제 관측: `no_robots` 9,500행 중 1개, `ultrachat` 207,865행 중 31개.
- 제공된 smoke는 위 No Robots revision을 요구하므로, 다른 preset으로 바꾸려면 실험의 데이터 ID와 revision도 함께 바꿔야 합니다.

Megatron의 `prepare_spark_data.sh`는 별도의 UltraChat 전용 이전 경로이고, viewer 기반 경로는 `megatron_lab.prepare_data`에서 revision을 생략한 경우입니다 — 둘 다 `prepare_public_data.sh`의 동작이 아닙니다.

<a id="internal-data"></a>

## Internal Data

여기서 말하는 "검수한 내부 대화"는 공개 데이터셋이 아니라 **사내 서비스에 쌓인 대화 로그**를 학습 데이터로 만드는 경우입니다.
공개 데이터 경로가 "고정 revision 재현"에 초점을 둔다면, 이 경로는 "무엇을 학습해도 되는지 사람이 판단하고 남기는" 것에 초점을 둡니다.
서비스 로그는 그 자체로 정답이 아니라 학습 예제를 만들기 위한 원본입니다.

시작 전에 **이용 권한**(해당 대화를 학습에 써도 되는지), **민감정보 제거**(개인정보·비밀 키·내부 주소), **응답 검수**(과거 모델 응답을 복사하지 말고 앞으로 해야 할 응답으로 교정)를 먼저 정리합니다.

### 입력 형식

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

- `label_source`는 `expert`, `human_reviewed`, `verified_execution` 중 하나여야 합니다.
- `split`을 생략하면 세션과 seed로 분할하며, 같은 세션에 서로 다른 split을 지정하면 실패합니다.
- 메시지는 비어 있지 않은 `system`·`user`·`assistant` 대화이고 마지막은 `assistant`여야 합니다.
- 중복 `trace_id`는 오류이며 같은 입력 대화의 반복은 제외됩니다.

### 변환 실행

저장소의 예제 입력으로 동작을 먼저 확인할 수 있습니다.

```bash
trace_output="$(mktemp -d)"
python -m datasets_lab.prepare_service_data \
  --input datasets_lab/examples/service-traces.jsonl \
  --output-dir "$trace_output"
python -m json.tool "$trace_output/manifest.json"
```

예상 결과는 학습 2개, 검증 1개, test 1개와 미승인 기록 제외 1개입니다.
`test.jsonl`의 `grader`는 평가 정보를 담지만 평가기를 자동 실행하지 않습니다.
`prepare_service_data.sh` 래퍼는 기본적으로 `.venv/bin/python`을 사용하며 `PYTHON`으로 바꿀 수 있습니다.

## Connect to Training

| 실행 경로 | 입력 연결 | 제한 |
| --- | --- | --- |
| TRL Spark | setup의 `nodes[].data_dir` | 공개 데이터 형식 manifest와 `prompt_id` 필요 |
| Megatron Spark | setup의 `nodes[].data_dir` | 고정 ID·revision과 native completion 전처리 필요 |

사내 대화 변환 결과는 두 Spark 백엔드의 manifest 검증을 그대로 통과하지 못합니다.
임의 revision을 채워 우회하지 말고, 불변 버전과 호환 manifest를 만드는 구현이 별도로 필요하다는 제한으로 취급합니다.

Spark 전처리는 모델의 native template으로 마지막 assistant 이전 prompt와 마지막 응답·EOS를 분리합니다.
TRL은 길이 초과 행을 stderr에 기록한 뒤 제외하고, **Megatron은 길이 초과 시 중단합니다**(truncate하지 않음).
Test 입력, `chosen`/`rejected` 선호도 쌍, 구조화된 `tool_calls`는 현재 SFT 입력 경로의 대체물이 아닙니다.

변환 후에도 작은 학습으로 로딩과 loss를 확인해야 하며, 데이터 간 validation loss만으로 품질 순위를 정하지 않습니다.
