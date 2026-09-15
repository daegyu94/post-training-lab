# Datasets

SFT는 사용자 요청과 목표 응답을 짝지어 학습합니다.
원본을 두 백엔드가 읽는 대화형 JSONL로 바꾸는 경로는 다음과 같습니다.

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

변환 규칙·고정 revision은 `datasets_lab/public_data.py`에, 공통 진입점은 `scripts/prepare_public_data.sh`에 있습니다.

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

**한 노드에서만 실행한 뒤 JSONL 두 개와 manifest를 나머지 노드에 복사합니다.**
노드별 재생성은 라이브러리 버전에 따른 내용 차이를 만들 수 있으며, `validate_dataset_manifest()`의 ID·revision 비교로는 이를 잡지 못합니다.
다운로드 중 `client has been closed`가 발생하면 직전의 CA 오류를 확인합니다([문제 해결](getting-started.md#troubleshooting)).
대규모 배포의 node-local 경로 원칙은 [Spark cluster setup](../setups/spark/README.md)을 따릅니다.

### 옵션과 주의점

- `--revision`을 생략하면 preset의 고정 revision을 사용합니다.
- `--max-scan`은 기본 10000이며 이 범위에서 요청한 개수를 못 채우면 실패합니다.
- 중복 ID와 동일 입력 대화는 제외하지만 의미상 유사한 대화까지 검사하지는 않습니다.
- 형식이 깨진 행(예: 빈 메시지 content)은 그 행만 건너뛰고 stderr에 이유를 남깁니다. 실제 관측: `no_robots` 9,500행 중 1개, `ultrachat` 207,865행 중 31개.
- 제공된 smoke는 위 No Robots revision을 요구하므로, 다른 preset으로 바꾸려면 실험의 데이터 ID와 revision도 함께 바꿔야 합니다.

Megatron의 `prepare_spark_data.sh`는 별도의 UltraChat 전용 이전 경로이고, viewer 기반 경로는 `megatron_lab.prepare_data`에서 revision을 생략한 경우입니다 — 둘 다 `prepare_public_data.sh`의 동작이 아닙니다.

<a id="internal-data"></a>

## Internal Data

사내 서비스 로그는 검수를 거쳐 학습 예제로 만듭니다.
먼저 이용 권한을 확인하고 개인정보·비밀 키·내부 주소를 제거한 뒤, 과거 모델 응답을 목표 응답으로 교정합니다.

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

사내 변환 결과는 Spark manifest 검증을 그대로 통과하지 못합니다.
임의 revision으로 우회하지 말고 불변 버전·호환 manifest 생성 기능을 별도로 구현해야 합니다.

Spark 전처리는 모델의 native template으로 마지막 assistant 이전 prompt와 마지막 응답·EOS를 분리합니다.
TRL은 길이 초과 행을 stderr에 기록한 뒤 제외하고, **Megatron은 길이 초과 시 중단합니다**(truncate하지 않음).
Test 입력, `chosen`/`rejected` 선호도 쌍, 구조화된 `tool_calls`는 현재 SFT 입력 경로의 대체물이 아닙니다.

변환 후에도 작은 학습으로 로딩과 loss를 확인해야 하며, 데이터 간 validation loss만으로 품질 순위를 정하지 않습니다.
