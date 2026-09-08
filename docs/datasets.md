# Datasets

이 문서는 공개 데이터와 승인된 서비스 기록을 대화형 SFT JSONL로 변환하는 방법을 설명합니다.
JSONL은 한 줄에 예제 하나를 저장하며, 마지막 메시지가 학습할 목표 응답입니다.

## Output Formats

| 파일 | 공개 데이터 | 서비스 기록 |
| --- | --- | --- |
| `training.jsonl` | `messages`, `prompt_id`, `provenance` | 승인된 학습 대화 |
| `validation.jsonl` | 원본 학습 split의 결정적 holdout | 세션 단위 검증 대화 |
| `test.jsonl` | 생성하지 않음 | 응답을 `reference_answer`로 분리한 평가 입력 |
| `manifest.json` | 원본 ID·revision, seed, 선택 ID, 개수, 출력 해시 | 입력 해시, seed, 개수와 제외 사유 |

공개 데이터 변환기는 원본 test split을 읽지 않습니다.
두 manifest 형식은 다르므로 서로 교환해서 사용할 수 없습니다.

## Public Data

두 백엔드의 `scripts/prepare_public_data.sh`는 같은 네 preset과 CLI를 제공합니다.
변환 규칙과 고정 revision은 각 백엔드의 `public_data.py`가 정의합니다.

| Preset | 원본 ID | 변환 대상 |
| --- | --- | --- |
| `ultrachat` | `HuggingFaceH4/ultrachat_200k` | 다중 대화 |
| `no_robots` | `HuggingFaceH4/no_robots` | 지시와 응답 |
| `self_oss` | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | `instruction`, `response` |
| `xlam` | `Salesforce/xlam-function-calling-60k` | 도구와 호출 결과의 JSON 문자열 |

원본 이용 권한과 라이선스는 사용 전에 확인합니다.
`xlam`은 코드에서 gated 데이터로 표시하며 스크립트가 접근 조건에 대신 동의하지 않습니다.
SWE-agent 기록은 참고 항목일 뿐 CLI preset이 아닙니다.

[백엔드 환경](getting-started.md#prepare-the-nodes)을 준비한 뒤 해당 노드에서 실행합니다.
다음은 TRL 예시이며, Megatron은 첫 줄만 `cd backends/megatron`으로 바꿉니다.
`<backend-python>`은 의존성이 설치된 Python 절대 경로로 바꿉니다.
명령은 Hub에 접속하고 출력 파일을 교체할 수 있으므로 새 출력 디렉터리를 사용합니다.

```bash
cd backends/trl
PYTHON='<backend-python>' bash scripts/prepare_public_data.sh \
  --preset no_robots \
  --revision e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b \
  --output-dir data/public/no_robots \
  --train-count 8 --eval-count 2 --seed 42
```

성공하면 마지막 출력에 `train=8 validation=2 overlap=0`이 표시됩니다.
`manifest.json`의 `dataset`, `dataset_revision`, `train_count`, `eval_count`, `files`를 확인합니다.
파일을 공유하지 않는다면 JSONL 두 개와 manifest를 함께 각 노드에 복사합니다.
제공된 smoke는 위 No Robots revision을 요구하므로 UltraChat으로 바꾸려면 실험의 데이터 ID와 revision도 함께 변경해야 합니다.

`--revision` 생략 시 preset의 고정 revision을 사용합니다.
`--max-scan`은 기본 10000이며 이 범위에서 요청한 개수를 채우지 못하면 실패합니다.
중복 ID와 동일 입력 대화는 제외하지만 의미상 유사한 대화까지 검사하지는 않습니다.

Megatron의 `prepare_spark_data.sh`는 별도의 UltraChat 전용 이전 경로입니다.
Viewer 기반 경로는 `megatron_lab.prepare_data`에서 revision을 생략한 경우이며 `prepare_public_data.sh`의 동작이 아닙니다.

## Reviewed Service Traces

서비스 기록은 학습 이용 권한을 확인하고 민감정보를 제거한 뒤 승인된 응답만 변환합니다.
다음은 입력 한 줄을 펼친 예시입니다.

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

`label_source`는 `expert`, `human_reviewed`, `verified_execution` 중 하나여야 합니다.
`split`을 생략하면 세션과 seed로 분할하며 같은 세션에 서로 다른 split을 지정하면 실패합니다.
메시지는 비어 있지 않은 `system`, `user`, `assistant` 대화이고 마지막 메시지는 `assistant`여야 합니다.
중복 `trace_id`는 오류이며 같은 입력 대화의 반복은 제외됩니다.

다음 가상 예제는 Python 표준 라이브러리만 필요하며 GPU나 네트워크를 사용하지 않습니다.
저장소 루트에서 새 임시 디렉터리에 실행합니다.

```bash
trace_output="$(mktemp -d)"
PYTHONPATH=backends/trl python -m trl_lab.prepare_service_data \
  --input backends/trl/examples/service-traces.jsonl \
  --output-dir "$trace_output"
python -m json.tool "$trace_output/manifest.json"
```

예상 결과는 학습 2개, 검증 1개, test 1개와 미승인 기록 제외 1개입니다.
Megatron은 `PYTHONPATH=backends/megatron`과 `megatron_lab.prepare_service_data`로 같은 예제를 실행할 수 있습니다.
`test.jsonl`의 `grader`는 평가 정보를 담지만 평가기를 자동 실행하지 않습니다.
`prepare_service_data.sh` 래퍼는 `.venv/bin/python`을 사용하므로 다른 Python을 쓰려면 위 모듈 진입점을 사용합니다.

## Connect to Training

| 실행 경로 | 입력 연결 | 제한 |
| --- | --- | --- |
| TRL 단일 GPU | `--dataset-jsonl-dir` | Qwen용 template과 assistant mask |
| TRL Spark | setup의 `nodes[].data_dir` | 공개 데이터 형식 manifest와 `prompt_id` 필요 |
| Megatron Spark | setup의 `nodes[].data_dir` | 고정 데이터 ID·revision과 native completion 전처리 필요 |

서비스 변환 결과는 두 Spark 백엔드의 manifest 검증을 그대로 통과하지 못합니다.
임의 revision을 채워 우회하지 말고 불변 버전과 호환 manifest를 만드는 구현이 별도로 필요하다는 제한으로 취급합니다.
TRL 단일 GPU 연결은 [TRL 가이드](backends/trl.md#single-gpu-qlora)를 따릅니다.

Spark 전처리는 모델의 native template으로 마지막 assistant 이전 prompt와 마지막 응답·EOS를 분리합니다.
토큰 경계와 supervised token을 검사하며 길이를 초과하면 조용히 자르지 않고 중단합니다.
Test 입력, `chosen`/`rejected` 선호도 쌍, 구조화된 `tool_calls`는 현재 SFT 입력 경로의 대체물이 아닙니다.
변환 후에도 작은 학습으로 로딩과 loss를 확인해야 하며 데이터 간 validation loss만으로 품질 순위를 정하지 않습니다.
