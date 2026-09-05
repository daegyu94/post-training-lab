# Data Lifecycle

## Data Sources

운영 LLM을 개선하기 위한 candidate는 service trace, 명시적 사용자 feedback, offline benchmark, human-authored example과 domain expert review에서 수집할 수 있습니다. raw trace를 곧바로 SFT target으로 사용해서는 안 됩니다. 사용자 입력과 기존 model 응답이 포함됐다는 사실만으로 그 응답이 정확하거나 학습에 적합하다는 의미가 아니기 때문입니다.

## Curation Pipeline

1. access policy에 따라 수집 가능한 event만 선택합니다.
2. 개인정보, credential, source code secret과 보안 등급이 높은 내용을 탐지하고 제거합니다.
3. request, response, tool call, tool result와 outcome을 하나의 trace 단위로 복원합니다.
4. 실패 원인, task category, language, difficulty와 provenance를 기록합니다.
5. 정답성, instruction following, tool 사용, safety와 형식 준수 기준으로 target을 검토합니다.
6. 중복과 benchmark contamination을 제거합니다.
7. stable identity를 기준으로 train, validation과 test split을 만든 뒤 immutable revision으로 등록합니다.

split은 개별 message가 아니라 conversation, user, task family, source document와 시간 구간처럼 leakage를 방지할 수 있는 단위로 수행합니다. 최종 test label은 training pipeline에서 읽을 수 없는 별도 권한과 위치에 보관하는 것이 안전합니다.

## Canonical Record

canonical schema는 특정 framework의 tokenizer나 binary dataset 형식에 종속되지 않아야 합니다.

```json
{
  "schema_version": "1",
  "sample_id": "<stable-id>",
  "source": {
    "type": "service-trace",
    "revision": "<source-revision>",
    "collected_at": "<timestamp>"
  },
  "task": {
    "category": "<task-category>",
    "difficulty": "<difficulty>",
    "language": "ko"
  },
  "messages": [
    {"role": "system", "content": "<system-instruction>"},
    {"role": "user", "content": "<request>"},
    {"role": "assistant", "content": "<reviewed-target>"}
  ],
  "tools": [],
  "quality": {
    "review_status": "approved",
    "reviewer_type": "human"
  },
  "split": "train"
}
```

실제 운영 schema에는 retention class, consent 또는 collection policy, redaction status, source lineage와 benchmark exclusion tag가 추가될 수 있습니다. 민감한 원문을 hash만으로 익명화했다고 간주해서는 안 됩니다.

## Backend Adapters

`trl` branch는 canonical record를 Hugging Face Dataset과 chat template 입력으로 변환하고 loss mask가 의도한 assistant token에만 적용되는지 검증합니다.

`megatron` branch는 같은 record를 Megatron 전처리 형식으로 변환하고 tokenizer revision, sequence packing, data index와 distributed consumption 조건을 검증합니다.

adapter는 원본 `sample_id`와 dataset revision을 보존해야 합니다. 이를 통해 두 backend의 학습 결과가 동일한 source population을 사용했는지 추적할 수 있습니다.

## Dataset Versioning

dataset revision에는 최소한 다음 정보가 필요합니다.

| Field | Purpose |
| --- | --- |
| `dataset_id` | 논리 dataset 식별자 |
| `revision` | immutable content revision |
| `schema_version` | record contract version |
| `source_revisions` | 입력 trace와 benchmark lineage |
| `split_policy` | leakage 방지 기준과 seed |
| `filter_policy` | quality, safety와 deduplication rule |
| `record_counts` | split과 task category별 sample 수 |
| `content_digest` | artifact 무결성 확인 |
| `created_at` | 생성 시각 |

dataset을 수정하면 기존 revision을 덮어쓰지 않고 새 revision을 생성합니다. training request는 floating path가 아니라 immutable revision과 digest를 참조해야 합니다.
