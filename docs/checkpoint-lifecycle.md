# Checkpoint Lifecycle

## Training Request

training request는 backend별 command를 직접 노출하는 대신 공통 provenance를 먼저 고정합니다.

```yaml
request_id: <run-id>
dataset:
  id: <dataset-id>
  revision: <immutable-revision>
  digest: <content-digest>
base_model:
  id: <model-id>
  revision: <immutable-revision>
backend:
  name: trl
  recipe_revision: <git-commit>
evaluation:
  suite_revision: <evaluation-revision>
output:
  registry_path: <candidate-registry-path>
```

TRL 또는 Megatron-LM의 상세 option은 `backend` 아래 별도 configuration artifact로 저장합니다. 공통 필드는 model lineage와 결과 비교에 사용하고, backend-specific field를 억지로 하나의 schema로 통합하지 않습니다.

## Artifact Manifest

candidate artifact를 registry에 등록할 때 다음 항목을 함께 보관합니다.

| Field | Purpose |
| --- | --- |
| Run identity | training request, source commit과 recipe 연결 |
| Base model | 원본 model과 revision |
| Dataset lineage | dataset revision, digest와 split policy |
| Artifact format | full checkpoint, distributed checkpoint, adapter 또는 serving format |
| Tokenizer | tokenizer와 chat template revision |
| Model configuration | architecture와 context length |
| Numeric format | BF16, FP16, FP8 또는 quantization 정보 |
| Evaluation results | quality, safety와 regression result |
| Runtime compatibility | 검증한 serving engine과 version |
| Integrity | artifact file 목록, size와 digest |

## Promotion Gates

candidate는 다음 gate를 모두 통과해야 serving 대상으로 승격할 수 있습니다.

1. artifact와 manifest의 digest가 일치합니다.
2. tokenizer와 model configuration이 서로 호환됩니다.
3. 새 process에서 artifact를 load하고 deterministic validation sample을 실행할 수 있습니다.
4. held-out evaluation과 고정 benchmark에서 승인 기준을 만족합니다.
5. 기존 production revision 대비 safety와 핵심 task regression이 허용 범위 안에 있습니다.
6. target serving engine에서 startup, memory capacity와 latency 기준을 만족합니다.

짧은 training subset에서 loss가 감소했다는 사실만으로 promotion해서는 안 됩니다. loss와 perplexity는 training implementation 검증에는 유용하지만 production 품질, safety와 agent task 성공률을 대신하지 않습니다.

## Deployment and Rollback

deployment controller는 다음 순서를 사용합니다.

1. candidate revision을 production revision과 다른 immutable location에 배치합니다.
2. artifact load, tokenizer, health endpoint와 representative inference를 검사합니다.
3. 제한된 replica 또는 traffic percentage로 canary를 시작합니다.
4. quality proxy, error rate, latency, memory usage와 task success를 관찰합니다.
5. 기준을 만족하면 traffic을 단계적으로 확대하고 production alias를 갱신합니다.
6. regression이 발생하면 이전 revision으로 alias를 되돌리고 candidate를 격리합니다.

rollback 가능성을 유지하려면 이전 production artifact, serving configuration과 evaluation result를 retention 기간 동안 함께 보관해야 합니다. rollback 이후에는 deployment event와 원인을 dataset 및 training lineage와 연결해 남깁니다.
