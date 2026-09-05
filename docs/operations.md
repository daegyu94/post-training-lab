# Operations

## Observability

post-training run, checkpoint와 serving revision은 공통 identifier로 연결해야 합니다. metric은 [`profiling` branch](https://github.com/daegyu94/sft-lab/tree/profiling)의 schema와 수집 방식을 사용하고, 이 브랜치에서는 lifecycle별 요구사항만 정의합니다.

| Stage | Required signals |
| --- | --- |
| Data preparation | input/output record 수, reject reason, deduplication rate, throughput |
| Training | loss, tokens per second, GPU utilization, memory, collective time |
| Checkpoint | save/load duration, bytes, storage throughput, metadata operation, digest validation |
| Evaluation | dataset revision, metric, threshold, pass/fail과 regression delta |
| Deployment | revision, replica 상태, rollout percentage, startup time와 failure reason |
| Serving | request success, task success, latency, throughput, GPU memory와 rollback signal |

metric label에 raw prompt, response, credential, user identifier처럼 cardinality와 보안 위험이 큰 값을 넣지 않습니다. 상세 sample은 접근 제어된 artifact로 분리하고 metric에는 stable run identifier와 집계 가능한 category만 기록합니다.

## Storage and Network

large-scale training에서는 dataset read와 checkpoint I/O가 shared storage에 집중됩니다. checkpoint save가 training step과 겹치는지, 모든 rank가 작은 file을 생성하는지, metadata service와 data path가 분리되는지 확인해야 합니다.

training cluster와 serving cluster 사이의 artifact transfer는 명시적인 publish 단계로 관리합니다. serving node가 training 중인 checkpoint directory를 직접 읽지 않도록 하고, upload 완료와 digest validation 이후에만 registry revision을 visible 상태로 전환합니다.

multi-node run에서는 collective communication과 storage traffic이 같은 NIC 또는 fabric을 공유하는지 기록합니다. network utilization만으로 병목을 판단하지 않고 GPU idle time, collective duration, I/O queueing과 checkpoint phase를 같은 timeline에서 비교합니다.

## Failure Handling

| Failure | Expected response |
| --- | --- |
| Dataset validation failure | revision 등록을 중단하고 reject evidence 보존 |
| Training worker failure | run을 failed로 종료하거나 검증된 checkpoint에서 재시작 |
| Incomplete checkpoint | registry에서 candidate로 노출하지 않음 |
| Evaluation regression | promotion 차단과 metric delta 기록 |
| Canary startup failure | traffic 전환 전 candidate 격리 |
| Serving regression | 이전 production revision으로 rollback |
| Registry or storage outage | 현재 serving revision 유지, promotion 일시 중단 |

자동 retry는 idempotent한 단계에만 적용합니다. dataset publish, registry alias 변경과 production traffic 전환처럼 상태를 변경하는 단계는 request identity와 compare-and-swap 조건을 사용해 중복 실행을 방지해야 합니다.

## Validation Checklist

production integration을 구현할 때 다음을 확인합니다.

- dataset, training request, checkpoint와 serving revision의 lineage를 한 방향과 역방향으로 추적할 수 있습니다.
- floating model 또는 dataset 이름 대신 immutable revision과 digest를 사용합니다.
- framework-specific configuration은 해당 backend artifact로 분리돼 있습니다.
- incomplete artifact가 registry나 serving system에 노출되지 않습니다.
- canary와 rollback 절차가 실제 serving environment에서 검증돼 있습니다.
- profiling metric과 log가 같은 run identifier와 timestamp 기준을 사용합니다.
- 개인정보와 secret이 dataset, log, metric 또는 checkpoint metadata에 포함되지 않습니다.
