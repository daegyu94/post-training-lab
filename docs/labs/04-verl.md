# Lab 04: verl Agentic RL Resource Profiling

## Goal

verl/Ray cluster에서 rollout 생산, trainer 소비, actor/critic/reference/reward role과 tool/environment wait를 분리합니다.
평균 GPU utilization만 보는 대신 비동기 pipeline의 backpressure와 critical path를 찾습니다.

## 1. Start With Ray and Rollout Metrics

Ray는 node, actor, task, object store와 scheduling metric을 Prometheus format으로 노출합니다.
verl의 async rollout monitoring을 사용하는 version에서는 다음 설정을 기존 launch command에 추가합니다.

```text
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.disable_log_stats=False
actor_rollout_ref.rollout.prometheus.enable=True
actor_rollout_ref.rollout.prometheus.port=9090
actor_rollout_ref.rollout.prometheus.file=/tmp/ray/session_latest/metrics/prometheus/prometheus.yml
```

Ray가 생성한 Prometheus target configuration을 확인하고 중앙 Prometheus의 application target에 병합합니다.
이미 중앙 Prometheus가 있다면 verl 예제처럼 두 번째 Prometheus를 무조건 띄우지 말고 Ray/rollout endpoint를 기존 service discovery에 추가합니다.
version마다 metric 이름과 port discovery가 달라질 수 있으므로 실제 `/metrics` endpoint와 generated configuration을 기준으로 합니다.

## 2. Build the Role Dashboard

다음 signal을 `run_id`, `role`, node와 replica 기준으로 연결합니다.

| Area | Signals |
| --- | --- |
| End to end | RL step time, accepted samples/s, prompt/output tokens/s, reward distribution |
| Rollout | queued/running request, batch size, generation latency, TTFT/TPOT, KV cache utilization, preemption |
| Training roles | actor/critic/reference/reward duration, optimizer time, role별 GPU utilization |
| Agent loop | turn count, tool-call count, environment/tool latency, timeout, error, retry와 wait ratio |
| Ray | pending/running/failed actor/task, scheduling delay, object-store memory, spill, restart와 placement |
| Backpressure | rollout production rate, trainer consumption rate, queue depth/age와 idle reason |

개별 request 또는 episode ID를 Prometheus label로 만들지 않습니다.
count, rate와 latency histogram으로 집계하고 일부 episode만 distributed trace로 보존합니다.

## 3. Run Without a Profiler

첫 기준 run에서는 `global_profiler.steps=null`로 두고 다음 관계를 확인합니다.

- rollout queue가 비어 있으면서 trainer GPU가 idle이면 rollout 공급 부족인지 확인합니다.
- queue가 계속 증가하면 trainer update 또는 weight synchronization이 소비 병목인지 확인합니다.
- rollout GPU utilization이 낮을 때 tool/environment wait와 timeout이 증가하는지 확인합니다.
- Ray scheduling delay, actor restart 또는 object spill이 같은 시점에 발생하는지 확인합니다.
- reward 증가가 throughput 또는 response length 변화로 인한 착시가 아닌지 함께 확인합니다.

## 4. Use verl's Built-in Selected Profiler

metric으로 느린 step과 role을 찾은 뒤 [`torch-profiler.yaml`](../../examples/verl/torch-profiler.yaml)을 사용 중인 verl configuration에 병합합니다.
예제는 global step 5에서 actor rank 0과 rollout rank 0만 PyTorch Profiler로 수집합니다.

Agent Loop에서는 `discrete: true`가 필요하며 rollout의 `profile_token_start`와 `profile_token_end`로 response-token window를 더 좁힐 수 있습니다.
rollout rank는 한 GPU process만 독립적으로 동작한다는 뜻이 아니라 해당 inference replica로 매핑되므로 TP/DP/PP 크기와 trace file의 rank metadata를 함께 확인합니다.

production configuration에 적용하기 전에 현재 verl version의 profiler schema를 확인합니다.
최신 verl은 `global_profiler.steps`, role별 `enable`, `all_ranks`, `ranks`와 PyTorch tool option을 제공하지만 config 위치와 지원 engine은 release에 따라 달라질 수 있습니다.

## 5. Trace Agent and Tool Wait

PyTorch trace는 외부 tool 또는 environment가 왜 늦었는지 보여주지 않습니다.
agent loop, tool gateway와 environment service에 OpenTelemetry context를 전파하고 다음 span 관계를 sampled trace로 기록합니다.

```text
rl_step
+-- rollout_batch
|   +-- model_generate
|   +-- agent_turn
|       +-- tool_call
|           +-- environment_request
+-- reward
+-- actor_update
+-- weight_sync
```

span에는 `run_id`, role, sampled episode class, tool name, status와 token count 정도만 기록합니다.
prompt, response와 tool payload는 기본적으로 제외하고 필요한 경우 명시적인 redaction/allowlist를 적용합니다.

## Common Patterns

| Pattern | Likely hypothesis | Next evidence |
| --- | --- | --- |
| trainer idle, rollout queue empty | rollout/tool 공급 부족 | rollout batch, tool span과 engine queue |
| rollout queue 증가, trainer busy | trainer consumption 병목 | actor update trace와 collective/checkpoint |
| rollout GPU idle, tool spans long | environment-bound agent loop | tool별 latency/error와 service resource metric |
| Ray pending actor 증가 | resource/placement 부족 | placement group, logical resource와 node health |
| object spill과 step latency 동시 증가 | object-store memory pressure | object store, disk I/O와 object size |
| weight sync 구간만 길어짐 | model transfer/collective 병목 | network, rank trace와 NCCL baseline |

## Expected Result

end-to-end RL step 지연을 rollout, agent/tool wait, reward, actor update, weight synchronization 또는 Ray scheduling 중 하나 이상으로 좁힐 수 있어야 합니다.
이후 선택한 role/rank/token window의 trace만 수집합니다.

## References

- [verl Prometheus and Grafana rollout monitoring](https://verl.readthedocs.io/en/latest/advance/grafana_prometheus.html)
- [verl profiler system](https://verl.readthedocs.io/en/latest/perf/verl_profiler_system.html)
- [PyTorch profiling in verl](https://verl.readthedocs.io/en/latest/perf/torch_profiling.html)
- [Ray metrics](https://docs.ray.io/en/latest/cluster/metrics.html)
