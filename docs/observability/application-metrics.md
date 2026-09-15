# Application Metrics Guide

`MetricEmitter`는 TRL, Megatron, agentic RL application이 같은 형식으로 metric을 기록하도록 돕습니다.
Application은 로컬 JSON snapshot만 갱신하고, 별도 collector가 이를 Node Exporter와 Prometheus에 전달하므로 metric server 장애가 workload를 중단시키지 않습니다.

## Use the Built-in Training Adapters

공통 experiment runner로 TRL이나 Megatron을 실행하면 별도 Python 코드 없이 metric 수집이 활성화됩니다.
Runner가 `OBSERVATORY_RUN_ID`를 설정하고 각 backend launcher가 `<output>/observatory-metrics/`를 사용합니다.

TRL과 Megatron은 다음 metric을 기본 기록합니다.

- `training_loss`
- `training_step_time_seconds`
- `training_tokens_per_second`
- `training_step`
- Megatron의 `training_timer_seconds{timer="..."}`

Hugging Face `Trainer`를 직접 만드는 application은 공용 adapter를 callback으로 전달합니다.

```python
from profiling_lab.trainer_metrics import make_trainer_callback

callback = make_trainer_callback(producer="my-agent-app")
trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_dataset,
}
if callback is not None:
    trainer_kwargs["callbacks"] = [callback]

trainer = SFTTrainer(**trainer_kwargs)
```

`execution_feedback.train`의 SFT와 DPO 경로도 이 adapter를 사용합니다.

## Instrument a Custom Loop

먼저 application process에서 observability package와 출력 위치를 지정합니다.
`worker_id`별 파일을 쓰므로 같은 디렉터리를 사용하는 worker에는 서로 다른 ID가 필요합니다.

```bash
export PYTHONPATH="/path/to/post-training-lab/observability${PYTHONPATH:+:$PYTHONPATH}"
export OBSERVATORY_RUN_ID="agentic-rl-001"
export OBSERVATORY_METRICS_DIR="/path/to/output/observatory-metrics"
```

Application 시작 시 emitter를 한 번 만들고 step이나 episode가 끝날 때 최신 snapshot을 기록합니다.

```python
from profiling_lab.app_metrics import Metric, MetricEmitter

emitter = MetricEmitter.from_env(
    producer="verl",
    role="trainer",
    worker_id="0",
)

if emitter is not None:
    emitter.emit(
        step=global_step,
        samples=[
            Metric("training_loss", float(loss)),
            Metric("training_tokens_per_second", tokens_per_second),
            Metric(
                "agent_tool_call_errors_total",
                tool_error_count,
                kind="counter",
                labels={"tool": "python"},
            ),
        ],
    )
```

`kind`는 `gauge`가 기본이며 누적값에는 `counter`를 사용합니다.
Metric 이름과 의미는 [Metrics Contract](metrics.md)를 따르고, `request_id`, `episode_id`, prompt처럼 계속 늘어나는 값은 label에 넣지 않습니다.

권장 role은 다음과 같습니다.

| Application component | `producer` 예시 | `role` |
| --- | --- | --- |
| Policy training | `verl`, `openrlhf`, `trl` | `trainer` |
| Response generation | `verl`, `vllm` | `rollout` |
| Reward evaluation | application 이름 | `reward` |
| Tool-using agent | application 이름 | `agent` |
| Scheduling | `ray` 또는 application 이름 | `orchestrator` |

`producer`, `role`, `worker_id`에는 64자 이하의 영문자, 숫자, `.`, `_`, `-`만 사용합니다.
Metric 기록이 실패하면 emitter가 한 번 경고한 뒤 비활성화되며 training은 계속됩니다.

## Publish Metrics to Prometheus

각 compute node에서 application과 같은 node-local metrics 디렉터리를 collector에 전달합니다.

```bash
cd /path/to/post-training-lab/observability
NODE_ADDR='<node-management-address>' \
OBSERVATORY_METRICS_DIR='/path/to/output/observatory-metrics' \
DURATION=3600 \
  bash scripts/run_observability.sh node
```

Collector는 2초마다 snapshot을 `observatory.prom`으로 변환합니다.
주기를 바꾸려면 `OBSERVATORY_METRICS_INTERVAL`을 초 단위로 설정합니다.

여러 node가 하나의 NFS metrics 디렉터리를 읽으면 같은 worker가 여러 Prometheus instance에 중복됩니다.
각 node의 application과 collector는 같은 node-local 디렉터리를 사용합니다.

## Check the Result

Collector 없이도 output directory에서 마지막 snapshot을 확인할 수 있습니다.

```bash
PYTHONPATH=observability python -m profiling_lab.show_run /path/to/output
```

정상이라면 다음과 같이 producer, role, worker와 마지막 metric이 표시됩니다.

```text
[verl/trainer worker 0] step 10: training_loss=1.25 training_tokens_per_second=420.0
```

Dashboard에서는 `run_id`, `producer`, `role`, `worker_id`로 시계열을 구분합니다.
Snapshot은 worker별 최신값만 보존하므로 전체 step history가 필요하면 Prometheus 또는 application log를 사용합니다.

vLLM과 Ray가 자체 Prometheus endpoint를 제공하는 경우 해당 metric을 `MetricEmitter`로 복제하지 않습니다.
Request latency histogram과 trace도 native exporter나 OpenTelemetry를 사용합니다.
