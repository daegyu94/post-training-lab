# Verl Agentic RL

이 backend는 `spark1` rollout·`spark2` trainer를 분리한 Qwen3-30B-A3B LoRA GRPO smoke를 준비한다.
실제 GPU 실행은 Spark1 복구 후의 별도 단계이며, 이 문서의 정적 preflight는 Spark 노드에 접속하거나 GPU를 쓰지 않는다.

## Static preflight

Controller에서 다음 명령으로 role, immutable model revision, GRPO group size, LoRA 및 smoke task 설정을 확인한다.

```bash
backends/verl/scripts/preflight.sh experiments/verl/qwen3-30b-agentic-grpo-smoke.json
```

이 계획은 trainer와 rollout을 서로 다른 host에 둔다.
GRPO는 prompt당 최소 두 sample이 필요하고, 초기 reward는 별도 reward model 대신 calculator tool의 정확한 최종 정답으로 계산한다.

## Deferred Spark work

Spark1이 복구된 뒤에만 다음을 수행한다.

1. 양 노드의 node-local `verl` virtual environment에 호환되는 verl, vLLM, Ray를 고정 설치한다.
2. 0.5B 모델로 Ray·tool-call·weight-sync를 1 step 검증한다.
3. Qwen3-30B-A3B rollout-only와 LoRA update-only를 각각 확인한다.
4. 두 역할을 연결해 30B GRPO 1 step을 실행한다.

30B는 LoRA와 1024-token 이하 smoke만 대상으로 한다.
이 저장소의 기존 full SFT는 한 노드에서 OOM이었으므로 full-parameter GRPO와 별도 reward model은 이 topology 범위에 넣지 않는다.
