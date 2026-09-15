# Verl Agentic RL

목표 topology는 `spark1` rollout·`spark2` trainer의 Qwen3-30B-A3B LoRA GRPO다.
현재 backend는 그 전제 조건을 검증하는 0.5B hybrid agentic smoke와 30B rollout-only 검증을 제공한다.
이 문서의 정적 preflight는 Spark 노드에 접속하거나 GPU를 쓰지 않는다.

## Static preflight

Controller에서 다음 명령으로 role, immutable model revision, GRPO group size, LoRA 및 smoke task 설정을 확인한다.

```bash
backends/verl/scripts/preflight.sh experiments/verl/qwen3-30b-agentic-grpo-smoke.json
```

목표 계획은 trainer와 rollout을 서로 다른 host에 둔다.
현재 V1 smoke의 generic Ray placement는 host affinity를 제공하지 않으므로 이 역할 고정을 보장하지 않는다.
GRPO는 prompt당 최소 두 sample이 필요하고, 초기 reward는 별도 reward model 대신 calculator tool의 정확한 최종 정답으로 계산한다.

## Deferred Spark work

Spark 노드에서 다음 순서로 실행한다.

1. 양 노드의 node-local `verl` virtual environment에 호환되는 verl, vLLM, Ray를 고정 설치한다. 완료.
2. 0.5B 모델로 Ray·tool-call·weight-sync를 1 step 검증한다. 완료.
3. Qwen3-30B-A3B rollout-only와 LoRA update-only를 각각 확인한다. rollout-only 완료.
4. 두 역할을 연결해 30B GRPO 1 step을 실행한다. host-affinity가 있는 fully-async resource-isolation launcher를 추가한 뒤 수행한다.

프로젝트 최상위 `.venv`는 NFS로 공유된다.
현재 이 환경은 controller의 Python 3.13으로 생성됐지만 Spark 노드에서는 같은 실행 파일 경로가 Python 3.12를 가리킨다.
따라서 ABI가 있는 `torch`와 `vLLM`은 공유 `.venv`에 설치하지 않고, 각 Spark 노드의 `/home/spark/.local/ptl/venvs/verl`에 Python 3.12 전용 환경으로 설치한다.
기존 SFT용 가상환경도 변경하지 않는다.
설치 기준은 [`backends/verl/requirements-spark.txt`](../../backends/verl/requirements-spark.txt)의 고정된 `verl`·`vLLM` 버전이다.
Verl v1 `separate_async`는 PyPI 배포물에 포함되지 않은 `TransferQueue==0.1.8`와 NCCL weight-sync용 `cupy-cuda13x`도 필요하므로 같은 파일에서 함께 설치한다.
vLLM의 FlashInfer sampler JIT에는 같은 환경의 `ninja` 실행 파일이 필요하므로 launcher가 가상환경의 `bin`을 `PATH` 앞에 둔다.

Calculator smoke는 Verl의 stateless function tool을 사용한다.
`verl_lab.function_tools`가 안전한 정수 계산기만 노출하고, `verl_lab.reward`가 마지막 `#### <integer>`를 rule reward로 채점한다.
`backends/verl/scripts/run_agentic_grpo_smoke.sh`는 Ray head가 `spark2`, worker가 `spark1`로 시작된 뒤 `spark2`에서 실행하는 0.5B·1-step 검증 명령이다.
이는 two-node component smoke이며 V1의 generic Ray placement는 trainer와 rollout을 특정 host에 고정하지 않는다.
학습 데이터는 NFS 공유 worktree의 `data/verl/`에 만들고, Hydra 로그와 checkpoint는 trainer의 `/mnt/post-training/verl/`에 쓴다.
현재 두 노드에서 공통으로 연결된 RoCE 포트만 쓰도록 `NCCL_SOCKET_IFNAME=enp1s0f0np0`와 `NCCL_IB_HCA=rocep1s0f0`를 기본 설정한다.
기본 0.5B smoke는 rank-8 LoRA를 base weight에 merge해 동기화하고, vLLM LoRA sync에 필요한 `load_format=safetensors`를 고정한다.

30B는 LoRA와 1024-token 이하 smoke만 대상으로 한다.
Spark1에서 128-token vLLM rollout은 BF16 weight 56.9GiB와 KV cache 26.3GiB로 생성까지 확인했다.
30B GRPO는 rollout과 trainer를 서로 다른 GPU에 고정하는 fully-async resource-isolation launcher를 추가한 뒤에만 실행한다.
현재 smoke의 hybrid engine은 두 역할을 한 GPU에 colocate하므로 30B 실행 대상이 아니다.
이 저장소의 기존 full SFT는 한 노드에서 OOM이었으므로 full-parameter GRPO와 별도 reward model은 이 topology 범위에 넣지 않는다.
