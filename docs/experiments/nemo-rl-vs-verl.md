# NeMo RL과 verl 비교

이 실습은 같은 2노드 Spark cluster에서 NeMo RL과 verl의 GRPO 실행 경로를 비교한다.
장기 수렴이나 최대 처리량이 아니라, 동일한 작은 입력으로 rollout과 LoRA update가 한 번 끝나는지와 그 실행 비용을 확인하는 smoke 실험이다.

## 비교 계획

1. 프레임워크 릴리스, 모델 파일, 데이터, seed와 학습 조건을 고정한다.
2. `spark1`과 `spark2`의 GPU를 하나씩 사용하고 generation과 trainer를 서로 다른 GPU에 둔다.
3. 각 프레임워크에서 GRPO 1 step을 실행해 종료 코드, global step, reward 분산, gradient, step 시간과 peak allocation을 수집한다.
4. 두 실행이 성공하면 기능 경로 비교를 멈춘다.
   반복 측정과 장기 품질 평가는 별도 실험으로 남긴다.

실행 경로의 성공 조건은 exit code 0, global step 1과 finite metric이다.
Reward 분산과 gradient는 별도로 기록하며, 작은 GRPO group에서 prompt별 reward가 같아 zero advantage가 나오면 실제 update 성공으로 해석하지 않는다.
Checkpoint와 validation은 1-step smoke의 범위에 포함하지 않는다.

## 고정 조건

| 항목 | 값 |
| --- | --- |
| 날짜 | verl: 2026-09-16, NeMo RL: 2026-09-17 |
| Hardware | `spark1`, `spark2`의 NVIDIA GB10 각 1개 |
| NeMo RL | v0.7.0 공식 ARM64 NGC image |
| verl | v0.9.0 node-local virtual environment |
| Model | `Qwen2.5-0.5B-Instruct`, BF16 |
| Model SHA-256 | `model.safetensors`: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| Algorithm | GRPO, LoRA rank 8, alpha 16 |
| Batch | prompt 2개, prompt당 generation 2개, global batch 4 |
| Length | prompt 최대 256, response 최대 128 token |
| Placement | trainer 1 GPU, vLLM generation 1 GPU, non-colocated |
| Data | [`math-smoke.jsonl`](../../experiments/rl-framework-comparison/math-smoke.jsonl)의 정수 산술 2문제 |
| Reward | 응답의 마지막 정수가 정답과 같은지 판정 |
| Seed | 42 |

두 노드에서 `config.json`, `model.safetensors`, `tokenizer.json`의 SHA-256이 각각 일치하는 것을 실행 전에 확인했다.
NeMo RL은 공식 [`v0.7.0`](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.7.0) container를 사용하고, verl은 공식 [`v0.9.0`](https://github.com/verl-project/verl/releases/tag/v0.9.0) package를 사용한다.
NeMo RL image ID는 두 노드에서 `sha256:439116e3af3142f908d07563d7c9cb7d81277d35934b636819ec48a4e617ac0d`로 일치했다.
Image 내부 조합은 PyTorch 2.11.0+cu130, Ray 2.55.1, vLLM 0.20.0이며 `nemo_rl` package metadata는 `0.6.0+19244a094`를 보고했다.
verl 환경은 PyTorch 2.13.0+cu130, Ray 2.58.0, vLLM 0.29.0이므로 결과에는 framework 외 dependency 차이도 포함된다.

## 실행

verl은 두 노드에서 Ray head와 worker를 시작한 뒤 `spark2`에서 실행한다.

```bash
cd /home/spark/shared/post-training-lab
backends/verl/scripts/run_grpo_comparison.sh
```

NeMo RL은 두 노드에서 `nvcr.io/nvidia/nemo-rl:v0.7.0` container로 같은 Ray cluster를 구성한다.
공식 image에 포함된 `/opt/nemo-rl` source와 dependency를 사용하고, repository와 node-local model 및 output directory만 mount한다.

```bash
cd /workspace/post-training-lab
backends/nemo_rl/scripts/run_grpo_comparison.sh
```

두 launcher는 이미 만들어진 Ray cluster에 연결한다.
Ray 시작·종료와 Docker mount의 전체 명령은 이 문서의 [재현 명령](#재현-명령)을 따른다.

## 2026-09-16–17 결과

| Metric | verl 0.9.0 | NeMo RL 0.7.0 |
| --- | ---: | ---: |
| Exit code / global step | 0 / 1 | 0 / 1 |
| Reward mean, min, max | 0.25, 0, 1 | 0.50, 0, 1 |
| Loss | -0.3421 | 0.0000 |
| Gradient norm | 3.8286 | 0.0000 |
| Framework step time | 24.86 s | 73.72 s |
| End-to-end wall time | 139.18 s | 124.04 s |
| Mean generated tokens | 15.5 | 85.0 |
| Trainer peak allocated | 4.65 GiB | 내장 metric 미제공 |

verl은 두 노드에서 rollout, reward 계산, nonzero-gradient LoRA update와 weight sync를 완료했다.
측정 실행에서는 final metric 출력 뒤 DataLoader worker shutdown 경고가 있었지만 process exit code는 0이었고, 재현 launcher는 `data.dataloader_num_workers=0`으로 이 불필요한 worker를 만들지 않는다.

NeMo RL도 두 노드에서 rollout, reward, logprob, policy training과 weight sync 경로를 완료했다.
다만 reward가 prompt별로 `[0, 0]`, `[1, 1]`이어서 leave-one-out GRPO advantage와 gradient가 모두 0이었고 실제 LoRA weight update의 증거는 얻지 못했다.
Generation worker는 engine 초기화 직후 58.54 GiB를 allocated하고 59.34 GiB를 reserved했다고 진단했지만 trainer worker의 peak allocation은 내장 metric으로 제공되지 않았다.

이 숫자는 각 framework의 첫 1-step 실행 한 번에서 얻은 값이다.
Model load, Ray actor 생성, vLLM warmup과 JIT가 wall time에 포함되므로 steady-state throughput으로 해석하면 안 된다.
두 framework의 내장 metric 정의도 완전히 같지 않으므로 framework step time과 peak allocation은 실행 경로의 규모를 확인하는 참고값이다.
특히 NeMo RL이 평균 85 token, verl이 평균 15.5 token을 생성했으므로 이 한 번의 step 시간을 framework 처리량 순위로 해석할 수 없다.

## 재현 명령

먼저 두 노드에 다른 Ray 또는 GPU workload가 없는지 확인한다.
실행 중인 공유 작업이 있으면 종료하지 말고 실험을 미룬다.

### verl

`spark2`에서 head를 시작한다.

```bash
NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0 \
  /home/spark/.local/ptl/venvs/verl/bin/ray start \
  --head --node-ip-address=192.168.0.12 --port=6379 \
  --dashboard-host=127.0.0.1 --num-gpus=1 --num-cpus=16 \
  --disable-usage-stats
```

`spark1`에서 worker를 연결한다.

```bash
NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0 \
  /home/spark/.local/ptl/venvs/verl/bin/ray start \
  --address=192.168.0.12:6379 --node-ip-address=192.168.0.11 \
  --num-gpus=1 --num-cpus=16 --disable-usage-stats
```

`spark2`에서 시간을 포함해 실행한다.

```bash
cd /home/spark/shared/post-training-lab
/usr/bin/time -f 'comparison_wall_seconds=%e' \
  backends/verl/scripts/run_grpo_comparison.sh
```

이번 실행으로 시작한 Ray만 양 노드에서 `ray stop`으로 종료한다.

### NeMo RL

두 노드에 ARM64 image를 준비한다.

```bash
docker pull nvcr.io/nvidia/nemo-rl:v0.7.0
```

각 노드에서 model과 repository 경로가 실제 mount 위치와 일치하는지 확인한 뒤 같은 이름의 container를 시작한다.
`spark1`은 head, `spark2`는 worker다.

```bash
docker run -d --rm --name nemo-rl-compare \
  --network host --ipc host --gpus all --cap-add IPC_LOCK --ulimit memlock=-1 \
  -v /dev/infiniband:/dev/infiniband \
  -v /home/spark/.local/ptl/models/Qwen2.5-0.5B-Instruct:/models/Qwen2.5-0.5B-Instruct:ro \
  -v /home/spark/shared/post-training-lab:/workspace/post-training-lab:ro \
  -v /mnt/post-training/rl-framework-comparison:/results \
  -e NEMO_RL_PY_EXECUTABLES_SYSTEM=0 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_HCA=rocep1s0f0 \
  nvcr.io/nvidia/nemo-rl:v0.7.0 \
  ray start --head --node-ip-address=192.168.0.11 --port=6379 \
  --dashboard-host=127.0.0.1 --num-gpus=1 --num-cpus=16 \
  --disable-usage-stats --block
```

`spark2`에서는 마지막 Ray 명령만 다음과 같이 바꾼다.

```bash
ray start --address=192.168.0.11:6379 \
  --node-ip-address=192.168.0.12 --num-gpus=1 --num-cpus=16 \
  --disable-usage-stats --block
```

`spark1`에서 시간을 포함해 실행한다.

```bash
/usr/bin/time -f 'comparison_wall_seconds=%e' \
  docker exec nemo-rl-compare \
  /workspace/post-training-lab/backends/nemo_rl/scripts/run_grpo_comparison.sh
```

이번 실행으로 만든 양 노드의 container만 `docker stop nemo-rl-compare`로 종료한다.

## 다음 단계

Nonzero update 비교가 필요하면 먼저 prompt당 generation을 늘려 각 GRPO group에 mixed reward가 생기는지 확인한다.
처리량 비교가 필요할 때만 같은 생성 token 수를 강제하고 warmup 1회 뒤 10 step을 3회 반복해 중앙값을 사용한다.
학습 품질 비교가 필요할 때는 최소 수백 prompt의 같은 holdout과 reward implementation을 사용하며, 이 smoke 결과를 품질 근거로 재사용하지 않는다.
