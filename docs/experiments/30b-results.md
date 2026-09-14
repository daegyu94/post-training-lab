# 30B Results

30B Qwen과 GLM의 검증 실행, GPU·메모리 결과와 framework 호환성 비교입니다.
현재 판단은 아래 2026-09-15 통제 실험을 기준으로 하며, 뒤의 2026-09-14 이전 결과는 historical context입니다.

## Controlled Results (2026-09-15)

Checkpoint·memory·recompute matrix는 commit `bee4520e05a04aaf48f8fe971a6b2a1726f6250b`의 깨끗한 worktree에서 실행했습니다.
두 모델은 같은 UltraChat revision, TP=1·PP=1·EP=2, BF16, micro/global batch 1/2, seed 42와 `transformer_engine`을 사용했습니다.
NFS는 repository checkout에만 사용했고 checkpoint는 node-local NVMe에 저장했으므로 distributed restore는 측정하지 않았습니다.

비교 cell은 별도 warmup 또는 pilot 뒤 3회 측정했습니다.
Save-call relative MAD가 10%를 넘으면 8회로 늘리기로 했지만 모든 checkpoint cell이 기준 이하여서 추가 반복은 필요하지 않았습니다.

### Distributed checkpoint write

Qwen과 GLM은 각각 12/12 run을 통과했습니다.
완료 시간은 save/enqueue와 async blocking finalization을 합한 host 시간이며 storage throughput으로 해석하지 않습니다.

| Model | Variant | 논리 크기 median | Save/enqueue median | Finalize median | 완료 median | rMAD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | sync `torch_dist`, EP2 | 10.64 MB | 1.045 s | 0.000 s | 1.046 s | 0.30% |
| Qwen | async `torch_dist`, EP2 | 10.64 MB | 0.204 s | 0.769 s | 0.972 s | 2.15% |
| Qwen | phase-2 `torch_dist`, EP2 | 10.64 MB | 1.065 s | 0.000 s | 1.066 s | 1.38% |
| Qwen | phase-2 `fsdp_dtensor`, DP2 | 20.92 MB | 0.441 s | 0.000 s | 0.441 s | 2.21% |
| GLM | sync `torch_dist`, EP2 | 22.04 MB | 1.064 s | 0.000 s | 1.064 s | 0.92% |
| GLM | async `torch_dist`, EP2 | 22.04 MB | 0.173 s | 0.689 s | 0.862 s | 0.83% |
| GLM | phase-2 `torch_dist`, EP2 | 22.04 MB | 1.063 s | 0.000 s | 1.064 s | 1.27% |
| GLM | phase-2 `fsdp_dtensor`, DP2 | 43.14 MB | 0.410 s | 0.000 s | 0.410 s | 1.14% |

Async는 API blocking 시간을 줄였지만 finalization을 포함한 차이는 Qwen 7.0%, GLM 19.0%였습니다.
Format/layout 행은 payload와 memory placement가 달라 framework 우열 비교에 쓰지 않습니다.
Raw manifests는 `results/refresh-distributed-{qwen|glm}-bee4520/manifest.json`입니다.

### Multi-step checkpoint impact

각 측정은 8 optimizer steps 중 2 step마다 저장해 총 4개 checkpoint를 만들었고, 앞의 2 step은 steady-step 집계에서 제외했습니다.
두 모델 모두 warmup 2회와 측정 6회, 합계 8/8 run을 통과했습니다.

| Model | Variant | 4개 논리 크기 median | Save/enqueue 합 median | Finalize median | 완료 median | Steady step median |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | sync | 291.86 MB | 2.441 s | 0.000 s | 2.441 s | 3239.25 ms |
| Qwen | async | 291.86 MB | 1.119 s | 0.774 s | 1.881 s | 3304.50 ms |
| GLM | sync | 602.56 MB | 2.822 s | 0.000 s | 2.822 s | 3410.05 ms |
| GLM | async | 602.56 MB | 1.573 s | 0.818 s | 2.391 s | 3442.35 ms |

Async 완료 시간은 sync보다 Qwen 23.0%, GLM 15.3% 짧았지만 steady-step median은 각각 2.0%, 0.9% 길었습니다.
8-step run은 장기 throughput 근거가 아닙니다.
Raw manifests는 `results/refresh-checkpoint-impact-{qwen|glm}-bee4520/manifest.json`입니다.

### Transformer Engine memory

두 모델 모두 length별 pilot 1회와 측정 3회, 합계 8/8 run을 통과했습니다.
표는 측정 run마다 두 rank 중 큰 값을 취한 뒤의 median입니다.

| Model | Length | Peak allocated median | Peak reserved median |
| --- | ---: | ---: | ---: |
| Qwen | 4096 | 34.320 GiB | 40.738 GiB |
| Qwen | 8192 | 38.662 GiB | 48.500 GiB |
| GLM | 4096 | 34.537 GiB | 41.311 GiB |
| GLM | 8192 | 38.942 GiB | 49.859 GiB |

4096에서 8192로 늘릴 때 allocated 증가는 Qwen 12.65%, GLM 12.75%로 유사했습니다.
Raw manifests는 `results/refresh-memory-te-{qwen|glm}-bee4520/manifest.json`입니다.

### Qwen recompute

각 cell은 warmup 1회와 측정 3회로 총 16/16 run을 통과했습니다.
표는 4-step run에서 첫 step을 제외한 median과 측정 run의 최대 CUDA peak입니다.

| Length | Recompute | Steady step median | Peak allocated max | Peak reserved max |
| ---: | --- | ---: | ---: | ---: |
| 2048 | full | 3148.1 ms | 32.150 GiB | 36.564 GiB |
| 2048 | selective | 2304.3 ms | 43.413 GiB | 46.270 GiB |
| 4096 | full | 6343.5 ms | 34.321 GiB | 39.990 GiB |
| 4096 | selective | 4572.3 ms | 55.905 GiB | 59.492 GiB |

Selective는 step median을 26.8%와 27.9% 줄이는 대신 peak allocated를 35.0%와 62.9% 늘렸습니다.
실행 뒤 read는 raw local-shard probe이며 checkpoint API restore가 아닙니다.
Raw manifest는 `results/refresh-recompute-qwen-bee4520/manifest.json`입니다.

### Full-SFT capacity

`experiments/megatron/qwen3-30b-full-sgd-pilot.json`은 Qwen full parameter, distributed SGD, TE, EP2, length 2048와 1 optimizer step을 고정합니다.
Commit `ed40ca702f07470d2378c441d392a2a89d71ddfa`에서 runner와 launcher가 `OPTIMIZER=sgd`를 전달하는 것을 검증했습니다.

Node-local 2048 코호트 실행은 설정 로그에서 SGD와 full recompute를 확인했지만 FP32 main gradient 구성 중 NVIDIA allocation failure와 global OOM으로 첫 optimizer step 전에 종료됐습니다.
따라서 96.6 GiB 사전 추정은 이 장치의 fit을 보장하지 않았고 같은 2노드 설정은 재실행하지 않습니다.
Controller 로그는 `results/refresh-full-sft-sgd-qwen-ed40ca7-cohort1/rank-0.log`이며 spark2 kernel log에서 실행 Python process의 OOM reaping을 확인했습니다.

### Remaining experiments

1. Full-SFT는 FP32 main gradient까지 shard하는 지원 topology를 확인하고 다시 추정한 뒤 1-step pilot부터 시작합니다.
2. Recompute를 모델 간 결론으로 쓸 필요가 생기면 같은 4-cell matrix를 GLM에서 3회 반복합니다. 현재 결과는 Qwen trade-off로만 해석합니다.
3. Async 장기 throughput이 운영 결정에 필요할 때만 100 step 이상 고정 workload를 추가합니다.
4. Shared checkpoint store가 생기기 전에는 distributed restore를 계획하지 않습니다.

0.5B smoke, NFS checkpoint와 backend-mixed exploratory 결과는 이 통제 비교에 포함하지 않습니다.

<a id="실제로-확인한-조합"></a>

## Historical Results (through 2026-09-14)

아래는 이전 조건에서 `build.py --execute`로 확인한 기록입니다.
Dataset·attention backend·반복 수가 위 통제 실험과 다르므로 최신 표와 합쳐 비교하지 않습니다.

| Case | Backend | Model | Dataset | knob | 결과 |
| --- | --- | --- | --- | --- | --- |
| 1 | TRL DDP | Qwen2.5-0.5B-Instruct | ultrachat (256/32) | `--epochs 1` | base/train/tuned 통과 |
| 2 | TRL DeepSpeed | Qwen3-30B-A3B | no_robots | `--offload nvme --max-steps 1` | train 통과, 별도 `tuned` 평가 통과 |
| 3 | Megatron | Qwen3-30B-A3B | self_oss (40,000/8,000) | `--max-steps 1 --stage all` | base `1.402232` → tuned `1.342737` |
| 4 | Megatron | GLM-4.7-Flash | ultrachat (160,000/30,000) | `--max-steps 1 --max-length 4096 --stage all` | base `2.269922` → tuned `2.012836` |

**Case 1**은 256샘플·16 step에서 `base` 1.4340 → `train` 1.4308 → `tuned` 1.4308을 확인했습니다.
저장 전후 eval loss 일치는 이 실행의 adapter 재로딩 증거이며 모델 품질 지표는 아닙니다.

**Case 3·4**는 node-local에 흩어진 `torch_dist` metadata·shard를 NFS checkpoint 경로로 모은 뒤 base→train→tuned와 iteration 1 재로딩을 통과했습니다.
NaN·skipped iteration은 0, `checkpoint_reload_verified`는 `true`였습니다.
Case 4는 UltraChat 길이 초과로 `--max-length 2048`에서 실패한 뒤 4096으로 통과했습니다.

<a id="30b-gpu-results"></a>

## 30B GPU Results

2026-09-09에 commit `54a5fe0b69d168a86746d7de576d5dcae61c36b9`의 깨끗한 checkout으로 `spark1`·`spark2`에서 GPU당 process 하나를 실행했습니다.
**이 표는 1-step 실행 가능성만 보여주며 장기 안정성이나 학습 품질을 뜻하지 않습니다.**

Megatron은 TP=1·PP=1·EP=2·DP=2, BF16, sequence length 2048, global batch 2, attention LoRA를 사용했고 두 모델 모두 1 optimizer step, 평가, async `torch_dist` checkpoint와 양 rank exit 0을 확인했습니다.

| Model | Train result | Peak allocated |
| --- | --- | --- |
| Qwen3-30B-A3B | loss `4.110986`, grad norm `15.480`, step `7.28 s` | `34.759 GiB` |
| GLM-4.7-Flash | loss `2.497179`, grad norm `7.883`, step `6.37 s` | `34.671 GiB` |

TRL은 Qwen3-30B-A3B, 같은 precision·길이·batch와 attention LoRA를 사용했습니다.

| Backend | Train / eval loss | Peak allocated / reserved | Update evidence |
| --- | --- | --- | --- |
| DDP | `3.156563` / `2.587339` | `58.825 / 58.971 GiB` | sampled parameter delta가 0이 아님 |
| FSDP2 | `3.156250` / `2.566406` | `32.147 / 34.188 GiB` | optimizer step과 sharded checkpoint |

같은 조건은 `experiments/megatron/{qwen3-30b-lora,glm-4.7-flash-30b-lora}.json`으로 재실행할 수 있습니다.
[Checkpoint and Memory Experiment](checkpoint-io.md#checkpoint-and-memory-experiment)는 같은 모델을 반복 측정한 별도 실험이므로, 이 표와 그쪽의 median은 서로 다른 run의 값입니다.

### NVMe and Full SFT

두 노드의 `/mnt/post-training`은 로컬 NVMe root filesystem에 있고 backend별 디렉터리에 `spark` 쓰기 권한이 있습니다.
Megatron Qwen3-30B-A3B와 GLM-4.7-Flash LoRA는 로컬 NVMe에 데이터 cache와 로그를 쓰고, NFS의 공통 `torch_dist` checkpoint에서 iteration 1을 새 process로 재로딩해 `STAGE=all`을 완료했습니다 — 정확한 수치는 위 [Verified Runs](#verified-runs)의 Case 3·4와 같습니다.
Megatron은 NVMe를 native training state offload 대상으로 지원하지 않으므로 이 결과는 dataset cache와 checkpoint I/O 검증입니다.

30B full SFT의 DDP·FSDP2 실패는 dataset 전체 적재가 원인이 아닙니다.
DDP는 parameter와 gradient가 unified memory 한도에 근접하고, 설치된 Accelerate의 FSDP2 준비 과정은 sharding 전에 trainable BF16 parameter를 FP32로 올립니다.
남은 후보였던 TRL DeepSpeed ZeRO-3 NVMe offload는 2노드에서 Qwen과 GLM 모두 1 optimizer step, 별도 `tuned` process 평가와 복구 가능한 native ZeRO checkpoint까지 확인했습니다.
기존 Qwen 검증 실행에서 두 노드 모두 `/mnt/post-training/trl/zero_stage_3` swap footprint가 약 256GiB로 각 노드 물리 RAM 119GiB보다 컸습니다 — NVMe offload 없이는 이 구성이 노드 RAM만으로 성립하지 않는다는 근거이며, 자세한 수치와 한계는 [30B NVMe 실습](../../labs/nvme-30b/README.md#why-nvme-offload-is-necessary)을 따릅니다.

#### UltraChat Full-SFT Topology Comparison (2026-09-14)

두 모델 모두 UltraChat revision `8049631c405ae6576f93f445c6b8166f76f5505a`, length 512, train/eval 4/1, BF16, AdamW, ZeRO-3 NVMe와 1 optimizer step을 사용했습니다.
2-node의 restore는 학습 process가 종료된 뒤 새 `tuned` process가 model skeleton과 DeepSpeed engine을 준비하고 native checkpoint를 적용할 때까지의 end-to-end elapsed time입니다.

| Model | 1-node | 2-node | Checkpoint / save | New-process restore | 2-node min `MemAvailable` / peak swap |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen3-30B-A3B | OOM before checkpoint | Passed | 451.7 GiB / 720.2 s | 994.1 s | 24.6 GiB / 3.15 GiB |
| GLM-4.7-Flash | OOM before checkpoint | Passed | 167.3 GiB / 300.6 s | 420.3 s | 25.5 GiB / 0.57 GiB |

![UltraChat full-SFT checkpoint lifecycle](../figures/full-sft-checkpoint-restore.svg)

Qwen 2-node는 restore 중 peak swap이 시작값 0.56 GiB보다 높아졌지만 최소 `MemAvailable`이 24.6 GiB였고 restore와 finite eval을 완료했으므로 유효 결과로 유지합니다.
GLM 2-node의 peak swap은 시작값과 같아 측정 중 swap 증가가 없었습니다.
반면 1-node는 Qwen이 `MemAvailable` 0과 swap 16.0 GiB, GLM이 각각 0.13 GiB와 16.0 GiB에 도달한 뒤 kernel OOM으로 종료되어 checkpoint와 restore 값이 없습니다.

Qwen 2-node raw 결과는 `results/trl-ultrachat-fullsft-2node-20260914`, GLM 2-node 실패·성공 raw 결과는 각각 `results/trl-ultrachat-glm-fullsft-2node-20260914`와 `results/trl-ultrachat-glm-fullsft-2node-retry1-20260914`에 있습니다.
1-node 실패 근거는 `results/single-node-io-{qwen,glm}-zero3-full-*-20260914`에 있으며 GLM은 첫 2-node 시도의 stale `zero_stage_3` 파일 누락 실패 후 해당 임시 경로를 비우고 재실행한 결과입니다.
고정 NVMe root를 쓰는 `deepspeed-zero3-nvme.json`은 이전 process의 `zero_stage_3`가 남아 있으면 다음 실행과 충돌할 수 있으므로 동시 실행하지 않고, 비활성 상태를 확인한 뒤 임시 offload 경로를 정리해야 합니다.

## Qwen and GLM Comparison

2026-09-14까지의 반복 측정은 조건이 혼재했으며, 위 2026-09-15 matrix에서 Qwen·GLM checkpoint와 TE memory를 같은 조건으로 다시 측정했습니다.
아래 adapter reload와 TRL 결과는 별도 historical run이고 LoRA 비율 sweep과 recompute는 Qwen 결과입니다.

### Adapter Reload

`experiments/trl/glm-4.7-flash-30b-lora.json`(no_robots, MAX_STEPS=1)으로 처음 실행했습니다.

| Stage | eval_loss | Peak allocated |
| --- | --- | --- |
| base | 1.926144 | 57.699 GiB |
| train | 1.928212 | 57.843 GiB |
| tuned | 1.928212 | 57.699 GiB |

`train`과 `tuned`의 eval_loss가 소수점까지 일치 — Qwen Case 1과 같은 adapter 재로딩 신뢰성 증거입니다.
Qwen의 별도 1-step 실행 수치와는 run·입력 조건이 다르므로 이 표로 모델별 메모리 우열을 판단하지 않습니다.

### Memory Comparison

| 조건 | Qwen CUDA / host | GLM CUDA / host | CUDA 차이 |
| --- | ---: | ---: | ---: |
| `LEN-4096`(Megatron LoRA) | 39.5 / 53.7 GB | 34.54 / 48.65 GB | −12.6% ⚠ |
| `LEN-8192`(Megatron LoRA) | 57.6 / 72.6 GB | 38.94 / 57.30 GB | −32.4% ⚠ |
| `MEM-TRL-DDP` | 59.8 / 69.7 GB | 58.24 / 109.50 GB | −2.6% (host 값은 아래 참고, 모델 신호 아님) |
| `MEM-TRL-Z3-NVME` | 6.4 / 91.8 GB | 7.27 / 85.88 GB | +13.6% |
| `MEM-TRL-FSDP2` | 34.1 / 104.0 GB(통과) | **3/3 실패**(아래) | — |

⚠ 두 `LEN-*` 행은 attention backend가 서로 다릅니다(Qwen `local`, GLM `transformer_engine`). 모델 차이로 읽으면 안 되며, 통제된 비교는 [아래 대조 실험](#sequence-length-기울기)을 따릅니다.

<a id="sequence-length-기울기"></a>

**Sequence length 기울기 차이는 모델이 아니라 attention 구현 때문입니다.**
표면적으로 Qwen은 4096→8192에서 CUDA peak가 +45.8%(39.5→57.6 GB), GLM은 +12.7%(34.54→38.94 GB)만 증가해 아키텍처 차이처럼 보입니다.
하지만 두 측정은 **애초에 같은 attention 경로가 아니었습니다** — `megatron_lab/config.py`의 `select_transformer_impl()`은 `TRANSFORMER_IMPL=auto`를 family별로 다르게 해석합니다.

```python
selected = ("transformer_engine" if requested == "auto" and family == "glm4_moe_lite"
            else "local" if requested == "auto" else requested)
```

즉 Qwen은 `local`, GLM은 `transformer_engine`으로 측정됐습니다(GLM은 MLA provider 제약으로 `local`을 아예 거부합니다).
Qwen만 backend를 바꿔 같은 조건으로 다시 측정한 결과입니다.

| 조건 | 4096 | 8192 | 기울기 |
| --- | ---: | ---: | ---: |
| Qwen + `local`(기존 측정) | 39.5 GB | 57.6 GB | **+45.8%** |
| Qwen + `transformer_engine`(대조 실험) | 36.85 GB | 41.51 GB | **+12.6%** |
| GLM + `transformer_engine` | 34.54 GB | 38.94 GB | **+12.7%** |

**이 두 길이에서 같은 backend를 사용한 관측 증가율은 +12.6%와 +12.7%로 유사합니다.** 다른 길이·batch·recompute 정책에서도 같은 증가율을 보장하지는 않습니다.
따라서 원래 세웠던 "GLM의 MLA가 KV를 압축해 완만하다"는 가설은 **기각됩니다** — 차이를 만든 것은 Megatron `local` 경로가 attention 행렬을 materialize해 sequence length에 제곱으로 증가하는 항을 남기는 반면, Transformer Engine의 fused attention은 그렇지 않다는 점입니다.

이 비교에서 배운 운영상의 교훈은 모델 비교 자체보다 큽니다: `auto`처럼 **입력에 따라 조용히 다른 구현을 고르는 설정은 A/B 비교의 통제 변수를 깨뜨립니다.**
위 memory comparison 표의 `LEN-*` 행도 Qwen은 `local`, GLM은 `transformer_engine` 측정이므로 두 값을 모델 차이로 읽으면 안 됩니다.

TRL DDP의 host memory pressure 차이는 노드 전체 상태에 크게 흔들려 모델 고유 사용량으로 해석할 수 없었습니다.
CPU-only loading에서도 두 모델의 peak RSS가 parameter bytes의 약 1.9배로 유사했으므로 기존의 “GLM만 두 벌을 쓴다”와 shard 원인 가설은 철회합니다.
이 historical 지표는 최신 통제 결과에 사용하지 않습니다.

### FSDP2 Compatibility

설치된 Accelerate 1.14의 GLM FSDP2 load는 persistent buffer를 `DTensor`로 가정해 3회 모두 학습 전에 실패했습니다.
CPU-efficient loading을 끄는 우회는 sharding 전 full replica 이동에서 OOM이 발생해 채택하지 않았습니다.
이는 historical TRL compatibility 기록이며 최신 Megatron matrix의 성공 여부와 무관합니다.

### Checkpoint Comparison

| Variant | Qwen(3회 재계산) | GLM(3회) | 비율 |
| --- | ---: | ---: | ---: |
| Sync run당 save 호출 합 | 2.3697s | 3.1121s | ×1.31 |
| Async run당 save 호출 합 | 1.2824s | 1.5805s | ×1.23 |
| Checkpoint 크기 | 72.8 MB | 150.2 MB | ×2.06 |

두 모델 모두 **async의 run당 save 호출 누적 시간이 더 짧습니다**(sync/async 비율: Qwen 1.85배, GLM 1.97배). Background 저장 완료나 학습 throughput이 같은 배수로 개선됐다는 뜻은 아닙니다.
rMAD는 GLM sync 0.31%, async 0.90%로 10% 기준을 크게 밑돌아 사전에 정한 규칙상 추가 반복을 요구하지 않았습니다.

**Checkpoint 크기 2.06배는 우연이 아니라 LoRA target module 구성 차이입니다.**
`backends/megatron/megatron_lab/config.py`는 family별로 다른 target module을 씁니다.

```python
target_modules = (
    ["linear_q_down_proj", "linear_q_up_proj", "linear_kv_down_proj", "linear_kv_up_proj", "linear_proj"]
    if spec.family == "glm4_moe_lite"
    else ["linear_qkv", "linear_proj"]
)
```

Qwen은 fused QKV 1개 + proj 1개(2종류), GLM은 MLA의 Q/KV down·up projection 4개 + proj 1개(5종류)입니다.
같은 `lora_dim=8`에서 실제 로그도 이를 뒷받침합니다: Qwen `Trainable parameters: 5,111,808`(0.0319%), GLM `Trainable parameters: 10,515,968`(0.07%) — 비율 **2.057배**로 checkpoint 크기 비율(2.063배)과 거의 일치합니다.
[LoRA trainable-ratio 실험](checkpoint-io.md#lora-ratio-and-checkpoint-io)에서 확인한 "checkpoint 크기는 trainable parameter 수에 거의 비례한다"는 관계와 두 모델의 관측값이 부합합니다. 다만 GLM의 여러 LoRA dimension을 sweep한 결과는 아니므로, 새 target module·저장 형식에서의 선형성은 별도 확인이 필요합니다.

### Supported Conclusions

| 관측 | 현재 해석 | 추가로 확인할 것 |
| --- | --- | --- |
| 두 모델의 async save 호출 합이 sync보다 작음 | Host가 save API 안에 머무는 시간 감소 | Finalization 포함 완료 시간, step time, 장기 throughput |
| Qwen LoRA 비율 약 10배 → checkpoint 크기 약 10배 | 고정 target module에서 trainable 수와 저장 크기가 거의 비례 | 다른 target module·checkpoint 형식 |
| GLM은 같은 LoRA dimension에서 checkpoint 약 2.06배 | Target module 구성과 trainable parameter 수가 다름 | 같은 `LORA_DIM`을 동일 학습 비율로 취급하지 않기 |
| TE에서 Qwen·GLM의 길이 증가율이 유사 | 기존 +45.8% 대 +12.7% 차이는 backend가 혼재한 비교 | 더 긴 sequence·batch·recompute별 실측 |
| GLM host pressure가 더 크다는 결론 철회 | 노드 전체 pressure를 모델 고유 사용량으로 해석할 수 없음 | 동일 초기 상태와 process RSS·loading 구간별 계측 |
| GLM FSDP2 실패 | 설치된 로딩 경로의 buffer 호환성과 우회 시 OOM | 수정된 로딩 경로에서 정확성·메모리 재검증 |
| Train/tuned eval loss 일치 | 해당 입력에서 adapter 재로딩을 뒷받침 | 학습 품질·장기 안정성은 별도 평가 |

비교 전에 모델·데이터 revision, tokenizer cohort, 실제 attention backend, optimizer, padding, 저장 주기와 집계 범위를 맞춥니다. `auto`라는 같은 문자열만으로 같은 구현 경로라고 판단하지 않습니다.


## Repeated Megatron Measurements

`experiments/benchmarks.py`는 [benchmark plan](../../experiments/megatron/benchmark-plan.json)의 cell·variant를 읽어 Qwen2.5-0.5B와 No Robots를 고정한 A/B 비교를 수행합니다.
여기서 얻은 시간·메모리 차이를 30B 성능으로 일반화하지 않습니다.

기본 계획은 8개 cell, variant별 4회 측정과 별도 warmup입니다.
길이 비교는 `MAX_LENGTH` 상한만 바꾸지 않고 고정 길이 padding을 씁니다.
30B Qwen full/selective recompute는 위 전용 matrix에서 완결했습니다.
이 절의 0.5B benchmark plan은 smoke 도구 설명으로만 남기며 현재 30B 결과에는 사용하지 않습니다.

```bash
python experiments/benchmarks.py \
  --setup setups/spark/local.json \
  --output results/benchmarks-16x2 \
  --steps 16 --repeats 2 --within-run-warmup 4 \
  --checkpoint-intervals 4 8 \
  --execute
```

이 축소 계획은 warmup 포함 48개 run을 선택하고 첫 4개 within-run step을 측정에서 제외합니다.
Cell 이름의 checkpoint interval 16/32는 유지되지만 실제 interval은 인자의 4/8이므로 run config를 기준으로 해석합니다.
`--plan`으로 계획을, `--timeout`으로 개별 실행 제한을 바꿀 수 있고 실패한 variant의 후속 반복은 생략될 수 있습니다.
`--execute` 없이 같은 인자로 먼저 계획을 확인합니다.
