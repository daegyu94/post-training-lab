# 30B Results

30B Qwen과 GLM의 검증 실행, GPU·메모리 결과와 framework 호환성 비교입니다.

<a id="실제로-확인한-조합"></a>

## Verified Runs

`build.py --execute`로 확인한 조합입니다.

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

여기까지의 모든 반복 측정(checkpoint I/O, memory footprint, LoRA ratio)은 Qwen3-30B-A3B 하나에서만 실행됐습니다.
GLM-4.7-Flash는 [30B GPU Results](#30b-gpu-results)에서 Megatron 1-step 실행 가능성만 확인됐고, TRL은 `spark_config.py`·`spark_train.py`에 `glm4_moe_lite` family와 전용 LoRA target module이 이미 있었지만 **한 번도 실행된 적이 없었습니다.**
2026-09-12에 GLM의 TRL adapter 재로딩, memory matrix와 checkpoint I/O를 확인했습니다. Memory·checkpoint 반복 측정은 조건당 3회이며, 아래 1-step adapter 재로딩 표는 별도 실행입니다. LoRA 비율 sweep은 Qwen 결과이며 GLM에서 반복한 것으로 해석하지 않습니다.

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

**TRL DDP의 host memory pressure 차이는 모델 차이가 아니라 측정 방법의 문제였습니다.**
처음에는 CUDA peak가 거의 같은데(59.8 vs 58.24 GB) host pressure만 GLM이 57% 높다고(109.50 vs 69.7 GB) 기록했습니다.
통제된 재측정에서 이 결론은 **철회됩니다.**

먼저 로딩 자체를 격리해 CPU로만 모델을 올리며 RSS를 0.2초 간격으로 샘플링했습니다(`AutoModelForCausalLM.from_pretrained`, GPU 미사용).

| | 로드된 parameter | peak RSS | parameter 대비 |
| --- | ---: | ---: | ---: |
| Qwen | 56.87 GiB | 111.02 GiB | **×1.95** |
| GLM | 55.77 GiB | 105.47 GiB | **×1.89** |

**이 격리 실험에서 두 모델의 peak RSS는 각각 로드된 parameter bytes의 약 1.9배였습니다.** 따라서 "GLM만 두 벌을 쓴다"는 해석은 성립하지 않습니다.

그다음 Qwen DDP를 오늘 같은 조건으로 다시 돌려 host pressure의 rank별 분포를 봤습니다.

| 조건 | rank별 host pressure | 편차 |
| --- | --- | ---: |
| GLM(3 run × 2 rank) | 99.3 / 103.0 / 104.0 / 117.0 / 117.6 / 118.4 GB | 19.1 GB |
| Qwen(오늘 재측정, 2 rank) | 69.1 / **122.3** GB | **53.2 GB** |

같은 모델·같은 run 안에서도 두 rank가 69 GB와 122 GB로 갈렸고, Qwen의 최대값이 오히려 GLM보다 높습니다.
즉 이 지표는 노드 상태에 크게 흔들려 **모델을 구분하는 근거로 쓸 수 없습니다.**
원래의 57% 격차는 GLM은 rank 최대값, Qwen은 다른 세션에서 측정된 값을 비교한 데서 생긴 것이었습니다.

조사 과정에서 배제한 후보도 함께 남깁니다.

- **모델 크기 아님.** safetensors 헤더 실측으로 GLM 58.2 GiB(~31.2B), Qwen 56.9 GiB(~30.5B)로 오히려 GLM이 2.3% 큽니다. GLM의 `model.safetensors.index.json`은 `total_size`를 실제의 정확히 절반인 29.1 GiB로 기록하므로 **이 필드를 메모리 추정에 쓰면 2배 틀립니다.**
- **CUDA allocator 단편화 아님.** reserved−allocated 격차가 두 모델 모두 정확히 0.54 GiB입니다.
- **shard 분할 아님.** GLM은 레이어별 expert 텐서가 정확히 1개 shard에 모여 있고(median 1, max 1) Qwen은 최대 2개로 오히려 더 흩어져 있습니다.
- **dtype 변환 아님.** 두 모델 모두 파일과 목표 dtype이 BF16으로 같습니다.
- **expert fusion 자체도 아님.** 두 모델 모두 디스크에는 per-expert 텐서로 저장되고 메모리에서는 fused 3D 파라미터(`experts.gate_up_proj`)를 쓰므로, 조립 비용은 양쪽 다 발생합니다.

기존의 “shard 분할 때문에 GLM host memory가 더 크다”는 가설도 유지하지 않습니다. 비교의 전제인 모델별 pressure 격차가 통제된 재측정에서 성립하지 않았기 때문입니다.

### FSDP2 Compatibility

**FSDP2는 정도가 아니라 종류가 다른 실패입니다.** Qwen은 34.1 GB로 통과하지만 GLM은 3회 모두 같은 지점에서 실패했습니다:

```
accelerate/utils/fsdp_utils.py:543, fsdp2_load_full_state_dict()
AttributeError: 'Tensor' object has no attribute 'device_mesh'
```

**원인은 persistent buffer입니다.** 두 모델을 meta device에 올려 accelerate와 같은 순서로 `fully_shard`를 적용한 뒤 `state_dict()`에서 `DTensor`가 아닌 항목을 센 결과입니다.

| 모델 | 감싼 decoder layer | non-DTensor 항목 |
| --- | ---: | --- |
| Qwen3-30B-A3B | 48 | **0개** |
| GLM-4.7-Flash | 47 | **46개** — 전부 `model.layers.N.mlp.gate.e_score_correction_bias` (shape 64) |

`modeling_glm4_moe_lite.py:376`이 MoE 라우터의 expert-score correction bias를 `register_buffer(...)`로 등록하는데 `persistent=False`가 없어 `state_dict()`에 포함됩니다.
FSDP2의 `fully_shard`는 `nn.Parameter`만 샤딩하고 buffer는 평범한 텐서로 남기는데, accelerate(`fsdp_utils.py:543`)는 `state_dict()`의 모든 항목이 `DTensor`라고 가정하고 `.device_mesh`를 읽습니다.
Qwen3 MoE가 통과하는 이유도 같은 지점에서 설명됩니다 — 이 모델이 등록하는 buffer는 `inv_freq` 계열뿐이고 전부 `persistent=False`라 `state_dict()`에 아예 들어가지 않습니다.

즉 이것은 GLM의 결함이 아니라 **persistent buffer를 가진 모델 전반에 적용되는 accelerate FSDP2 경로의 가정 오류**이며, MoE·MLA 여부와는 무관합니다.
`first_k_dense_replace: 1`이라 MoE 레이어가 47개 중 46개인 것과 실패 항목 46개가 정확히 일치합니다.
**우회를 실제로 시도했고, 두 번째 장벽이 나왔습니다.**
문제의 `fsdp2_load_full_state_dict()`는 accelerate에서 `cpu_ram_efficient_loading`이 켜져 있을 때만 호출되므로, 이 플래그를 꺼서 해당 경로를 건너뛰어 봤습니다.
첫 장벽은 실제로 사라져 처음으로 `accelerator.prepare`를 통과하고 가중치 로딩까지 끝냈지만, 곧바로 다음에서 실패했습니다.

```
fsdp2_prepare_model → fully_shard → _move_states_to_device → tensor.to(device)
torch.OutOfMemoryError: 119.69 GiB 중 659 MiB만 남은 상태에서 768 MiB 할당 실패
(해당 process가 73.56 GiB 사용 중)
```

이 플래그를 끄면 **샤딩하기 전에** 각 모듈의 전체 가중치를 device로 올리는데, 이는 `cpu_ram_efficient_loading`이 애초에 막으려던 동작입니다.
즉 버그 하나를 메모리 폭발과 맞바꾸는 셈이라 119 GiB unified memory에서는 쓸 수 없습니다.

정리하면 이 조합에는 **관찰된 실패 경로가 두 개**입니다. 현재 설치 버전과 시도한 설정에서 성공하지 못했으며, 하드웨어 자체의 영구적인 불가능성을 입증한 것은 아닙니다.

| 경로 | 결과 |
| --- | --- |
| 기본값(`cpu_ram_efficient_loading` on) | accelerate의 DTensor 가정이 persistent buffer에서 깨짐 — 3/3 실패 |
| 우회(`cpu_ram_efficient_loading` off) | 샤딩 전 전체 가중치를 device로 이동하다 CUDA OOM |

호환성을 개선하려면 persistent buffer의 값을 rank 간 올바르게 전달하는 로딩 경로가 필요합니다. 단순히 buffer를 `persistent=False`로 바꾸는 우회는 `cpu_ram_efficient_loading` 경로에서 rank 0만 실제 가중치를 읽으므로 다른 rank가 이 라우터 bias를 못 받아 **조용히 다른 routing 결과를 낼 위험**이 있어 채택하지 않았습니다.
이 저장소에서 GLM을 쓸 때는 DDP 또는 DeepSpeed를 사용합니다.

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
Selective recompute는 현재 설정 오류로 warmup에서 실패하므로 full/selective 비교가 완결된다고 기대하지 않습니다.

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
