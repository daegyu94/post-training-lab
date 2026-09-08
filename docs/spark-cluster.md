# Spark-cluster setup2

이 문서는 `spark1`과 `spark2` 두 NVIDIA DGX Spark GB10 노드에서 Megatron Bridge setup2를 bring-up하기 위한 경로입니다. 두 노드의 shared repository는 controller의 `/home/daegyu/shared/post-training-lab`가 Spark 노드에서 `/home/spark/shared/post-training-lab`로 보이는 NFS 경로를 사용합니다.

## Scope and status

Setup2 target은 `Qwen/Qwen3-30B-A3B`와 `zai-org/GLM-4.7-Flash`의 BF16 SFT입니다. `megatron_lab.config`는 local Hugging Face `config.json`의 `model_type`을 읽어 `qwen3_moe`와 `glm4_moe_lite`를 구분하며, GLM 경로에 Qwen model-name 조건을 재사용하지 않습니다. GLM은 Bridge 0.6.0의 `GLM47FlashBridge`를 통해 provider를 만들고, 공통 ConfigContainer sections만 local snapshot에서 구성합니다.

두 30B target은 2026-09-08에 node-local snapshot을 사용하는 two-node EP=2 LoRA 1-step integration으로 model load, native data preparation, forward/backward, optimizer update, validation과 sync DCP save를 검증했습니다. 이는 짧은 bring-up evidence이며 장기 수렴, model quality, fault durability 또는 안정적인 throughput 결과가 아닙니다. 소형 Qwen2.5-0.5B two-node 분산·DCP·feature A/B 경로도 별도 runtime smoke로 검증했습니다.

2026-09-08에 다음 host/network facts를 확인했습니다.

| Check | Result | Boundary |
| --- | --- | --- |
| `spark1`, `spark2` | 각 노드 SSH 접속, GB10 확인 | training evidence가 아님 |
| 각 노드 memory | 총 약 119 GiB, 가용 약 115 GiB | 두 노드가 자동 single memory pool이 아님 |
| native software stack | Torch 2.10.0+cu130, Core 0.19.0, Bridge 0.6.0, TE 2.18.0, ModelOpt 0.46.0 stable import 확인 | TE는 ARM64 source build, ModelOpt는 Bridge 선언 rc1과 다름 |
| RoCE link | `100 Gb/sec (4X EDR)`, data-IP ping 성공 | link configuration 확인이며 GPU/NCCL 결과가 아님 |
| `ib_write_bw` | 1 QP, 64 KiB, RoCEv2 GID 3에서 평균 92.57 Gb/s, exit 0 | application throughput 일반화 금지 |
| NCCL | 두 노드 GPU all-reduce/all-to-all correctness smoke exit 0, RoCE `NET/IB` 경로 | throughput benchmark가 아님; GDR은 지원되지 않음 |

`ib_write_bw` 결과는 RDMA write path가 측정된 한 번의 환경 결과입니다. 100Gbps line-rate 목표의 증거로만 기록하며, NCCL·Megatron throughput 또는 model training speed로 해석하지 않습니다.

NCCL smoke는 `NCCL 2.28.9`, `rocep1s0f0`, RoCEv2 GID 3과 `enp1s0f0np0` data interface를 사용해 두 노드에서 all-reduce와 all-to-all 결과를 검증한 것입니다. 로그의 `GDR 0`은 실패가 아니며, [NVIDIA DGX Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)에 설명된 것처럼 DGX Spark의 GPUDirect RDMA 미지원 환경에서 host staging을 사용하는 결과입니다. 이 smoke는 GPU collective correctness만 입증하고 NCCL bandwidth나 LLM training throughput은 입증하지 않습니다.

추가로 Qwen2.5-0.5B-Instruct를 사용한 2-node BF16 LoRA 2-step integration smoke가 두 rank에서 exit 0으로 완료되었고, 로그는 `NET/IB` 경로와 `torch_dist` checkpoint 저장을 확인했습니다(session `13489`, `71164`). 별도 reload smoke(session `26774`)도 동일한 첫 validation loss `3.152711`을 확인했습니다. 이는 checkpoint/reload 경로의 작은 모델 integration evidence일 뿐이며, Qwen3/GLM 30B weight 또는 30B SFT 성공 증거가 아닙니다.

## GLM loss-boundary gate

GLM과 Qwen3의 generic assistant-mask inference는 사용하지 않습니다. Setup2는 각 model의 native `apply_chat_template`로 마지막 assistant 앞부분을 prompt로 렌더링하고 마지막 응답과 EOS를 completion으로 분리한 뒤, joint token prefix·completion boundary·EOS·non-empty supervision을 검증합니다. 실제 두 tokenizer에 대한 8-row preparation probe는 통과했지만, 이것은 model forward/backward 또는 30B SFT 성공 증거가 아닙니다. Native template이 바뀌거나 검증이 실패하면 training을 중단하며 Qwen용 grammar를 GLM에 덧씌우지 않습니다.

GLM-4.7-Flash의 actual one-layer diagnostic에서는 local MLA layer spec이 `DotProductAttention` 인자 오류로 실패했고, captured Transformer Engine layer spec과 `attention_backend=auto` 경로가 finite forward/backward를 통과했습니다. 따라서 setup2 GLM baseline은 `transformer_impl=transformer_engine`을 요구하며 TE layer spec이 없으면 fail-fast합니다. Qwen2/Qwen3의 local layer-spec baseline은 변경하지 않습니다.

## Verified 30B integration smoke

2026-09-08의 두 run은 `spark1`, `spark2` 각각의 `~/.cache/huggingface` snapshot을 사용했고 `TP=1`, `PP=1`, `EP=2`, `DP=2`, BF16, sequence length 2048, micro/global batch 1/2, attention LoRA, distributed Adam과 sync `torch_dist` DCP로 실행했습니다. 양 rank process는 모두 exit 0이었고 각 checkpoint의 `latest_checkpointed_iteration.txt`는 `1`입니다. Metadata에는 config/index SHA-256, shard/tensor count와 실제 local snapshot path가 남습니다.

| Model and data | Observed step | DCP and memory | Boundary |
| --- | --- | --- | --- |
| Qwen3-30B-A3B + `HuggingFaceH4/no_robots` | 30.52B parameters, 7.49 s, LM loss 4.123573, grad norm 15.432, skipped/nan 0 | save rank range 2795.43 ms, rank 0 max allocated 34.759 GiB | one step; quality/speedup claim 아님 |
| GLM-4.7-Flash + `bigcode/self-oss-instruct-sc2-exec-filter-50k` | 29.41B parameters, 6.01 s, LM loss 0.784571, grad norm 1.950, skipped/nan 0 | save rank range 2601.34 ms, rank 0 max allocated 34.670 GiB | 서로 다른 domain loss를 model 간 비교하지 않음 |

Qwen DCP는 `results/qwen3-30b-no-robots-smoke/checkpoints/iter_0000001`, GLM DCP는 `results/glm-4.7-flash-self-oss-smoke/checkpoints/iter_0000001`에 생성됐습니다. Result directory는 재현을 위한 runtime artifact이며 Git source의 일부가 아닙니다. 이 smoke는 DCP 생성까지 확인했지만 30B checkpoint의 별도-process reload나 async save A/B를 검증한 것은 아니며, 그 효과 실습은 아래의 작은 full-parameter paired run과 구분합니다.

## Software prerequisites

Spark 노드에는 ARM64 호환 Python/CUDA stack, kernel smoke check, Megatron Bridge `0.6.0`, Megatron Core `0.19.0`, Transformer Engine `2.18.0` 조합을 먼저 확인합니다. 양 노드에서 Torch `2.10.0+cu130`, Transformers `5.12.1`, ModelOpt `0.46.0` stable을 import한 것은 확인했으며, Bridge가 선언하는 ModelOpt `0.46.0rc1`과 stable 후보의 차이는 run provenance에 남겨야 합니다. Cluster dependency는 [requirements-spark.txt](../requirements-spark.txt)에 고정합니다.

Docker socket과 passwordless sudo를 전제로 하지 않습니다. 두 30B model snapshot은 NFS에서 읽지 않고 `spark1`, `spark2` 각각의 node-local `~/.cache/huggingface`에 같은 pinned revision 전체를 둡니다. Dataset, checkpoint, log와 summary처럼 두 rank가 공유해야 하는 run artifact만 NFS repository 경로를 사용합니다. 이 문서에는 host-specific certificate override나 credential을 저장하지 않습니다.

사용할 interpreter와 userspace Python headers 경로는 노드별로 환경변수로 지정하고, 공유 repository에서 다음 helper를 source합니다. Helper는 system package 설치나 driver 변경을 수행하지 않으며, venv의 NCCL/cuDNN library path와 CPython extension suffix만 설정합니다.

```bash
export PYTHON=/path/to/arm64-venv/bin/python
export PYTHON_HEADERS="/path/to/extracted-python-dev/usr/include/python3.12:/path/to/extracted-python-dev/usr/include"
source scripts/spark_runtime_env.sh
${PYTHON} -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Transformer Engine `2.18.0`은 ARM64에서 source build한 wheel이며 build 시점에 `NVTE_WITH_NCCL_EP=0`을 사용했습니다. 설치된 Torch tree에 optional `symm_mem/nccl_dev_cap.hpp`가 없어 NCCL EP extension을 빌드하지 않았습니다. 이 환경변수는 build-time 선택이므로 이미 설치된 wheel에 나중에 `NVTE_WITH_NCCL_EP=1`을 export해도 extension이 추가되지 않습니다. Native helper wheel이 재빌드되면 `CPATH`에 extracted `usr/include/python3.12`과 상위 `usr/include`를 모두 추가하고 `MAKEFLAGS`의 `LIBEXT`가 현재 interpreter의 `EXT_SUFFIX`와 일치해야 합니다.

## Model and data preparation

각 model은 두 노드의 기본 HF cache에 준비하고, run마다 model revision과 snapshot manifest를 stage별 `run-metadata-<stage>.json`에 기록합니다. 현재 PoC에서 사용하는 known revisions는 Qwen `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, GLM `7dd20894a642a0aa287e9827cb1a1f7f91386b67`입니다. Launcher는 `HF_HOME`, `XDG_CACHE_HOME`, `~/.cache/huggingface` 순서로 node-local snapshot을 찾고, safetensors index가 참조하는 모든 shard가 없으면 rendezvous 전에 중단합니다. 2026-09-08 검증 기준으로 양 노드에서 Qwen 16 shards와 GLM 48 shards, config/index hash, cache file manifest가 일치하며 NFS model copy는 제거했습니다.

기본 smoke dataset은 `HuggingFaceH4/no_robots`의 `train` split이며 revision `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`를 pin합니다. Preparation은 source test split을 읽지 않고 train에서 deterministic validation holdout을 만들며, 생성한 `training.jsonl`, `validation.jsonl`, `manifest.json`을 함께 보관합니다. UltraChat, Self-OSS와 xLAM 변환은 [public dataset guide](public-datasets.md)를 따릅니다.

```bash
cd /home/spark/shared/post-training-lab
./scripts/prepare_public_data.sh --preset no_robots \
  --output-dir data/public-smoke/no_robots \
  --revision e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b \
  --train-count 8 --eval-count 2 --seed 42
```

이 command는 mutable dataset viewer를 사용하지 않고 pinned Hub revision을 streaming으로 읽습니다. Dataset manifest에는 split, seed `42`, count와 prompt IDs가 남습니다.

## Explicit two-node launch

각 노드에서 같은 command를 실행하되 `NODE_RANK`만 `spark1=0`, `spark2=1`로 지정합니다. `MASTER_ADDR`는 spark1의 rendezvous 주소로 지정하고, `MASTER_PORT`는 양쪽에서 같게 둡니다. 실행 전 각 노드에서 `PYTHON`을 선택하고 위 runtime helper를 source합니다.

```bash
export MASTER_ADDR=<spark1-data-address>
export MASTER_PORT=29500
export NNODES=2
export NPROC_PER_NODE=1
export NODE_RANK=0  # spark2에서는 1
export PYTHON=/path/to/arm64-venv/bin/python
source scripts/spark_runtime_env.sh
export MODEL_ID=Qwen/Qwen3-30B-A3B
export MODEL_REVISION=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export DATASET_ID=HuggingFaceH4/no_robots
export DATASET_REVISION=e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b
export DATA_DIR=data/public-smoke/no_robots
export MAX_LENGTH=2048
./scripts/run_spark_cluster.sh
```

GLM run은 `MODEL_ID`와 `MODEL_REVISION`을 GLM snapshot으로 바꿉니다. `MODEL_DIR`를 생략하면 각 노드의 기본 HF cache에서 pinned snapshot을 해석하며, 비표준 local snapshot을 사용할 때만 명시합니다. Launcher는 `torchrun --nnodes --node-rank --master-addr --master-port`를 사용하며 `--standalone`을 사용하지 않습니다. 각 rank의 stage log는 `results/.../logs/rank-<global-rank>-<stage>.log`에 기록되고, `summary.json`은 global rank 0만 씁니다.

기본 topology는 `TP=1`, `PP=1`, `EP=2`, `expert_tensor_parallel_size=1`입니다. Dense data parallel size는 `world/(TP*PP*CP)`로 계산되어 world size 2에서 DP=2이고, expert data parallel size는 `world/(ETP*EP*PP)`로 별도 기록되어 1입니다. 따라서 sequence 2048, micro batch 1, global batch 8의 gradient accumulation은 DP 기준 4입니다.

Bring-up baseline은 BF16, sequence 2048, micro batch 1, global batch 8, optimizer step 5, attention-only LoRA입니다. 실행 전에 선택된 sample이 sequence limit 안에 있고 assistant supervision token이 남는지 preflight합니다. DeepEP, grouped-GEMM/permutation fusion, shared-expert overlap, CUDA graphs와 fused cross-entropy는 baseline에서 명시적으로 끕니다. Setup2 config는 `gradient_accumulation_fusion=False`와 scheduler `min_lr=0.0`도 명시해 ARM runtime의 unsupported/falsy defaults에 의존하지 않습니다. GLM의 MLA attention LoRA target은 `linear_q_down_proj`, `linear_q_up_proj`, `linear_kv_down_proj`, `linear_kv_up_proj`, `linear_proj`입니다.

`FINETUNING_MODE=full`은 명시적으로 선택할 수 있지만 30B full-parameter Adam은 baseline으로 권장하지 않습니다. BF16 parameter와 standard FP32 master/gradient, Adam state를 합치면 대략 18 bytes/parameter, 30B에서 약 540 GB 수준이고, 두 노드의 약 238 GiB 합계보다 큽니다. SGD 같은 optimizer로 조용히 바꾸지 않으며, full mode의 optimizer와 memory feasibility는 별도 evidence가 필요합니다.

Checkpoint path는 training resume와 adapter evaluation을 구분합니다. `CHECKPOINT_MODE=async`는 train/resume stage에서만 허용되고 persistent worker와 `torch_dist`를 사용합니다. `RESUME_AFTER_TRAIN=true`를 사용하면 optimizer·scheduler·RNG를 포함한 checkpoint load를 별도 resume stage로 실행하며, tuned adapter evaluation 성공을 resume 검증으로 대신하지 않습니다.

## Required evidence order

1. ARM64 stack과 kernel smoke check를 각 노드에서 확인합니다.
2. NCCL over RoCE process-group smoke test를 확인합니다.
3. Pinned model snapshot과 shared canonical dataset manifest를 준비합니다.
4. Base held-out evaluation → 5-step LoRA train → checkpoint/adapter reload → 같은 held-out evaluation 순서로 실행합니다.
5. Rank logs, run metadata, topology, model·dataset revision과 실패 원인을 보관합니다.

현재 저장소의 Megatron implementation은 setup1 Qwen2.5-7B single-GPU 결과를 보존하면서 setup2 경로를 추가했습니다. NCCL prerequisite와 두 30B model의 one-step train/eval/DCP evidence는 확보했으며, 더 긴 repeat와 checkpoint reload 결과가 생기면 이 문서와 [feature labs](megatron-feature-labs.md)의 범위를 함께 갱신합니다.
