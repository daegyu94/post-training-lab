# TRL Spark 30B path

이 문서는 `spark1`, `spark2` DGX Spark GB10 두 노드에서 `Qwen/Qwen3-30B-A3B` 또는 `zai-org/GLM-4.7-Flash` local snapshot을 native BF16으로 준비하는 TRL 경로입니다. 기존 single-GPU Qwen2.5-14B NF4 QLoRA `sft_lab.train`은 변경하지 않습니다.

현재 구현은 같은 `SFTTrainer` 진입점에서 DDP, FSDP2, DeepSpeed ZeRO-2/3를 선택합니다. Transformers 5.12.1, TRL 1.12.0, Accelerate 1.14.0와 Torch 2.10.0+cu130 조합에서 소형 모델의 local, two-node DDP와 two-node FSDP2 경로를 실제 검증했습니다. Qwen3-30B-A3B와 GLM-4.7-Flash도 node-local snapshot을 사용한 two-node DDP LoRA 1-step integration을 통과했지만, 짧은 smoke를 장기 수렴·품질·성능 결과로 해석하지 않습니다. DeepSpeed는 configuration/launcher unit test만 통과한 구현 가정이고 30B FSDP2/DeepSpeed는 아직 미검증입니다.

## Runtime boundary

두 Spark 노드의 GB10과 약 119 GiB physical memory, RoCE link와 NCCL allreduce/alltoall smoke는 별도로 확인했지만 GPUDirect RDMA는 GB10 platform limitation이라 사용할 수 없고, model training throughput이나 convergence evidence는 아닙니다. NVIDIA의 [DGX Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)는 host-pinned `cudaHostAlloc` buffer와 `ib_reg_mr` registration을 권장하며, 검증된 NCCL NET/IB host-staging 경로는 이 제한과 일치합니다. driver나 configuration으로 GDR을 enable하려는 조치는 필요하지 않습니다. 각 노드의 memory는 자동 single pool이 아니며 native BF16 full 30B는 parameter 약 60 GB와 full gradient 약 60 GB를 사용합니다. AdamW가 BF16 moment 두 개를 만들면 optimizer state는 약 120 GB(4 bytes/parameter)이고 세 성분 합은 약 240 GB입니다. 이는 activation, temporary buffer, CUDA allocator reserve, 분산 shard를 제외한 단순 global byte estimate이며 fit claim이 아닙니다. FP32 AdamW moments라면 state는 약 240 GB입니다.

Full mode는 `--optimizer sgd`(momentum 0, optimizer state 0 byte) 또는 `--optimizer adamw`를 명시적으로 선택합니다. SGD의 BF16 update numerics와 AdamW state dtype/peak physical memory는 실제 run metadata에서 확인해야 하며, 어떠한 optimizer도 full 30B가 맞는다고 보장하지 않습니다. DDP는 model replica를 rank마다 유지하고, FSDP2와 DeepSpeed ZeRO는 model/gradient/optimizer state 중 해당 stage가 담당하는 상태를 shard합니다. 이 차이는 설정상의 의도이며 아직 Spark 30B fit evidence가 아닙니다.

## Data and model prerequisites

두 노드에는 같은 model revision의 완전한 snapshot을 각각 local `~/.cache/huggingface/hub`에 둡니다. Model weight를 NFS에 두지 않으므로 한 노드의 cache만 채워서는 안 됩니다. `HF_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME`을 설정했다면 Hugging Face의 같은 우선순위로 cache root를 해석합니다. Launcher는 `MODEL_DIR`을 생략하면 model ID와 immutable revision으로 현재 노드의 snapshot을 찾아 사용합니다. 다른 local cache를 의도적으로 사용할 때만 `MODEL_DIR`을 지정합니다.

Model snapshot의 `config.json`은 `qwen3_moe` 또는 `glm4_moe_lite`여야 하며 command의 model ID와 family가 불일치하면 중단합니다. Model revision과 dataset revision은 모두 immutable 40-hex SHA여야 하고, public adapter가 만든 `manifest.json`의 dataset ID/revision과 일치해야 합니다. Model preflight는 revision 이름의 snapshot directory, safetensors index, index가 열거한 모든 non-empty shard와 `.incomplete` 부재를 model load 전에 각 노드에서 확인합니다. Manifest에 split SHA-256 또는 count가 있으면 local JSONL bytes와 line count를 검증합니다. Summary에는 local `config.json`과 weight index hash, indexed tensor/shard count, snapshot directory revision을 기록합니다. Canonical data preparation은 [public dataset guide](public-datasets.md)를 참고하세요.

각 노드에서 동일하게 내려받는 예시는 다음과 같습니다. `hf download`가 출력하는 snapshot path는 node-local 경로이며 NFS에 복사할 필요가 없습니다.

```bash
hf download Qwen/Qwen3-30B-A3B --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
hf download zai-org/GLM-4.7-Flash --revision 7dd20894a642a0aa287e9827cb1a1f7f91386b67
```

```bash
./scripts/prepare_public_data.sh --preset ultrachat --output-dir data/setup2-ultrachat --revision 8049631c405ae6576f93f445c6b8166f76f5505a --train-count 128 --eval-count 16 --seed 42
```

Canonical row는 원본 messages를 보존하지만 training collator는 마지막 assistant를 제외한 native prompt를 `apply_chat_template(add_generation_prompt=True, enable_thinking=False)`로 렌더링하고 final assistant content와 tokenizer EOS를 completion으로 붙입니다. Qwen template을 GLM에 덧씌우지 않으며, prompt token IDs가 prompt+completion token IDs의 정확한 prefix이고 EOS가 마지막 supervised token인지 확인합니다. completion-only loss의 supervised token이 0이거나 `max_length`를 넘으면 조용히 truncate하지 않고 fail-fast합니다. Unknown role과 final assistant tool/function call도 fail-fast합니다.

LoRA는 Qwen의 실제 `q_proj/k_proj/v_proj/o_proj` leaf modules와 GLM MLA의 `q_a_proj/q_b_proj/kv_a_proj_with_mqa/kv_b_proj/o_proj` 중 snapshot에 실제 존재하는 attention modules만 선택합니다. Module discovery가 비어 있으면 model-name을 추측하지 않고 중단합니다.

## Explicit two-node launch

각 노드에서 같은 command를 실행하고 `NODE_RANK`만 `spark1=0`, `spark2=1`로 바꿉니다. `PYTHON`은 NFS 밖 native venv도 가리킬 수 있고, launcher는 `torchrun`의 explicit rendezvous를 사용하며 `--standalone`을 사용하지 않습니다.

분산 계층은 `torchrun -> TRL SFTTrainer -> Transformers Trainer -> Accelerate -> DDP/FSDP2/DeepSpeed`입니다. `torchrun`은 rank와 rendezvous 환경을 만들고, 실제 model wrapping과 gradient synchronization/sharding은 Trainer가 생성한 Accelerate backend가 담당합니다. TRL의 [distributed training guide](https://huggingface.co/docs/trl/distributing_training), Transformers의 [FSDP2 guide](https://huggingface.co/docs/transformers/en/fsdp)와 [DeepSpeed guide](https://huggingface.co/docs/transformers/en/deepspeed)를 기준으로 구성했습니다.

한 GPU local smoke는 `NNODES=1 NPROC_PER_NODE=1 NODE_RANK=0`과 `MASTER_ADDR=127.0.0.1`로 같은 launcher를 사용할 수 있고, 직접 `python -m sft_lab.spark_train`을 호출해도 됩니다. Qwen2.5 small feature model은 tokenizer/data/pipeline smoke에만 사용 가능하며 30B target evidence를 대체하지 않습니다. 이는 multi-node evidence도 대체하지 않습니다.

```bash
export MASTER_ADDR=<spark1-data-address> MASTER_PORT=29500
export NNODES=2 NPROC_PER_NODE=1 NODE_RANK=0  # spark2에서는 1
export MODEL_ID=Qwen/Qwen3-30B-A3B
export MODEL_REVISION=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export DATASET_ID=HuggingFaceH4/no_robots DATASET_REVISION=e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b
export DATA_DIR=/home/spark/shared/post-training-lab/data/public-smoke/no_robots
export MAX_LENGTH=2048 PYTHON=/home/spark/ptl-envs/trl/bin/python
./scripts/run_spark_cluster.sh
```

GLM run은 `MODEL_ID=zai-org/GLM-4.7-Flash`와 revision `7dd20894a642a0aa287e9827cb1a1f7f91386b67`로 바꿉니다. Coding example은 `DATASET_ID=bigcode/self-oss-instruct-sc2-exec-filter-50k`, revision `356bb069eee815daa6e23e9a282eeefe1490ad44`와 준비한 Self-OSS directory를 사용합니다. 각 노드의 default Hugging Face cache에서 같은 model revision을 자동으로 찾습니다. 현재 collator는 tokenizer별 native prompt와 마지막 assistant completion을 직접 구성하고, prompt-prefix·EOS·supervised-token 조건을 model load 전에 검증합니다. 이 preflight만으로 training 성공을 주장하지 않고 아래 실제 one-step evidence와 구분합니다.

DDP의 기본 `STAGE=all`은 `base`, `train`, `tuned`를 각각 새 process로 실행합니다. 개별 `STAGE=base|train|tuned`도 지원합니다. `base`는 held-out eval, `train`은 LoRA adapter 또는 full model 저장, `tuned`는 별도 process에서 저장물을 읽어 eval하며, adapter reload 성공을 full optimizer resume 증거로 사용하지 않습니다. FSDP2와 DeepSpeed는 검증하지 않은 sharded export/reload를 성공한 것처럼 취급하지 않도록 현재 `STAGE=base|train`만 허용합니다. Rank logs는 `RANK_LOG_DIR` 아래에 남고 summary는 rank 0만 기록합니다.

```bash
STAGE=base ./scripts/run_spark_cluster.sh
STAGE=train MAX_STEPS=5 FINETUNING_MODE=lora ./scripts/run_spark_cluster.sh
STAGE=tuned ALLOW_EXISTING_OUTPUT=true ./scripts/run_spark_cluster.sh
```

## DDP, FSDP2, and DeepSpeed exercises

`DISTRIBUTED_BACKEND`은 다음 세 경로 중 하나를 선택합니다. 비교 run은 model revision, dataset manifest, selected rows, precision, per-device batch, gradient accumulation, optimizer와 step 수를 같게 유지하고 backend별 output directory를 분리합니다.

| Backend | Configuration | State and boundary |
| --- | --- | --- |
| `ddp` | Trainer/Accelerate DDP, NCCL | 소형 모델 DP=2 actual smoke 완료; 각 rank에 model replica 유지 |
| `fsdp2` | `fsdp=True`, config `version=2`, full reshard, transformer auto-wrap, CPU-RAM-efficient load, sharded state dict | 소형 full-parameter 2-node 1-step Spark smoke 완료; 30B runtime·export/reload 미검증 |
| `deepspeed` | `SFTConfig.deepspeed`에 ZeRO-2/3 JSON 전달 | 구현·unit test 완료; DeepSpeed 0.19.6 설치 및 Spark GPU runtime 미검증 |

DDP 기준 실습은 별도 backend 설정 없이 실행합니다.

```bash
DISTRIBUTED_BACKEND=ddp STAGE=train \
  OUTPUT_DIR=results/trl-ddp ./scripts/run_spark_cluster.sh
```

Backend 자체의 차이를 LoRA에 의존하지 않고 비교하려면 작은 local causal-LM snapshot으로 세 run 모두 `FINETUNING_MODE=full`과 동일 optimizer를 사용합니다. 30B DDP full training은 각 rank가 model replica를 유지하므로 memory fit을 가정하지 않으며, 먼저 작은 모델로 execution path를 확인한 뒤 FSDP2/ZeRO의 30B 가능성을 별도 측정합니다.

FSDP2는 Transformers 5.12.1의 `fsdp_config["version"]=2` API를 사용합니다. `SFTConfig`를 model load 전에 생성해 rank 0만 pretrained checkpoint를 읽는 초기화가 적용될 수 있게 하고, model을 수동으로 GPU에 옮기지 않습니다. `activation_checkpointing=true`를 사용하므로 중복 all-gather를 유발할 수 있는 Trainer gradient checkpointing은 이 backend에서 끕니다.

```bash
DISTRIBUTED_BACKEND=fsdp2 FINETUNING_MODE=full OPTIMIZER=sgd STAGE=train \
  OUTPUT_DIR=results/trl-fsdp2 ./scripts/run_spark_cluster.sh
```

DeepSpeed는 제공된 JSON에서 batch size, gradient accumulation과 BF16 값을 `auto`로 두어 `SFTConfig`와 어긋나는 이중 설정을 방지합니다. ZeRO-2는 optimizer state와 gradient를, ZeRO-3는 parameter까지 shard하는 비교 후보입니다.

```bash
DISTRIBUTED_BACKEND=deepspeed \
  DEEPSPEED_CONFIG=configs/deepspeed-zero2.json \
  FINETUNING_MODE=full OPTIMIZER=sgd STAGE=train OUTPUT_DIR=results/trl-zero2 \
  ./scripts/run_spark_cluster.sh

DISTRIBUTED_BACKEND=deepspeed \
  DEEPSPEED_CONFIG=configs/deepspeed-zero3.json \
  FINETUNING_MODE=full OPTIMIZER=sgd STAGE=train OUTPUT_DIR=results/trl-zero3 \
  ./scripts/run_spark_cluster.sh
```

FSDP2는 `SHARDED_STATE_DICT`를 사용하고 ZeRO-3 profile은 `stage3_gather_16bit_weights_on_model_save=false`이므로, full-parameter 결과를 일반 Hugging Face directory로 즉시 재로딩할 수 있다고 가정하지 않습니다. 30B에서 full state gather는 peak memory를 다시 키울 수 있습니다. 우선 `STAGE=train`으로 sharded checkpoint와 rank별 정상 종료를 확인하고, adapter/full checkpoint export와 별도-process reload는 backend별 실제 smoke 후 검증된 절차를 추가해야 합니다.

Sharded backend에서는 단순 parameter slice 비교가 collective gather를 유발하거나 local shard만 관측할 수 있어 현재 summary가 직접 delta를 기록하지 않습니다. 대신 finite loss와 실제 optimizer step을 기록하며 `parameter_update_evidence`를 `optimizer steps only`로 명시합니다. `memory_components_bytes`는 wrapping 전 logical model view이며 rank-local shard allocation이 아니고, 실제 device 관측치는 peak CUDA memory를 따릅니다. 이 evidence는 DDP의 sampled nonzero parameter delta보다 약하며 backend-aware checksum 검증을 추가할 대상입니다.

실행 후에는 finite loss, actual optimizer steps, trained parameter count/fraction, 실제 선택된 prompt IDs, completion supervised-token totals, model·dataset revision, checkpoint path와 peak physical GPU memory를 함께 보관합니다. Summary는 실제 parameter/gradient dtype별 byte count와 initialized optimizer state bytes를 구분해 기록합니다. Train stage는 non-finite loss 또는 관측한 trainable parameter update가 0이면 summary를 `training_result_verified: false`로 기록하고 실패합니다. 서로 다른 dataset의 held-out loss를 순위화하지 않고, 공통 held-out suite와 domain별 natural-language/code/JSON/tool metrics를 별도로 비교합니다.

## Verified integration smoke

2026-09-08에 두 30B target의 native BF16 LoRA, two-node DDP, sequence length 2048, per-device batch 1, gradient accumulation 1, 1 optimizer-step run을 완료했습니다. Qwen은 `HuggingFaceH4/no_robots`, GLM은 `bigcode/self-oss-instruct-sc2-exec-filter-50k`의 canonical subset을 사용했고 양 rank가 모두 exit 0이었습니다. 각 summary는 local snapshot config/index SHA-256, shard count, selected prompt IDs, supervised-token count, optimizer state dtype, finite loss와 sampled nonzero parameter update를 기록합니다.

| Model and data | Observed train/eval | Rank 0 peak CUDA | Boundary |
| --- | --- | --- | --- |
| Qwen3-30B-A3B + No Robots | 30,538,807,296 parameters, train loss 4.099994, grad norm 5.445, eval loss 2.590571 | allocated 58.825 GiB, reserved 58.971 GiB | DDP LoRA one step; quality/throughput claim 아님 |
| GLM-4.7-Flash + Self-OSS | 29,953,906,944 parameters, train loss 0.778345, grad norm 0.9788, eval loss 0.922483 | allocated 57.844 GiB, reserved 58.186 GiB | 다른 domain의 loss를 model 간 비교하지 않음 |

Runtime evidence는 각각 `results/trl-qwen3-30b-ddp-no-robots-smoke/summary-train.json`과 `results/trl-glm-4.7-flash-ddp-self-oss-smoke/summary-train.json`에 생성됐고 adapter도 같은 run directory에 저장됐습니다. Result directory는 Git source가 아닌 실행 artifact입니다. 이 결과는 DDP에서 각 rank가 전체 BF16 model replica를 유지하면서 LoRA optimizer state만 작게 유지한 구성이고, full-parameter 30B 또는 FSDP2/DeepSpeed fit evidence가 아닙니다.

30B 실행 전에 동일 launcher의 분산 동작을 `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`와 No Robots 4 train/2 validation row로 검증했습니다. `spark1`, `spark2`에서 각각 한 process를 실행한 DP=2 LoRA AdamW 2-step run이며, NCCL log는 `NET/IB` RoCE 경로를 사용했습니다. 두 rank가 모두 정상 종료했고 rank 0 summary는 `world_size=2`, `actual_optimizer_steps=2`, finite train/eval loss와 nonzero sampled parameter update를 기록했습니다.

동일 held-out data에서 train 직후와 별도 process로 adapter를 다시 읽은 tuned evaluation loss는 모두 `2.243760585784912`였습니다. Base evaluation loss `2.2450199127197266`과의 작은 차이는 짧은 integration smoke의 품질 향상 근거로 사용하지 않습니다. 실행 증거는 `results/trl-two-node-smoke-a/summary-{base,train,tuned}.json`과 `data/trl-two-node-smoke-a-rank{0,1}.log`이며, 이 소형 모델 결과는 두 30B target의 메모리 적합성·성능·학습 성공을 입증하지 않습니다.

같은 소형 model과 No Robots 4 train/2 validation row로 full-parameter FSDP2, SGD, sequence length 2048, 1-step smoke도 양 rank exit 0으로 완료했습니다. NCCL은 RoCE `NET/IB`와 GDR 0 경로를 사용했고 train loss `2.828125`, grad norm `23.29`, eval loss `2.24609375`가 finite였으며 rank 0 peak CUDA allocated/reserved는 약 `3.01/5.23 GiB`였습니다. `results/trl-fsdp2-small-b/checkpoints/checkpoint-1`에는 두 rank의 `pytorch_model_fsdp_0` distcp shard와 RNG·scheduler state가 남았고 process group cleanup도 완료됐습니다. Summary의 wrapping 전 logical parameter view는 FP32였으며 sampled parameter checksum을 수집하지 않았으므로, 이 결과는 FSDP2 small integration과 sharded-save evidence일 뿐 nonzero parameter equality, reload correctness, 30B fit·training 또는 성능 우위를 입증하지 않습니다.
