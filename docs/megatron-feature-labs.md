# Megatron feature labs

이 문서는 setup2의 model-quality PoC와 분리된 A/B 실험 계획입니다. Feature 효과는 30B MoE LoRA run이 아니라 작은 dense full-parameter model에서 먼저 측정할 수 있으며, 결과 summary에는 `model_scope=small-dense-feature-model`을 남겨 30B evidence와 섞지 않습니다.

현재 harness는 variant를 고정하고 rank-local metadata·timing record를 모으는 scaffold입니다. 0.5B paired run에서 checkpoint/restart correctness의 제한된 evidence는 확보했지만, crash·power-loss durability는 아직 검증하지 않았습니다. raw Megatron rank log parser로 full 0.5B 2-node 6-step logs의 step 3–6 steady median을 계산하면 overlap off `315.2 ms`, on `281.0 ms`였지만 single pair이므로 stable benchmark나 speedup claim으로 보고하지 않습니다. 양 노드의 NCCL collective correctness와 host RoCE `ib_write_bw`는 별도 prerequisite smoke입니다.

## Matrix

| Feature | Variants | Topology or requirement |
| --- | --- | --- |
| Distributed checkpoint save | sync / async | `torch_dist`, fully parallel save/load, optimizer state 저장 |
| Restart correctness | fresh / resumed | optimizer·scheduler·RNG·iteration state load, adapter eval과 별도 |
| Grad-reduce overlap | off / on | EP=1, TP=1, PP=1, dense DP=2; LoRA gradient가 작으면 speedup 없음 가능 |
| Recompute | full / selective | 같은 batch·precision·seed, warmup 제외 repeat |
| Sequence parallel | off / on | 두 variant 모두 TP=2; on만 sequence-parallel 활성화 |
| Expert parallel | EP=1 / EP=2 | MoE model에서만 비교; network/NCCL evidence 선행 |

공개 [Megatron-LM release 목록](https://github.com/NVIDIA/Megatron-LM/releases)과 [Megatron Core 최신 문서](https://docs.nvidia.com/megatron-core/developer-guide/latest/)를 함께 확인합니다. 2026-09-08 기준 최신 공개 release와 이 환경의 설치 stack은 모두 Megatron Core `0.19.0`이지만, release note에 보이는 기능을 현재 Bridge provider·Transformer Engine build와 GB10에서 자동으로 실행 가능하다고 가정하지 않습니다.

| 최신 설계 후보 | 기대 효과와 관측값 | Spark 실습 상태 |
| --- | --- | --- |
| [NVRx async DCP와 `torch_dist` reshard](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/dist_checkpointing.html) | training과 checkpoint I/O의 겹침, topology 변경 load correctness | 현재 legacy `mcore` async와 fully-reshardable DP→TP만 측정함. NVRx dependency를 설치·고정한 별도 A/B 전에는 두 async 전략의 시간을 섞지 않음 |
| [`--overlap-param-gather`](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.distributed.param_and_grad_buffer.html) | distributed optimizer의 parameter all-gather를 forward와 겹침 | 현재 harness는 grad-reduce off/on만 지원함. 작은 full-parameter DP=2에서 동일 bucket 설정을 고정한 다음 후보로 추가 가능 |
| Quantile balancing과 routing analysis | auxiliary-loss-free expert balance, routing concentration·predictability 관측 | 0.19.0 후보. 기존 aux-loss router와 같은 MoE model/data/seed로 expert load histogram과 task metric을 함께 수집해야 하며 2-rank one-step run으로 효과를 주장하지 않음 |
| Fused shared-expert MLP | grouped GEMM·SwiGLU·quantized shared-expert 경로의 kernel overhead 감소 | 0.19.0 후보. 현재 30B provider의 shared-expert 구조와 TE kernel 지원 여부를 먼저 확인해야 함 |
| HybridModel·MLA | 차세대 model composition과 MLA projection 처리 | 0.19.0 후보. 현재 GLM Bridge provider가 기존 GPTModel 경로를 사용하므로 checkpoint conversion·numerical parity 실습을 별도로 설계해야 함 |
| [fine-grained activation offloading](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html) | 선택한 activation을 host로 이동해 peak device allocation 감소 | 최신 MoE 문서의 후보임. GB10 unified-memory 동작과 transfer overhead를 별도 측정하기 전에는 memory saving을 주장하지 않음 |
| CUDA Graph-compatible offloading과 partial graph memory 최적화 | 반복 구간의 launch overhead와 graph-retained buffer 감소 | 0.19.0 후보. dynamic MoE shape와 capture 범위를 먼저 확인하고 peak allocation·step time을 별도 관측해야 함 |
| Streaming quantized checkpoint load | 전체 BF16 scratch allocation 없이 FP8/MXFP8/NVFP4 weight를 순차 dequantize | 0.19.0 후보. quantized source checkpoint가 없으므로 현재 BF16 local snapshot load와 비교할 수 없음 |
| MXFP8 training과 Lion distributed optimizer | Blackwell 저정밀 compute·memory 또는 single-moment optimizer state 효과 | 0.19.0 후보. Transformer Engine capability, optimizer semantics와 numerical baseline을 각각 통과해야 하며 BF16 Adam 결과와 한 번에 비교하지 않음 |
| HybridEP/DeepEP 또는 NCCL EP dispatcher | expert token dispatch 통신 경로 변경 | 현재 Transformer Engine은 `NVTE_WITH_NCCL_EP=0`으로 빌드되어 실행 대상이 아님. 지원 hardware·dependency 조합을 확보한 뒤 별도 실습으로 추가 |

[Parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)의 여러 parallelism 조합도 후보 선택 기준으로 사용합니다. 현재 runnable matrix는 DCP save/restart, grad-reduce overlap, recompute, sequence parallel과 expert parallel이며, 위 표의 나머지 항목은 구현·dependency·smoke evidence가 추가되기 전까지 screening 상태입니다.

Setup2의 `TRANSFORMER_IMPL=auto`는 Qwen을 `local`, GLM-4.7-Flash를 `transformer_engine`으로 선택합니다. `local` backend는 sequence parallel에서 거부되므로 sequence-parallel off/on 두 variant 모두 `TRANSFORMER_IMPL=transformer_engine`을 명시하고 같은 backend를 유지합니다. Qwen의 Transformer Engine 경로는 Bridge의 공식 `transformer_engine_layer_spec`와 `AttnBackend.auto`를 사용하며, GLM에 `local`을 지정하면 fail-fast합니다.

## Small dense feature workload

기본 feature model은 `Qwen/Qwen2.5-0.5B-Instruct` local snapshot이며, model revision을 반드시 pin합니다. Full mode, BF16, TP=1, PP=1, EP=1, world size 2를 사용하면 dense DP=2가 되어 overlap-grad-reduce와 distributed optimizer의 효과를 관찰할 수 있습니다. `expert-parallel` feature만 Qwen3/GLM MoE snapshot으로 바꾸고 EP=1/2를 비교합니다.

```bash
export PYTHON=/path/to/arm64-venv/bin/python
source scripts/spark_runtime_env.sh
export NODE_RANK=0  # spark2에서는 1
export MASTER_ADDR="<spark1-data-address>"
export FEATURE=overlap-grad-reduce
export FEATURE_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
export FEATURE_MODEL_DIR=models/Qwen2.5-0.5B-Instruct
export FEATURE_MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
./scripts/run_feature_lab.sh
```

`PYTHON_HEADERS`가 필요한 native helper 또는 Transformer Engine 재빌드는 extracted userspace Python development headers를 별도로 지정합니다. `scripts/spark_runtime_env.sh`는 system package 설치나 driver 변경을 수행하지 않습니다. 검증된 Transformer Engine은 `2.18.0` ARM64 source build이며 optional NCCL EP는 `NVTE_WITH_NCCL_EP=0`으로 제외했습니다.

`run_feature_lab.sh`는 각 variant에 대해 warmup 1회와 measured repeats를 수행하고, 기본 sequence length 2048 및 동일한 seed·micro/global batch·precision을 전달합니다. 각 run은 실제 `run_spark_cluster.sh`를 거치므로 rank logs와 summary를 남깁니다. `REPEATS=3`은 variant별 세 measured run을 의미하며 warmup은 timing summary에서 제외합니다.

Raw rank log는 within-run warmup을 명시해 다음처럼 파싱합니다. `--feature`, `--variant`, `--run-index`와 log path를 반드시 지정하며, parser의 기본 warmup은 첫 2개 iteration입니다. parser는 iteration ID 누락·중복, non-finite loss/grad/time을 거부하고 raw log SHA-256, step별 loss/grad/skip/nan counts, steady median을 기록합니다.

```bash
python -m megatron_lab.feature_lab \
  --log data/megatron-full-baseline-rank1.log \
  --feature overlap-grad-reduce --variant off --run-index 0 \
  --warmup-steps 2
```

`save-checkpoint` timer는 rank-range milliseconds로 별도 보존하며 steady step median에 섞지 않습니다. 이 timer만으로 checkpoint durability 또는 restart correctness를 주장하지 않습니다.

Checkpoint A/B에서는 `CHECKPOINT_MODE=sync|async`, `FULLY_PARALLEL_SAVE=true`, `FULLY_PARALLEL_LOAD=true`, `SAVE_OPTIMIZER=true`, `LOAD_OPTIMIZER=true`를 유지합니다. Async save는 process 종료 시 pending write가 완료되고 durable checkpoint directory가 존재하는지 확인해야 하며, 파일 일부가 보인다는 사실만으로 성공으로 표시하지 않습니다.

Restart correctness는 train stage와 별도의 `resume` stage를 실행해 resumed iteration, optimizer state, scheduler state와 RNG state가 기대한 값인지 확인합니다. Layout reshard candidate는 `RESUME_TP`, `RESUME_PP`, `RESUME_EP`를 저장 topology와 다르게 지정하고 fully parallel load 결과를 확인합니다. 저장한 adapter를 다시 읽어 held-out evaluation을 통과한 것은 inference/reload evidence일 뿐 optimizer-state resume evidence가 아닙니다.

## Measured DCP/restart evidence

2026-09-08에 `data/megatron-mid-async-rank{0,1}.log`와 `data/megatron-mid-sync-rank{0,1}.log`의 paired run을 확인했습니다. 두 run 모두 Qwen/Qwen2.5-0.5B-Instruct, HuggingFaceH4/no_robots, BF16 full mode, DP=2, global batch 2, micro batch 1, 8 optimizer steps, save interval 4를 사용했고 process exit 0으로 완료했습니다. Async run은 iteration 4에서 save를 schedule한 뒤 iteration 5–8을 계속 수행했으며, iteration 8의 pending-save finalization/backpressure timer는 `60117.55–60117.59 ms`였습니다. Sync run의 save timer는 iteration 4에서 `65191.37–65191.40 ms`, iteration 8에서 `61735.60–61735.66 ms`였습니다. 이 수치는 단일 paired run의 rank-log 관측값이며 async/sync speedup이나 일반적인 I/O 성능을 뜻하지 않습니다.

`data/megatron-resume-async4-rank{0,1}.log`는 async run의 `results/megatron-mid-async/checkpoints/iter_0000004`를 읽어 새 output `results/megatron-resume-from-async4/checkpoints`에 저장했습니다. 로그에는 distributed optimizer와 scheduler state를 포함한 load 성공, iteration 4에서 시작해 5–8을 수행한 사실과 iteration 8 저장이 남아 있습니다. Resume의 step 5–8 LR/loss/grad는 연속 run과 print precision에서 일치했고, `data/resume-tensor-comparison.log`의 사전 선택된 norm/bias tensor 4개도 `mismatch []`였습니다. 이는 선택된 4개 model tensor의 equality만 확인한 것이며, whole model·optimizer state의 bitwise equality나 crash/power-loss durability를 입증하지 않습니다.

동일한 8-step schedule로 새 output에 resume하는 직접 실행 예시는 다음과 같습니다. 두 노드에서 `NODE_RANK`만 0과 1로 바꾸고, 입력 checkpoint 경로는 보존한 채 output 경로를 새 디렉터리로 지정합니다.

```bash
export NNODES=2
export NPROC_PER_NODE=1
export NODE_RANK=0  # spark2에서는 1
export MASTER_ADDR=<spark1-data-address>
export MASTER_PORT=29500
export PYTHON=/path/to/arm64-venv/bin/python
source scripts/spark_runtime_env.sh
"$PYTHON" -m torch.distributed.run \
  --nnodes "${NNODES}" --nproc-per-node "${NPROC_PER_NODE}" \
  --node-rank "${NODE_RANK}" --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  -m megatron_lab.sft --setup spark-cluster --stage resume \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --dataset-id HuggingFaceH4/no_robots \
  --dataset-revision e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b \
  --model-dir models/Qwen2.5-0.5B-Instruct \
  --train-data data/public-smoke/no_robots/training.jsonl \
  --eval-data data/public-smoke/no_robots/validation.jsonl \
  --output-dir results/megatron-resume-from-async4-repro \
  --load-checkpoint results/megatron-mid-async/checkpoints/iter_0000004 \
  --max-steps 8 --schedule-steps 8 --save-interval 8 \
  --max-length 2048 --global-batch-size 2 --micro-batch-size 1 \
  --tp 1 --pp 1 --ep 1 --finetuning-mode full \
  --checkpoint-mode sync
```

이 명령은 checkpoint iteration 4 이후 네 번의 update를 수행하도록 `max-steps 8`을 사용하며, `--load-checkpoint`의 source를 수정하지 않고 새 output에 결과를 씁니다. 위 evidence와 정확히 일치시키려면 model/data snapshot, seed, topology, optimizer와 checkpoint 설정도 동일하게 유지하고, 실제 실행 metadata와 rank log를 함께 보관해야 합니다. 0.5B smoke는 30B Qwen3/GLM SFT 성공이나 model quality 결과를 대신하지 않습니다.

## Measured sequence-parallel candidate

2026-09-08에 `data/megatron-sp-te-off-tp2-rank{0,1}.log`, `data/megatron-sp-te-on-tp2-rank{0,1}.log`와 각각의 `results/sp-te-off-evidence.json`, `results/sp-te-on-evidence.json`을 확인했습니다. 두 paired process run 모두 Qwen/Qwen2.5-0.5B-Instruct full BF16, Transformer Engine 2.18, TP=2, PP=1, EP=1, DP=1, global batch 2, sequence length 2048, seed와 data가 동일한 6-step run이며 process exit 0입니다. Optimizer state는 저장하지 않았고, 첫 2 step을 warmup으로 제외했습니다. 모든 step의 loss와 grad는 finite이고 skipped/nan iteration은 0이었으며, 첫 2 step loss는 일치한 뒤 steady 구간에서 작은 수치 차이가 생겼습니다.

Rank 1의 steady step 3–6 median은 sequence-parallel off `239.85 ms`, on `252.1 ms`로 관측되어 on이 약 5.1% 느렸습니다. 이는 한 paired run의 descriptive result일 뿐 stable regression, speedup 또는 model-quality claim이 아닙니다. Rank 1의 2-step 이후 max allocated memory는 off `4.8374 GiB`, on `4.8175 GiB`로 on이 약 `0.0199 GiB` 낮았지만, 이 작은 차이를 일반적인 memory saving으로 해석하지 않습니다. 두 로그 모두 `CUDA_DEVICE_MAX_CONNECTIONS=1` 권고를 출력했지만 변수는 양쪽에서 동일하게 unset이었습니다. 재현 가능한 비교에는 해당 변수를 명시하고 더 긴 sequence, 여러 warmup 제외 repeat가 필요합니다.

이전에 Qwen local backend로 sequence-parallel on을 시도한 `data/megatron-sp-on-tp2-rank{0,1}.log`는 TransformerLayer의 torch LayerNorm 제약으로 process exit 1을 기록했습니다. 이 실패는 backend 선택을 fail-fast해야 한다는 근거이며 TE off/on paired comparison에 포함하지 않았습니다. 이 section의 evidence도 0.5B smoke이며 30B Qwen3/GLM 실행이나 quality 결과를 뜻하지 않습니다.

현재 default distributed-optimizer checkpoint를 DP=2에서 TP=2로 layout reshard하는 것은 검증되지 않았습니다. `data/megatron-reshard-dp2-to-tp2-rank{0,1}.log`의 시도는 training 전에 양 rank에서 exit 1로 종료되었고, Core 0.19의 optimizer sharding type `dp_reshardable`이 checkpoint의 `(TP, PP)=(1,1)`과 새 `(2,1)` mismatch를 지원하지 않는다고 보고했습니다. `fully_parallel_save=True`만으로 이 optimizer state reshard가 활성화되지는 않으므로, `dist_ckpt_optim_fully_reshardable`을 사용하는 별도의 source checkpoint가 필요합니다. 아래 fully reshardable source/resume 결과는 이 default format의 결과가 아니며, 현재 default 결과를 layout-reshard correctness로 보고하지 않습니다.

`--dist-ckpt-optim-fully-reshardable`은 `fully_parallel_save`와 다른 옵션입니다. 기본값 `false`는 일반적인 `dp_reshardable` optimizer format을 사용하며, 빠른 저장 경로와 TP/PP reshard 불가라는 현재 관측 경계를 보존합니다. `true`는 distributed optimizer를 요구하고, 새 source를 만드는 `train` stage에서는 optimizer state 저장도 요구합니다. 공유 launcher는 항상 `base` 다음 `train`을 실행하므로 이 flag와 `SAVE_OPTIMIZER=false`를 함께 주면 fail-fast합니다. Direct `sft --stage resume`에서는 source optimizer state를 읽기만 하는 실험을 위해 save optimizer를 끌 수 있지만, 실제 reshard 결과가 검증되었다는 뜻은 아닙니다. Fully reshardable format은 일반적으로 더 느리고 저장 비용·시간이 달라질 수 있으므로 sync/async 결과와 섞어 해석하지 않습니다.

다음은 DP=2 source를 두 optimizer step 저장한 뒤 TP=2로 새 output에 resume하는 재현용 명령 쌍입니다. 두 노드에서 `NODE_RANK`만 바꾸고, `TRANSFORMER_IMPL=auto`와 동일한 model/data snapshot을 사용합니다. 이 명령은 별도 축소 재현 예시이며, 아래 measured run의 성공·성능·durability를 자동으로 보장하지 않습니다.

```bash
export NNODES=2
export NPROC_PER_NODE=1
export NODE_RANK=0  # spark2에서는 1
export MASTER_ADDR="<spark1-data-address>"
export MASTER_PORT=29500
export PYTHON=/path/to/arm64-venv/bin/python
source scripts/spark_runtime_env.sh
common=(
  --setup spark-cluster
  --model-id Qwen/Qwen2.5-0.5B-Instruct
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775
  --dataset-id HuggingFaceH4/no_robots
  --dataset-revision e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b
  --model-dir models/Qwen2.5-0.5B-Instruct
  --train-data data/public-smoke/no_robots/training.jsonl
  --eval-data data/public-smoke/no_robots/validation.jsonl
  --max-length 2048 --global-batch-size 2 --micro-batch-size 1
  --tp 1 --pp 1 --ep 1 --finetuning-mode full
  --dist-ckpt-optim-fully-reshardable
  --schedule-steps 2 --save-interval 2
)
"$PYTHON" -m torch.distributed.run --nnodes "$NNODES" \
  --nproc-per-node "$NPROC_PER_NODE" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  -m megatron_lab.sft --stage train "${common[@]}" \
  --max-steps 2 --output-dir results/reshard-source-dp2
"$PYTHON" -m torch.distributed.run --nnodes "$NNODES" \
  --nproc-per-node "$NPROC_PER_NODE" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  -m megatron_lab.sft --stage resume "${common[@]}" \
  --tp 2 --pp 1 --ep 1 --schedule-steps 4 --max-steps 4 \
  --output-dir results/reshard-resume-tp2 \
  --load-checkpoint results/reshard-source-dp2/checkpoints/iter_0000002
```

실제 fully-reshardable evidence는 `data/megatron-fully-reshardable-dp2-source-rank{0,1}.log`와 `data/megatron-fully-reshardable-tp2-resume-rank{0,1}.log`에서 확인했습니다. Qwen/Qwen2.5-0.5B-Instruct full BF16 source는 DP=2, TP=1, PP=1, EP=1에서 iteration 4 checkpoint를 저장했고 양 rank process가 exit 0으로 완료했습니다. Metadata와 rank log의 optimizer sharding type은 `fully_reshardable`, `save_optim=true`였으며 source save timer는 `122480.32–122480.37 ms`였습니다.

같은 source의 `iter_0000004`를 새 output에서 TP=2, DP=1로 resume한 paired process도 양 rank exit 0으로 완료했습니다. Resume log는 fully-reshardable distributed optimizer와 scheduler state를 load했고 load timer는 `63888.07–63888.10 ms`였습니다. `save_optim=false`인 새 output에서 iteration 5–8을 수행했으며, 모든 step이 finite이고 skipped/nan iteration은 0이었고 iteration 8 final save timer는 `10264.65–10264.68 ms`였습니다.

TP/PP topology mismatch 때문에 resume 과정에서 RNG state와 rerun state는 명시적으로 ignored 되었습니다. 따라서 이 결과는 optimizer/scheduler state load와 DP→TP layout reshard 및 후속 training의 제한된 integration evidence일 뿐, bitwise continuity, whole-model equality, optimizer equality, power-loss durability, model quality 또는 30B 실행 성공을 주장하지 않습니다.

## Result interpretation

Feature summary는 raw elapsed seconds, median과 run metadata를 보존하지만 자동 speedup claim을 만들지 않습니다. 작은 workload에서 overlap 또는 async checkpoint가 느려질 수 있고, LoRA는 gradient payload가 작아 overlap 효과가 관찰되지 않을 수 있습니다. 차이가 없거나 결과가 incomplete이면 그대로 기록합니다.

모든 비교는 같은 model snapshot, dataset manifest, data order, seed, batch, precision, git commit과 topology에서 수행합니다. rank별 log를 합친 평균을 단일 throughput으로 만들지 않고, network, checkpoint durability, resumed iteration과 model quality를 서로 다른 validation signal로 보고합니다.

DGX Spark의 GPUDirect RDMA는 지원되지 않으므로 NCCL의 `GDR 0`과 host-staged RoCE는 현재 expected boundary입니다. `ib_write_bw`의 단일 64 KiB host result와 NCCL correctness를 feature speedup 또는 GPU-to-GPU bandwidth로 일반화하지 않습니다.
