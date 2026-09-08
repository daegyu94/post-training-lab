# Megatron 개념과 기능 실습 가이드

이 가이드는 Megatron의 구성 요소와 병렬화 개념을 설명하고, Spark 두 노드에서 기능을 실습하는 방법과 기존 측정 결과를 정리합니다.
처음 읽는다면 구성 요소와 CPU 실습부터, 실행 방법이나 결과를 찾는다면 아래 표에서 해당 항목으로 이동하세요.
Spark 환경 준비와 30B 모델 실행 절차는 [Spark 실행 가이드](spark-cluster.md)에서 관리합니다.

문서의 shell 명령은 별도 안내가 없으면 `backends/megatron`에서 실행합니다.
CPU 개념 실습은 controller에서 실행할 수 있으며, 실제 학습 명령은 Spark 노드에서 실행합니다.

## 읽는 순서

| 목적 | 이동할 곳 | 현재 기록의 범위 |
| --- | --- | --- |
| 구성 요소와 병렬화 이해 | [Stack 구성](#stack-overview), [병렬화 용어](#parallelism-terms) | 개념 설명 |
| GPU 없이 개념 확인 | [CPU 실습](#cpu-only-concept-exercise) | 논리적 rank 배치만 확인 |
| 실행할 기능 선택 | [실습 matrix](#matrix) | DCP, restart, overlap, recompute, sequence/expert parallel |
| A/B 실행과 로그 수집 | [작은 dense workload](#small-dense-feature-workload) | 고정 variant와 rank별 metadata·timing 수집 scaffold |
| 저장 후 같은 topology로 재개 | [DCP/restart](#measured-dcprestart-evidence) | 0.5B paired run; 선택된 tensor 4개 equality |
| Sequence parallel 비교 | [Sequence parallel](#measured-sequence-parallel-candidate) | TE off/on 한 쌍; 안정적인 성능 결론 없음 |
| DP=2에서 TP=2로 재개 | [Layout reshard](#layout-reshard) | fully-reshardable source로 제한된 integration 확인 |
| 후속 기능 검토 | [설계 후보](#future-candidates) | dependency·구현·smoke 확인 전 screening 단계 |

## Stack overview

이 저장소의 학습 코드는 Megatron Bridge를 통해 Megatron Core 기반 모델을 사용합니다.
Megatron-LM은 참조 학습 애플리케이션, Core는 모델·병렬화 라이브러리, Bridge는 모델 변환과 학습 설정을 연결하는 계층입니다.
이 저장소에서 스크립트를 읽을 때는 Bridge가 설정을 만들고 Core가 분산 모델을 실행한다고 이해하면 됩니다.

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)은 대규모 Transformer 학습을 위한 참조 애플리케이션입니다. Megatron Core와 실행 스크립트를 함께 제공합니다.
- [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/)는 Transformer 블록과 병렬화 전략을 조합하는 라이브러리입니다.
- [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/)는 Hugging Face 체크포인트와 Megatron 형식 사이의 변환, 모델별 recipe, 학습 진입점을 제공하는 연결 계층입니다. 이 backend는 local Hugging Face snapshot에서 Bridge provider를 만들어 분산 학습 설정을 적용합니다.

## How Setup2 Uses the Stack

~~~text
Local HF Snapshot (0.5B dense / 30B MoE)
        |
        v
Megatron Bridge AutoBridge + ConfigContainer
        |
        v
Megatron Core Distributed Model
        |
        +-------------------+
        |                   |
        v                   v
      spark1              spark2
~~~

Setup2는 두 노드에서 실제 분산 process group을 사용합니다.
작은 dense 모델의 DP=2를 시작점으로 TP=2와 checkpoint 재분할을 실습하고, MoE baseline은 EP=2로 확장합니다.
실행 환경은 [Spark 실행 가이드](spark-cluster.md), 기능별 비교는 아래 [실습 matrix](#matrix)를 참고합니다.
아래 CPU 개념 실습은 논리적 rank 배치만 보여 주며 분산 학습 성능이나 통신 동작을 검증하지 않습니다.

## Parallelism Terms

병렬화는 여러 GPU에 무엇을 나눌지 정하는 방법입니다.
예를 들어 DP는 같은 모델에 서로 다른 데이터를 주고, TP는 한 레이어의 계산 자체를 나눕니다.

| 방식 | 나누는 대상 | 주로 해결하는 문제 |
| --- | --- | --- |
| TP (Tensor Parallelism) | 한 레이어의 텐서 연산 | 한 GPU에 레이어 연산/가중치를 모두 두기 어려운 경우 |
| PP (Pipeline Parallelism) | 모델의 레이어 구간 | 모델 깊이를 여러 stage에 분할 |
| DP (Data Parallelism) | 데이터 배치 | 같은 모델 복제본으로 처리량 확장 |
| CP (Context Parallelism) | 한 샘플의 시퀀스 길이 | 긴 context의 activation 및 attention 메모리 부담 완화 |

CP는 한 시퀀스의 token 구간을 CP rank에 나누어 둡니다.
attention을 계산하려면 다른 구간의 key/value 정보도 필요하므로, 실제 구현에서는 CP group 내부의 통신이 필요합니다.
따라서 CP는 단순히 배치를 나누는 DP와 다르며, 긴 context에 특히 의미가 있습니다.
현재 Setup2 학습 경로는 CP=1을 사용하며, 아래 CPU 실습에서 CP>1의 논리적 배치를 살펴볼 수 있습니다.

일반적인 논리적 관계는 다음과 같습니다.

~~~text
world_size = TP × PP × CP × DP
~~~

실제 가능한 조합과 통신 방식은 모델, sequence length, GPU 메모리, 네트워크에 따라 달라집니다.
[Megatron Core 병렬화 문서](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html)를 실제 설정의 기준으로 삼습니다.

## CPU-only Concept Exercise

다음 명령은 GPU, CUDA, NCCL, torch.distributed를 초기화하지 않습니다.

~~~bash
./scripts/run_megatron_practice.sh
~~~

이 스크립트는 두 작업을 합니다.

1. Qwen2.5-7B LoRA Bridge recipe의 기본 병렬화 값을 참고용으로 출력합니다.
   이는 CPU recipe inspection 예제이며 Setup2 실행 설정이 아닙니다.
   체크포인트도 내려받지 않습니다.
2. TP=2, PP=2, CP=2, DP=2, 총 16개 논리 rank의 group 배치를 JSON으로 출력합니다.

직접 조합을 바꿔 보려면 다음과 같이 실행합니다.

~~~bash
WORLD_SIZE=16 TP_SIZE=2 PP_SIZE=2 CP_SIZE=2 ./scripts/simulate_parallelism.sh
~~~

출력되는 group은 개념 설명용입니다.
실제 멀티 GPU 또는 멀티 노드 실행에는 launcher, process group 초기화, GPU 자원, 네트워크 설정이 별도로 필요합니다.

## Spark 기능 실습

아래 실습은 [Spark 두 노드](spark-cluster.md)에서 한 가지 설정만 바꿔 결과를 비교합니다(A/B 비교).
예를 들어 gradient 통신과 계산을 겹치는 overlap을 끈 실행과 켠 실행의 시간을 비교합니다.
30B MoE SFT integration과 feature 효과 측정은 별도의 실험으로 구분합니다.
기능 효과는 먼저 작은 dense 모델의 full SFT에서 측정합니다.
결과에는 `model_scope=small-dense-feature-model`을 남겨 30B MoE LoRA 실행 기록과 구분합니다.

```text
Setup2 + NCCL / host RoCE prerequisite smoke
                    |
                    v
Select Feature + Pin Model / Data / Config
                    |
                    v
Variant A / B: Warmup Run -> Measured Repeats
                    |
                    v
Rank Logs + Metadata -> Exclude Warmup Steps
                    |
                    v
Compare Timing / Memory / Checkpoint Correctness
                    |
                    v
Record Observations + Validation Limits
```

### Recorded observations

| 실험 | 관측값 | 해석 범위 |
| --- | --- | --- |
| Grad-reduce overlap | 0.5B 2-node 6-step, step 3–6 median: off `315.2 ms`, on `281.0 ms` | raw rank log parser 결과; single pair로 speedup 확정 불가 |
| DCP/restart | async iteration 4에서 저장한 checkpoint로 5–8 재개 | optimizer/scheduler load와 선택된 tensor 비교; crash·power-loss durability 미검증 |
| Sequence parallel | rank 1 median: off `239.85 ms`, on `252.1 ms` | single pair; stable regression이나 speedup 근거 아님 |
| DP→TP reshard | default format 실패, fully-reshardable source 재개 성공 | RNG/rerun state는 보존되지 않음 |

30B MoE integration 성공 여부, feature 성능, checkpoint correctness는 서로 다른 검증 항목입니다.
양 노드의 NCCL collective correctness와 host RoCE `ib_write_bw`도 별도 prerequisite smoke로 취급합니다.

통합 후 반복 측정의 실행 설정과 결과는 [공통 측정 가이드](../../experiments/repeated-measurements.md)와 [통합 검증 기록](../../verification/integration-20260908/README.md)에서 관리합니다.

## Matrix

비교할 기능을 고른 뒤 표의 모델·병렬화 조건을 함께 맞춥니다.
Recompute는 중간 계산값을 다시 계산해 메모리를 줄이는 방식이고, sequence parallel은 시퀀스 방향으로 일부 연산과 데이터를 나누는 방식입니다.
분산 checkpoint(DCP)는 여러 rank의 학습 상태를 저장합니다.

| Feature | Variants | Topology or requirement |
| --- | --- | --- |
| Distributed checkpoint save | sync / async | `torch_dist`, fully parallel save/load, optimizer state 저장 |
| Restart correctness | fresh / resumed | optimizer·scheduler·RNG·iteration state load, adapter eval과 별도 |
| Grad-reduce overlap | off / on | EP=1, TP=1, PP=1, dense DP=2; LoRA gradient가 작으면 speedup 없음 가능 |
| Recompute | full / selective | full은 실행 확인; selective는 [현재 설정 검증 실패](../../verification/integration-20260908/README.md#recompute-limitation), 성능 비교 미완료 |
| Sequence parallel | off / on | 두 variant 모두 TP=2; on만 sequence-parallel 활성화 |
| Expert parallel | EP=1 / EP=2 | MoE model에서만 비교; network/NCCL evidence 선행 |

Setup2의 `TRANSFORMER_IMPL=auto`는 Qwen을 `local`, GLM-4.7-Flash를 `transformer_engine`으로 선택합니다.
`local` backend는 sequence parallel에서 거부되므로 sequence-parallel off/on 두 variant 모두 `TRANSFORMER_IMPL=transformer_engine`을 명시하고 같은 backend를 유지합니다.
Qwen의 Transformer Engine 경로는 Bridge의 공식 `transformer_engine_layer_spec`와 `AttnBackend.auto`를 사용하며, GLM에 `local`을 지정하면 fail-fast합니다.

## Small dense feature workload

기본 feature model은 `Qwen/Qwen2.5-0.5B-Instruct` local snapshot이며, model revision을 반드시 pin합니다.
Full mode, BF16, TP=1, PP=1, EP=1, world size 2를 사용하면 dense DP=2가 되어 overlap-grad-reduce와 distributed optimizer의 효과를 관찰할 수 있습니다.
`expert-parallel` feature만 Qwen3/GLM MoE snapshot으로 바꾸고 EP=1/2를 비교합니다.

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

`PYTHON_HEADERS`가 필요한 native helper 또는 Transformer Engine 재빌드는 extracted userspace Python development headers를 별도로 지정합니다.
`scripts/spark_runtime_env.sh`는 system package 설치나 driver 변경을 수행하지 않습니다.
검증된 Transformer Engine은 `2.18.0` ARM64 source build이며 optional NCCL EP는 `NVTE_WITH_NCCL_EP=0`으로 제외했습니다.

### A/B runs and timing

처음 실행할 때의 초기화 비용을 비교에서 빼기 위해 준비 실행(warmup)을 따로 수행합니다.
측정 실행 안에서도 첫 몇 step을 제외하고 나머지 구간의 시간을 비교합니다.

`run_feature_lab.sh`는 각 variant에 대해 warmup 1회와 measured repeats를 수행하고, 기본 sequence length 2048 및 동일한 seed·micro/global batch·precision을 전달합니다.
각 run은 실제 `run_spark_cluster.sh`를 거치므로 rank logs와 summary를 남깁니다.
`REPEATS=3`은 variant별 세 measured run을 의미하며 warmup은 timing summary에서 제외합니다.

Raw rank log는 within-run warmup을 명시해 다음처럼 파싱합니다.
`--feature`, `--variant`, `--run-index`와 log path를 반드시 지정하며, parser의 기본 warmup은 첫 2개 iteration입니다.
parser는 iteration ID 누락·중복, non-finite loss/grad/time을 거부하고 raw log SHA-256, step별 loss/grad/skip/nan counts, steady median을 기록합니다.

```bash
python -m megatron_lab.feature_lab \
  --log data/megatron-full-baseline-rank1.log \
  --feature overlap-grad-reduce --variant off --run-index 0 \
  --warmup-steps 2
```

`save-checkpoint` timer는 rank-range milliseconds로 별도 보존하며 steady step median에 섞지 않습니다.
이 timer만으로 checkpoint durability 또는 restart correctness를 주장하지 않습니다.

### Checkpoint and restart checks

Checkpoint A/B에서는 `CHECKPOINT_MODE=sync|async`, `FULLY_PARALLEL_SAVE=true`, `FULLY_PARALLEL_LOAD=true`, `SAVE_OPTIMIZER=true`, `LOAD_OPTIMIZER=true`를 유지합니다.
Async save는 process 종료 시 pending write가 완료되고 durable checkpoint directory가 존재하는지 확인해야 하며, 파일 일부가 보인다는 사실만으로 성공으로 표시하지 않습니다.

Restart correctness는 train stage와 별도의 `resume` stage를 실행해 resumed iteration, optimizer state, scheduler state와 RNG state가 기대한 값인지 확인합니다.
Layout reshard candidate는 `RESUME_TP`, `RESUME_PP`, `RESUME_EP`를 저장 topology와 다르게 지정하고 fully parallel load 결과를 확인합니다.
저장한 adapter를 다시 읽어 held-out evaluation을 통과한 것은 inference/reload evidence일 뿐 optimizer-state resume evidence가 아닙니다.

## Measured DCP/restart evidence

### Conditions and results

2026-09-08의 paired run 기록입니다.

| 공통 조건 | 값 |
| --- | --- |
| Model / data | Qwen/Qwen2.5-0.5B-Instruct / HuggingFaceH4/no_robots |
| Training | BF16 full mode, DP=2, global batch 2, micro batch 1 |
| Schedule | 8 optimizer steps, save interval 4 |
| Completion | sync/async 모두 process exit 0 |

| 경로 | 관측값 |
| --- | --- |
| Async save | iteration 4 save schedule 후 5–8 계속 실행; iteration 8 pending-save finalization/backpressure `60117.55–60117.59 ms` |
| Sync save | iteration 4 `65191.37–65191.40 ms`; iteration 8 `61735.60–61735.66 ms` |
| Resume | async iteration 4 checkpoint에서 optimizer/scheduler load, step 5–8 실행 후 iteration 8 저장 |
| Continuity check | step 5–8 LR/loss/grad가 연속 run과 print precision에서 일치; 사전 선택한 norm/bias tensor 4개 `mismatch []` |

단일 paired run이므로 async/sync speedup이나 일반적인 I/O 성능을 뜻하지 않습니다.
Tensor 비교는 선택된 model tensor 4개에 한정되며, whole model·optimizer state의 bitwise equality 또는 crash/power-loss durability를 입증하지 않습니다.

### Evidence files

| 기록 | 경로 |
| --- | --- |
| Async / sync rank logs | `data/megatron-mid-async-rank{0,1}.log`, `data/megatron-mid-sync-rank{0,1}.log` |
| Resume rank logs | `data/megatron-resume-async4-rank{0,1}.log` |
| Source checkpoint | `results/megatron-mid-async/checkpoints/iter_0000004` |
| Resume output | `results/megatron-resume-from-async4/checkpoints` |
| Tensor comparison | `data/resume-tensor-comparison.log` |

### Reproduce same-topology resume

동일한 8-step schedule로 새 output에 resume하는 직접 실행 예시는 다음과 같습니다.
두 노드에서 `NODE_RANK`만 0과 1로 바꾸고, 입력 checkpoint 경로는 보존한 채 output 경로를 새 디렉터리로 지정합니다.

```bash
export NNODES=2
export NPROC_PER_NODE=1
export NODE_RANK=0  # spark2에서는 1
export MASTER_ADDR="<spark1-data-address>"
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

이 명령은 checkpoint iteration 4 이후 네 번의 update를 수행하도록 `max-steps 8`을 사용하며, `--load-checkpoint`의 source를 수정하지 않고 새 output에 결과를 씁니다.
위 evidence와 정확히 일치시키려면 model/data snapshot, seed, topology, optimizer와 checkpoint 설정도 동일하게 유지하고, 실제 실행 metadata와 rank log를 함께 보관해야 합니다.
0.5B smoke는 30B Qwen3/GLM SFT 성공이나 model quality 결과를 대신하지 않습니다.

## Measured sequence-parallel candidate

2026-09-08에 `data/megatron-sp-te-off-tp2-rank{0,1}.log`, `data/megatron-sp-te-on-tp2-rank{0,1}.log`와 각각의 `results/sp-te-off-evidence.json`, `results/sp-te-on-evidence.json`을 확인했습니다.
두 paired process run 모두 Qwen/Qwen2.5-0.5B-Instruct full BF16, Transformer Engine 2.18, TP=2, PP=1, EP=1, DP=1, global batch 2, sequence length 2048, seed와 data가 동일한 6-step run이며 process exit 0입니다.
Optimizer state는 저장하지 않았고, 첫 2 step을 warmup으로 제외했습니다.
모든 step의 loss와 grad는 finite이고 skipped/nan iteration은 0이었으며, 첫 2 step loss는 일치한 뒤 steady 구간에서 작은 수치 차이가 생겼습니다.

| Rank 1 관측값 | SP off | SP on | 단일 pair 차이 |
| --- | --- | --- | --- |
| Step 3–6 steady median | `239.85 ms` | `252.1 ms` | on이 약 5.1% 느림 |
| 2-step 이후 max allocated memory | `4.8374 GiB` | `4.8175 GiB` | on이 약 `0.0199 GiB` 낮음 |

한 paired run의 descriptive result이며 stable regression, speedup, 일반적인 memory saving 또는 model-quality claim이 아닙니다.
두 로그 모두 `CUDA_DEVICE_MAX_CONNECTIONS=1` 권고를 출력했지만 변수는 양쪽에서 동일하게 unset이었습니다.
재현 가능한 비교에는 해당 변수를 명시하고 더 긴 sequence, 여러 warmup 제외 repeat가 필요합니다.

이전에 Qwen local backend로 sequence-parallel on을 시도한 `data/megatron-sp-on-tp2-rank{0,1}.log`는 TransformerLayer의 torch LayerNorm 제약으로 process exit 1을 기록했습니다.
이 실패는 backend 선택을 fail-fast해야 한다는 근거이며 TE off/on paired comparison에 포함하지 않았습니다.
이 section의 evidence도 0.5B smoke이며 30B Qwen3/GLM 실행이나 quality 결과를 뜻하지 않습니다.

## Layout reshard

Layout reshard는 checkpoint를 저장할 때와 다른 병렬 배치로 학습 상태를 다시 나누는 작업입니다.
여기서는 DP=2로 저장한 상태를 TP=2로 읽는 경우를 확인합니다.
Optimizer 저장 형식에 따라 가능 여부가 달라집니다.

| Source optimizer format | DP=2 → TP=2 결과 | 경계 |
| --- | --- | --- |
| Default `dp_reshardable` | training 전 양 rank exit 1 | TP/PP 변경을 지원하지 않음; fully parallel save만으로 해결되지 않음 |
| `fully_reshardable` | optimizer/scheduler load 후 step 5–8 실행, 양 rank exit 0 | RNG/rerun state ignored; bitwise continuity 미검증 |

### Format requirements

현재 default distributed-optimizer checkpoint를 DP=2에서 TP=2로 layout reshard하는 것은 검증되지 않았습니다.
`data/megatron-reshard-dp2-to-tp2-rank{0,1}.log`의 시도는 training 전에 양 rank에서 exit 1로 종료되었고, Core 0.19의 optimizer sharding type `dp_reshardable`이 checkpoint의 `(TP, PP)=(1,1)`과 새 `(2,1)` mismatch를 지원하지 않는다고 보고했습니다.
`fully_parallel_save=True`만으로 이 optimizer state reshard가 활성화되지는 않으므로, `dist_ckpt_optim_fully_reshardable`을 사용하는 별도의 source checkpoint가 필요합니다.
아래 fully reshardable source/resume 결과는 이 default format의 결과가 아니며, 현재 default 결과를 layout-reshard correctness로 보고하지 않습니다.

`--dist-ckpt-optim-fully-reshardable`은 `fully_parallel_save`와 다른 옵션입니다.
기본값 `false`는 일반적인 `dp_reshardable` optimizer format을 사용하며, 빠른 저장 경로와 TP/PP reshard 불가라는 현재 관측 경계를 보존합니다.

`true`는 distributed optimizer를 요구하고, 새 source를 만드는 `train` stage에서는 optimizer state 저장도 요구합니다.
공유 launcher는 항상 `base` 다음 `train`을 실행하므로 이 flag와 `SAVE_OPTIMIZER=false`를 함께 주면 fail-fast합니다.

Direct `sft --stage resume`에서는 source optimizer state를 읽기만 하는 실험을 위해 save optimizer를 끌 수 있지만, 실제 reshard 결과가 검증되었다는 뜻은 아닙니다.
Fully reshardable format은 일반적으로 더 느리고 저장 비용·시간이 달라질 수 있으므로 sync/async 결과와 섞어 해석하지 않습니다.

### Reproduce DP-to-TP resume

다음은 DP=2 source를 두 optimizer step 저장한 뒤 TP=2로 새 output에 resume하는 재현용 명령 쌍입니다.
두 노드에서 `NODE_RANK`만 바꾸고, `TRANSFORMER_IMPL=auto`와 동일한 model/data snapshot을 사용합니다.
이 명령은 별도 축소 재현 예시이며, 아래 measured run의 성공·성능·durability를 자동으로 보장하지 않습니다.

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

### Measured fully-reshardable evidence

실제 fully-reshardable evidence는 `data/megatron-fully-reshardable-dp2-source-rank{0,1}.log`와 `data/megatron-fully-reshardable-tp2-resume-rank{0,1}.log`에서 확인했습니다.
Qwen/Qwen2.5-0.5B-Instruct full BF16 source는 DP=2, TP=1, PP=1, EP=1에서 iteration 4 checkpoint를 저장했고 양 rank process가 exit 0으로 완료했습니다.
Metadata와 rank log의 optimizer sharding type은 `fully_reshardable`, `save_optim=true`였으며 source save timer는 `122480.32–122480.37 ms`였습니다.

같은 source의 `iter_0000004`를 새 output에서 TP=2, DP=1로 resume한 paired process도 양 rank exit 0으로 완료했습니다.
Resume log는 fully-reshardable distributed optimizer와 scheduler state를 load했고 load timer는 `63888.07–63888.10 ms`였습니다.
`save_optim=false`인 새 output에서 iteration 5–8을 수행했으며, 모든 step이 finite이고 skipped/nan iteration은 0이었고 iteration 8 final save timer는 `10264.65–10264.68 ms`였습니다.

TP/PP topology mismatch 때문에 resume 과정에서 RNG state와 rerun state는 명시적으로 ignored 되었습니다.
따라서 이 결과는 optimizer/scheduler state load와 DP→TP layout reshard 및 후속 training의 제한된 integration evidence일 뿐, bitwise continuity, whole-model equality, optimizer equality, power-loss durability, model quality 또는 30B 실행 성공을 주장하지 않습니다.

## Result interpretation

Feature summary는 raw elapsed seconds, median과 run metadata를 보존하지만 자동 speedup claim을 만들지 않습니다.
작은 workload에서 overlap 또는 async checkpoint가 느려질 수 있고, LoRA는 gradient payload가 작아 overlap 효과가 관찰되지 않을 수 있습니다.
차이가 없거나 결과가 incomplete이면 그대로 기록합니다.

모든 비교는 같은 model snapshot, dataset manifest, data order, seed, batch, precision, git commit과 topology에서 수행합니다.
rank별 log를 합친 평균을 단일 throughput으로 만들지 않고, network, checkpoint durability, resumed iteration과 model quality를 서로 다른 validation signal로 보고합니다.

DGX Spark의 GPUDirect RDMA는 지원되지 않으므로 NCCL의 `GDR 0`과 host-staged RoCE는 현재 expected boundary입니다.
`ib_write_bw`의 단일 64 KiB host result와 NCCL correctness를 feature speedup 또는 GPU-to-GPU bandwidth로 일반화하지 않습니다.

## Future candidates

아래 항목은 후속 실습 후보이며 현재 실행 가능한 기능 목록이 아닙니다.
후보를 추가하려면 설치된 라이브러리, 모델 구조와 GPU 지원 여부를 확인하고 작은 실행으로 먼저 검증해야 합니다.

공개 [Megatron-LM release 목록](https://github.com/NVIDIA/Megatron-LM/releases)과 [Megatron Core 최신 문서](https://docs.nvidia.com/megatron-core/developer-guide/latest/)를 함께 확인합니다.
이 환경의 설치 stack은 Megatron Core `0.19.0`이며, 공개 release와 최신 문서의 version은 검증 시점에 각각 확인합니다.
release note에 보이는 기능을 현재 Bridge provider·Transformer Engine build와 GB10에서 자동으로 실행 가능하다고 가정하지 않습니다.

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

[Parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)의 여러 parallelism 조합도 후보 선택 기준으로 사용합니다.
현재 실행 경로는 DCP save/restart, grad-reduce overlap, full recompute, sequence parallel과 expert parallel이며, 위 표의 나머지 항목은 구현·dependency·smoke evidence가 추가되기 전까지 screening 상태입니다.
