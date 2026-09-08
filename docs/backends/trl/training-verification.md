# Spark Training Verification — 2026-09-08

The measurements and raw logs below are historical verification from the TRL branch.
These measurements predate the layout migration.
For the separate GPU checks after migration, see the [integration verification record](../../verification/integration-20260908/README.md).

이 기록은 `trl_lab`으로 패키지 이름을 변경한 뒤 Spark 두 노드에서 실행한 검증입니다.
기준 source commit은 `867ef3d`이며 이번 실행의 training logic은 해당 commit과 같습니다.
이름 변경 후 `spark1`과 최종 `spark2` TRL 환경의 `python -m pytest -q`는 각각 49 passed였습니다.
Python compileall, launcher의 `bash -n`, `git diff --check`도 통과했습니다.
과거 DDP LoRA 결과는 [기존 실행 기록](spark-cluster.md#verified-integration-smoke)에 별도로 보존합니다.

개별 노드의 BF16 full SFT(SGD)는 이번에 검증한 두 조합 모두 첫 optimizer step을 완료하지 못했습니다.
`spark1`의 Qwen3-30B는 메모리 allocation 오류가 관측된 뒤 20분 제한으로 종료됐고, `spark2`의 GLM-4.7-Flash는 첫 backward에서 CUDA OOM으로 실패했습니다.
두 모델을 각 노드에 교차 배치한 네 조합을 모두 시험한 것은 아닙니다.
현재 검증 구성에서는 30B full SFT를 사용 가능한 경로로 확인하지 못했으며, 두 모델 모두 two-node DDP LoRA는 1 step·저장·평가를 완료했습니다.
이 결과는 다른 optimizer·offload·sequence length 등 모든 가능한 구성의 불가능을 뜻하지 않습니다.

## Conditions

아래 결과는 같은 데이터와 짧은 학습 조건에서 backend·학습 모드를 바꿔 확인한 기록입니다.
모델 크기뿐 아니라 초기화 방식과 optimizer에 따라 메모리 사용량과 실패 지점이 달라집니다.

각 노드에서 Torch 2.10.0+cu130, Transformers 5.12.1, TRL 1.12.0, Accelerate 1.14.0을 사용했습니다.
DeepSpeed 0.19.6은 양 노드의 기존 TRL 환경에 `DS_BUILD_OPS=0 python -m pip install deepspeed==0.19.6`으로 설치했습니다.
각 run은 별도 output directory를 사용합니다.

Two-node training은 GPU당 한 process, native BF16 configuration, full-parameter SGD 또는 LoRA AdamW, learning rate `2e-5`, max length 2048, micro batch 1, gradient accumulation 1, optimizer step 1, seed 42로 실행합니다.
BF16 configuration만으로 FSDP2의 parameter storage나 DeepSpeed의 master weight dtype까지 BF16이라고 해석하지 않습니다.

Training pool은 4 rows이며 1 step의 effective global batch는 two-node에서 2, single-node에서 1입니다.
No Robots revision `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`의 동일한 canonical 4 train / 2 validation rows를 사용합니다.
모델 revision과 선택한 prompt IDs는 각 summary에 기록됩니다.

표의 CUDA allocated는 PyTorch가 실제 할당한 메모리, reserved는 재사용을 위해 확보한 메모리입니다.
둘 다 노드 전체 사용량과는 다르므로 아래 시스템 메모리 관측값과 구분해 읽으세요.

## Results

| Model / backend | Result | Train loss | Eval loss | Rank 0 peak CUDA allocated / reserved |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B / ZeRO-2 | 양 rank exit 1; optimizer 초기화 전 NVML 오류 | — | — | summary 생성 전 실패 |
| Qwen2.5-0.5B / ZeRO-3 | 양 rank exit 0; 1 optimizer step, checkpoint 저장, evaluation | 2.828331709 | 2.243357897 | 4.367538 / 7.298828 GiB |
| Qwen3-30B / DDP LoRA | 양 rank exit 0; 1 step, nonzero sampled update, adapter 저장 | 3.156562805 | 2.574227333 | 58.824724 / 58.970703 GiB |
| GLM-4.7-Flash / DDP LoRA | 양 rank exit 0; 1 step, nonzero sampled update, adapter 저장 | 2.949144363 | 1.931783199 | 57.843072 / 58.185547 GiB |
| Qwen3-30B / FSDP2 LoRA | 양 rank exit 0; 1 step, sharded model·optimizer checkpoint 저장 | 3.156250000 | 2.570312500 | 32.147320 / 34.187500 GiB |
| GLM-4.7-Flash / FSDP2 LoRA | state-dict load에서 `Tensor.device_mesh` AttributeError; step 전 실패 | — | — | summary 생성 전 실패 |
| Qwen3-30B / ZeRO-3 LoRA | model 초기화 중 rank 0 kernel OOM kill; optimizer step 없음 | — | — | summary 생성 전 실패 |
| GLM-4.7-Flash / ZeRO-3 LoRA | 첫 optimizer step 완료 전 rank 0 kernel OOM kill | — | — | summary 생성 전 실패 |
| Qwen3-30B / spark1 개별 full SGD | 첫 step 미완료; 1200초 제한 종료, exit 124; driver 메모리 allocation 오류 관측 | — | — | summary 생성 전 종료 |
| GLM-4.7-Flash / spark2 개별 full SGD | 첫 backward에서 CUDA OOM, exit 1 | — | — | 오류 시 PyTorch allocated 103.03 GiB; peak summary 없음 |
| Qwen3-30B / full FSDP2 | 최초 실행과 GPU 단독 재실행 모두 rank 1 kernel OOM kill | — | — | summary 생성 전 실패 |

Qwen3의 spark1 개별 full SFT는 첫 step이 `0/1`인 상태에서 1200초 제한에 도달해 exit 124로 종료됐습니다.
종료 처리까지 포함한 controller 관측 시간은 1251.23초였습니다.

실행 중 17:02:21~17:04:33 KST에 NVIDIA driver의 `Out of memory [NV_ERR_NO_MEMORY]` allocation 오류가 기록됐고, 관측 최대 system used memory는 119.693 GiB였습니다.
이는 `free -b` 관측값이며 CUDA allocator peak가 아닙니다.

이 run의 kernel OOM kill은 확인되지 않았으므로 메모리 오류를 동반한 시간 초과로 분류합니다.
종료 후 양 노드에 CUDA compute process가 남아 있지 않음을 확인했습니다.

GLM의 spark2 개별 full SFT는 BF16·SGD 구성에서도 첫 `loss.backward()`에서 CUDA OOM으로 실패했습니다.
오류 당시 PyTorch allocated는 103.03 GiB, free memory는 621.67 MiB였으며 추가 768 MiB allocation을 확보하지 못했습니다.
이 수치는 오류 시점의 allocator 보고이며 성공한 run의 peak summary와 구분합니다.

GLM ZeRO-3 LoRA는 model 초기화 후 training loop에 진입했지만 첫 optimizer step을 완료하기 전 16:59:00 KST에 rank 0 PID 1214575가 kernel OOM kill로 종료됐습니다.
따라서 두 30B target의 ZeRO-3 실패 시점은 서로 다릅니다.

Qwen3 ZeRO-3 LoRA는 16:51:28 KST에 model 초기화 중 rank 0 PID 1212398이 kernel OOM kill로 종료됐습니다.
LoRA는 base weight를 없애지 않으며, 이 실행에서는 학습 step에 도달하기 전 메모리가 부족했습니다.

GLM FSDP2 LoRA는 Accelerate의 `fsdp2_load_full_state_dict()`에서 plain Tensor의 `device_mesh` 접근으로 실패했습니다.
이는 이번 실행에서 관측한 sharded state-dict 호환성 오류이며 OOM으로 분류하지 않습니다.
Rank 1이 exit 1로 실패한 뒤 rank 0 launcher를 정리해 exit 143이 기록됐고, rank 0 로그에도 같은 AttributeError가 있습니다.

ZeRO-2는 `DeepSpeed Stage1And2ZeroOptimizer -> accelerator.available_memory() -> nvmlDeviceGetMemoryInfo()`에서 `pynvml.NVMLError_NotSupported`로 실패했습니다.
이 실행은 0.5B에서도 실패했으므로 모델 크기만으로 설명되는 OOM이 아닙니다.
소형 모델의 ZeRO-3는 같은 dependency와 SGD 조건에서 실행을 완료했습니다.

ZeRO-3 summary의 `total_parameter_count`와 `trainable_parameter_count`는 partition placeholder에 `numel()`을 적용해 0으로 기록되는 기존 계측 문제이므로 논리적 model size로 사용하지 않습니다.
`optimizer_state_actual_bytes=0`도 SGD momentum state에 대한 관측이며 DeepSpeed FP32 master parameter나 전체 engine allocation이 0이라는 뜻은 아닙니다.

Sharded backend의 성공 판정은 finite loss, optimizer step, 저장과 정상 종료 범위이며, parameter checksum·optimizer resume·장기 수렴 검증을 대체하지 않습니다.

Qwen3 full FSDP2 격리 재실행도 16:18:18 KST에 rank 1 kernel OOM kill로 실패했습니다.
시작 시 양 노드에 CUDA process가 없었고, 관측된 CUDA PID는 각 노드의 학습 process 하나뿐이었습니다.
Rank 0/1의 관측 최대 system used memory는 116.950/119.093 GiB, swap은 9.547/16.000 GiB였습니다.
이는 CUDA allocator peak가 아닌 `free -b`의 시스템 관측값입니다.

Rank 1에서는 메모리 압박 중 관측 간격이 최대 196초로 늘어났으므로 연속적인 process 부재 증명은 아닙니다.

설치된 Accelerate 1.14.0의 `fsdp2_prepare_model`은 `fully_shard` 전에 trainable parameter를 FP32로 upcast하고 full state dictionary를 보유합니다.
현재 구현·설정에서 full 30B 초기화가 완료되지 않았다는 결과이며 모든 가능한 sharding·offload 구현에 대한 불가능 판정은 아닙니다.

첫 Qwen3 FSDP2 run에서 `spark2` 커널은 16:02:06 KST에 학습 PID 430934의 OOM kill을 기록했습니다.
학습 process의 anon RSS는 106,146,352 KiB였고, 다른 Python/torchrun process 두 개의 RSS 합계는 약 656 MiB였습니다.
따라서 이 첫 실행을 다른 작업이 없었던 독립 실험으로 분류하지 않습니다.

## FSDP2 Small Checkpoint Export

이 검사는 분산 저장한 가중치를 하나의 모델로 내보낸 뒤 다시 읽을 수 있는지 확인합니다.
Optimizer 상태까지 읽어 학습을 이어가는 resume 검사와는 목적이 다릅니다.

기존 `results/trl-fsdp2-small-b/checkpoints/checkpoint-1/pytorch_model_fsdp_0`를 `accelerate merge-weights`로 새 output directory에 export했습니다.
원본 checkpoint는 수정하지 않았습니다.
원본 model config를 export directory에 복사하고, export된 tensor와 base snapshot에서 각 차원의 처음 최대 4개 원소를 CPU에서 비교했습니다.
290 tensor 중 185 tensor의 표본에서 nonzero delta를 관측했고 최대 absolute delta는 `9.406358003616333e-08`이었습니다.

이는 export된 FP32 가중치의 표본 변화이며 전체 tensor equality나 재학습 resume 검증은 아닙니다.

Export 후 새 two-node DDP process에서 full model을 BF16으로 읽어 평가했고 양 rank exit 0, eval loss `2.2439277172088623`을 기록했습니다.
원래 FSDP2 eval loss `2.24609375`와 차이는 `-0.0021660327911376953`입니다.

Backend와 loading dtype 경로가 달라졌으며, export·reload 실행 성공을 동일 backend 수치 일치나 optimizer resume 성공으로 해석하지 않습니다.

## Reproduce the Training Commands

[Setup 2 가이드](spark-cluster.md)의 node-local model cache와 canonical data를 준비한 뒤, 각 Spark 노드의 TRL worktree에서 실행합니다.
다음은 Qwen3 two-node DDP LoRA command이며 `spark2`에서는 `NODE_RANK=1`로 바꿉니다.

```bash
export PYTHON=/home/spark/ptl-envs/trl/bin/python
export MASTER_ADDR=12.201.48.11 MASTER_PORT=29641 NNODES=2 NPROC_PER_NODE=1 NODE_RANK=0
export MODEL_ID=Qwen/Qwen3-30B-A3B MODEL_REVISION=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export DATASET_ID=HuggingFaceH4/no_robots DATASET_REVISION=e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b
export DATA_DIR=/home/spark/shared/post-training-lab/data/public-smoke/no_robots
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0 NCCL_DEBUG=INFO
STAGE=train DISTRIBUTED_BACKEND=ddp FINETUNING_MODE=lora OPTIMIZER=adamw \
  MAX_STEPS=1 MAX_LENGTH=2048 GRADIENT_ACCUMULATION_STEPS=1 TRAIN_SAMPLES=4 EVAL_SAMPLES=2 \
  LEARNING_RATE=2e-5 SEED=42 OUTPUT_DIR=/home/spark/trl-verification-results/trl-verification-20260908-qwen30b-ddp-lora \
  ./scripts/run_spark_cluster.sh
```

GLM은 model ID를 `zai-org/GLM-4.7-Flash`, revision을 `7dd20894a642a0aa287e9827cb1a1f7f91386b67`로 바꿉니다.
FSDP2는 `DISTRIBUTED_BACKEND=fsdp2`, ZeRO-3는 `DISTRIBUTED_BACKEND=deepspeed DEEPSPEED_CONFIG=configs/deepspeed-zero3.json`을 사용합니다.
Full 비교는 `FINETUNING_MODE=full OPTIMIZER=sgd`이며 single-node 비교에서는 `NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1`을 사용합니다.

각 configuration은 결과 디렉터리 이름을 바꿔 독립 실행합니다.
30B LoRA 비교 run의 output은 node-local directory이므로 rank별 checkpoint shard도 해당 노드에 있습니다.
재사용하려면 필요한 rank 파일을 모두 수집해야 하며 한 노드의 directory만으로 complete distributed checkpoint라고 가정하지 않습니다.

## Evidence

[실행 상태](verification-20260908/execution.json), [ZeRO-3 summary](verification-20260908/small-zero3-summary.json), [첫 30B FSDP2 OOM 기록](verification-20260908/qwen30b-fsdp2-oom.txt)을 함께 보존합니다.
[Qwen3 개별 full SFT driver 기록](verification-20260908/qwen30b-single-full-kernel.txt)은 해당 실행 시간대의 로그만 포함합니다.
Rank logs는 같은 evidence directory의 `<run>-rank{0,1}.txt`에 있습니다.
모델 weight와 대형 checkpoint는 Git에 포함하지 않습니다.
