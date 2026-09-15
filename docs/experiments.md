# 실험

이 문서는 저장소에서 유지하는 실험과 현재 지원 상태의 단일 기준입니다.
모든 GPU 실험은 `spark1`·`spark2`에서 노드당 process 하나를 사용하며, checkpoint는 각 노드의 local NVMe에 저장합니다.

## 지원 상태

| 대상 | 상태 | 검증 범위 |
| --- | --- | --- |
| TRL DDP LoRA | 지원 | 분산 학습과 node-local 저장 |
| TRL DeepSpeed ZeRO-3 NVMe full SFT | 지원 | 2노드 분산 학습, runtime offload와 node-local 저장 |
| QLoRA | 추정만 제공 | 구현·GPU 검증 없이 [규모별 용량](experiments/scaling-estimates.md)만 비교 |
| Megatron Qwen·GLM LoRA | 지원 | 2노드 EP=2 학습, sync/async node-local 저장 |
| Megatron recompute 비교 | 지원 | Qwen LoRA의 full/selective 비교 |
| Verl agentic GRPO | 정적 검증만 완료 | GPU runtime 검증 전에는 지원으로 간주하지 않음 |
| 분산 restore·resume | TODO | 모든 rank shard가 보이는 shared checkpoint storage가 필요함 |
| Single-node 30B full SFT | 미지원 | NVMe runtime offload를 사용해도 host memory와 swap 소진으로 checkpoint 전에 OOM |

`지원`은 짧은 실행의 성공과 저장 경로를 뜻하며 장기 수렴이나 모델 품질을 보장하지 않습니다.
분산 restore·resume과 NFS checkpoint는 현재 실행·검증 범위에 포함하지 않습니다.

## 실측 결과

| 비교 | 결과 |
| --- | --- |
| Sequence length별 memory | [Qwen·GLM 실측](experiments/measured-results.md#sequence-memory) |
| Full·selective recompute | [Qwen 실측](experiments/measured-results.md#recompute) |
| Sync·async checkpoint | [LoRA rank 8·251의 100-step 실측](experiments/measured-results.md#async-checkpoint-scaling) |
| LoRA 비율별 checkpoint 크기 | [Qwen 실측](experiments/measured-results.md#lora-ratio) |

## 유지하는 preset

| 파일 | 용도 |
| --- | --- |
| `experiments/trl/{single-node-,}smoke.json` | 작은 TRL LoRA 실행 경로 확인 |
| `experiments/megatron/smoke.json` | 작은 Megatron full SFT 실행 경로 확인 |
| `experiments/trl/nvme-offload-ultrachat{,-glm}-30b.json` | Qwen·GLM 30B full SFT와 NVMe runtime offload |
| `experiments/megatron/{qwen3-30b-lora,glm-4.7-flash-30b-lora}.json` | Qwen·GLM 30B LoRA와 local checkpoint |
| `experiments/megatron/checkpoint-memory-30b.json` | 모델을 `--model`로 선택하는 sync/async 반복 측정 |
| `experiments/megatron/{recompute-30b,lora-ratio-checkpoint-io}.json` | 재계산과 LoRA checkpoint 크기 비교 |
| `experiments/megatron/async-checkpoint-scale-30b.json` | 100-step에서 작은·큰 LoRA checkpoint의 sync/async 비교 |

분산 preset은 모두 `STAGE=train`을 사용합니다.
Megatron preset의 `CHECKPOINT_PLACEMENT=local`은 rank마다 `<OUTPUT_DIR>/checkpoints`에 shard를 저장하며 cross-node 복원을 보장하지 않습니다.

## 실행

기본 runner는 dry-run으로 계획만 출력합니다.
계획의 모델·데이터 revision, host, node-local 경로를 확인한 뒤 `--execute`를 추가합니다.

```bash
python experiments/run.py \
  --backend megatron \
  --setup setups/spark/local.json \
  --experiment experiments/megatron/qwen3-30b-lora.json \
  --output results/megatron-qwen-30b
```

Checkpoint 반복 측정은 하나의 plan과 모델 선택 옵션을 사용합니다.

```bash
python experiments/checkpoint_memory_30b.py \
  --setup setups/spark/local.json \
  --model qwen \
  --phase checkpoint \
  --output results/checkpoint-qwen-30b
```

`--model glm`은 GLM preset과 cohort를 선택합니다.
`experiments/build.py`는 개별 knob로 일반 실험을 만들며 기본 stage와 Megatron checkpoint 배치는 각각 `train`, `local`입니다.

## 판정 기준

다음 조건을 모두 만족해야 GPU 실행을 성공으로 판정합니다.

1. Controller manifest의 `status`가 `passed`이고 모든 rank의 exit code가 0입니다.
2. 실제 optimizer step 수가 계획과 같고 loss가 finite입니다.
3. 각 rank의 local checkpoint directory에 shard가 생성됩니다.
4. Async save는 종료 전에 blocking finalization을 완료합니다.
5. Application metric과 resource measurement 파일이 비어 있지 않습니다.

Checkpoint 크기는 rank별 최신 shard의 logical byte 합으로 집계합니다.
Save 시간과 blocking finalization은 따로 기록하며, 둘을 restore 시간이나 storage throughput으로 해석하지 않습니다.

메모리 추정은 실행 전 하한 점검입니다.
CUDA context, allocator 단편화와 framework 임시 buffer를 모두 포함하지 않으므로 `FITS`만으로 실행 가능성을 확정하지 않습니다.

```bash
python experiments/estimate_memory.py \
  --model-dir '<snapshot-dir>' \
  --backend megatron --finetuning-mode full \
  --ep 2 --world-size 2 --budget-gib 119
```

100B 이상 state·checkpoint 계산은 [Scaling Estimates](experiments/scaling-estimates.md)를 따릅니다.
