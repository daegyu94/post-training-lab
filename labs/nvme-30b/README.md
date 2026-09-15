# 30B NVMe 실습

이 실습은 두 Spark 노드에서 TRL의 DeepSpeed ZeRO-3 runtime offload와 Megatron의 node-local checkpoint save를 확인합니다.
Megatron 경로는 runtime state offload가 아니며, 두 backend 모두 restore를 완료 조건에 포함하지 않습니다.

## 준비

각 노드의 `/mnt/post-training`이 local NVMe이고 `spark` 사용자가 backend별 디렉터리에 쓸 수 있어야 합니다.
공통 준비는 [Spark cluster setup](../../setups/spark/README.md#prepare-each-spark-node)을 따릅니다.

```bash
sudo install -d -o spark -g spark -m 700 \
  /mnt/post-training/trl /mnt/post-training/megatron
```

TRL의 async I/O build와 경로를 확인합니다.

```bash
for host in spark1 spark2; do
  ssh "spark@$host" \
    'findmnt -T /mnt/post-training/trl; test -w /mnt/post-training/trl; test -w /mnt/post-training/megatron'
  ssh "spark@$host" \
    '$HOME/.local/ptl/venvs/trl/bin/python -c "from deepspeed.ops.op_builder import AsyncIOBuilder; AsyncIOBuilder().load(verbose=True)"'
done
```

`setups/spark/local.json`의 backend별 `output_root`는 다음처럼 지정합니다.

```json
"output_root": {
  "trl": "/mnt/post-training/trl",
  "megatron": "/mnt/post-training/megatron"
}
```

각 노드에는 선택한 모델 snapshot과 준비된 UltraChat cohort가 있어야 합니다.

## 실행

먼저 dry-run 계획을 확인합니다.

```bash
python experiments/run.py --backend trl \
  --setup setups/spark/local.json \
  --experiment experiments/trl/nvme-offload-ultrachat-30b.json \
  --output results/trl-qwen-30b

python experiments/run.py --backend megatron \
  --setup setups/spark/local.json \
  --experiment experiments/megatron/qwen3-30b-lora.json \
  --output results/megatron-qwen-30b
```

모든 rank의 output이 `/mnt/post-training/<backend>` 아래인지 확인한 뒤 `--execute`를 추가합니다.
GLM은 각 backend의 `glm` preset으로 바꿉니다.

## 성공 조건

- Controller manifest가 `passed`이고 모든 rank가 정상 종료합니다.
- 계획한 optimizer step을 완료하고 loss가 finite입니다.
- TRL은 parameter·optimizer state를 local NVMe에 offload하고 학습 결과를 저장합니다.
- Megatron은 async save finalization과 각 노드의 local checkpoint shard 생성을 완료합니다.
- Application metric과 resource measurement 파일이 생성됩니다.

Single-node 30B full SFT는 NVMe runtime offload를 사용해도 host memory와 swap을 소진해 checkpoint 전에 OOM이 발생하므로 지원하지 않습니다.
같은 조건은 다시 실행하지 않습니다.

## 제한과 정리

분산 restore·resume은 shared checkpoint storage가 준비된 뒤 검증할 TODO입니다.
현재 node-local checkpoint는 cross-node 복원과 노드 장애 복구를 보장하지 않습니다.

실행이 끝난 뒤 필요한 summary와 metric을 확인하고 run directory를 정리합니다.
DeepSpeed의 `zero_stage_3` swap 파일은 관련 process가 없는지 확인한 뒤 제거합니다.
