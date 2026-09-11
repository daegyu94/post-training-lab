# Lab: 30B NVMe Data Movement

두 Spark 노드의 로컬 NVMe가 30B post-training에 미치는 영향을 관찰합니다.
TRL은 DeepSpeed ZeRO-3 parameter·optimizer state offload를 실행하고, Megatron은 async distributed checkpoint를 로컬 NVMe에 기록합니다.
**Megatron 경로는 학습 state offload가 아니라 checkpoint I/O**라는 차이를 결과 해석에서 유지합니다.

두 backend 모두 canonical JSONL을 로컬 NVMe에 순차 전처리하고 disk-backed Arrow cache에서 batch 단위로 읽습니다.

## Prerequisites

두 노드 모두 `/`가 로컬 NVMe ext4인지 확인하고 `spark` 사용자가 backend별 디렉터리에 쓸 수 있어야 합니다.
각 노드에서 [공통 준비 스크립트](../../setups/spark/README.md#prepare-each-spark-node)를 먼저 실행합니다 — `libaio-dev`, Python headers와 32GiB memlock 설정을 이 스크립트가 준비합니다.
NVMe 경로가 없다면 관리자가 한 번 생성합니다.

```bash
sudo install -d -o spark -g spark -m 700 \
  /mnt/post-training/trl /mnt/post-training/megatron
```

TRL 환경에는 `requirements-spark.txt`에 고정된 `ninja`도 필요합니다(launcher가 `PYTHON` 환경의 `bin`을 `PATH`에 추가하므로 DeepSpeed JIT가 같은 환경의 `ninja`를 씁니다).
그다음 controller에서 경로와 DeepSpeed async I/O를 확인합니다.

```bash
for host in spark1 spark2; do
  ssh "spark@$host" \
    'findmnt -T /mnt/post-training/trl; test -w /mnt/post-training/trl; test -w /mnt/post-training/megatron'
  ssh "spark@$host" \
    '$HOME/.local/ptl/venvs/trl/bin/python -c "from deepspeed.ops.op_builder import AsyncIOBuilder; AsyncIOBuilder().load(verbose=True)"'
done
```

설정 후 기존 SSH 연결을 끊고 다시 접속해 `ulimit -l`이 `33554432` 이상인지 확인합니다.

두 노드에 `Qwen/Qwen3-30B-A3B` revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` snapshot과 No Robots revision `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b` 데이터가 필요하며, `output_root`는 backend별 로컬 NVMe 경로를 씁니다.

```json
"output_root": {
  "trl": "/mnt/post-training/trl",
  "megatron": "/mnt/post-training/megatron"
}
```

## Run

controller의 저장소 루트에서 두 계획을 먼저 확인합니다.

```bash
python experiments/run.py --backend trl \
  --setup setups/spark/local.json \
  --experiment experiments/trl/nvme-offload-30b.json \
  --output results/trl-nvme-30b

python experiments/run.py --backend megatron \
  --setup setups/spark/local.json \
  --experiment experiments/megatron/qwen3-30b-lora.json \
  --output results/megatron-nvme-30b
```

각 rank의 output이 `/mnt/post-training/<backend>/...`인지 확인한 뒤 같은 명령에 `--execute`를 붙여 한 번에 하나씩 실행합니다.

<a id="expected-results-and-verification"></a>

## Expected Results and Verification

**TRL 성공 조건**은 두 rank의 정상 종료, 1 optimizer step, `/mnt/post-training/trl`의 DeepSpeed NVMe read/write 발생입니다.
이 preset은 전체 parameter와 optimizer state를 NVMe training-time memory tier로 쓰는 full SFT입니다.
현재 고정된 TRL·DeepSpeed 조합에서 LoRA와 ZeRO-3 NVMe parameter offload를 함께 쓰는 것은 과거 관측된 실패를 근거로 권장하지 않습니다(원인으로 적어뒀던 reentrant gradient checkpointing 설명은 `spark_train.py`가 `use_reentrant=False`로 고정된 현재 코드와 맞지 않으므로, 제한은 유지하되 원인 설명은 stale로 표시합니다).

`train` stage는 학습 직후 같은 process에서 평가하지 않습니다.
ZeRO-3 parameter coordinator는 forward+backward에서 기록한 실행 trace를 기준으로 NVMe swap buffer 반납 시점을 정하는데, backward가 없는 평가 forward는 이 trace와 어긋나 buffer가 반납되지 않고 소진됩니다(`buffer_count`를 늘려도 소진 시점만 미뤄집니다).
대신 `STAGE=tuned`를 별도 process로 실행합니다 — 새 process는 학습 trace가 없는 새 ZeRO-3 엔진을 만들고 `trainer.save_model()`이 남긴 native ZeRO checkpoint를 `deepspeed_load_checkpoint`로 복원한 뒤 평가합니다.
이 복원은 rank별 partition을 그대로 읽는 저메모리 경로만 쓰고 30B 전체를 한 process에 fp32로 모으는 변환은 쓰지 않습니다.
`tuned`는 checkpoint를 만든 `train`과 동일한 topology로 실행해야 하며, `STAGE=all`이면 같은 환경변수를 재사용하므로 자동으로 맞습니다.

**Megatron 성공 조건**은 두 rank의 정상 종료, 1 optimizer step, async save finalization과 각 노드의 checkpoint shard 생성입니다.
실행 중 `iostat -dx 1 nvme0n1`로 장치 I/O를 관찰합니다.
Arrow dataset은 memory map과 page cache를 쓰므로 같은 batch를 다시 읽을 때 물리 NVMe I/O가 생략될 수 있습니다.

<a id="why-nvme-offload-is-necessary"></a>

## Why NVMe Offload Is Necessary Here

2026-09-09 실행에서 두 노드 모두 `/mnt/post-training/trl/zero_stage_3`가 **약 256GiB**까지 자랐고 같은 시점 각 노드의 물리 RAM은 **119GiB**였습니다.
즉 이 구성(Qwen3-30B-A3B full SFT, ZeRO-3, SGD, world size 2)에서 parameter·optimizer state footprint는 노드 RAM보다 2배 이상 큽니다.
`offload_param`/`offload_optimizer`의 `device: nvme`가 있어야 학습이 성립하며, NVMe offload는 처리량 최적화가 아니라 이 footprint를 RAM만으로 담을 수 없다는 사실 자체가 근거입니다.

이 결론은 측정된 footprint와 RAM 용량을 비교한 것이지 `device: cpu`나 `none`으로 돌려 OOM을 직접 관찰한 결과가 아닙니다 — **간접적이지만 정량적인 근거**이며 그 control 실행은 하지 않았습니다.
또한 이 수치는 이 모델 크기·optimizer·병렬 구성에 한정되며 AdamW 등 다른 optimizer로 일반화하지 않습니다.

### 규모가 커지면: local offload, remote checkpoint

아래는 실측 비교가 아니라 **방향성 논의**입니다.
Local NVMe와 NFS의 throughput·contention 비교는 2노드·30B 규모에서는 결론에 의미가 없다고 보고 보류했습니다.

- **Runtime offload는 node-local이 맞습니다.** 매 step 접근하는 고빈도·latency-sensitive I/O이고 각 rank의 shard는 다른 노드와 공유할 이유가 없습니다. DeepSpeed 자체가 "heavy write traffic ... prefer enterprise/datacenter SSDs"라고 경고할 만큼 무거운 트래픽이라 network filesystem을 얹으면 그대로 병목이 됩니다.
- **Checkpoint는 remote shared filesystem 쪽입니다.** 쓰는 빈도는 훨씬 낮지만 node-local에만 있으면 그 노드가 죽을 때 checkpoint도 사라져 재개할 수 없습니다. 모델이 커지면 checkpoint 크기도 커져 언젠가 node-local 용량(현재 노드당 3.7TB, 30B 학습은 다 합쳐 1TB 이내)을 넘습니다.
- 실제 distributed filesystem 선택(pNFS, 3FS 등)과 성능 검증은 이 저장소의 범위 밖입니다.

<a id="training-data-storage"></a>

### 학습 데이터 저장: 일반 원칙과 이 PoC

**일반적인 대규모 환경의 원칙**은 계층을 나누는 것입니다: object storage가 정본, parallel/distributed filesystem이 학습 시점의 읽기 경로, node-local NVMe는 정본이 아니라 다음 shard를 당겨두는 prefetch cache입니다.
데이터는 노드마다 독립적으로 다시 만드는 게 아니라 **한 곳에서 한 번 만들고 checksum 검증하며 배포**합니다.

**이 PoC는 구성이 다릅니다.**
별도 object storage나 parallel filesystem이 없는 2노드 환경이고 controller는 NFS로 코드 checkout만 내줍니다.
학습에 쓰는 무거운 자원(모델 가중치, 학습 데이터)까지 controller의 NFS를 거치게 하면 그 한 대가 병목이자 단일 장애점이 됩니다.
그래서 **모델 가중치와 학습 데이터를 둘 다 node-local**로 두되, 방법은 "각 노드가 Hub에서 재생성"이 아니라 **한 노드에서 한 번 준비한 뒤 검증된 결과물을 복사**하는 방식입니다([이유](../../docs/datasets.md#public-data)).

이건 "모든 걸 로컬로 복제"가 답이라는 뜻이 아닙니다 — 이 데이터가 32KB 수준이라 복사 비용이 0에 가까울 뿐입니다.
데이터 크기가 전체 복제로 감당이 안 되는 지점부터는 위의 일반 원칙(정본 + PFS hot path + 부분 prefetch)을 적용해야 하며, "한 번 만들고 검증해서 배포한다"는 원칙만 그대로 유지됩니다.

## Cleanup

실행이 끝나고 보존할 로그·요약을 옮긴 뒤 run 디렉터리를 삭제합니다.
DeepSpeed가 남긴 `zero_stage_3` swap 파일은 실행 process가 없는지 확인한 뒤 정리합니다 — 학습이 끝나면 필요 없는 임시 데이터이며 노드당 수백 GB를 차지합니다.

## Limitations

DGX Spark의 CPU와 GPU는 unified memory를 쓰므로 일반적인 discrete GPU의 CPU offload와 같은 의미로 해석하지 않습니다.
Megatron Core의 native training offload 대상은 CPU memory이고 NVMe state offload는 지원하지 않으므로 이 실습의 Megatron 경로는 checkpoint I/O까지만 다룹니다.
Node-local Megatron checkpoint의 cross-node reload와 장애 복구는 완료 조건이 아닙니다.
