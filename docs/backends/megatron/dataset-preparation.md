# Megatron Dataset Preparation

공통 schema, revision·manifest, 정제와 split 기준은 [Dataset Guides](../../../README.md#dataset-guides)에서 관리합니다.
이 문서의 명령은 `backends/megatron` 디렉터리에서 실행합니다.

## Public Datasets

[공개 데이터 기준](../../datasets/public-datasets.md)을 적용한 뒤 아래 명령으로 준비합니다.

Megatron backend:

```bash
./scripts/prepare_public_data.sh --preset no_robots --output-dir data/public/no_robots --revision e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b --train-count 32 --eval-count 8 --seed 42
```

생성된 `training.jsonl`/`validation.jsonl`은 `--train-data`/`--eval-data`로 지정하는 Megatron Bridge JSON loader 입력 형식입니다.
Setup2 launcher를 사용할 때는 `DATA_DIR=data/public/no_robots DATASET_ID=HuggingFaceH4/no_robots DATASET_REVISION=e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`를 함께 지정하고, cluster/NCCL/model/tokenizer evidence가 없는 실행은 runnable 결과로 보고하지 않습니다.

다른 SFT source도 같은 형태로 준비합니다.

```bash
./scripts/prepare_public_data.sh --preset ultrachat --output-dir data/public/ultrachat --revision 8049631c405ae6576f93f445c6b8166f76f5505a --train-count 32 --eval-count 8 --seed 42
./scripts/prepare_public_data.sh --preset self_oss --output-dir data/public/self_oss --revision 356bb069eee815daa6e23e9a282eeefe1490ad44 --train-count 32 --eval-count 8 --seed 42
./scripts/prepare_public_data.sh --preset xlam --output-dir data/public/xlam --revision 26d14ebfe18b1f7b524bd39b404b50af5dc97866 --train-count 32 --eval-count 8 --seed 42
```

XLAM preparation may require an already authorized Hub account; this script does not accept gated terms or bypass access controls.
위 command는 preparation example이지 GPU-training evidence가 아니며, `no_robots`는 non-commercial license이므로 commercial use 전에 별도 법무·권리 검토가 필요합니다.

## Internal Service Data

[사내 데이터 기준](../../datasets/internal-data-guide.md)에 따라 검수한 trace를 변환합니다.

```bash
./scripts/setup.sh
./scripts/prepare_service_data.sh
```

기본 입력은 `examples/service-traces.jsonl`, 출력은 `data/service-sft`입니다.
예제에는 train 2개, validation 1개, test 1개와 승인되지 않아 제외되는 trace 1개가 들어 있습니다.

```text
data/service-sft/
├── training.jsonl
├── validation.jsonl
├── test.jsonl
└── manifest.json
```

`training.jsonl`과 `validation.jsonl`은 이 backend의 `DirectHFSFTDatasetConfig`가 Hugging Face `json` loader로 읽는 `messages` column을 포함합니다.
Setup2 학습 전처리는 native chat template으로 마지막 assistant 응답을 prompt/completion으로 분리하고, `PromptCompletionSFTPreprocessingConfig(loss_mode="completion")`로 completion token에 loss를 계산합니다.

`test.jsonl`은 마지막 assistant 응답을 prompt에서 제거하고 `reference_answer`로 분리합니다.
현재 실험은 `do_test=False`이므로 이 파일을 학습 recipe에 전달하지 않습니다.
`grader`는 `exact_match`, `required_phrases`, schema validator, executable test, human rubric처럼 task에 맞는 독립 평가기로 연결하기 위한 metadata입니다.

실제 trace와 출력 위치는 환경 변수로 바꿀 수 있습니다.

```bash
TRACE_FILE=<reviewed-traces.jsonl> DATA_DIR=<prepared-data-dir> \
  ./scripts/prepare_service_data.sh
```

변환 결과는 canonical `messages` JSONL입니다.
현재 service-data manifest는 Setup2가 요구하는 pinned dataset revision 형식과 다르므로, `DATA_DIR`만 지정해 바로 학습할 수는 없습니다.
학습에 연결하려면 승인된 dataset의 불변 버전과 manifest를 준비하고 [Setup2 guide](spark-cluster.md)의 dataset identity 및 native completion 검증을 통과해야 합니다.

`manifest.json`에서 입력 파일 hash, split별 개수, 제외 사유를 확인합니다.
실제 데이터 pipeline에서는 승인 데이터가 갑자기 줄거나 특정 제외 사유가 늘면 build를 실패시키는 기준을 추가해야 합니다.

## Scaling Considerations

작은 JSONL은 이 backend의 기능 확인에 적합합니다.
데이터가 커지면 version별 immutable shard, node-local cache, worker별 deterministic sharding, offline packing을 고려합니다.
sequence length 분포와 truncation 비율을 먼저 측정한 뒤 packing을 적용하고, packed artifact가 어떤 원본 dataset version과 tokenizer에서 생성되었는지 함께 기록해야 합니다.

Megatron Bridge의 data API는 version에 따라 바뀔 수 있으므로 현재 고정한 `megatron-bridge==0.6.0`의 `DirectHFSFTDatasetConfig`를 이 backend의 source of truth로 사용합니다.
다른 version에서 `GPTSFTDatasetConfig`나 prompt-completion `input`/`output` 형식을 사용할 때는 recipe와 preprocessing config를 함께 변경하고 smoke test를 다시 수행해야 합니다.

## References

- [Megatron Bridge data preparation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/data-preparation.html)
- [Megatron Bridge text SFT data tutorial](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/data-preparation.html#text-sft-or-peft)
