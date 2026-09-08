# TRL Dataset Preparation

공통 schema, revision·manifest, 정제와 split 기준은 [Dataset Guides](../../../README.md#dataset-guides)에서 관리합니다.
이 문서의 명령은 `backends/trl` 디렉터리에서 실행합니다.

학습에 전달하는 옵션은 Setup 1의 `--dataset-jsonl-dir`, Setup 2의 `DATA_DIR`입니다.

## Public Datasets

[공개 데이터 기준](../../datasets/public-datasets.md)을 적용한 뒤 아래 명령으로 준비합니다.

TRL branch:

```bash
./scripts/prepare_public_data.sh --preset no_robots --output-dir data/public/no_robots --revision e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b --train-count 32 --eval-count 8 --seed 42
./scripts/prepare_public_data.sh --preset self_oss --output-dir data/public/self_oss --revision 356bb069eee815daa6e23e9a282eeefe1490ad44 --train-count 32 --eval-count 8 --seed 42
./scripts/prepare_public_data.sh --preset xlam --output-dir data/public/xlam --revision 26d14ebfe18b1f7b524bd39b404b50af5dc97866 --train-count 32 --eval-count 8 --seed 42
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m trl_lab.train --model Qwen/Qwen2.5-14B-Instruct --dataset bigcode/self-oss-instruct-sc2-exec-filter-50k --dataset-jsonl-dir data/public/self_oss --local-files-only --train-samples 32 --eval-samples 8 --max-steps 1 --output-dir results/public-self-oss
```

XLAM preparation may require an already authorized Hub account; this script does not accept gated terms or bypass access controls.
`no_robots`는 non-commercial license이므로 commercial use 전에 별도 법무·권리 검토가 필요합니다.

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

`training.jsonl`과 `validation.jsonl`은 TRL conversational format인 `messages`를 포함합니다.
`SFTTrainer`는 이 브랜치의 `assistant_only_loss=True` 설정과 Qwen chat template을 사용해 assistant token에만 loss를 계산합니다.

`test.jsonl`은 마지막 assistant 응답을 prompt에서 제거하고 `reference_answer`로 분리합니다.
이 파일은 `SFTTrainer`에 전달하지 않습니다.
`grader`는 `exact_match`, `required_phrases`, schema validator, executable test, human rubric처럼 task에 맞는 평가기로 연결하기 위한 metadata입니다.

실제 trace와 출력 위치는 환경 변수로 바꿀 수 있습니다.

```bash
TRACE_FILE=<reviewed-traces.jsonl> DATA_DIR=<prepared-data-dir> \
  ./scripts/prepare_service_data.sh
```

변환 결과를 이 브랜치의 학습 경로에서 사용하려면 다음처럼 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m trl_lab.train \
  --dataset-jsonl-dir data/service-sft \
  --local-files-only \
  --train-samples 2 \
  --eval-samples 1 \
  --max-steps 1 \
  --output-dir results/service-data-practice
```

`manifest.json`에서 입력 파일 hash, split별 개수, 제외 사유를 확인합니다.
실제 데이터 pipeline에서는 승인 데이터가 갑자기 줄거나 특정 제외 사유가 늘면 build를 실패시키는 기준을 추가해야 합니다.

## References

- [TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
