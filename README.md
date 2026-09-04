# Megatron Bridge Qwen2.5 SFT Reproduced Result

이 브랜치는 Megatron Bridge로 `Qwen/Qwen2.5-14B-Instruct`를 `HuggingFaceH4/ultrachat_200k`에 LoRA fine-tuning하고, 학습 전후의 held-out loss와 checkpoint 재로딩을 확인하는 reproduced result 실행 절차만 제공합니다.

UltraChat의 `train_sft`와 `test_sft`를 각각 학습·평가 데이터로 사용하며, assistant token에만 loss를 적용합니다.

## 재현 범위

실행 스크립트는 다음 순서를 고정합니다.

1. 실행 환경과 입력 경로를 preflight로 확인합니다.
2. pretrained checkpoint를 새 프로세스에서 읽고 평가 데이터의 base loss를 측정합니다.
3. `train_sft`로 LoRA adapter를 학습하고 native Megatron checkpoint를 저장합니다.
4. 새 프로세스에서 base checkpoint와 저장된 adapter checkpoint를 다시 읽고 같은 평가 데이터의 tuned loss를 측정합니다.
5. 두 평가 log를 비교해 `summary.json`을 생성합니다.

학습과 마지막 평가는 서로 다른 프로세스에서 실행되므로 checkpoint 저장과 재로딩도 함께 확인할 수 있습니다.

## 요구 환경

- Python 3.12
- BF16을 지원하는 GPU 한 장
- 단일 GPU 기준 최소 40GiB GPU memory
- Qwen checkpoint를 저장할 약 28GB 이상의 disk space
- Hugging Face Hub와 Hugging Face Datasets에 접근 가능한 네트워크

Megatron Bridge의 14B LoRA 실행은 quantized base model을 사용하지 않습니다. 따라서 40GiB 미만의 GPU에서는 실행 전에 preflight가 중단합니다.

## 실행 방법

### 환경 설치

~~~bash
./scripts/setup.sh
~~~

### UltraChat 데이터 준비

기본값은 `train_sft`에서 32개, `test_sft`에서 8개의 유효한 대화를 선택합니다. 생성된 JSONL 파일은 Git에 포함하지 않습니다.

~~~bash
./scripts/prepare_data.sh
~~~

환경 변수로 샘플 수와 출력 경로를 변경할 수 있습니다.

~~~bash
TRAIN_SAMPLES=32 EVAL_SAMPLES=8 DATA_DIR=data/ultrachat_200k ./scripts/prepare_data.sh
~~~

### Qwen checkpoint 준비

다음 명령은 `Qwen/Qwen2.5-14B-Instruct`의 BF16 checkpoint를 기본 경로에 내려받습니다.

~~~bash
./scripts/download_model.sh
~~~

이미 checkpoint가 있으면 `MODEL_DIR`로 경로를 지정할 수 있습니다.

~~~bash
MODEL_DIR=<model-dir> ./scripts/download_model.sh
~~~

### Reproduced result 실행

프로세스에 GPU 한 장만 노출한 상태에서 전체 절차를 실행합니다.

~~~bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_reproduced_result.sh
~~~

다른 GPU를 사용하려면 `CUDA_VISIBLE_DEVICES` 값만 변경합니다.

~~~bash
CUDA_VISIBLE_DEVICES=<gpu-index> ./scripts/run_reproduced_result.sh
~~~

기본 실행 설정은 다음과 같습니다.

| 항목 | 기본값 |
| --- | --- |
| `MODEL_DIR` | `models/Qwen2.5-14B-Instruct` |
| `DATA_DIR` | `data/ultrachat_200k` |
| `OUTPUT_DIR` | `results/qwen2.5-14b-megatron-reproduced-result` |
| `MAX_STEPS` | `5` |
| `EVAL_ITERS` | `8` |
| `MAX_LENGTH` | `512` |
| `GLOBAL_BATCH_SIZE` | `8` |
| `SEED` | `42` |

재현 조건을 바꾸지 않으려면 위 환경 변수를 지정하지 않고 실행합니다.

## 결과 확인

실행 결과는 다음 경로에 저장됩니다.

~~~text
results/qwen2.5-14b-megatron-reproduced-result/
|-- base-eval.log
|-- train.log
|-- tuned-eval.log
|-- checkpoints/
`-- summary.json
~~~

`summary.json`에는 base와 tuned의 held-out loss 및 perplexity, loss 변화율, checkpoint 재로딩 확인 결과가 기록됩니다.

핵심 비교 지표는 같은 평가 데이터에 대한 `tuned_eval_loss`와 `base_eval_loss`입니다. 일반적으로 `tuned_eval_loss < base_eval_loss`인지 확인하고, 각 log에 `nan`이나 무한대가 없는지도 함께 확인합니다.

## 주요 구현

- `scripts/prepare_data.sh`: 공개 UltraChat split에서 재현 가능한 학습·평가 subset을 생성합니다.
- `scripts/download_model.sh`: Hugging Face checkpoint를 준비합니다.
- `scripts/run_reproduced_result.sh`: preflight, base 평가, 학습, checkpoint 재로딩 평가, 결과 비교를 순서대로 실행합니다.
- `megatron_lab/preflight.py`: GPU memory, BF16 지원, package, model/data 경로를 확인합니다.
- `megatron_lab/config.py`: Qwen2.5-14B LoRA와 UltraChat 데이터 구성을 생성합니다.
- `megatron_lab/sft.py`: base 평가, train, tuned 평가 stage를 실행합니다.
- `megatron_lab/compare.py`: 평가 log에서 loss를 추출해 `summary.json`을 생성합니다.

## 참고 자료

- [Megatron Bridge Qwen 지원 및 recipe](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/qwen/qwen.html)
- [Megatron Bridge text SFT data](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/hf-text-only/README.md)
- [UltraChat 200k dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
