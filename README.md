# Megatron Bridge Qwen2.5-7B SFT Experiment

이 저장소는 Qwen/Qwen2.5-7B-Instruct에 LoRA SFT를 적용하고, 결과를 원본 모델과 비교하는 단일 GPU 실험 예제입니다. 학습 모델은 Hugging Face 체크포인트를 Megatron Bridge의 AutoBridge로 불러오고, Qwen용 PEFT recipe로 구성합니다.

실험 환경과 결과는 [환경 및 실험 기록](docs/experiment.md)에 정리합니다.

## 요구 사항

- Linux와 NVIDIA GPU
- CUDA를 사용할 수 있는 PyTorch 환경
- BF16을 지원하는 GPU
- 실험용 GPU 1개
- VRAM 20GiB 이상 권장
- Hugging Face에서 모델을 내려받을 수 있는 네트워크
- 모델과 결과를 위한 약 16GB 이상의 디스크 공간

기본 설정은 sequence length 512, micro batch size 1, activation recomputation을 사용합니다. 24GiB GPU에서 시도할 수 있는 구성이나, 드라이버·CUDA·할당자 상태에 따라 OOM이 날 수 있으므로 실제 실행으로 확인해야 합니다.

## 설치

~~~bash
git clone -b megatron-lab https://github.com/daegyu94/sft-lab.git
cd sft-lab
./scripts/setup.sh
~~~

설치 스크립트는 가상환경을 만들고, PyTorch·Megatron Bridge·Transformers·데이터셋 도구를 설치합니다.

## 모델과 데이터 준비

Hugging Face CLI 인증이 필요한 환경이라면 먼저 로그인합니다.

~~~bash
.venv/bin/hf auth login
~~~

그 다음 Qwen2.5-7B 모델과 Alpaca 데이터셋을 준비합니다.

~~~bash
./scripts/download_model.sh
./scripts/prepare_data.sh
~~~

기본 경로는 아래와 같습니다.

| 항목 | 기본 경로 |
| --- | --- |
| 모델 | models/Qwen2.5-7B-Instruct |
| 학습 데이터 | data/alpaca_zh_100.jsonl |
| 결과 | results/qwen2.5-7b-megatron-experiment |

환경 변수로 경로와 학습 조건을 바꿀 수 있습니다.

~~~bash
MODEL_DIR=/path/to/Qwen2.5-7B-Instruct \
OUTPUT_DIR=/path/to/output \
MAX_STEPS=10 \
CUDA_VISIBLE_DEVICES=0 \
./scripts/run_experiment.sh
~~~

## 실험 실행

아래 명령은 preflight 검사를 거친 뒤 모델을 불러와 LoRA SFT를 수행하고, 고정 프롬프트에서 원본 모델과 adapter 적용 모델의 출력을 비교합니다.

~~~bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_experiment.sh
~~~

기본 학습 횟수는 3 step입니다. 우선 작은 step 수로 환경과 메모리를 확인한 뒤 늘리는 편이 안전합니다. 결과물은 지정한 OUTPUT_DIR에 저장되며, adapter와 비교 결과는 각각 adapters, comparison.json 파일에서 확인할 수 있습니다.

## Megatron 개념 실습

이 저장소의 실제 SFT는 단일 GPU(TP=1, PP=1, CP=1, DP=1) 실행입니다. GPU 없이 Megatron Bridge recipe와 병렬화 group 배치를 살펴보려면 다음을 실행합니다.

~~~bash
./scripts/run_megatron_practice.sh
~~~

이 실습은 체크포인트를 내려받지 않고, CUDA·NCCL·torch.distributed·멀티 GPU·멀티 노드 통신을 시작하지 않습니다. TP, PP, DP와 Context Parallelism(CP)의 역할 및 사용법은 [Megatron-LM과 Megatron Core 개요](docs/megatron-overview.md)를 참고하세요.

## 주요 구현

- megatron_lab/config.py: Qwen2.5-7B LoRA recipe와 단일 GPU SFT 설정
- megatron_lab/sft.py: Bridge 기반 학습과 adapter 저장
- megatron_lab/compare.py: 원본 모델과 adapter 모델의 생성 결과 비교
- megatron_lab/preflight.py: GPU와 실행 환경 확인
- megatron_lab/inspect_recipe.py: GPU 초기화 없이 Bridge recipe 요약
- megatron_lab/parallelism.py: TP/PP/CP/DP 논리적 rank group 시뮬레이션
- scripts/run_experiment.sh: 전체 실험 실행
- scripts/run_megatron_practice.sh: recipe 확인과 병렬화 개념 실습
