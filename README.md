# Large-Scale LLM Post-Training Lab

이 저장소는 LLM post-training workflow를 작은 단일 GPU 검증에서 multi-GPU·multi-node 확장까지 단계적으로 확인하기 위한 실습 저장소입니다. TRL은 SFT의 입력·loss·adapter·평가 흐름을 빠르게 검증하는 기준 구현으로 사용하고, Megatron-LM 계열은 같은 workflow를 분산 학습과 대규모 모델·클러스터 환경으로 확장하는 구현으로 사용합니다.

두 프레임워크는 경쟁하는 실습이 아닙니다. TRL로 post-training의 의미론과 end-to-end 경로를 먼저 확인한 뒤, Megatron-LM으로 parallelism, checkpoint, 통신, storage를 포함한 large-scale 실행 조건을 검증합니다.

## Getting Started

이 브랜치의 전체 목표와 확장 경로는 [docs/overview.md](docs/overview.md), 현재 실행 가능한 SFT workflow와 두 프레임워크의 역할은 [docs/step1.md](docs/step1.md)를 참고하세요.

| 경로 | 역할 | 실행 시작점 |
| --- | --- | --- |
| `sft_lab/`, `scripts/trl/` | TRL 기반 QLoRA SFT의 단일 GPU 기준 실행 | `./scripts/trl/setup.sh`, `./scripts/trl/run_experiment.sh` |
| `megatron_lab/`, `scripts/megatron/` | Megatron Bridge 기반 SFT와 checkpoint workflow 검증 | `./scripts/megatron/setup.sh`, `./scripts/megatron/prepare_data.sh`, `./scripts/megatron/run_experiment.sh` |
| `requirements/` | 프레임워크별 독립 Python 의존성 | 각 setup script에서 설치 |

각 workflow는 별도의 virtual environment를 사용해야 합니다. 기본값은 `.venv-trl`과 `.venv-megatron`이며, 같은 환경에 두 requirements를 함께 설치하지 마세요.

## Scope and Scale-up Boundary

현재 포함된 실행은 Qwen2.5와 UltraChat을 사용해 dataset → tokenization → SFT → checkpoint/adapter 저장 → 새 프로세스 재로딩 → held-out evaluation 경로를 확인합니다. TRL 실행은 4-bit QLoRA로 24GiB GPU에서도 검증할 수 있게 구성되어 있으며, Megatron 실행은 native distributed backend로 옮기기 전의 Bridge workflow 검증을 제공합니다.

대규모 환경에서는 model size, GPU 수, parallelism 구성, checkpoint 형식, 데이터·storage·통신 경로를 실제 클러스터 조건에 맞춰 바꿔야 합니다. 작은 실행에서의 loss 변화나 처리량을 일반적인 모델 품질 또는 cluster-scale 성능으로 해석하면 안 됩니다.

## Outputs and Verification

model weight, dataset cache, checkpoint, profiler trace처럼 큰 산출물은 Git에 저장하지 않습니다. 각 실행은 `results/` 아래에 summary와 log를 생성하며, 확인 방법은 [docs/step1.md](docs/step1.md)에 정리되어 있습니다. 실제 TRL 실행 기록은 [docs/trl-run-record.md](docs/trl-run-record.md)에서 확인할 수 있습니다.
