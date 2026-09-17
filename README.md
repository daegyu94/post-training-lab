# Post-Training Lab

이미 학습된 LLM에 SFT를 적용하면서 데이터 준비, 분산 실행, checkpoint 저장과 자원 계측을 실습하는 저장소입니다.
백엔드는 TRL과 Megatron Bridge이며, verl·NeMo RL 비교 경로는 0.5B 1-step GPU smoke를 제공합니다.
실행 환경은 NVIDIA DGX Spark 2노드(`spark1`, `spark2`)입니다.
Controller가 실행을 조율하고 GPU 연산은 Spark 노드에서만 수행합니다.

첫 실행은 [Getting Started](docs/getting-started.md)에서 시작합니다.
Controller에는 Python 3.10 이상과 Git이, 학습에는 준비된 Spark 노드·CUDA 환경·모델·데이터가 필요합니다.

## Documentation

| 질문 | 문서 |
| --- | --- |
| 환경을 준비하고 첫 SFT를 돌리려면? | [Getting Started](docs/getting-started.md) |
| 어떤 구성 요소가 무엇을 책임지는가? | [Architecture](docs/architecture.md) |
| 학습 입력을 어떻게 만드는가? | [Datasets](docs/datasets.md) |
| 공개 코드 문제로 execution feedback cycle을 돌리려면? | [Execution Feedback](docs/execution-feedback.md) |
| TRL로 어떻게 학습하는가? | [TRL Backend](docs/backends/trl.md) |
| Megatron으로 어떻게 학습·저장하는가? | [Megatron Backend](docs/backends/megatron.md) |
| Verl agentic-RL smoke를 준비하려면? | [Verl Backend](docs/backends/verl.md) |
| NeMo RL과 verl을 같은 Spark 조건에서 비교하려면? | [NeMo RL vs verl](docs/experiments/nemo-rl-vs-verl.md) |
| 실험 결과가 무엇을 의미하고 어디까지 비교 가능한가? | [Experiments](docs/experiments.md) |
| 자원·통신·저장소를 어떻게 관측하는가? | [Post-Training Telemetry](third_party/post-training-telemetry/README.md) (submodule) |
| 클러스터 토폴로지와 노드 준비는? | [Spark Cluster Setup](setups/spark/README.md) |

디렉터리별 책임과 각 경로가 포함하지 않는 책임은 [Architecture](docs/architecture.md#responsibilities)에 있습니다.

## Scope

기본 범위는 SFT workflow와 그 관측·baseline입니다.
추가로 공개 Python 코드 문제에서 한 번의 execution-filtered SFT/DPO cycle을 검증하는 독립 실험 경로를 제공합니다.
사내 framework 성능 검증, 반복적인 feedback cycle, repository-level build, 여러 언어, 장기 GPU RL 학습, artifact registry와 serving 배포는 구현하지 않습니다.

정적 검사, CPU 테스트, dry-run, 실제 GPU 실행은 서로 다른 증거이며 한 단계의 성공이 다른 단계를 보장하지 않습니다.
검증 명령은 [AGENTS.md](AGENTS.md), 판정 기준은 [Getting Started](docs/getting-started.md#6-verify-the-result)를 따릅니다.
