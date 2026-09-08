# Post-Training Lab

Post-Training Lab은 LLM post-training을 데이터 준비, 학습, 평가, serving integration과 resource profiling까지 연결해 실습하는 저장소입니다.
SFT를 시작점으로 preference optimization과 reinforcement learning까지 확장합니다.
`main`은 실습 목록과 공통 작성 기준을 제공하고, 실행 코드와 환경 설정은 각 실습 브랜치가 관리합니다.

## Overview

Post-training은 사전 학습된 모델을 데이터와 피드백으로 추가 학습해 지시 수행 능력이나 특정 작업 성능을 개선하는 과정입니다.
정답 예시로 학습하는 supervised fine-tuning(SFT), 선호 데이터로 학습하는 preference optimization(예: DPO), 보상 신호를 활용하는 reinforcement learning(RL)이 여기에 포함됩니다.
아래는 학습 결과를 서비스에 반영하는 개념적 흐름입니다.

```text
Pretrained Model + Training Data / Feedback
                    |
                    v
           Post-training (SFT / DPO / RL)
                    |
                    v
           Checkpoint / Adapter
                    |
                    v
           Evaluation & Validation
                    |
                    v
           Versioned Model Artifact
                    |
                    v
           Serving Instance Rollout
                    |
                    v
           Monitoring & Feedback
```

평가를 통과한 checkpoint나 adapter를 서빙에 필요한 형식으로 준비하고 버전을 부여한 뒤, 서빙 인스턴스가 해당 버전을 로드하도록 배포합니다.
배포 방식은 전체 모델 재로딩, 인스턴스 교체, adapter 로딩 등으로 달라지며, 배포 후에는 품질과 성능을 관측하고 문제가 생기면 이전 버전으로 rollback합니다.
이 저장소의 학습 실습은 `trl`과 `megatron`, 자원 측정은 `profiling`에서 다룹니다.
서빙 반영과 rollback은 현재 [`system-integration`](https://github.com/daegyu94/post-training-lab/tree/system-integration)의 설계 범위이며, 위 흐름 전체가 자동화되어 있지는 않습니다.

## Learning Paths

목적에 맞는 시작점을 선택하세요.
각 실습은 독립적으로 진행할 수 있습니다.

| 목적 | 시작점 | 실습 내용 |
| --- | --- | --- |
| GPU 없이 학습 결과 관측하기 | [Observatory CPU 실습](https://github.com/daegyu94/post-training-lab-observatory/blob/main/docs/labs/01-observe-runs.md) | 합성 SFT·RL 데이터를 dashboard에서 관측하고 비교 |
| SFT 시작하기 | [`trl`](https://github.com/daegyu94/post-training-lab/tree/trl) | QLoRA와 분산 SFT |
| 분산 학습 구조 익히기 | [`megatron`](https://github.com/daegyu94/post-training-lab/tree/megatron) | parallelism과 checkpoint 저장·재개 |
| 실행 자원 측정하기 | [`profiling`](https://github.com/daegyu94/post-training-lab/tree/profiling) | GPU·host·network·storage 측정 |
| 학습과 serving 연결 설계하기 | [`system-integration`](https://github.com/daegyu94/post-training-lab/tree/system-integration) | data lifecycle, checkpoint promotion과 rollback 설계 문서; 실행 구현 없음 |

DPO와 RL 학습은 [확장 계획](labs/README.md#planned-exercises) 단계입니다.
Observatory의 RL 데이터는 합성 예시입니다.

## Dataset Guides

데이터 선택·schema 변환·revision·manifest와 사내 데이터 정제·split 기준은 `main`에서 공통으로 관리합니다.
실행 코드와 변환 명령은 각 실습 브랜치에서 관리합니다.

- [공개 데이터 가이드](docs/datasets/public-datasets.md): No Robots, Self-OSS, xLAM 등의 schema와 provenance
- [사내 데이터 가이드](docs/datasets/internal-data-guide.md): 승인된 service trace의 정제, split과 test 정답 분리
- [TRL 데이터 준비](https://github.com/daegyu94/post-training-lab/blob/trl/docs/dataset-preparation.md), [Megatron 데이터 준비](https://github.com/daegyu94/post-training-lab/blob/megatron/docs/dataset-preparation.md): 브랜치별 변환·학습 연결

## PoC Setups

| Setup | Hardware | 실습 범위 |
| --- | --- | --- |
| Setup 1 | RTX PRO 4000 Blackwell 24 GiB, single GPU | TRL: Qwen2.5-14B QLoRA; Megatron: Qwen2.5-7B LoRA |
| Setup 2 | DGX Spark GB10 2대 (`spark1`, `spark2`), 노드당 memory 약 119 GiB, 100Gbps RoCE | Qwen3-30B-A3B·GLM-4.7-Flash LoRA와 분산 학습 |

설치와 실행 방법은 선택한 브랜치의 README를 따르세요.
Setup 2의 검증 범위와 제약은 [TRL Spark guide](https://github.com/daegyu94/post-training-lab/blob/trl/docs/spark-cluster.md)와 [Megatron feature labs](https://github.com/daegyu94/post-training-lab/blob/megatron/docs/megatron-feature-labs.md)에서 확인하세요.
30B 모델은 짧은 LoRA 실행을 검증했으며, 장기 학습과 품질·성능 검증은 별도입니다.

각 실습 브랜치는 다른 실습 브랜치의 파일을 전제로 하지 않습니다.
