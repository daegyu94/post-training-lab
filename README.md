# Post-Training Lab

Post-Training Lab은 LLM post-training을 데이터 준비, 학습, 평가, serving integration과 resource profiling까지 연결해 실습하는 저장소입니다.
SFT를 시작점으로 preference optimization과 reinforcement learning까지 확장합니다.
`main`은 실습 목록과 공통 작성 기준을 제공하고, 실행 코드와 환경 설정은 각 실습 브랜치가 관리합니다.

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

## PoC Setups

| Setup | Hardware | 실습 범위 |
| --- | --- | --- |
| Setup 1 | RTX PRO 4000 Blackwell 24 GiB, single GPU | TRL: Qwen2.5-14B QLoRA; Megatron: Qwen2.5-7B LoRA |
| Setup 2 | DGX Spark GB10 2대 (`spark1`, `spark2`), 노드당 memory 약 119 GiB, 100Gbps RoCE | Qwen3-30B-A3B·GLM-4.7-Flash LoRA와 분산 학습 |

설치와 실행 방법은 선택한 브랜치의 README를 따르세요.
Setup 2의 검증 범위와 제약은 [TRL Spark guide](https://github.com/daegyu94/post-training-lab/blob/trl/docs/spark-cluster.md)와 [Megatron feature labs](https://github.com/daegyu94/post-training-lab/blob/megatron/docs/megatron-feature-labs.md)에서 확인하세요.
30B 모델은 짧은 LoRA 실행을 검증했으며, 장기 학습과 품질·성능 검증은 별도입니다.

```bash
git clone https://github.com/daegyu94/post-training-lab.git
cd post-training-lab
git switch trl
```

각 실습 브랜치는 다른 실습 브랜치의 파일을 전제로 하지 않습니다.

## Add an Exercise

[실습 목록과 확장 기준](labs/README.md)에서 다음 실습의 범위와 구현 위치를 정하고, [실습 템플릿](labs/template/README.md)에 따라 command, 예상 결과, 검증 자료와 정리 방법을 기록합니다.
공통 안내는 `main`, backend별 구현은 해당 브랜치, dashboard 관측 실습은 Observatory에 둡니다.
