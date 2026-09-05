# Post-Training System Integration

이 브랜치는 특정 training framework의 사용법이 아니라, 사내 LLM 서비스에서 수집한 데이터를 학습 가능한 형태로 준비하고, training backend가 생성한 checkpoint를 검증한 뒤 serving system에 안전하게 반영하는 framework-independent post-training lifecycle을 다룹니다.

TRL과 Megatron-LM의 설치, 학습 명령, framework별 configuration과 실행 결과는 각각 [`trl` branch](https://github.com/daegyu94/sft-lab/tree/trl)와 [`megatron` branch](https://github.com/daegyu94/sft-lab/tree/megatron)가 담당합니다. 이 브랜치는 두 backend를 복제하지 않고 공통 입력·출력 contract와 production integration만 정의합니다.

## Scope

이 브랜치가 다루는 범위는 다음과 같습니다.

- LLM service trace와 benchmark 결과에서 training 및 evaluation candidate를 수집하는 과정
- 개인정보, 보안 정보, 잘못된 응답을 제거하는 data curation과 dataset versioning
- training backend에 독립적인 canonical dataset과 training request contract
- TRL 또는 Megatron-LM backend 선택과 artifact handoff
- checkpoint 등록, offline evaluation, compatibility validation과 promotion gate
- canary deployment, serving traffic 전환, monitoring과 rollback
- multi-node GPU, network, storage를 포함한 production architecture

다음 항목은 이 브랜치에서 다루지 않습니다.

- TRL `SFTTrainer`, QLoRA, PEFT adapter의 상세 사용법
- Megatron Core의 TP, PP, DP, CP 설정과 distributed training 명령
- 특정 model과 GPU 환경에서 수행한 framework별 실행 기록
- framework package 설치와 독립적인 smoke test

## Documentation

| Document | Description |
| --- | --- |
| [System Architecture](docs/architecture.md) | training system과 serving system을 분리하고 연결하는 전체 구조 |
| [Data Lifecycle](docs/data-lifecycle.md) | trace 수집부터 canonical dataset, split, versioning까지의 공통 기준 |
| [Checkpoint Lifecycle](docs/checkpoint-lifecycle.md) | checkpoint 등록, 평가, promotion, deployment와 rollback |
| [Operations](docs/operations.md) | observability, failure handling, storage와 network 운영 기준 |

## Branch Responsibilities

| Branch | Primary responsibility |
| --- | --- |
| `main` | 프로젝트 목적과 활성 branch 안내 |
| `trl` | TRL 기반 SFT/QLoRA 구현과 단일 노드 기준 실험 |
| `megatron` | Megatron 기반 distributed SFT와 parallelism 검증 |
| `post-training-system` | framework-independent production lifecycle과 serving integration |
| `profiling` | 공통 resource metric 수집, schema와 dashboard 연동 |

Framework를 TRL에서 Megatron-LM으로 바꿔도 유효한 정책과 interface는 이 브랜치에 둡니다. 특정 Python API, CLI option, configuration 또는 checkpoint 형식에 의존하는 내용은 해당 framework 브랜치에 둡니다.

## Current Status

이 브랜치는 production architecture와 interface contract를 정의하는 설계 기준입니다. 실제 사내 trace ingestion, model registry, evaluation service, deployment controller와 serving control plane은 환경별 구현이 필요합니다. 문서의 예시 값은 interface를 설명하기 위한 것이며 실제 운영 endpoint나 credential을 포함하지 않습니다.
