# Megatron Backend

Megatron Bridge와 Core 기반의 Spark SFT·checkpoint 코드입니다.
Launcher 기본값은 LoRA이며 소형 smoke preset은 full SFT를 명시합니다.
실행 절차·설치 한계는 [Megatron 가이드](../../docs/backends/megatron.md), 공통 시작 경로는 [Getting Started](../../docs/getting-started.md)를 따릅니다.

| 진입점 | 역할 |
| --- | --- |
| `megatron_lab/config.py` | Model provider와 학습 설정 |
| `megatron_lab/sft.py` | Base/train/resume/tuned |
| `megatron_lab/cluster.py` | Topology 검증 |
| `megatron_lab/feature_lab.py` | 기능 비교 로그 파싱 |
| `scripts/` | 환경·데이터 준비와 실행 |

CPU 테스트와 실행 판정은 [Verification](../../docs/verification.md)을 확인합니다.
