# TRL Backend

단일 GPU QLoRA와 Spark SFT를 위한 학습·데이터·launcher 코드입니다.
실행 절차와 제한은 [TRL 가이드](../../docs/backends/trl.md), 공통 시작 경로는 [Getting Started](../../docs/getting-started.md)를 따릅니다.

| 진입점 | 역할 |
| --- | --- |
| `trl_lab/train.py` | 단일 GPU QLoRA |
| `trl_lab/infer.py` | Adapter 재로딩 추론 |
| `trl_lab/spark_train.py` | Spark SFT |
| `trl_lab/spark_config.py` | 설정·snapshot·manifest 검증 |
| `scripts/` | 환경·데이터 준비와 실행 |
| `configs/` | DeepSpeed 예제 설정 |

CPU 테스트와 실행 판정은 [Verification](../../docs/verification.md)을 확인합니다.
