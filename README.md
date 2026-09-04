# SFT Lab

SFT Lab은 LLM supervised fine-tuning(SFT)을 직접 실행하고 학습 품질과 시스템 동작을 함께 확인하기 위한 실습 저장소입니다.

## Learning Paths

| Branch | Purpose | Main exercise |
| --- | --- | --- |
| `trl-qwen2.5-14b-ultrachat-qlora` | TRL 기반 SFT | Qwen2.5-14B-Instruct와 UltraChat을 이용한 QLoRA 학습, 평가, adapter 재로딩 |
| `megatron-lab` | Megatron Core 기반 분산 학습 원리 | single GPU와 single-host multi-GPU에서 tensor/data parallel 실습, multi-node 구성 검증 |
| `profiling` | 공통 측정·profiling 환경 | training, GPU, host, communication, storage 지표 수집과 PyTorch trace 분석 |
| `backup` | 개편 전 저장소 보존 | 기존 통합형 TRL·Megatron 실습 자료 열람 |

각 실습은 다른 실습 브랜치의 파일을 전제로 하지 않습니다. 원하는 브랜치로 전환한 뒤 해당 브랜치의 `README.md`부터 진행하세요.

```bash
git fetch origin
git switch <branch>
```

## Scope

이 저장소는 작은 예제로 명령 실행 여부만 확인하는 데 그치지 않고, 가능한 환경에서는 실제 GPU workload를 실행해 결과를 남기는 것을 목표로 합니다. 고비용 또는 multi-node 환경이 필요한 항목은 같은 개념을 검증하는 축소 실습과 configuration dry run을 제공합니다.

학습 결과는 모델·dataset·hardware·seed·실행 시간에 따라 달라질 수 있습니다. 브랜치에 기록된 결과는 재현 기준점이며 절대적인 품질 benchmark가 아닙니다.

## Requirements

- Linux
- Git
- NVIDIA GPU와 호환 driver(실제 GPU 실습 시)
- Python 3.10 이상(브랜치별 세부 버전은 해당 문서 참고)

## Repository Policy

모델 weight, dataset cache, checkpoint, profiler trace처럼 크기가 큰 실행 산출물은 Git에 포함하지 않습니다. 각 브랜치는 재현 가능한 command, 작은 summary, 환경 정보와 해석만 저장합니다.
