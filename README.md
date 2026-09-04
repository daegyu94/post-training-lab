# SFT Lab

SFT Lab은 LLM supervised fine-tuning(SFT)과 large-scale LLM post-training으로 확장되는 workflow를 직접 실행하고 검증하기 위한 실습 저장소입니다. 각 브랜치는 하나의 목적과 실행 환경을 독립적으로 가지며, `main`은 이 브랜치들을 찾기 위한 안내 역할만 합니다.

## Active Branches

| Branch | 목적 | 시작점 |
| --- | --- | --- |
| `qlora` | TRL 기반 QLoRA SFT를 실제 실행하고 held-out 평가·adapter 재로딩까지 확인 | 해당 브랜치의 `README.md` |
| `megatron` | Qwen2.5-7B와 Megatron Bridge를 사용한 LoRA SFT 실험, Megatron-LM/Megatron Core 병렬화 개념 실습 | 해당 브랜치의 `README.md` |
| `post-training` | TRL 기준 workflow와 Megatron-LM 계열 확장 경로를 함께 정리한 large-scale post-training baseline | 해당 브랜치의 `README.md` |
| `profiling` | Megatron 또는 verl post-training workload의 GPU·host·network·storage resource profiling | 해당 브랜치의 `README.md` |

원하는 실습 브랜치로 전환한 뒤, 그 브랜치의 `README.md`부터 진행하세요.

```bash
git fetch origin
git switch <branch>
```

## Branch Relationship

- `qlora`는 SFT의 dataset, loss, adapter, evaluation 경로를 빠르게 검증하는 기준 실습입니다.
- `megatron`은 Megatron Bridge와 Megatron-LM/Megatron Core의 개념을 바탕으로, 더 큰 model과 분산 학습 환경으로 확장하기 위한 실습입니다.
- `post-training`은 두 framework를 경쟁 관계가 아닌 기준 검증과 scale-up의 연결 경로로 다룹니다.
- `profiling`은 실제 multi-GPU·multi-node 실행에서 병목을 관찰하고 원인을 좁히기 위한 공통 resource profiling 환경입니다.

## Repository Policy

model weight, dataset cache, checkpoint, profiler trace처럼 큰 실행 산출물은 Git에 저장하지 않습니다. 각 실습 브랜치는 재현 가능한 command, 작은 summary, 환경 정보와 결과 해석을 저장합니다.

실행 결과는 model, dataset, hardware, seed, 실행 시간에 따라 달라질 수 있습니다. 브랜치에 기록된 결과는 해당 configuration의 실행 기록이며, 일반적인 모델 품질이나 cluster-scale 성능을 보장하지 않습니다.
