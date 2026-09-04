# SFT Lab

SFT Lab은 LLM supervised fine-tuning(SFT), large-scale post-training과 resource profiling workflow를 직접 실행하고 검증하기 위한 브랜치 기반 실습 저장소입니다. `main`은 실행 코드를 포함하지 않고 각 실습의 목적과 시작점을 안내합니다.

## Branches

| Branch | Purpose |
| --- | --- |
| `trl` | TRL 기반 QLoRA SFT, held-out evaluation과 adapter 재로딩 검증 |
| `megatron` | Qwen2.5-7B Megatron Bridge LoRA SFT와 Megatron parallelism 개념 실습 |
| `post-training` | TRL baseline에서 Megatron 기반 large-scale post-training으로 확장하는 workflow |
| `profiling` | Megatron·verl workload의 GPU, host, network와 storage resource profiling |

원하는 실습 브랜치로 전환한 뒤, 그 브랜치의 `README.md`부터 진행하세요.

```bash
git fetch origin
git switch <branch>
```

## Repository Policy

model weight, dataset cache, checkpoint, profiler trace처럼 큰 실행 산출물은 Git에 저장하지 않습니다. 각 브랜치에는 재현 가능한 command, configuration, 작은 summary와 결과 해석만 저장합니다. 기록된 결과는 해당 실행 환경의 검증 자료이며, 일반적인 model quality나 cluster-scale 성능을 보장하지 않습니다.
