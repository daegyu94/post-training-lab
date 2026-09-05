# SFT Lab

SFT Lab은 LLM supervised fine-tuning(SFT), large-scale post-training과 resource profiling workflow를 직접 실행하고 검증하기 위한 브랜치 기반 실습 저장소입니다. `main`은 실행 코드를 포함하지 않고 각 실습의 목적과 시작점을 안내합니다.

## Branches

| Branch | Purpose |
| --- | --- |
| `trl` | TRL 기반 QLoRA SFT, held-out evaluation과 adapter 재로딩 검증 |
| `megatron` | Qwen2.5-7B Megatron Bridge LoRA SFT와 Megatron parallelism 개념 실습 |
| `post-training` | framework-independent data lifecycle, checkpoint promotion, serving integration과 rollback |
| `profiling` | Megatron·verl workload의 GPU, host, network와 storage resource profiling |

브랜치 사이에 반드시 따라야 하는 순서는 없습니다. 목적에 맞는 브랜치를 선택하세요.

- 단일 GPU에서 실행 가능한 SFT와 QLoRA workflow를 확인하려면 `trl`
- Megatron 기반 model·dataset·checkpoint workflow와 parallelism 개념을 확인하려면 `megatron`
- training backend와 무관한 data-to-production lifecycle을 설계하려면 `post-training`
- distributed workload의 GPU, host, network와 storage 병목을 분석하려면 `profiling`

`trl`과 `megatron`은 서로 다른 training backend를 다루는 독립 실습입니다. `post-training`은 두 backend가 production lifecycle에 연결될 때 필요한 공통 contract를 설명하며, 실행 가능한 training implementation은 포함하지 않습니다. `profiling`은 framework 실행 방법 대신 이미 실행 중인 Megatron 또는 verl workload를 관측하고 분석하는 방법에 집중합니다.

선택한 실습 브랜치로 전환한 뒤, 해당 브랜치의 `README.md`부터 진행하세요.

```bash
git fetch origin
git switch <branch>
```

## Repository Policy

model weight, dataset cache, checkpoint, profiler trace처럼 큰 실행 산출물은 Git에 저장하지 않습니다. 각 브랜치에는 재현 가능한 command, configuration, 작은 summary와 결과 해석만 저장합니다. 기록된 결과는 해당 실행 환경의 검증 자료이며, 일반적인 model quality나 cluster-scale 성능을 보장하지 않습니다.
