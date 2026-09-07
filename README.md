# Post-Training Lab

Post-Training Lab은 LLM post-training을 데이터 준비, 학습, 평가, serving integration과 resource profiling까지 연결해 실습하는 저장소입니다. SFT를 시작점으로 preference optimization과 reinforcement learning까지 확장합니다. `main`은 실습 목록과 공통 작성 기준을 제공하고, 실행 코드와 환경 설정은 각 실습 브랜치가 관리합니다.

## Learning Paths

| Track | Branch / entry point | Current status |
| --- | --- | --- |
| SFT · QLoRA | [`trl`](https://github.com/daegyu94/post-training-lab/tree/trl) | 기존 학습, held-out evaluation과 adapter 재로딩 실습 |
| Distributed training foundations | [`megatron`](https://github.com/daegyu94/post-training-lab/tree/megatron) | Qwen2.5-7B LoRA SFT와 parallelism 개념 실습; 실제 multi-node 학습은 미구현 |
| System integration | [`system-integration`](https://github.com/daegyu94/post-training-lab/tree/system-integration) | data lifecycle, checkpoint promotion, serving과 rollback 설계 문서; 실행 구현 없음 |
| Resource profiling | [`profiling`](https://github.com/daegyu94/post-training-lab/tree/profiling) | GPU·host·network·storage 측정 실습; 각 실습별 환경 필요 |
| Observatory | [CPU 관측 실습](https://github.com/daegyu94/post-training-lab-observatory/blob/main/docs/labs/01-observe-runs.md) | GPU 없이 실행하는 합성 SFT·agentic RL dashboard 실습 |
| Preference optimization · DPO | [확장 계획](labs/README.md#planned-exercises) | 계획 단계; 학습 코드 없음 |
| Reinforcement learning | [확장 계획](labs/README.md#planned-exercises) | 계획 단계; RL 학습 코드 없음. Observatory의 RL 데이터는 합성 예시 |

GPU 없이 시작하려면 Observatory 실습을, 실제 SFT를 실행하려면 `trl` 또는 `megatron`을 선택하세요. SFT → preference optimization → RL은 학습 범위를 넓히는 방향이며 필수 실행 순서는 아닙니다. `system-integration`과 `profiling`은 여러 training backend에 공통으로 적용됩니다.

```bash
git clone https://github.com/daegyu94/post-training-lab.git
cd post-training-lab
git switch trl
```

선택한 브랜치의 `README.md`에서 설치와 실행 조건을 확인하세요. 각 실습 브랜치는 다른 실습 브랜치의 파일을 전제로 하지 않습니다.

## Add an Exercise

[실습 목록과 확장 기준](labs/README.md)에서 다음 실습의 범위와 구현 위치를 정하고, [실습 템플릿](labs/template/README.md)에 따라 command, 예상 결과, 검증 자료와 정리 방법을 기록합니다. 공통 안내는 `main`, backend별 구현은 해당 브랜치, dashboard 관측 실습은 Observatory에 둡니다.

## Repository Migration

저장소 이름은 `sft-lab`에서 `post-training-lab`으로, 기존 `post-training` 브랜치는 역할을 명확히 하기 위해 `system-integration`으로 변경했습니다. 기존 clone에서는 remote를 갱신합니다.

```bash
git remote set-url origin git@github.com:daegyu94/post-training-lab.git
git fetch origin --prune
```

로컬에 기존 `post-training` 브랜치가 있고 `system-integration` 브랜치가 아직 없는 경우에만 아래 명령을 실행하세요. 미완료 작업은 먼저 보존하고, 다른 worktree가 해당 브랜치를 사용 중이면 그 worktree에서 진행합니다.

```bash
git branch -m post-training system-integration
git branch --set-upstream-to=origin/system-integration system-integration
```

Observatory 저장소는 [`post-training-lab-observatory`](https://github.com/daegyu94/post-training-lab-observatory)로 변경했습니다. [새 dashboard 주소](https://daegyu94.github.io/post-training-lab-observatory/)를 사용하세요. 기존 GitHub Pages 주소는 자동 리디렉션되지 않습니다.

## Repository Policy

model weight, dataset cache, checkpoint, profiler trace처럼 큰 실행 산출물은 Git에 저장하지 않습니다. 각 브랜치에는 재현 가능한 command, configuration, 작은 summary와 결과 해석만 저장합니다. 합성 데이터, dry run과 실제 GPU 실행 결과를 구분하며, 기록된 결과는 해당 환경의 검증 자료로만 해석합니다.
