# Step 1: SFT Workflow and Baseline

이 저장소는 더 큰 multi-stage 로드맵(Step 1 → Step 2-1 medium-scale distributed post-training → Step 2-2 long-running agent workload → Step 3 1T+ model·4+ node Agentic RL)의 **Step 1(workflow·monitoring·profiling·baseline)만** 다룹니다.
Step 2 이후의 distributed filesystem 선정, train/rollout 분리, agent workload, RL 학습은 이 저장소의 범위가 아니며 별도 단계에서 다룹니다.

## Goal-to-Implementation Map

| Step 1 목표 | 상태 | 근거 |
| --- | --- | --- |
| GPU/CPU memory·utilization·disk I/O 자동 수집 + 공통 지표 정의 | 완전 반영 | [Observability](observability.md), `observability/config/metrics.json` |
| Container 포함 sandbox 자원 사용 규칙 수립 | 부분 반영 (CPU만 검증) | [Sandbox resource limits](../labs/sandbox-resource-limits/README.md) |
| 30B급 모델 GPU/CPU memory·storage baseline | 완전 반영 | [Verification](verification.md), [30B NVMe](../labs/nvme-30b/README.md) |
| 모델 확장 시 local disk vs remote storage pool 판단 기준 | 부분 반영 (memory-tier 근거는 실측, storage-topology 방향은 설계 논의만) | [30B NVMe: Why NVMe Offload Is Necessary](../labs/nvme-30b/README.md#why-nvme-offload-is-necessary-here), [Direction at Larger Scale](../labs/nvme-30b/README.md#direction-at-larger-scale-local-offload-remote-checkpoint) |
| 이후 단계에 재사용 가능한 profiling·analysis 기술 확보 | 완전 반영 | [Observability](observability.md) |
| 중형 모델급 환경 → 향후 simulation·scaling study baseline | 부분 반영 (방법론만, 모델 크기 다변화 없음) | [Experiments](experiments.md)의 Repeated Megatron Measurements |
| SFT 대표 workload로 end-to-end·distributed 확장 경로 검증 | 완전 반영 | [Verification](verification.md) |

"부분 반영"인 항목들은 알려진 gap이며, 각 문서의 Limitations 절(또는 해당 절)에 원인과 남은 범위가 적혀 있습니다.
Local disk vs remote storage pool 항목의 실제 throughput·contention 실측과 distributed filesystem 선택은 의도적으로 보류했습니다 — 그 결론은 checkpoint·node 규모가 훨씬 커지는 로드맵 Step 2-1/Step 3에서 의미가 생깁니다.
