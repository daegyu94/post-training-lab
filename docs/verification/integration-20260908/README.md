# Unified Layout Verification

이 기록은 backend 디렉터리 이동 후 Spark에서 실제 실행한 검증입니다.
통합 전 30B 결과는 [TRL 기록](../../backends/trl/training-verification.md)과 [Megatron 기록](../../backends/megatron/spark-cluster.md)에 보존하며 이번 결과와 구분합니다.

## Layout Regression

| 실행 | Commit | 결과 |
| --- | --- | --- |
| TRL, two-node DDP LoRA, Qwen2.5-0.5B | `e8e04b8` | base 평가, 학습 1 step, adapter 저장, tuned 재로딩 평가; 양 rank exit 0 |
| Megatron, two-node full SFT, Qwen2.5-0.5B | `e8e04b8` | base 평가, 학습 2 step, checkpoint 저장, tuned 재로딩 평가, summary; 양 rank exit 0 |
| Observability telemetry | `e8e04b8` | 양 Spark 노드에서 3초 수집 완료 |

두 backend 모두 No Robots의 pinned revision과 max length 2048을 사용했습니다.
Megatron의 base/tuned eval loss는 각각 `3.152960`과 `3.145808`입니다.
이는 저장·재로딩 경로의 실행 확인이며 모델 품질 개선 검증이 아닙니다.

[TRL 실행 설정·종료 코드](trl-layout-execution.json), [Megatron 실행 설정·종료 코드](megatron-layout-execution.json), [Megatron summary](megatron-layout-summary.json)를 함께 보존합니다.
TRL의 stage별 summary는 같은 디렉터리의 `trl-layout-summary-{base,train,tuned}.json`입니다.
원본 rank logs와 checkpoint timing JSONL도 같은 디렉터리에 있습니다.
대형 checkpoint와 model weight는 Git에 포함하지 않습니다.

## Findings and Fixes

Megatron의 첫 설정에서 max length 256은 canonical row 길이 검사에 의해 학습 전에 거부됐습니다.
이 loader는 임의 truncation을 허용하지 않으며, 기존 검증 조건인 2048로 맞췄습니다.

Bridge는 평가 loss를 마지막 global rank에 출력합니다.
기존 launcher가 rank 0 로그만 읽어 summary 생성에 실패하던 문제를 수정하고, 마지막 노드에서 해당 rank의 로그와 rank별 metadata를 읽도록 변경했습니다.
`e8e04b8` 실행의 원본 summary에는 이전 `summary_writer_rank=0` 필드와 잘못된 training log 경로가 남아 있습니다.
후속 commit `74c7097`에서 writer node와 metrics rank를 구분하고 실제 rank training log 경로 및 micro batch 전달을 수정했습니다.
원본 실행 기록은 수정하지 않습니다.

## Runtime

양 노드의 기존 TRL 환경 안에 Megatron도 설치되어 있었으며, 이를 복제한 독립 Megatron 환경에서 검증했습니다.
[Megatron 환경 버전](megatron-environment.json)과 [TRL 핵심 버전](trl-environment.json)을 기록합니다.
새로운 전체 dependency resolution은 선택 패키지의 build dependency 문제를 만났으므로 fresh install 전체가 검증됐다고 표시하지 않습니다.
ARM64 Transformer Engine wheel과 userspace library 설정은 [Spark backend 가이드](../../backends/megatron/spark-cluster.md)를 따릅니다.

[Telemetry spark1](telemetry-spark1.jsonl), [Telemetry spark2](telemetry-spark2.jsonl)에서 GB10의 일부 NVML memory field는 unavailable로 기록됩니다.
이 null 값을 memory 사용량 0으로 해석하지 않습니다.
GPU memory peak는 학습 process의 CUDA allocator 계측과 별도로 다룹니다.

## Interpretation

성능 측정, optimizer resume correctness와 checkpoint durability는 서로 다른 검증입니다.
`save` timing은 synchronous 저장 호출 또는 asynchronous enqueue 호출의 host 시간입니다.
`finalize_async_saves` timing은 blocking 여부를 함께 기록하며 fsync나 장애 후 복구 보장을 의미하지 않습니다.

## Common Runner Regression

| 실행 | Commit | 결과 |
| --- | --- | --- |
| TRL, Spark1 single-node DDP LoRA, 1 step | `6d42a4f` | base/train/tuned 완료, exit 0, 49.147초 |
| Megatron, Spark1+2 full SFT, 2 → 3 steps | `ccfb787` | base/train/resume/tuned 및 summary 완료, 양 rank exit 0, 272.115초 |

실행 manifest, rank log, summary와 계측은 [runner/](runner/)에 보존합니다.
TRL preset의 최초 실행은 dataset revision 불일치로 학습 전에 거부됐고, No Robots revision으로 수정한 뒤 통과했습니다.
Megatron의 최초 공통 실행기 실행은 NFS에서 peer session 식별자를 5초 안에 확인하지 못해 학습 전에 중단됐습니다.
대기 상한을 run timeout 이내 최대 70초로 늘린 뒤 같은 모델·학습 조건이 완료됐습니다.

Megatron 재개 실행은 iteration 2 checkpoint를 로드해 iteration 3을 수행하고 optimizer state 로딩 경로를 실행했습니다.
이는 중단 없는 3-step 학습과의 수치 동등성 검증이나 장애 후 checkpoint durability 검증은 아닙니다.
최종 base/tuned eval loss는 `3.152960`과 `3.152162`이며, 작은 smoke의 품질 개선으로 해석하지 않습니다.

## CPU and Static Checks

통합 commit `ecd1e50`에서 Spark Python 환경으로 전체 CPU 테스트 173개가 통과했습니다.
Controller에서는 Python compileall, shell script 23개의 `bash -n`, 추적 중인 Markdown의 로컬 링크 존재 검사를 통과했습니다.
CPU·정적 검사 성공은 GPU 실행이나 새로운 dependency 환경 설치 성공을 의미하지 않습니다.

## Sequence Length Measurement Correction

Bridge 0.6의 `DirectHFSFTDatasetConfig`는 `pad_to_max_length=False`가 기본값입니다.
짧은 동일 dataset에서 `MAX_LENGTH` 상한만 2048/4096으로 바꾸는 것은 실제 tensor 길이 비교가 되지 않으므로, 길이 실험에는 고정 길이 padding을 명시합니다.
초기 matrix는 이 문제를 확인한 뒤 중단했고, 완료된 micro batch 1 overlap 비교만 유지합니다.
나머지 조건은 수정된 설정으로 별도 실행하여 commit과 결과 경로를 구분합니다.
이 의도적인 중단은 backend 지원 실패로 해석하지 않습니다.

Padding 및 중단 처리 수정 후 commit `8626241`에서 전체 CPU 테스트 176개가 통과했습니다.
원본 결과는 [cpu-tests-padding.txt](cpu-tests-padding.txt)에 보존합니다.

현재 설치된 Bridge의 `training/train.py` checkpoint wrapper는 저장 호출 전 `interval-time`을 멈추고 저장 후 재시작합니다.
따라서 native iteration timer와 저장 호출 계측을 별도로 보고합니다.
Async I/O의 자원 경쟁이 학습 시간에 주는 영향은 제거하지 않으며, 저장 호출 계측에는 동기화·filesystem 대기가 포함될 수 있습니다.

## Recompute Limitation

고정 길이 2048과 4096의 selective recompute warmup은 학습 전에 `ValueError: When using recompute_granularity: selective recompute_num_layers must be None.`으로 실패했습니다.
현재 backend는 selective에도 `recompute_num_layers=1`을 설정하므로 Bridge의 configuration 검증을 통과하지 못합니다.
이는 GPU 메모리 부족이나 Spark에서 selective recompute가 원천적으로 불가능하다는 증거가 아닙니다.
이 variant는 반복 비교에서 제외하고 full 측정만 별도로 기록합니다.

## Repeated Measurements

[반복 측정 결과와 원본 기록](benchmarks/README.md)에 8개 조건의 결과를 정리했습니다.
실제 실행안은 16 optimizer step, variant별 2회 반복, within-run warmup 4 step 제외이며 기본 후보인 64 step·4회 반복을 실행한 것은 아닙니다.
Checkpoint 크기와 공유 저장소 비용을 고려해 이 축소 실행안을 사용했습니다.

선택된 48개 run 중 42개가 통과했고, selective warmup 2개가 설정 오류로 실패했으며 해당 variant의 후속 반복 4개는 생략했습니다.
성공한 측정 반복은 28개이며 나머지 성공 14개는 별도 warmup입니다.
Overlap, full recompute, sequence parallel과 sync/async checkpoint 경로는 이 작은 모델의 실행을 확인했습니다.
Selective가 실패했으므로 full/selective 비교 완료로 표시하지 않습니다.

Async checkpoint는 짧은 저장 주기에서 이전 저장이 끝나기 전에 다음 저장 요청이 발생했다는 경고를 출력했습니다.
일부 종료 로그에는 CUDA IPC tensor 해제 경고도 남아 있으나 해당 run의 종료 코드는 0이었고 저장 finalization을 완료했습니다.
이 기록은 fsync 또는 장애 후 복구 보장을 의미하지 않습니다.
