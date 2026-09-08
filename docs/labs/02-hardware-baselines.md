# Lab 02: Hardware Baselines

## Goal

실제 training과 같은 node allocation에서 collective network와 checkpoint storage의 정상 범위를 먼저 측정합니다.
이후 application 성능이 낮을 때 framework 문제와 infrastructure 문제를 구분할 비교 기준을 만듭니다.

## 1. NCCL All-reduce Baseline

`nccl-tests`를 MPI 지원으로 build한 뒤 scheduler가 할당한 node에서 실행합니다.
예시는 node당 같은 수의 GPU가 있는 homogeneous allocation을 가정합니다.

```bash
export NCCL_TEST_BINARY=/opt/nccl-tests/build/all_reduce_perf
export HOSTS=node-a,node-b
export GPUS_PER_NODE=8
export OUTPUT_DIR=artifacts/nccl-2nodes-16gpus
./scripts/run_nccl_baseline.sh
```

보존할 값은 message size별 latency, `algbw`, `busbw`, NCCL/CUDA/driver version, NIC와 GPU topology입니다.
최소한 single-node와 multi-node 결과를 따로 측정해 node 내부 fabric과 node 간 fabric을 분리합니다.

확인 질문은 다음과 같습니다.

- multi-node `busbw`가 동일 cluster의 과거 정상 범위보다 낮은가?
- 특정 message size에서만 급락하는가?
- GPU/NIC binding 또는 topology 변경과 결과가 같이 바뀌는가?
- 반복 run의 분산이 커서 shared fabric contention이 의심되는가?

## 2. Checkpoint-like Storage Baseline

예제 fio job은 4 MiB block, queue depth 16의 direct sequential write/read를 수행합니다.
실제 checkpoint writer의 block size, concurrency와 shared/local filesystem 특성에 맞게 job file을 수정해야 합니다.

```bash
export FIO_DIRECTORY=/path/on/filesystem-under-test
export FIO_SIZE=16G
export FIO_RUNTIME=60
export OUTPUT_DIR=artifacts/fio-node-a
./scripts/run_fio_baseline.sh
```

스크립트는 read test와 재검증을 위해 `profiling-lab-checkpoint.bin`을 자동 삭제하지 않습니다.
테스트가 끝난 뒤 출력된 정확한 경로를 확인하고 수동으로 제거합니다.

각 node에서 동시에 수행하면 shared filesystem contention을 측정할 수 있지만 production workload와 같은 storage를 방해할 수 있으므로 격리된 profiling allocation과 승인된 시간대에서만 실행합니다.

## 3. Compare With Application Metrics

NCCL Tests와 fio 값은 training throughput이 아닙니다.
다음처럼 원인 분류에 사용합니다.

| Observation | Baseline | Likely next step |
| --- | --- | --- |
| communication time 증가 | NCCL baseline도 저하 | fabric, routing, NIC/GPU binding과 error 확인 |
| communication time 증가 | NCCL baseline 정상 | rank skew, tensor size, overlap와 framework trace 확인 |
| checkpoint 지연 증가 | fio도 저하 | filesystem/backend contention 확인 |
| checkpoint 지연 증가 | fio 정상 | serialization, rank coordination, page cache와 writer count 확인 |

## Expected Result

application을 실행하지 않은 상태의 network/storage 기준과 topology가 artifact로 남아야 합니다.
이후 run에서 같은 allocation과 조건의 baseline만 비교합니다.
