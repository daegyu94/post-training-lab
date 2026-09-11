# Post-Training Dashboard Plan

이 문서는 post-training 실행을 관측하는 Grafana 개선 계획이며, 현재 구현된 기능을 설명하는 사용 설명서가 아닙니다.
SFT부터 향후 agentic RL까지 실행 진행, 연산·통신, 데이터·스토리지 병목을 같은 실행 맥락에서 조사하도록 구성합니다.
Checkpoint와 NFS는 각각 작업 단계와 저장소 구현의 한 사례이며 전체 화면 구조를 결정하지 않습니다.

## Baseline and Reuse

계획의 코드 기준은 로컬 `main`의 `3c06f31`입니다.
현재 동작과 실행 방법은 [Observability](observability.md), 지표 정의는 [Observability Reference](observability-reference.md)를 따릅니다.
실행 중인 Grafana의 배포 버전·노출 지표·렌더링은 별도 검증 대상입니다.

| 확인 대상 | 현재 구현 | 개선 방향 |
| --- | --- | --- |
| `observability/examples/observability/spark-resources.json` | GPU·메모리·TCP·RDMA·학습 시계열, 단일 UID `spark-profiling` | Overview 진입점을 유지하고 상세 화면 분리 |
| `observability/scripts/run_spark_observability.sh` | Spark 대시보드 한 파일 복사, 두 노드 scrape, retention 1일 | 여러 대시보드 provisioning, 실제 확장 시 scrape 대상 목록화 |
| `observability/profiling_lab/spark_telemetry.py` | GPU 사용률·전력·온도·클럭, 일부 process memory | 기존 지표로 GPU 행렬과 health 상세 구성 |
| `observability/profiling_lab/framework_metrics.py` | rank별 JSON에 `local_rank`, `observed_at` 기록 | 실제 GPU 식별자·실행 역할·상태 연결 보완 |
| `observability/profiling_lab/framework_metrics_textfile.py` | loss·throughput·step·timer 노출, 읽는 시각을 freshness로 사용 | 원본 sample 시각과 collector 상태 분리 |
| 일반 Compose 예시 | 별도 `cluster-resources.json`에 CPU·disk busy 포함 | 쿼리를 참고하되 Spark provisioning 대상과 혼동하지 않음 |

기존 `post-training-lab-observatory`의 `index.html`과 `api/runs/*/{resources,interconnect,storage,data-movement}.json`에서 GPU tile·inspector, 연결 행렬, storage 상세의 탐색 방식을 참고합니다.
해당 화면의 `static-mock-api` 데이터는 합성 예시이며 실제 계측의 증거가 아닙니다.
별도 frontend나 mock API를 이식하지 않고 기존 exporter·textfile writer·Grafana 기본 패널을 우선 재사용합니다.
다른 작업 브랜치의 미병합 계측 코드는 이 계획의 의존성으로 삼지 않으며, 구현 시 병합 여부와 재사용 가능성을 확인합니다.

## Dashboard Responsibilities

평소에는 Run Overview 하나를 열고, 이상 구간에서 상세 화면으로 이동합니다.
같은 시계열을 여러 화면에 복제하기보다 Overview에는 요약과 이동 링크를, 상세에는 비교·추이를 둡니다.

| 화면 | 질문 | 소유하는 내용 |
| --- | --- | --- |
| Run Overview | 실행이 진행되고 있으며 어디서 지연되는가? | run·workload 요약, 사용 가능한 품질·처리율 지표, 단계별 시간, GPU 사용률 행렬, 메모리·수집 상태 요약 |
| Compute & Communication | 어떤 역할·노드·GPU·rank·연결이 병목인가? | allocation, GPU health, rank·역할별 timer, TCP·RDMA 포트별 상세, compute interconnect topology와 측정 행렬 |
| Data & Storage | 어떤 데이터 경로·저장 계층이 병목인가? | client I/O, local device·filesystem, remote storage topology, metadata·data 등 역할별 서비스 상태, 데이터 작업별 지연 |

SFT에서는 loss·tokens/s·step time을 먼저 연결합니다.
Agentic RL이 구현되면 rollout·reward·episode 완료·tool 대기·queue·policy update·weight sync 중 실제 제공되는 지표를 연결합니다.
Tool 실행이나 외부 서비스 대기는 GPU 또는 storage 사용량과 별개의 실행 지연으로 다루며, 현재 없는 지표를 위한 빈 패널이나 RL backend를 이번 작업에서 만들지 않습니다.
SFT와 RL의 공통 상세 화면은 공유하고, 실제 관측 흐름이 달라질 때 Overview 분리를 검토합니다.

## Context and Measurement Semantics

화면 이동은 시간 범위와 관련 선택값을 보존해야 합니다.
공통 변수는 `cluster`, `run_id`, `node`를 우선하고, 상세 화면에서 필요한 `gpu`, `role`, `device`, `mount`, `storage_system`을 실제 지표에 맞춰 추가합니다.
기존 `instance`·`nodename`·framework의 `node`가 같은 노드를 가리키도록 명시적으로 매핑하고, 빈 선택·복수 선택·All도 검증합니다.

- Run에 연결한 host 지표는 해당 노드·시간대의 관측값입니다.
  프로세스나 cgroup 귀속 근거가 없으면 run 단독 사용량으로 표시하지 않습니다.
- GPU 사용률 0%는 미할당을 뜻하지 않습니다.
  Allocation은 run·역할·rank와 실제 GPU 식별자의 명시적 매핑으로 표시하며, `local_rank`를 물리 GPU index로 간주하지 않습니다.
- GPU UUID 매핑은 장치 visibility를 반영합니다.
  여러 역할·프로세스가 같은 GPU를 사용하는 관계를 허용하고, 종료 상태 또는 관측이 끊긴 상태를 구분합니다.
- 원본 `observed_at`, collector 생존, run lifecycle은 서로 다른 신호입니다.
  오래된 snapshot을 반복해서 읽어도 최신 실행이나 활성 allocation으로 보이지 않아야 하며, 긴 step을 종료로 단정하지 않습니다.
- 미수집·미지원·오래된 값·실제 0을 구분합니다.
  GPU inventory를 기준으로 셀을 유지하고, Spark UMA의 host memory와 GPU allocation을 합산하지 않습니다.
- Snapshot 행렬은 instant query로 중복 시점을 제거합니다.
  노드마다 GPU 수가 다를 수 있고, 큰 행렬은 선택한 그룹 범위로 좁힙니다.
- 포트별 송수신 counter는 endpoint 쌍의 트래픽이 아닙니다.
  All-reduce 결과도 GPU 쌍별 bandwidth·latency로 변환하지 않습니다.
- 지표 단위는 계약을 따르며 bytes/s와 bits/s를 구분합니다.
  Disk counter의 평균 지연은 p95가 아니고, 공유 filesystem 용량은 client별로 합산하지 않습니다.
- Episode·request·trace ID와 전체 경로를 고카디널리티 Prometheus label로 추가하지 않습니다.
  상세 관계와 실행 설정은 manifest·진단 산출물에서 관리합니다.

## Storage and Topology Inputs

Data & Storage는 dataset, 모델 가중치, trajectory·replay 데이터, checkpoint, 실행 산출물 등 실제 데이터 작업을 관측합니다.
작업별 계측이 없다면 client·장치 단위 사용량으로 표시하고 특정 작업에 귀속하지 않습니다.
NFS·pNFS·3FS 등의 연결은 실제 사용 시스템의 구성 입력과 계측 기능을 확인한 뒤 진행합니다.

Topology와 성능 지표는 별도 입력으로 받아 안정적인 구성요소 ID로 연결합니다.
다음은 필요한 정보의 초안이며, 실제 입력 제공자와 예시를 확인하기 전 공통 adapter framework나 최종 schema를 구현하지 않습니다.

| 입력 | 필요한 정보 | 표시 원칙 |
| --- | --- | --- |
| 구성요소 | storage system ID, component ID, 제공된 역할, host/service 관계 | metadata·data 역할을 고정된 두 종류의 물리 서버로 가정하지 않음 |
| 연결 | source·destination ID, 관계 종류 | 관계가 있다는 사실과 해당 경로의 실측 트래픽을 구분 |
| Client 경로 | client node, mount 또는 논리 경로, storage system ID | workload의 데이터 접근과 storage 상세를 연결 |
| 출처·시점 | topology 출처, 관측·갱신 시각 | 오래된 구성과 미확인 상태를 표시 |
| 성능 | 실제 exporter/API metric과 component ID 매핑 | 계측된 구성요소에만 수치 표시 |

구성만 있으면 구조와 역할을 보여주고, 지표만 있으면 해당 ID의 지표를 보여주되 관계를 추측하지 않습니다.
동일 서버의 여러 역할, 일부 누락된 구성요소, client·서버 추가를 허용합니다.
Topology 표 또는 Grafana 기본 시각화 중 실제 입력을 표현하는 가장 단순한 방식을 선택합니다.
Compute interconnect도 같은 증거 구분 원칙을 따르지만 storage topology와 별개 관계로 관리합니다.

## Implementation Sequence

각 단계는 독립적으로 검토 가능한 변경으로 진행합니다.
지표·설정 계약은 구현이 필요한 단계에서 갱신하고, 계획만으로 기존 실행 계약을 바꾸지 않습니다.

1. **기존 지표로 세 화면 구성**
   `spark-resources.json`의 UID를 유지해 Overview로 정리하고, Compute 및 Data 상세 JSON을 추가합니다.
   GPU 사용률 행렬, 온도·클럭 상세, 포트별 네트워크, CPU·로컬 disk·filesystem을 기존 지표로 연결합니다.
   `run_spark_observability.sh`는 명시한 세 dashboard 파일을 provisioning하고, 공통 변수·링크로 시간 범위를 전달합니다.
   GPU inventory가 아직 없는 단계에서는 사라진 시계열을 정상·미할당으로 해석하지 않도록 제한을 표시합니다.
   완료 기준은 기존 학습 지표 보존, 세 화면 탐색, GPU별 표시, 장치·mount별 구분입니다.
2. **Freshness와 allocation 신뢰성**
   기존 framework snapshot과 GPU sampler에 필요한 최소 정보만 추가합니다.
   원본 timestamp, inventory·UUID, run·역할·rank 매핑, 종료·관측 중단 처리의 데이터 흐름을 producer부터 panel까지 연결합니다.
   `framework_metrics.py`, `framework_metrics_textfile.py`, `spark_telemetry.py` 및 관련 backend 호출부를 함께 확인합니다.
   완료 기준은 stale snapshot, 장치 재매핑, GPU 공유, 종료 후 allocation이 올바르게 구분되는 것입니다.
3. **제공된 topology와 추가 계측 연결**
   실제 topology 입력 예시와 exporter/API를 확인하고 storage·compute 관계를 연결합니다.
   Storage 구현별 collector는 실제 대상 하나부터 추가하고, 기존에 병합된 계측 결과를 우선 재사용합니다.
   Endpoint별 성능 행렬은 측정 조건·시각·단위가 있는 결과를 확보한 경우에만 추가합니다.
   완료 기준은 구성 변경·부분 계측을 표시하고 counter 합계로 가짜 edge 성능을 만들지 않는 것입니다.
4. **Workload와 규모 확장**
   실제 agentic RL producer가 준비되면 역할·phase·데이터 작업 지표를 기존 화면에 연결합니다.
   노드 확장 시 scrape 대상을 목록화하고, 장시간 관측 시 retention·수집기 실행 기간을 요구에 맞춥니다.
   현재 GPU sampler의 유한 실행 기간과 `wait -n`에 따른 서비스 종료를 확인하며 dashboard를 상시 서비스로 오인하지 않도록 합니다.
   RL 구현이나 대규모 storage adapter 일괄 작성은 이 단계의 선행 작업이 아닙니다.

## Validation and Handoff

첫 구현은 저장소의 Grafana 버전과 실제 exporter 출력으로 쿼리·변환 지원을 확인하는 것부터 시작합니다.
문서상의 기능 가능성과 설치된 Grafana에서의 렌더링 성공을 구분합니다.

| 단계 | 검증 | 성공이 의미하지 않는 것 |
| --- | --- | --- |
| 문서 준비 | 상대 링크·경로, diff, 기준 코드와의 일치 | dashboard 구현·배포 완료 |
| 정적·CPU | JSON, UID·링크·변수·provisioning 검사, 변경 logic의 회귀 테스트 | Prometheus 실측값 존재 |
| 실제 monitoring | target 상태, metric label·단위, freshness, 쿼리 결과 | GPU workload 정상 수행 |
| 브라우저 | 세 화면, 선택·시간 전달, 행렬, N/A·stale, GPU 수가 다른 노드와 여러 storage | 병목 원인 확정 |
| 실제 workload | Spark에서 해당 workload 관측, phase·역할·자원 관계 확인 | 미실행 RL 또는 다른 storage 검증 |

코드 구현 후 기본 검사는 `python -m pytest -q`, `python -m compileall -q backends datasets_lab experiments observability tests`와 프로젝트 지침에 지정된 shell script의 `bash -n`입니다.
새 설정 allowlist가 필요하면 `docs/architecture.md`와 `docs/experiments.md`의 계약도 확인합니다.
측정 부하를 발생시키는 baseline·trace는 읽기 전용 monitoring 확인과 분리하고, 실제 LLM 연산은 Spark 노드에서 수행합니다.
모델·실행 산출물·로컬 설정은 커밋하지 않습니다.

계획 단계에서는 dashboard JSON·exporter·배포 설정을 변경하지 않습니다.
후속 구현자는 1단계부터 시작하고, 3단계에 필요한 실제 topology 입력과 4단계의 RL 지표는 제공 시점에 구체화합니다.
