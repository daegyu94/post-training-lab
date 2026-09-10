# Fresh Installation Audit

2026-09-11에 `1c0b03e`의 시작 가이드와 설치 명령을 검토했습니다.
현재 가이드만으로 두 백엔드의 새 GPU 환경을 문제없이 완성할 수 있다고 판정할 수 없습니다.
CUDA Torch 선택과 Megatron의 native dependency 설치 단계가 빠져 있으며 데이터 준비 경로에도 오류가 있습니다.
이번 변경은 요청한 검토 결과를 기록하며 설치 코드와 기존 가이드를 수정하지 않습니다.

## 검증 방법과 결과

Controller에서 별도 브랜치를 만들고 새 Python 가상환경으로 Getting Started의 TRL 실행 계획 명령을 실행했습니다.
Spark 노드에서는 기존 환경을 변경하지 않고 node-local 임시 디렉터리에 Python 3.12.3 가상환경을 만들어 문서의 설치 명령을 실행했습니다.
두 설치 모두 기본 인덱스의 `torch-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl` (146.0 MB)을 다운로드하는 단계에서 지연되어 해당 설치 process를 중단했습니다.
사용자 요청에 따라 각 노드의 기존 backend 가상환경을 별도 경로에 복사하고 복사본의 Python 절대 경로로 후속 검사를 실행했습니다.

| 검사 | 결과 | 판정 범위 |
| --- | --- | --- |
| Controller의 새 venv에서 TRL dry-run | 종료 코드 0, 두 rank 계획 생성 | 추가 Python package 없이 계획 생성 가능 |
| spark1의 새 TRL 설치 | Torch 다운로드 단계에서 중단 | 새 설치 완료 여부 미검증 |
| spark2의 새 Megatron 설치 | Bridge `--no-deps` 설치 후 requirements의 Torch 다운로드 단계에서 중단 | 새 설치 완료 여부 미검증 |
| TRL 복사본의 requirements 설치 | 종료 코드 0 | 기존 package를 포함한 환경에서 요구사항 충족 |
| Megatron 복사본의 requirements 설치 | 종료 코드 0 | 기존 package를 포함한 환경에서 요구사항 충족 |
| TRL 문서의 import 검사 | Torch `2.10.0+cu130`, CUDA `13.0`, CUDA available `True`; Transformers `5.12.1`, TRL `1.12.0`, Accelerate `1.14.0` | 복사 환경의 import와 GPU 탐지 성공 |
| Megatron 문서의 import 검사 | Torch `2.10.0+cu130`, CUDA available `True`; Bridge, Core, Transformer Engine, AutoBridge, LoRA import 성공 | 기존 native extension을 포함한 환경에서 성공 |
| 두 backend 복사본의 CUDA tensor 덧셈 | 각각 `[2.0, 2.0, 2.0, 2.0]` | 작은 CUDA 연산 성공, 학습·NCCL 검증 아님 |
| 기존 controller venv의 CPU 테스트 | `217 passed` | 저장소 회귀 검사 |
| `compileall` 및 지정된 shell script의 `bash -n` | 통과 | Python 컴파일·shell 문법 검사 |

가상환경 복사는 native library와 requirements 밖의 package도 함께 가져옵니다.
실제로 TRL 복사본에도 `megatron-bridge`가 들어 있었으므로 backend가 완전히 분리된 최소 환경 검증으로 해석할 수 없습니다.
복사된 `activate`와 실행 파일의 shebang은 원래 경로를 가리킬 수 있어 활성화나 bare `pip` 대신 복사본의 `bin/python -m pip`를 사용했고 `sys.prefix`가 복사본을 가리키는지 확인했습니다.

## 발견한 문제와 원인

### 1. CUDA Torch를 선택하는 실행 명령이 없음

[TRL 가이드](backends/trl.md#prepare-the-spark-environment)는 `pip install -r requirements-spark.txt`만 실행합니다.
[Megatron requirements](../backends/megatron/requirements-spark.txt)는 cu130 Torch를 먼저 설치하라고 주석에 적지만 [실제 설치 명령](backends/megatron.md#spark-environment)에는 그 단계가 없습니다.
두 파일의 `torch==2.10.0`은 CUDA build 선택을 고정하지 않으며, 이번 새 환경에서는 기존 환경의 `2.10.0+cu130`과 다른 wheel이 선택됐습니다.

기존 cu130 Torch가 설치되어 있으면 이 version 조건을 이미 만족하므로 재설치가 성공해도 누락된 선행 단계를 발견하지 못합니다.
설치 순서에 CUDA build 선택을 명시하고 설치 후 `torch.version.cuda`와 실제 CUDA 연산을 확인해야 합니다.
[PyTorch 공식 설치 안내](https://pytorch.org/get-started/previous-versions/)는 2.10.0의 CUDA 13.0 설치에 `--index-url https://download.pytorch.org/whl/cu130`을 지정합니다.
이번 기본 wheel의 다운로드는 중단했으므로 그 wheel을 import한 CUDA 판정 결과는 없습니다.

### 2. Megatron의 Transformer Engine 설치 단계가 없음

Megatron 가이드는 Transformer Engine `2.18.0`을 source build했다고 설명하지만 실제 설치 명령을 제공하지 않습니다.
requirements에도 이 package가 없고 Bridge를 `--no-deps`로 설치하므로 해당 단계에서 대신 설치되지 않습니다.
따라서 문서 끝의 `import transformer_engine` 검사를 통과하려면 별도 설치가 필요합니다.

기존 환경에는 `transformer_engine`과 `transformer_engine_torch` `2.18.0`이 모두 설치되어 있어 복사본 검사가 통과했습니다.
`spark_runtime_env.sh`는 library 경로와 `NVTE_WITH_NCCL_EP=0`을 설정할 뿐 package나 extension을 빌드하지 않습니다.
CUDA 개발 도구와 build dependency를 포함한 설치 순서를 문서화하고 빈 환경에서 다시 검증해야 합니다.
[NVIDIA 설치 문서](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html)의 build 절차는 참고할 수 있지만 이번 작업에서 해당 source build를 재현하지는 않았습니다.

### 3. 공개 데이터 준비 명령의 실행 위치가 모순됨

[Datasets의 변환 실행 설명](datasets.md#public-data)은 Megatron에서 첫 줄을 `cd backends/megatron`으로 바꾸라고 하지만 코드 블록에 그 첫 줄이 없습니다.
Getting Started의 모델 다운로드 단계도 backend 디렉터리에서 끝납니다.
그 위치에서 이어서 `bash scripts/prepare_public_data.sh --help`를 실행하면 실제로 `No such file or directory`와 종료 코드 127이 발생합니다.

해당 script는 저장소 루트의 `scripts/`에만 있습니다.
문서에서 저장소 루트로 돌아오는 명령을 명시해야 합니다.
또한 예시의 상대 경로 `data/public/no_robots`는 wrapper가 저장소 루트로 이동한 뒤 해석하므로 NFS checkout 아래에 데이터를 만듭니다.
node-local 데이터 원칙을 따르려면 `--output-dir`에 해당 노드의 로컬 절대 경로를 지정하고 그 경로를 setup의 `data_dir`에 연결해야 합니다.

### 4. 테스트용 venv가 원격 실행의 clean checkout 조건을 깨뜨림

[Verification](verification.md#repository-checks)은 저장소 내부에 `.venv-check`를 만듭니다.
하지만 `.gitignore`는 `.venv/`만 제외하며 `.venv-check/pyvenv.cfg`는 `git check-ignore`의 제외 대상이 아닙니다.
문서대로 테스트 환경을 만들면 추적되지 않은 파일이 생겨 checkout이 dirty 상태가 됩니다.

`experiments/run.py`의 `remote_command()`는 dirty checkout을 거부하므로 같은 NFS checkout에서 이후 smoke가 학습 전에 막힐 수 있습니다.
테스트 환경을 checkout 밖에 두거나 문서의 환경 이름과 ignore 규칙을 맞춰야 합니다.
이번 controller 검사는 checkout 밖의 임시 venv를 사용했습니다.

### 5. Bridge의 의존성 생략은 정상적인 전체 설치와 다름

두 복사본에서 `pip check`는 Bridge의 미설치 요구사항 9개를 보고했습니다.
목록은 `comet-ml`, `fast-hadamard-transform`, `flash-linear-attention`, `flashinfer-cubin`, `flashinfer-python`, `mlflow`, `open-clip-torch`, `qwen-vl-utils`, `timm`입니다.
이는 가이드가 채택한 `--no-deps` 경로의 결과이며, 이번 import 성공과 모순되지 않습니다.

특정 SFT 경로가 실행 가능하다는 사실과 package의 전체 의존성 계약을 충족한다는 사실은 다릅니다.
이 생략을 유지한다면 지원하는 모델·기능과 알려진 `pip check` 실패를 명시하고 해당 기능의 실행으로 검증해야 합니다.
기존 환경의 숨은 의존성이 새 설치에서도 제공된다고 가정해서는 안 됩니다.

## 남은 검증 범위

공통 준비 스크립트는 시스템 package, memlock 설정과 기존 가상환경에 영향을 주므로 기존 노드에서 재실행하지 않았습니다.
따라서 OS를 초기화한 머신에서의 전체 bootstrap, 새 환경의 전체 dependency 설치, 모델·데이터 신규 다운로드, 실제 SFT와 두 노드 NCCL은 이번 결과에 포함하지 않습니다.
두 backend의 CUDA 검사에는 GB10 capability `12.1`과 Torch의 지원 범위 `8.0–12.0`에 관한 경고가 있었지만 작은 덧셈 연산은 성공했습니다.
이 경고만으로 모든 학습 kernel의 성공이나 실패를 판정하지 않습니다.

추가로 `experiments/run.py`의 원격 명령은 Spark에서 `git rev-parse`와 `git status`를 실행합니다.
이는 모든 Git 명령을 controller에서만 실행하도록 한 현재 `AGENTS.md`와 충돌하므로 이번 검토에서는 공통 runner의 `--execute`를 실행하지 않았습니다.
운영 지침과 checkout 검증 구현을 일치시키는 별도 수정이 필요합니다.
