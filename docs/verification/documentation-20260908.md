# Documentation Review — 2026-09-08

기준 `main`은 `3e9a0a0dab0d79cc0636642f88fdbf19952a569d`입니다.
작업 브랜치는 `docs/restructure-20260908`이며 이전 문서 정리 commit `5d51b73c7e9087b5d8b71ba3c615ffec5a61b045`의 누락과 오류를 재검토했습니다.
구현을 문서에 맞춰 변경하지 않았습니다.

## Information Architecture

| 통합 전 영역 | 현재 위치 | 이유 |
| --- | --- | --- |
| README·setup·시작 안내 | [Getting Started](../getting-started.md) | GPU 없는 확인부터 결과 판정까지 하나의 진입 경로 |
| Backend별 설치·Spark 실행 | [TRL](../backends/trl.md), [Megatron](../backends/megatron.md) | 환경·실행·제약을 백엔드별로 연결 |
| 데이터 README | [Datasets](../datasets.md) | 양 백엔드 공통 변환을 한 곳에서 설명 |
| 반복 측정 가이드 | [Experiments](../experiments.md) | preset·실행 조건·측정 해석 분리 |
| Observability setup·실습 | [Observability](../observability.md) | monitoring·trace·baseline 실행 절차 통합 |
| 지표 schema·framework 설명 | [Reference](../observability-reference.md) | 실행 절차와 지표 계약의 역할 분리 |
| Lifecycle 설계 여러 파일 | [Design](../design.md) | 현재 미구현인 데이터·artifact·배포 제안을 하나로 통합 |
| 실측 결과가 섞인 backend 안내 | [Verification](../verification.md) 아래 날짜별 기록 | 현재 명령과 당시 환경·실패를 구분 |
| 통합·benchmark README 삭제분 | 해당 archive의 README 복구 | raw 파일만으로 알기 어려운 결과 해석 보존 |

기존 분산 문서와 SVG를 그대로 유지하지 않고 task guide와 두 Mermaid 흐름도로 재구성했습니다.
실험 수치·제약은 버리지 않고 날짜별 해설에 보존했으며 raw manifest·log·measurement는 변경하지 않았습니다.
통합 전 대형 checkpoint와 일부 runtime 로그는 원래부터 Git에 없으므로 존재하는 저장소 파일처럼 링크하지 않습니다.
삭제된 과거 문서는 Git history에서 복구할 수 있습니다.

## Corrections

Megatron의 공개 데이터 wrapper를 viewer 경로로 잘못 설명한 부분을 수정했습니다.
공개 변환은 두 백엔드가 같은 pinned preset을 지원하며, 서비스 변환의 manifest는 Spark 입력 형식과 다릅니다.
Megatron summary writer는 마지막 global rank의 로그를 사용하는 현재 동작에 맞췄습니다.
Grafana 비밀번호 예제의 잘못된 shell redirection을 제거하고 포트 공개·서비스 지속 실행·fio 쓰기 부작용을 명시했습니다.
설치 성공·학습 성공·수치 동등성·checkpoint durability를 구분했습니다.

## Validation Performed

GitHub에서 고정 commit의 코드·설정·테스트 파일을 가져와 검사했습니다.
로컬 Git checkout이 아닌 파일 snapshot이므로 원격 실행이나 commit provenance 검증을 수행한 것으로 보고하지 않습니다.

| 검사 | 결과 |
| --- | --- |
| 전체 CPU suite | Python 3.12, pytest 9.1.1, datasets 5.0.0에서 176 passed |
| Python compile | backend·experiments·observability·tests·summary helper 통과 |
| Shell script syntax | 23개 `bash -n` 통과 |
| Markdown | CommonMark parser와 table 확장으로 파싱 |
| 문서 code block | Bash 25개, JSON 2개, Python 1개 구문 검사 통과 |
| 내부 링크 | 최종 tree의 파일·디렉터리와 Markdown heading 대조 |
| Smoke preset | 네 개 dry-run 성공; 출력 디렉터리 생성 없음 |
| 서비스 가상 데이터 | 양 백엔드에서 train 2 / validation 1 / test 1 / excluded 1 |
| 공개 변환 CLI | 양 백엔드 `--help`와 예제 옵션 대조 |
| CPU parallelism | 논리 rank 16개, DP=2 확인 |
| Monitoring target | 3 files / 2 groups / 4 targets 형식 검사 |
| Benchmark schedule | 기본 80개, 축소 48개; 각각 별도 warmup 16개 확인 |
| 중복 | 현재 task guide의 긴 완전 중복 문단 없음 |

Dry-run의 `controller_commit`은 snapshot 환경에서 `unknown`이었습니다.
현재 runner는 이 경우 원격 commit 일치 검사를 생략하므로 실제 실행은 Git checkout에서 해야 합니다.
Benchmark CLI 전체 dry-run은 실제 Git HEAD가 필요하므로 실행하지 않았고 schedule 함수와 CPU 회귀 테스트를 확인했습니다.
문서의 다운로드 예제는 사용한 API와 인자를 검사했지만 모델·공개 데이터 다운로드를 실행하지 않았습니다.
Mermaid 두 개는 source를 검토했으며 렌더러 실행 검증은 하지 않았습니다.

## Not Executed

GPU 학습·NCCL, 모델 다운로드, Spark fresh install, Docker 서비스 시작, fio 쓰기 시험과 실제 checkpoint 재개는 수행하지 않았습니다.
과거 GPU 기록을 이번 작업의 실행 증거로 재사용하지 않습니다.
새 환경의 Megatron 의존성 충돌, selective recompute 설정, 서비스 manifest 호환성, TRL JSONL wrapper와 sharded 계측 제한은 [구현 한계](../verification.md#known-implementation-limits)에 별도로 남겼습니다.
