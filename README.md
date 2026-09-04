# Megatron Bridge Qwen2.5 SFT Lab

이 브랜치는 Megatron Bridge로 `Qwen/Qwen2.5-14B-Instruct`를 `HuggingFaceH4/ultrachat_200k`에 LoRA fine-tuning하는 SFT 예제만 포함합니다.
UltraChat의 `train_sft`와 `test_sft`를 분리하고, assistant token에만 loss를 적용합니다.

## 검증 범위

실제 학습 launcher는 `CUDA_VISIBLE_DEVICES`로 GPU 한 장만 노출합니다.
학습 전 held-out loss, optimizer step, 학습 후 같은 held-out loss, native Megatron checkpoint의 새 프로세스 재로딩을 순서대로 검증합니다.
TP와 DP는 실제 GPU를 점유하지 않고 rank topology만 시뮬레이션합니다.

현재 장비의 GPU는 장당 24GiB입니다.
Megatron Bridge의 14B LoRA는 quantized base model을 사용하지 않으므로 이 장비의 단일 GPU에는 들어가지 않습니다.
스크립트는 최소 40GiB를 보수적인 시작 조건으로 검사하지만, 통과가 학습 성공을 보장하지는 않습니다.
공식 recipe가 대상으로 하는 H100 80GiB급 GPU에서 실행하는 것을 권장합니다.

## 1. 환경 설치

Python 3.12 환경이 필요합니다.

```bash
./scripts/setup.sh
```

## 2. UltraChat 준비

기본값은 학습 32개와 평가 8개 대화입니다.
두 subset은 각각 `train_sft`와 `test_sft`에서 가져오므로 서로 섞이지 않습니다.

```bash
./scripts/prepare_data.sh
```

생성 파일은 `data/ultrachat_200k/training.jsonl`과 `validation.jsonl`이며 Git에 포함되지 않습니다.

## 3. Qwen checkpoint 준비

다음 명령은 약 28GB의 BF16 model weight를 내려받으므로 여유 공간을 먼저 확인합니다.

```bash
./scripts/download_model.sh
```

이미 받은 Hugging Face checkpoint가 있으면 `MODEL_DIR`로 지정할 수 있습니다.

## 4. 단일 GPU smoke run

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_smoke.sh
```

두 번째 물리 GPU를 쓰더라도 프로세스에는 그 GPU 한 장만 보입니다.

```bash
CUDA_VISIBLE_DEVICES=1 ./scripts/run_smoke.sh
```

## 실제 실행 흐름

`run_smoke.sh`는 다음 순서를 고정합니다.

1. GPU가 정확히 한 장만 보이는지, memory와 model/data 파일이 준비됐는지 검사합니다.
2. 새 프로세스에서 pretrained checkpoint를 읽고 `test_sft` loss를 측정합니다.
3. 새 프로세스에서 `train_sft`로 LoRA adapter를 5 optimizer step 학습하고 native Megatron checkpoint를 저장합니다.
4. 다시 새 프로세스를 시작해 base checkpoint와 저장된 adapter checkpoint를 불러오고 같은 `test_sft` loss를 측정합니다.
5. 두 log의 `lm loss value`와 perplexity를 `summary.json`에 기록합니다.

성공 판정의 핵심은 `tuned_eval_loss < base_eval_loss`입니다.
Step loss는 batch마다 달라 단조 감소하지 않아도 되지만 `nan`이나 무한대가 없어야 합니다.
마지막 평가는 학습 프로세스와 분리되어 있으므로 checkpoint 재로딩도 함께 확인합니다.

## 결과 파일

```text
results/qwen2.5-14b-megatron-smoke/
|-- base-eval.log
|-- train.log
|-- tuned-eval.log
|-- checkpoints/
`-- summary.json
```

## TP/DP simulation

다음 명령은 GPU를 초기화하지 않고 world size 2의 두 배치를 모두 출력합니다.

```bash
./scripts/simulate_parallelism.sh
```

TP 배치에서는 rank 0과 1의 `tensor_parallel_rank`가 다르고 `data_parallel_rank`는 같습니다.
DP 배치에서는 `tensor_parallel_rank`가 같고 `data_parallel_rank`가 다릅니다.
이는 process-group 산술 검증이며 NCCL 통신이나 처리량 검증은 아닙니다.

## 이 장비에서 실제로 확인한 콘솔

UltraChat subset 준비도 실제 공개 split을 대상으로 실행했습니다.

```console
$ ./scripts/prepare_data.sh
[data] dataset=HuggingFaceH4/ultrachat_200k train_split=train_sft eval_split=test_sft
[data] saved train=32 path=data/ultrachat_200k/training.jsonl
[data] saved evaluation=8 path=data/ultrachat_200k/validation.jsonl
[data] overlap=0
```

하드웨어 preflight는 GPU 한 장만 노출한 상태에서 실행했습니다.

```console
$ CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m megatron_lab.preflight --hardware-only
[preflight] CUDA_VISIBLE_DEVICES=1 visible_gpus=1
[preflight] gpu=NVIDIA RTX PRO 4000 Blackwell total_memory_gib=23.42
[blocked] Qwen2.5-14B Megatron LoRA requires at least 40GiB for this lab; found 23.42GiB
```

TP/DP simulation과 unit test도 실제로 실행한 결과입니다.

```console
$ ./scripts/simulate_parallelism.sh
{
  "tensor_parallel": {
    "world_size": 2,
    "tensor_parallel": 2,
    "data_parallel": 1,
    "ranks": [
      {
        "global_rank": 0,
        "tensor_parallel_rank": 0,
        "data_parallel_rank": 0
      },
      {
        "global_rank": 1,
        "tensor_parallel_rank": 1,
        "data_parallel_rank": 0
      }
    ]
  },
  "data_parallel": {
    "world_size": 2,
    "tensor_parallel": 1,
    "data_parallel": 2,
    "ranks": [
      {
        "global_rank": 0,
        "tensor_parallel_rank": 0,
        "data_parallel_rank": 0
      },
      {
        "global_rank": 1,
        "tensor_parallel_rank": 0,
        "data_parallel_rank": 1
      }
    ]
  }
}
```

```console
$ .venv/bin/python -m pytest -q
.........                                                                [100%]
9 passed in 0.02s
```

현재 장비에서는 preflight가 memory 부족을 올바르게 차단했으므로 14B 학습 loss나 checkpoint 재로딩 성공을 기록하지 않습니다.
실행하지 않은 수치를 예시 결과로 제시하지 않습니다.

## 참고 자료

- [Megatron Bridge Qwen 지원 및 recipe](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/qwen/qwen.html)
- [Megatron Bridge text SFT data](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/hf-text-only/README.md)
- [UltraChat 200k dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
