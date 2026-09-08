# Public SFT datasets

이 문서는 UltraChat 외 public dataset을 같은 bounded preparation 경로로 canonical conversational JSONL로 바꾸는 호환성 기준입니다. 실제 GPU training이나 GLM assistant-mask 통과를 주장하지 않으며, dataset schema·license·split provenance와 token/loss-mask 검증은 별도 단계입니다.

## Presets and source schema

| Preset | Task/domain | Hub source and source split | Source schema | License/access |
| --- | --- | --- | --- | --- |
| `ultrachat` | natural-language dialogue | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), `train_sft` | native `messages` | MIT per official card |
| `no_robots` | human-written general instruction | [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots), `train` | native `messages`, `prompt_id`, `category` | CC BY-NC 4.0; non-commercial limitation |
| `self_oss` | synthetic coding instruction | [bigcode/self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k), `train` | `instruction`, `response`, `id`, `concepts` | ODC-By |
| `xlam` | synthetic function calling | [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), config `dataset`, `train` | `query` string, `tools`/`answers` JSON strings | CC BY 4.0; gated access, do not accept terms automatically |

`bigcode`의 `prompt`는 source card가 설명하는 generation prompt이므로 adapter는 task instruction을 의미하는 `instruction`만 user content로 사용합니다. `no_robots`는 official `train`/`test` split을 제공하지만 preparation은 benchmark-test leakage를 피하기 위해 `train`에서만 deterministic validation holdout을 만듭니다.

XLAM adapter는 `tools`와 `answers`를 JSON parse 후 stable canonical JSON으로 보존하여 system/user prompt와 assistant JSON 응답으로 만듭니다. 이는 JSON-output instruction SFT이며 native tool-call protocol, 실행 가능한 agent rollout 또는 tool execution이 아니고, malformed JSON·tool schema·answer schema는 조용히 버리지 않고 즉시 실패합니다.

Source-to-canonical examples:

```jsonl
{"prompt_id": "n1", "messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}], "provenance": {"preset": "no_robots"}}
```

```jsonl
{"id": 7, "instruction": "Write a function.", "response": "def f(): pass"}
{"prompt_id": "7", "messages": [{"role": "user", "content": "Write a function."}, {"role": "assistant", "content": "def f(): pass"}], "provenance": {"preset": "self_oss"}}
```

```jsonl
{"query": "Find 42", "tools": "[{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}]", "answers": "[{\"name\":\"lookup\",\"arguments\":{\"value\":42}}]"}
{"prompt_id": "source-id", "messages": [{"role": "system", "content": "Available tools (JSON): [...]\\nReturn the selected call(s) as JSON."}, {"role": "user", "content": "Find 42"}, {"role": "assistant", "content": "[{\"name\":\"lookup\",\"arguments\":{\"value\":42}}]"}], "provenance": {"preset": "xlam"}}
```

SWE-agent trajectories는 [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)의 `instance_id`, `model_name`, `target`, `trajectory`, `exit_status`, `generated_patch`, `eval_logs` reference schema만 기록합니다. trajectory는 system/ai/user action-observation 구조이고 repository별 license와 model/data terms가 섞일 수 있으므로 이 repository에서는 canonical SFT adapter를 제공하지 않습니다. coding instruction dataset과 coding-agent trajectory를 같은 것으로 취급하지 않습니다.

## Canonical output and provenance

각 output row는 `prompt_id`, `messages`, `provenance`를 포함하며 모든 canonical message는 `role`과 non-empty string `content`를 갖고 assistant로 끝납니다. Structured `tool_calls`는 현재 adapter에서 허용하지 않으며 대상 tokenizer의 native template와 assistant loss mask는 training 전 실제 probe로 확인해야 합니다.

`prepare_public_data.py`는 Hub revision이 branch/tag이면 Hub API로 immutable 40-hex commit SHA를 resolve하고, 이미 SHA이면 그대로 사용합니다. Streaming은 source train split에서 bounded scan만 읽고 seed `42`와 prompt-content fingerprint로 중복을 제거하며, deterministic disjoint validation holdout을 만듭니다. Source test/benchmark split은 읽지 않습니다.

`manifest.json`에는 dataset/config, resolved revision, source split/schema, adapter version, license, seed, scan bound, source IDs, counts와 각 JSONL SHA-256이 기록됩니다. 같은 prompt가 서로 다른 source ID를 가져도 assistant target을 제외한 input fingerprint가 같으면 한 split에만 남습니다.

실제 pinned streaming smoke는 별도 Megatron Spark 환경에서 `no_robots`와 BigCode self-oss 각각 train 8/validation 2, overlap 0, exit 0으로 확인했으며 이 branch의 GPU training evidence는 아닙니다.

## Preparation commands

TRL branch:

```bash
./scripts/prepare_public_data.sh --preset no_robots --output-dir data/public/no_robots --revision e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b --train-count 32 --eval-count 8 --seed 42
./scripts/prepare_public_data.sh --preset self_oss --output-dir data/public/self_oss --revision 356bb069eee815daa6e23e9a282eeefe1490ad44 --train-count 32 --eval-count 8 --seed 42
./scripts/prepare_public_data.sh --preset xlam --output-dir data/public/xlam --revision 26d14ebfe18b1f7b524bd39b404b50af5dc97866 --train-count 32 --eval-count 8 --seed 42
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m sft_lab.train --model Qwen/Qwen2.5-14B-Instruct --dataset bigcode/self-oss-instruct-sc2-exec-filter-50k --dataset-jsonl-dir data/public/self_oss --local-files-only --train-samples 32 --eval-samples 8 --max-steps 1 --output-dir results/public-self-oss
```

XLAM preparation may require an already authorized Hub account; this script does not accept gated terms or bypass access controls. `no_robots`는 non-commercial license이므로 commercial use 전에 별도 법무·권리 검토가 필요합니다.

## Input compatibility

| Input form | Current TRL loader | Current Megatron loader | Boundary |
| --- | --- | --- | --- |
| canonical `messages` JSONL | compatible | compatible | project adapters require string content and assistant final turn |
| prompt/completion pair | not accepted directly | not accepted directly | upstream libraries may support it, but these project paths do not |
| raw text | not accepted directly | not accepted directly | requires an explicit adapter |
| preference `chosen/rejected` | not accepted | not accepted | requires a preference trainer/adapter |
| native structured tools | not accepted | not accepted | XLAM is flattened JSON-output SFT only |
| SWE trajectory | reference-only | reference-only | trajectory/action-observation schema is not canonical chat |

## Compatibility and evaluation boundaries

Canonical JSONL compatibility는 source schema 변환과 loader 입력을 뜻할 뿐, model-specific chat template, truncation, non-zero assistant mask, EOS alignment 또는 quality를 보장하지 않습니다. Preparation 후 sample별 token count·truncation·assistant mask를 확인하고, 같은 model/tokenizer/finetuning mode/optimizer/LR/sequence 및 supervised-token budget으로 base와 tuned를 비교합니다.

서로 다른 dataset의 자체 held-out loss를 순위화하지 않습니다. Cross-domain 비교는 공통 multi-domain held-out suite와 domain별 loss, 자연어 rubric, code held-out tests, JSON validity, tool name/argument accuracy를 별도 보고합니다. 이 adapter는 generated code를 실행하지 않으며 xLAM tool도 실행하지 않습니다.
