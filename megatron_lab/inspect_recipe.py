"""Qwen2.5-7B LoRA Megatron Bridge recipe를 실행 없이 요약한다."""

from __future__ import annotations

import json

from megatron.bridge.recipes.qwen import qwen25_7b_peft_config


def value(obj: object, name: str) -> object:
    return getattr(obj, name, None)


def main() -> None:
    cfg = qwen25_7b_peft_config(peft_scheme="lora")
    model = value(cfg, "model")
    train = value(cfg, "train")

    summary = {
        "mode": "recipe inspection only: no checkpoint download, CUDA, or distributed initialization",
        "recipe": "qwen25_7b_peft_config(peft_scheme='lora')",
        "model_parallel": {
            "tensor_model_parallel_size": value(model, "tensor_model_parallel_size"),
            "pipeline_model_parallel_size": value(model, "pipeline_model_parallel_size"),
            "context_parallel_size": value(model, "context_parallel_size"),
        },
        "training": {
            "micro_batch_size": value(train, "micro_batch_size"),
            "global_batch_size": value(train, "global_batch_size"),
        },
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
