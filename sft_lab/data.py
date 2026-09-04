"""Dataset loading and chat-template helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


QWEN_ASSISTANT_MASK_TEMPLATE = r"""
{%- if messages[0]['role'] != 'system' %}
    {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'assistant' %}
        {{- '<|im_start|>assistant\n' }}
        {%- generation %}{{- message['content'] + '<|im_end|>' }}{%- endgeneration %}
        {{- '\n' }}
    {%- else %}
        {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
""".strip()


def validate_conversation(example: dict[str, Any]) -> bool:
    """Return whether an example is a usable user/assistant conversation."""
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    roles = [message.get("role") for message in messages]
    if roles[0] not in {"system", "user"} or "assistant" not in roles:
        return False
    return all(
        isinstance(message.get("content"), str) and message["content"].strip()
        for message in messages
    )


def _load_split(dataset_name: str, split: str, parquet_dir: Path | None) -> Dataset:
    if parquet_dir is None:
        return load_dataset(dataset_name, split=split)
    files = sorted(parquet_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split}-*.parquet files under {parquet_dir}")
    return load_dataset("parquet", data_files={split: [str(path) for path in files]}, split=split)


def load_ultrachat(
    dataset_name: str,
    train_samples: int,
    eval_samples: int,
    seed: int,
    parquet_dir: Path | None = None,
) -> tuple[Dataset, Dataset]:
    """Load deterministic, disjoint UltraChat SFT subsets."""
    train = _load_split(dataset_name, "train_sft", parquet_dir)
    evaluation = _load_split(dataset_name, "test_sft", parquet_dir)
    train = train.select_columns(["messages"])
    evaluation = evaluation.select_columns(["messages"])
    train = train.filter(validate_conversation).shuffle(seed=seed)
    evaluation = evaluation.filter(validate_conversation).shuffle(seed=seed)
    if train_samples > len(train) or eval_samples > len(evaluation):
        raise ValueError("Requested sample count exceeds the available validated split")
    return train.select(range(train_samples)), evaluation.select(range(eval_samples))
