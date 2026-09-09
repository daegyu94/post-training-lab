import json
from pathlib import Path

import pytest

from trl_lab.spark_data import (
    prepare_prompt_completion_data,
    render_prompt_completion,
)


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 999

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        return "|".join(f"{item['role']}:{item['content']}" for item in messages) + "|assistant:"

    def __call__(self, text, **kwargs):
        ids = list(range(max(1, len(text))))
        if text.endswith(self.eos_token):
            ids[-1] = self.eos_token_id
        return {"input_ids": ids}


class FakeQwenTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return "<|im_start|>" + "".join(f"{item['role']}\\n{item['content']}<|im_end|>" for item in messages) + "<|im_start|>assistant\\n"


class FakeGlmTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return "<|user|>" + "".join(item["content"] for item in messages) + "<|assistant|>"


def row(prompt_id="one"):
    return {"prompt_id": prompt_id, "messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}], "provenance": {"dataset": "public/data"}}


def test_native_prompt_and_final_completion_are_separate() -> None:
    result = render_prompt_completion(row(), FakeTokenizer())
    assert result["prompt"].endswith("assistant:")
    assert result["completion"] == "Answer<eos>"
    assert result["supervised_tokens"] > 0


@pytest.mark.parametrize("tokenizer", [FakeQwenTokenizer(), FakeGlmTokenizer()])
def test_native_templates_keep_prompt_as_joint_tokenization_prefix(tokenizer) -> None:
    result = render_prompt_completion(row(), tokenizer)
    prompt_ids = tokenizer(result["prompt"], add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(result["prompt"] + result["completion"], add_special_tokens=False)["input_ids"]
    assert full_ids[:len(prompt_ids)] == prompt_ids
    assert full_ids[-1] == tokenizer.eos_token_id
    assert result["supervised_tokens"] == len(full_ids) - len(prompt_ids)


def test_non_assistant_final_turn_fails() -> None:
    bad = row()
    bad["messages"][-1]["role"] = "user"
    with pytest.raises(ValueError, match="end with assistant"):
        render_prompt_completion(bad, FakeTokenizer())


def test_unknown_prompt_role_and_boundary_merge_fail() -> None:
    bad_role = row()
    bad_role["messages"][0]["role"] = "tool"
    with pytest.raises(ValueError, match="unsupported message role"):
        render_prompt_completion(bad_role, FakeTokenizer())

    class BoundaryMergingTokenizer(FakeTokenizer):
        def __call__(self, text, **kwargs):
            if text.endswith("assistant:"):
                return {"input_ids": [1, 2]}
            return {"input_ids": [7, self.eos_token_id]}

    with pytest.raises(ValueError, match="not a prefix"):
        render_prompt_completion(row(), BoundaryMergingTokenizer())


def test_final_structured_call_fails_but_null_field_is_not_a_call() -> None:
    structured = row()
    structured["messages"][-1]["tool_calls"] = [{"name": "search"}]
    with pytest.raises(ValueError, match="structured tool calls"):
        render_prompt_completion(structured, FakeTokenizer())
    nullable = row()
    nullable["messages"][-1]["tool_calls"] = None
    assert render_prompt_completion(nullable, FakeTokenizer())["completion"] == "Answer<eos>"


def test_loader_checks_manifest_and_split_disjointness(tmp_path: Path) -> None:
    (tmp_path / "training.jsonl").write_text(json.dumps(row("train")) + "\n", encoding="utf-8")
    (tmp_path / "validation.jsonl").write_text(json.dumps(row("eval")) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"dataset": "public/data", "dataset_revision": "a" * 40}), encoding="utf-8")
    paths, metadata = prepare_prompt_completion_data(
        tmp_path,
        tmp_path / "prepared",
        FakeTokenizer(),
        "public/data",
        "a" * 40,
        512,
    )
    assert paths["train"].is_file() and paths["validation"].is_file()
    assert metadata["supervised_tokens"]["train"] > 0
    assert metadata["actual_selected_rows"] == {"train": ["train"], "validation": ["eval"]}


def test_requested_rows_must_be_available(tmp_path: Path) -> None:
    (tmp_path / "training.jsonl").write_text(json.dumps(row("train")) + "\n", encoding="utf-8")
    (tmp_path / "validation.jsonl").write_text(json.dumps(row("eval")) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"dataset": "public/data", "dataset_revision": "a" * 40}), encoding="utf-8")
    with pytest.raises(ValueError, match="fewer than requested"):
        prepare_prompt_completion_data(
            tmp_path,
            tmp_path / "prepared",
            FakeTokenizer(),
            "public/data",
            "a" * 40,
            512,
            train_samples=2,
        )


def test_preparation_streams_disk_backed_trainer_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "training.jsonl").write_text(json.dumps(row("train")) + "\n", encoding="utf-8")
    (source / "validation.jsonl").write_text(json.dumps(row("eval")) + "\n", encoding="utf-8")
    (source / "manifest.json").write_text(
        json.dumps({"dataset": "public/data", "dataset_revision": "a" * 40}),
        encoding="utf-8",
    )

    paths, metadata = prepare_prompt_completion_data(
        source,
        tmp_path / "prepared",
        FakeTokenizer(),
        "public/data",
        "a" * 40,
        512,
    )

    assert json.loads(paths["train"].read_text()) == {
        "prompt": "user:Question|assistant:",
        "completion": "Answer<eos>",
    }
    assert metadata["loading"].startswith("disk-backed Arrow")
    assert metadata["actual_selected_rows_truncated"] == {
        "train": False,
        "validation": False,
    }
