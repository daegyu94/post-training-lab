import json
from pathlib import Path

import pytest

from megatron_lab.cluster_data import prepare_cluster_data, render_prompt_completion


class CharTokenizer:
    eos_token = "<eos>"
    eos_token_id = 999

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "|".join(f"<{message['role']}>{message['content']}" for message in messages) + "<assistant>"

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        index = 0
        while index < len(text):
            if text.startswith(self.eos_token, index):
                ids.append(self.eos_token_id)
                index += len(self.eos_token)
            else:
                ids.append(ord(text[index]))
                index += 1
        return {"input_ids": ids}


def _row(prompt_id="row-1"):
    return {
        "prompt_id": prompt_id,
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."},
        ],
        "provenance": {"source": "fixture"},
    }


def test_native_render_preserves_prompt_completion_boundary_and_eos():
    example = render_prompt_completion(_row(), CharTokenizer())
    assert example["completion"].endswith("<eos>")
    assert example["supervised_tokens"] > 0
    assert example["prompt_tokens"] > 0
    assert example["provenance"] == {"source": "fixture"}


def test_native_render_rejects_structured_tool_calls():
    row = _row()
    row["messages"][1]["tool_calls"] = [{"name": "never-run"}]
    with pytest.raises(ValueError, match="structured tool calls"):
        render_prompt_completion(row, CharTokenizer())


def test_native_render_rejects_missing_final_assistant():
    row = _row()
    row["messages"][-1]["role"] = "user"
    with pytest.raises(ValueError, match="end with assistant"):
        render_prompt_completion(row, CharTokenizer())


def test_native_render_rejects_boundary_tokenization_change():
    class BoundaryTokenizer(CharTokenizer):
        def __call__(self, text, *, add_special_tokens=False):
            if text == "<eos>":
                return {"input_ids": [999]}
            if text.endswith("<eos>") and not text.startswith("<system>"):
                return {"input_ids": [2, 999]}
            if text.startswith("<system>"):
                return {"input_ids": [1, 7, 2, 999]}
            return super().__call__(text, add_special_tokens=add_special_tokens)

    with pytest.raises(ValueError, match="prefix|boundary"):
        render_prompt_completion(_row(), BoundaryTokenizer())


def test_native_preparation_rejects_overlong_without_truncation(tmp_path: Path):
    row = _row()
    row["messages"][-1]["content"] = "x" * 100
    data_dir = tmp_path / "source"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(
        json.dumps({"dataset": "fixture", "dataset_revision": "rev"}), encoding="utf-8"
    )
    (data_dir / "training.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (data_dir / "validation.jsonl").write_text(json.dumps(_row("eval")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="never truncates"):
        prepare_cluster_data(
            data_dir / "training.jsonl",
            data_dir / "validation.jsonl",
            tmp_path / "prepared",
            CharTokenizer(),
            32,
            dataset_id="fixture",
            dataset_revision="rev",
            model_revision="model-rev",
        )


def test_native_preparation_skips_boundary_tokenization_rows_instead_of_aborting(tmp_path: Path, capsys):
    class BoundaryTokenizer(CharTokenizer):
        """Returns an inconsistent id list for exactly row-1's rendered prompt; everything
        else (including row-2-ok's strings) falls through to real, consistent per-char coding."""

        BAD_PROMPT = "<system>Be concise.|<user>What is 2+2?<assistant>"

        def __call__(self, text, *, add_special_tokens=False):
            if text == self.BAD_PROMPT:
                return {"input_ids": [1, 2, 3]}
            return super().__call__(text, add_special_tokens=add_special_tokens)

    ok_row = {
        "prompt_id": "row-2-ok",
        "messages": [
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "6."},
        ],
        "provenance": {"source": "fixture"},
    }
    data_dir = tmp_path / "source"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(
        json.dumps({"dataset": "fixture", "dataset_revision": "rev"}), encoding="utf-8"
    )
    train = data_dir / "training.jsonl"
    valid = data_dir / "validation.jsonl"
    train.write_text(
        json.dumps(_row("row-1")) + "\n" + json.dumps(ok_row) + "\n",
        encoding="utf-8",
    )
    valid.write_text(json.dumps({**ok_row, "prompt_id": "eval"}) + "\n", encoding="utf-8")

    manifest = prepare_cluster_data(
        train,
        valid,
        tmp_path / "prepared",
        BoundaryTokenizer(),
        512,
        dataset_id="fixture",
        dataset_revision="rev",
        model_revision="model-rev",
    )

    assert manifest["counts"] == {"train": 1, "validation": 1}
    prepared_ids = {
        json.loads(line)["prompt_id"]
        for line in (tmp_path / "prepared" / "training.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert prepared_ids == {"row-2-ok"}
    assert "skipping train row" in capsys.readouterr().err


def test_native_preparation_keeps_source_and_writes_provenance(tmp_path: Path):
    data_dir = tmp_path / "source"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(
        json.dumps({"dataset": "fixture", "dataset_revision": "rev"}), encoding="utf-8"
    )
    train = data_dir / "training.jsonl"
    valid = data_dir / "validation.jsonl"
    train.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    valid.write_text(json.dumps(_row("eval")) + "\n", encoding="utf-8")
    source_before = train.read_bytes()
    manifest = prepare_cluster_data(
        train,
        valid,
        tmp_path / "prepared",
        CharTokenizer(),
        512,
        dataset_id="fixture",
        dataset_revision="rev",
        model_revision="model-rev",
    )
    assert train.read_bytes() == source_before
    assert manifest["counts"] == {"train": 1, "validation": 1}
    assert manifest["supervision"] == "final_assistant_completion_only"
    prepared = (tmp_path / "prepared" / "training.jsonl").read_text(encoding="utf-8")
    assert "<eos>" in prepared


def test_native_preparation_rejects_source_alias(tmp_path: Path):
    data_dir = tmp_path / "source"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(
        json.dumps({"dataset": "fixture", "dataset_revision": "rev"}), encoding="utf-8"
    )
    train = data_dir / "training.jsonl"
    valid = data_dir / "validation.jsonl"
    train.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    valid.write_text(json.dumps(_row("eval")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="separate"):
        prepare_cluster_data(
            train,
            valid,
            data_dir,
            CharTokenizer(),
            512,
            dataset_id="fixture",
            dataset_revision="rev",
            model_revision="model-rev",
        )
