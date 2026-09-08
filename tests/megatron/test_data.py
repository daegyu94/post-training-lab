import pytest

from megatron_lab.data import select_conversations, validate_conversation


def row(prompt_id: str, answer: str = "Answer") -> dict:
    return {
        "prompt_id": prompt_id,
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": answer},
        ],
    }


def test_validate_conversation_accepts_user_assistant_pair() -> None:
    assert validate_conversation(row("one"))


def test_validate_conversation_rejects_empty_assistant() -> None:
    assert not validate_conversation(row("one", answer=""))


def test_select_conversations_deduplicates_prompt_ids() -> None:
    selected = select_conversations([row("one"), row("one"), row("two")], 2)
    assert [item["prompt_id"] for item in selected] == ["one", "two"]


def test_select_conversations_rejects_insufficient_rows() -> None:
    with pytest.raises(ValueError, match="found 1"):
        select_conversations([row("one")], 2)
