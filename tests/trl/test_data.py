from trl_lab.data import QWEN_ASSISTANT_MASK_TEMPLATE, validate_conversation


def test_template_marks_assistant_generation() -> None:
    assert "generation %}" in QWEN_ASSISTANT_MASK_TEMPLATE
    assert "endgeneration %}" in QWEN_ASSISTANT_MASK_TEMPLATE


def test_validate_conversation_accepts_user_assistant_pair() -> None:
    example = {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }
    assert validate_conversation(example)


def test_validate_conversation_rejects_empty_content() -> None:
    example = {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": ""},
        ]
    }
    assert not validate_conversation(example)
