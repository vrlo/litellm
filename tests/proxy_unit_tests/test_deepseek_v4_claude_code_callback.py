import pytest

from examples.deepseek_v4_claude_code_mitigation.callback import (
    normalize_anthropic_system_messages,
    remove_unsupported_deepseek_thinking,
)


def test_normalizes_first_turn_before_fallback_selection() -> None:
    request: dict[str, object] = {
        "model": "primary-model-group",
        "system": [{"type": "text", "text": "base instructions", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "system", "content": [{"type": "text", "text": "available agents"}]},
        ],
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "medium"},
    }

    normalized = normalize_anthropic_system_messages(request)

    assert normalized["model"] == "primary-model-group"
    assert normalized["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert normalized["system"] == [
        {"type": "text", "text": "base instructions", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "available agents"},
    ]
    assert normalized["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert normalized["output_config"] == {"effort": "medium"}


def test_converts_later_system_attachment_to_user_reminder() -> None:
    request: dict[str, object] = {
        "model": "primary-model-group",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
            {
                "role": "system",
                "content": [{"type": "text", "text": "tokens remaining", "cache_control": {"type": "ephemeral"}}],
            },
        ],
    }

    normalized = normalize_anthropic_system_messages(request)

    assert normalized["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "next"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<system-reminder>tokens remaining</system-reminder>",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]


@pytest.mark.parametrize(
    "selected_model",
    ["azure_ai/DeepSeek-V4-Flash", "azure_ai/DeepSeek-V4-Flash-0731"],
)
def test_removes_thinking_after_direct_or_fallback_deployment_selection(selected_model: str) -> None:
    selected_deployment: dict[str, object] = {
        "model": selected_model,
        "messages": [{"role": "user", "content": "hello"}],
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": {"type": "object"}},
        },
    }

    normalized = remove_unsupported_deepseek_thinking(selected_deployment)

    assert "thinking" not in normalized
    assert normalized["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}


def test_leaves_non_deepseek_deployment_thinking_unchanged() -> None:
    selected_deployment: dict[str, object] = {
        "model": "azure_ai/another-model",
        "thinking": {"type": "adaptive"},
    }

    assert remove_unsupported_deepseek_thinking(selected_deployment) is selected_deployment
