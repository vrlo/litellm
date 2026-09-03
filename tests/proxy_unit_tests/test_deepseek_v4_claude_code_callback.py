import pytest

from examples.deepseek_v4_claude_code_mitigation.callback import (
    DeepSeekV4ClaudeCodeMitigation,
    deepseek_v4_claude_code_mitigation,
    normalize_anthropic_system_messages,
    normalize_deepseek_deployment,
    stamp_requested_model_group,
)
from litellm.types.utils import CallTypes


def test_callback_constructor_accepts_explicit_model_groups() -> None:
    callback = DeepSeekV4ClaudeCodeMitigation(frozenset({"primary-model-group"}))

    assert callback.allowed_model_groups == frozenset({"primary-model-group"})
    assert callback.message_logging is True


def test_normalizes_first_turn_after_fallback_selection() -> None:
    request: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "system": [{"type": "text", "text": "base instructions", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "system", "content": [{"type": "text", "text": "available agents"}]},
        ],
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "medium"},
        "litellm_params": {"metadata": {"model_group": "deepseek-v4-fallback"}},
    }

    normalized = normalize_deepseek_deployment(request)

    assert normalized["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<system-reminder>available agents</system-reminder>",
                }
            ],
        },
    ]
    assert normalized["system"] == [
        {"type": "text", "text": "base instructions", "cache_control": {"type": "ephemeral"}}
    ]
    assert normalized["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert normalized["output_config"] == {"effort": "medium"}


def test_converts_later_system_attachment_to_user_reminder() -> None:
    request: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
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

    normalized = normalize_deepseek_deployment(request)

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
def test_supports_both_selected_deployments(selected_model: str) -> None:
    selected_deployment: dict[str, object] = {
        "model": selected_model,
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "agents"},
        ],
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": {"type": "object"}},
        },
    }

    normalized = normalize_deepseek_deployment(selected_deployment)

    assert normalized["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert normalized["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }


def test_promotes_only_contiguous_leading_system_messages() -> None:
    request: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "messages": [
            {"role": "system", "content": "leading"},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "later"},
        ],
    }

    normalized = normalize_deepseek_deployment(request)

    assert normalized["system"] == [{"type": "text", "text": "leading"}]
    assert normalized["messages"] == [
        {"role": "user", "content": "hello"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "<system-reminder>later</system-reminder>"}],
        },
    ]


def test_empty_top_level_system_does_not_abort_normalization() -> None:
    request: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "system": [],
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "agents"},
        ],
    }

    normalized = normalize_anthropic_system_messages(request)

    assert "system" not in normalized
    assert normalized["messages"] == [
        {"role": "user", "content": "hello"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "<system-reminder>agents</system-reminder>"}],
        },
    ]


@pytest.mark.parametrize("empty_content", [[], "", "   "])
def test_empty_system_message_content_is_dropped(empty_content: object) -> None:
    request: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": empty_content},
        ],
    }

    normalized = normalize_anthropic_system_messages(request)

    assert "system" not in normalized
    assert normalized["messages"] == [{"role": "user", "content": "hello"}]


def test_empty_text_blocks_are_removed_without_dropping_nonempty_blocks() -> None:
    request: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "instructions", "cache_control": {"type": "ephemeral"}},
                ],
            },
            {"role": "user", "content": "hello"},
        ],
    }

    normalized = normalize_anthropic_system_messages(request)

    assert normalized["system"] == [{"type": "text", "text": "instructions", "cache_control": {"type": "ephemeral"}}]
    assert normalized["messages"] == [{"role": "user", "content": "hello"}]


def test_all_system_messages_fail_open_instead_of_producing_empty_messages() -> None:
    request: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "messages": [{"role": "system", "content": "instructions"}],
    }

    assert normalize_anthropic_system_messages(request) is request


def test_unsupported_system_blocks_fail_open_instead_of_being_stringified() -> None:
    request: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": [{"type": "image", "source": "unsupported"}]},
        ],
    }

    assert normalize_anthropic_system_messages(request) is request


def test_does_not_normalize_valid_messages() -> None:
    selected_deployment: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "system": "base instructions",
        "messages": [{"role": "user", "content": "hello"}],
    }

    assert normalize_deepseek_deployment(selected_deployment) is selected_deployment


@pytest.mark.asyncio
async def test_anthropic_deployment_pass_applies_mitigation() -> None:
    request: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "agents"},
        ],
        "thinking": {"type": "adaptive"},
    }

    result = await deepseek_v4_claude_code_mitigation.async_pre_call_deployment_hook(
        request,
        CallTypes.anthropic_messages,
    )

    assert result is not request
    assert result["thinking"] == {"type": "adaptive"}
    assert result["messages"] == [
        {"role": "user", "content": "hello"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "<system-reminder>agents</system-reminder>"}],
        },
    ]


@pytest.mark.asyncio
async def test_second_acompletion_pass_preserves_translated_system_prompt() -> None:
    translated_request: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "messages": [
            {"role": "system", "content": "important instructions"},
            {"role": "user", "content": "hello"},
        ],
    }

    result = await deepseek_v4_claude_code_mitigation.async_pre_call_deployment_hook(
        translated_request,
        CallTypes.acompletion,
    )

    assert result is translated_request
    assert result["messages"] == translated_request["messages"]


@pytest.mark.asyncio
async def test_native_chat_completion_is_not_modified() -> None:
    request: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "messages": [
            {"role": "system", "content": "important instructions"},
            {"role": "user", "content": "hello"},
        ],
        "thinking": {"type": "adaptive"},
    }

    result = await deepseek_v4_claude_code_mitigation.async_pre_call_deployment_hook(
        request,
        CallTypes.acompletion,
    )

    assert result is request
    assert "thinking" in result


def test_leaves_non_deepseek_deployment_unchanged() -> None:
    selected_deployment: dict[str, object] = {
        "model": "azure_ai/another-model",
        "messages": [{"role": "system", "content": "unchanged"}],
        "thinking": {"type": "adaptive"},
    }

    assert normalize_deepseek_deployment(selected_deployment) is selected_deployment


def test_optional_model_group_allowlist_accepts_fallback_origin() -> None:
    selected_deployment: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "agents"},
        ],
        "thinking": {"type": "adaptive"},
        "litellm_params": {
            "litellm_metadata": {
                "model_group": "deepseek-v4-fallback",
                "original_model_group": "primary-model-group",
            }
        },
    }

    normalized = normalize_deepseek_deployment(selected_deployment, frozenset({"primary-model-group"}))

    assert normalized["thinking"] == {"type": "adaptive"}
    assert normalized["messages"] == [
        {"role": "user", "content": "hello"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "<system-reminder>agents</system-reminder>"}],
        },
    ]


def test_ingress_stamp_overwrites_client_supplied_model_group_marker() -> None:
    ingress_request: dict[str, object] = {
        "model": "unlisted-group",
        "litellm_metadata": {"deepseek_v4_mitigation_requested_model_group": "primary-model-group"},
    }
    original_metadata = ingress_request["litellm_metadata"]
    stamped = stamp_requested_model_group(ingress_request)

    assert stamped is ingress_request
    assert stamped["litellm_metadata"] is original_metadata
    selected_deployment: dict[str, object] = {
        **stamped,
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "agents"},
        ],
        "thinking": {"type": "adaptive"},
    }

    assert normalize_deepseek_deployment(selected_deployment, frozenset({"primary-model-group"})) is selected_deployment


def test_optional_model_group_allowlist_rejects_unlisted_group() -> None:
    selected_deployment: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "agents"},
        ],
        "thinking": {"type": "adaptive"},
        "litellm_params": {"metadata": {"model_group": "unlisted-group"}},
    }

    assert normalize_deepseek_deployment(selected_deployment, frozenset({"primary-model-group"})) is selected_deployment
