import pytest

from examples.deepseek_v4_claude_code_mitigation.callback import (
    normalize_deepseek_deployment,
    stamp_requested_model_group,
)


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

    assert normalized["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert normalized["system"] == [
        {"type": "text", "text": "base instructions", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "available agents"},
    ]
    assert "thinking" not in normalized
    assert "output_config" not in normalized


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

    assert "thinking" not in normalized
    assert normalized["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}


def test_does_not_normalize_valid_messages() -> None:
    selected_deployment: dict[str, object] = {
        "model": "azure_ai/DeepSeek-V4-Flash-0731",
        "system": "base instructions",
        "messages": [{"role": "user", "content": "hello"}],
    }

    assert normalize_deepseek_deployment(selected_deployment) is selected_deployment


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

    assert "thinking" not in normalized
    assert normalized["messages"] == [{"role": "user", "content": "hello"}]


def test_ingress_stamp_overwrites_client_supplied_model_group_marker() -> None:
    ingress_request: dict[str, object] = {
        "model": "unlisted-group",
        "litellm_metadata": {"deepseek_v4_mitigation_requested_model_group": "primary-model-group"},
    }
    stamped = stamp_requested_model_group(ingress_request)
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
