from examples.deepseek_v4_claude_code_mitigation.callback import normalize_anthropic_request


def test_normalizes_first_turn_claude_code_request() -> None:
    request: dict[str, object] = {
        "model": "deepseek-v4-flash-0731",
        "system": [{"type": "text", "text": "base instructions", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "system", "content": [{"type": "text", "text": "available agents"}]},
        ],
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": {"type": "object"}},
        },
    }

    normalized = normalize_anthropic_request(request)

    assert normalized["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert normalized["system"] == [
        {"type": "text", "text": "base instructions", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "available agents"},
    ]
    assert "thinking" not in normalized
    assert normalized["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}


def test_converts_later_system_attachment_to_user_reminder() -> None:
    request: dict[str, object] = {
        "model": "deepseek-v4-flash-0731",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
            {
                "role": "system",
                "content": [{"type": "text", "text": "tokens remaining", "cache_control": {"type": "ephemeral"}}],
            },
        ],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
    }

    normalized = normalize_anthropic_request(request)

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
    assert "thinking" not in normalized
    assert "output_config" not in normalized


def test_leaves_other_models_unchanged() -> None:
    request: dict[str, object] = {
        "model": "another-model",
        "messages": [{"role": "system", "content": "unchanged"}],
        "thinking": {"type": "adaptive"},
    }

    assert normalize_anthropic_request(request) is request
