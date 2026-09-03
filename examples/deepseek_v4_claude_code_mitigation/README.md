# DeepSeek V4 Claude Code mitigation for LiteLLM 1.99.0

This custom callback works around malformed Claude Code requests without changing LiteLLM. It supports `azure_ai/DeepSeek-V4-Flash` and `azure_ai/DeepSeek-V4-Flash-0731`, whether either deployment is selected directly or through Router fallback

After Router selects a deployment, the callback runs only for the two affected Azure DeepSeek models. It normalizes a request only when `messages[]` contains an invalid `system` role. First-turn system content moves into the top-level Anthropic `system` field, while later dynamic system attachments become user-side `<system-reminder>` blocks. The same deployment hook removes `thinking` and `output_config.effort`. Other `output_config` settings, such as structured-output formats, are retained

Set `DEEPSEEK_V4_MITIGATION_MODEL_GROUPS` to add an optional model-group gate. The comma-separated comparison is case-insensitive and accepts either the selected fallback group or its original model group:

```bash
export DEEPSEEK_V4_MITIGATION_MODEL_GROUPS="primary-model-group,deepseek-v4-flash,deepseek-v4-flash-0731"
```

When the variable is unset, selected-deployment matching and the invalid system-role check remain the only normalization gates

For example, the newer deployment can be configured as a fallback without changing the callback:

```yaml
router_settings:
  fallbacks:
    - primary-model-group:
        - deepseek-v4-flash-0731
```

Run the proxy from the repository root:

```bash
export AZURE_AI_API_KEY="$(security find-generic-password -a cody-eu -s foundry-io -w)"
uv run litellm --config examples/deepseek_v4_claude_code_mitigation/config.yaml --port 4000
```

Test the affected request shape:

```bash
curl --no-buffer http://127.0.0.1:4000/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-api-key: local-test' \
  -d '{
    "model": "deepseek-v4-flash-0731",
    "max_tokens": 128,
    "stream": true,
    "system": [{"type": "text", "text": "You are a concise assistant."}],
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "Reply with exactly: MITIGATION_OK"}]},
      {"role": "system", "content": [{"type": "text", "text": "Available agent types: general-purpose"}]}
    ],
    "thinking": {"type": "adaptive", "display": "omitted"},
    "output_config": {"effort": "medium"}
  }'
```

The response should contain `MITIGATION_OK`, no unrelated task prefix, and no visible `</think>` text
