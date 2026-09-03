# DeepSeek V4 Claude Code mitigation for LiteLLM 1.99.0

This custom callback works around malformed Claude Code requests without changing LiteLLM. It supports `azure_ai/DeepSeek-V4-Flash` and `azure_ai/DeepSeek-V4-Flash-0731`, whether either deployment is selected directly or through Router fallback

Before routing, the callback moves first-turn `messages[].role == "system"` content into the top-level Anthropic `system` field. After an assistant turn, it changes dynamic system attachments into user-side `<system-reminder>` blocks. Once Router selects a deployment, a deployment hook removes `thinking` and `output_config.effort` only for the affected Azure DeepSeek models. Other `output_config` settings, such as structured-output formats, are retained

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
