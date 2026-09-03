# DeepSeek V4 Claude Code mitigation for LiteLLM 1.99.0

This custom callback adapts Claude Code's injected `messages[].role == "system"` entries for Azure DeepSeek without changing LiteLLM. LiteLLM 1.98.0 and 1.99.0 can translate those entries, but preserving them as mid-turn system messages can produce undesirable DeepSeek behavior. The callback supports `azure_ai/DeepSeek-V4-Flash` and `azure_ai/DeepSeek-V4-Flash-0731`, whether either deployment is selected directly or through Router fallback

After Router selects a deployment, the callback runs only for the two affected Azure DeepSeek models and only during the original Anthropic Messages pass. LiteLLM invokes deployment hooks again after translating the request to OpenAI format; the callback explicitly ignores that second `acompletion` pass so it cannot remove the translated system prompt or alter native `/v1/chat/completions` requests

The callback normalizes a request only when `messages[]` contains a `system` role. Contiguous leading system messages move into the top-level Anthropic `system` field, while later dynamic system attachments remain in position as user-side `<system-reminder>` blocks. Requests that would be left with no messages, non-dictionary message lists, and unsupported non-text system blocks are left unchanged. Unexpected callback errors are logged and fail open with the original request

The callback does not remove Anthropic thinking fields. LiteLLM translates them to OpenAI-compatible parameters before the Azure call. Each example deployment uses `additional_drop_params` to remove unsupported `reasoning_effort`, `thinking`, and `output_config` parameters at the provider boundary. This keeps provider compatibility in model configuration instead of disabling Anthropic request behavior in the callback

Set `DEEPSEEK_V4_MITIGATION_MODEL_GROUPS` to add an optional model-group gate. The comma-separated comparison is case-insensitive and accepts either the selected fallback group or its original model group:

```bash
export DEEPSEEK_V4_MITIGATION_MODEL_GROUPS="primary-model-group,deepseek-v4-flash,deepseek-v4-flash-0731"
```

When the variable is unset, selected-deployment matching, Anthropic call-type matching, and the system-role check remain the only normalization gates. When it is set, the ingress hook stamps the requested group into the existing `litellm_metadata` object so the identity survives fallback without breaking LiteLLM logging references

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
