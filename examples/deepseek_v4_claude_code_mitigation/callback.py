import json
from typing import Final, cast

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import DualCache
from litellm.types.utils import CallTypesLiteral

_MODEL_NAME: Final = "deepseek-v4-flash-0731"
_SYSTEM_REMINDER_OPEN: Final = "<system-reminder>"
_SYSTEM_REMINDER_CLOSE: Final = "</system-reminder>"


def _text_block(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        block: Final = cast(dict[str, object], value)
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            return {**block}
    if isinstance(value, str):
        return {"type": "text", "text": value}
    return {"type": "text", "text": json.dumps(value, ensure_ascii=False, default=str)}


def _system_blocks(content: object) -> tuple[dict[str, object], ...]:
    if isinstance(content, list):
        blocks: Final = cast(list[object], content)
        return tuple(_text_block(block) for block in blocks)
    return (_text_block(content),)


def _reminder_block(block: dict[str, object]) -> dict[str, object]:
    text: Final = block.get("text")
    reminder_text: Final = f"{_SYSTEM_REMINDER_OPEN}{text}{_SYSTEM_REMINDER_CLOSE}"
    return {**block, "text": reminder_text}


def _normalized_message(raw_message: object, promote_system: bool) -> tuple[dict[str, object], ...]:
    if not isinstance(raw_message, dict):
        return ()
    message: Final = cast(dict[str, object], raw_message)
    if message.get("role") != "system":
        return ({**message},)
    if promote_system:
        return ()
    blocks: Final = _system_blocks(message.get("content", ""))
    return (
        {
            **message,
            "role": "user",
            "content": [_reminder_block(block) for block in blocks],
        },
    )


def _normalize_messages(
    messages: list[object],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    first_assistant_index: Final = next(
        (
            index
            for index, raw_message in enumerate(messages)
            if isinstance(raw_message, dict) and cast(dict[str, object], raw_message).get("role") == "assistant"
        ),
        len(messages),
    )
    normalized_messages: Final = tuple(
        message
        for index, raw_message in enumerate(messages)
        for message in _normalized_message(raw_message, promote_system=index < first_assistant_index)
    )
    promoted_system_blocks: Final = tuple(
        block
        for index, raw_message in enumerate(messages)
        if index < first_assistant_index
        and isinstance(raw_message, dict)
        and cast(dict[str, object], raw_message).get("role") == "system"
        for block in _system_blocks(cast(dict[str, object], raw_message).get("content", ""))
    )
    return normalized_messages, promoted_system_blocks


def normalize_anthropic_request(data: dict[str, object]) -> dict[str, object]:
    model: Final = data.get("model")
    messages: Final = data.get("messages")
    if not isinstance(model, str) or model.lower() != _MODEL_NAME or not isinstance(messages, list):
        return data

    normalized_messages, promoted_system_blocks = _normalize_messages(cast(list[object], messages))
    existing_system_blocks: Final = _system_blocks(data["system"]) if "system" in data else ()
    output_config: Final = data.get("output_config")
    retained_output_config: Final = (
        {key: value for key, value in cast(dict[str, object], output_config).items() if key != "effort"}
        if isinstance(output_config, dict)
        else None
    )
    normalized: Final = {
        key: value for key, value in data.items() if key not in {"thinking", "output_config", "system", "messages"}
    }
    optional_system: Final = (
        {"system": [*existing_system_blocks, *promoted_system_blocks]}
        if existing_system_blocks or promoted_system_blocks
        else {}
    )
    optional_output_config: Final = {"output_config": retained_output_config} if retained_output_config else {}
    return {
        **normalized,
        **optional_system,
        **optional_output_config,
        "messages": list(normalized_messages),
    }


class DeepSeekV4ClaudeCodeMitigation(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:
        if call_type != "anthropic_messages":
            return data
        return normalize_anthropic_request(data)


deepseek_v4_claude_code_mitigation: Final = DeepSeekV4ClaudeCodeMitigation()
