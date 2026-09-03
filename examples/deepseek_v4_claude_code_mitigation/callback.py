import json
import os
from collections.abc import Mapping
from typing import Final, cast

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import DualCache
from litellm.types.utils import CallTypes, CallTypesLiteral

_AZURE_DEEPSEEK_MODELS: Final = frozenset(
    {
        "azure_ai/deepseek-v4-flash",
        "azure_ai/deepseek-v4-flash-0731",
    }
)
_MODEL_GROUP_MARKER: Final = "deepseek_v4_mitigation_requested_model_group"
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


def _has_system_role_message(messages: list[object]) -> bool:
    return any(
        isinstance(message, dict) and cast(dict[str, object], message).get("role") == "system" for message in messages
    )


def normalize_anthropic_system_messages(data: dict[str, object]) -> dict[str, object]:
    messages: Final = data.get("messages")
    if not isinstance(messages, list):
        return data
    typed_messages: Final = cast(list[object], messages)
    if not _has_system_role_message(typed_messages) or not all(isinstance(message, dict) for message in typed_messages):
        return data

    normalized_messages, promoted_system_blocks = _normalize_messages(typed_messages)
    existing_system_blocks: Final = _system_blocks(data["system"]) if "system" in data else ()
    normalized: Final = {key: value for key, value in data.items() if key not in {"system", "messages"}}
    optional_system: Final = (
        {"system": [*existing_system_blocks, *promoted_system_blocks]}
        if existing_system_blocks or promoted_system_blocks
        else {}
    )
    return {**normalized, **optional_system, "messages": list(normalized_messages)}


def remove_unsupported_deepseek_thinking(data: dict[str, object]) -> dict[str, object]:
    model: Final = data.get("model")
    if not isinstance(model, str) or model.lower() not in _AZURE_DEEPSEEK_MODELS:
        return data

    output_config: Final = data.get("output_config")
    has_effort: Final = isinstance(output_config, dict) and "effort" in output_config
    if "thinking" not in data and not has_effort:
        return data
    retained_output_config: Final = (
        {key: value for key, value in cast(dict[str, object], output_config).items() if key != "effort"}
        if isinstance(output_config, dict)
        else None
    )
    normalized: Final = {key: value for key, value in data.items() if key not in {"thinking", "output_config"}}
    optional_output_config: Final = {"output_config": retained_output_config} if retained_output_config else {}
    return {**normalized, **optional_output_config}


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return cast(Mapping[str, object], value)


def _requested_model_groups(data: dict[str, object]) -> frozenset[str]:
    litellm_params: Final = _mapping(data.get("litellm_params"))
    metadata_candidates: Final = (
        _mapping(data.get("metadata")),
        _mapping(data.get("litellm_metadata")),
        _mapping(litellm_params.get("metadata")),
        _mapping(litellm_params.get("litellm_metadata")),
    )
    return frozenset(
        value.casefold()
        for metadata in metadata_candidates
        for key in ("model_group", "original_model_group", _MODEL_GROUP_MARKER)
        if isinstance((value := metadata.get(key)), str)
    )


def stamp_requested_model_group(data: dict[str, object]) -> dict[str, object]:
    requested_model: Final = data.get("model")
    if not isinstance(requested_model, str):
        return data
    litellm_metadata: Final = _mapping(data.get("litellm_metadata"))
    return {
        **data,
        "litellm_metadata": {
            **litellm_metadata,
            _MODEL_GROUP_MARKER: requested_model.casefold(),
        },
    }


def _model_groups_from_env() -> frozenset[str] | None:
    value: Final = os.getenv("DEEPSEEK_V4_MITIGATION_MODEL_GROUPS")
    if value is None:
        return None
    return frozenset(group.strip().casefold() for group in value.split(",") if group.strip())


def normalize_deepseek_deployment(
    data: dict[str, object],
    allowed_model_groups: frozenset[str] | None = None,
) -> dict[str, object]:
    model: Final = data.get("model")
    if not isinstance(model, str) or model.casefold() not in _AZURE_DEEPSEEK_MODELS:
        return data
    if allowed_model_groups is not None and not (_requested_model_groups(data) & allowed_model_groups):
        return data
    normalized_messages: Final = normalize_anthropic_system_messages(data)
    return remove_unsupported_deepseek_thinking(normalized_messages)


class DeepSeekV4ClaudeCodeMitigation(CustomLogger):
    def __init__(self, allowed_model_groups: frozenset[str] | None = None) -> None:
        self.allowed_model_groups: Final = allowed_model_groups

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:
        if self.allowed_model_groups is None or call_type != "anthropic_messages":
            return data
        return stamp_requested_model_group(data)

    async def async_pre_call_deployment_hook(
        self,
        kwargs: dict[str, object],
        call_type: CallTypes | None,
    ) -> dict[str, object]:
        return normalize_deepseek_deployment(kwargs, self.allowed_model_groups)


deepseek_v4_claude_code_mitigation: Final = DeepSeekV4ClaudeCodeMitigation(_model_groups_from_env())
