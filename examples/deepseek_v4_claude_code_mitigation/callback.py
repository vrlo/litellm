import os
from collections.abc import Mapping
from typing import Final, cast

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import DualCache
from litellm.types.utils import CallTypes, CallTypesLiteral

_ANTHROPIC_CALL_TYPES: Final = frozenset(
    {
        CallTypes.anthropic_messages,
        CallTypes.aanthropic_messages,
    }
)
_AZURE_DEEPSEEK_MODELS: Final = frozenset(
    {
        "azure_ai/deepseek-v4-flash",
        "azure_ai/deepseek-v4-flash-0731",
    }
)
_MODEL_GROUP_MARKER: Final = "deepseek_v4_mitigation_requested_model_group"
_SYSTEM_REMINDER_OPEN: Final = "<system-reminder>"
_SYSTEM_REMINDER_CLOSE: Final = "</system-reminder>"


def _text_block(value: object) -> dict[str, object] | None:
    if isinstance(value, str):
        return {"type": "text", "text": value}
    if not isinstance(value, dict):
        return None
    block: Final = cast(dict[str, object], value)
    if block.get("type") != "text" or not isinstance(block.get("text"), str):
        return None
    return {**block}


def _system_blocks(content: object) -> tuple[dict[str, object], ...] | None:
    values: Final = cast(list[object], content) if isinstance(content, list) else [content]
    blocks: Final = tuple(_text_block(value) for value in values)
    if not blocks or any(block is None for block in blocks):
        return None
    return cast(tuple[dict[str, object], ...], blocks)


def _reminder_block(block: dict[str, object]) -> dict[str, object]:
    text: Final = block.get("text")
    return {**block, "text": f"{_SYSTEM_REMINDER_OPEN}{text}{_SYSTEM_REMINDER_CLOSE}"}


def _leading_system_count(messages: list[dict[str, object]]) -> int:
    count = 0
    for message in messages:
        if message.get("role") != "system":
            break
        count += 1
    return count


def _has_system_role_message(messages: list[dict[str, object]]) -> bool:
    return any(message.get("role") == "system" for message in messages)


def normalize_anthropic_system_messages(data: dict[str, object]) -> dict[str, object]:
    messages: Final = data.get("messages")
    if not isinstance(messages, list):
        return data
    raw_messages: Final = cast(list[object], messages)
    if not all(isinstance(message, dict) for message in raw_messages):
        return data
    typed_messages: Final = cast(list[dict[str, object]], raw_messages)
    if not _has_system_role_message(typed_messages):
        return data

    leading_system_count: Final = _leading_system_count(typed_messages)
    if leading_system_count == len(typed_messages):
        verbose_proxy_logger.debug(
            "Skipped DeepSeek V4 system-message normalization because it would leave no messages"
        )
        return data

    existing_system_blocks: tuple[dict[str, object], ...] = ()
    if "system" in data:
        parsed_existing_system: Final = _system_blocks(data["system"])
        if parsed_existing_system is None:
            verbose_proxy_logger.debug(
                "Skipped DeepSeek V4 system-message normalization due to unsupported top-level system content"
            )
            return data
        existing_system_blocks = parsed_existing_system

    promoted_system_blocks: list[dict[str, object]] = []
    normalized_messages: list[dict[str, object]] = []
    for index, message in enumerate(typed_messages):
        if message.get("role") != "system":
            normalized_messages.append({**message})
            continue

        blocks = _system_blocks(message.get("content", ""))
        if blocks is None:
            verbose_proxy_logger.debug(
                "Skipped DeepSeek V4 system-message normalization due to unsupported system-message content"
            )
            return data
        if index < leading_system_count:
            promoted_system_blocks.extend(blocks)
        else:
            normalized_messages.append(
                {
                    **message,
                    "role": "user",
                    "content": [_reminder_block(block) for block in blocks],
                }
            )

    normalized: Final = {key: value for key, value in data.items() if key not in {"system", "messages"}}
    all_system_blocks: Final = [*existing_system_blocks, *promoted_system_blocks]
    optional_system: Final = {"system": all_system_blocks} if all_system_blocks else {}
    return {**normalized, **optional_system, "messages": normalized_messages}


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

    existing_metadata: Final = data.get("litellm_metadata")
    metadata: dict[str, object]
    if isinstance(existing_metadata, dict):
        metadata = cast(dict[str, object], existing_metadata)
    else:
        metadata = {}
        data["litellm_metadata"] = metadata
    metadata[_MODEL_GROUP_MARKER] = requested_model.casefold()
    return data


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
    return normalize_anthropic_system_messages(data)


class DeepSeekV4ClaudeCodeMitigation(CustomLogger):
    allowed_model_groups: frozenset[str] | None = None

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:
        if self.allowed_model_groups is None or call_type not in {
            "anthropic_messages",
            "aanthropic_messages",
        }:
            return data
        try:
            return stamp_requested_model_group(data)
        except Exception:
            verbose_proxy_logger.exception("DeepSeek V4 model-group stamping failed open")
            return data

    async def async_pre_call_deployment_hook(
        self,
        kwargs: dict[str, object],
        call_type: CallTypes | None,
    ) -> dict[str, object]:
        # The Anthropic adapter invokes this hook again through acompletion after
        # translation. That pass contains a legitimate OpenAI system message.
        if call_type not in _ANTHROPIC_CALL_TYPES:
            return kwargs
        try:
            normalized: Final = normalize_deepseek_deployment(kwargs, self.allowed_model_groups)
            if normalized is not kwargs:
                verbose_proxy_logger.debug("Applied DeepSeek V4 Anthropic request mitigation")
            return normalized
        except Exception:
            verbose_proxy_logger.exception("DeepSeek V4 Anthropic request mitigation failed open")
            return kwargs


deepseek_v4_claude_code_mitigation: Final = DeepSeekV4ClaudeCodeMitigation()
deepseek_v4_claude_code_mitigation.allowed_model_groups = _model_groups_from_env()
