"""Minimal Anthropic client used by eval_document_level.py."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message, MessageParam


class LLMClient:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def message(
        self,
        messages: Iterable[MessageParam],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        cache_system: bool = False,
        **kwargs: Any,
    ) -> Message:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        if system:
            if cache_system:
                params["system"] = [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                params["system"] = system
        return self._client.messages.create(**params)
