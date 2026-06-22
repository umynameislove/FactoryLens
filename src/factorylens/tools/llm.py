"""Pluggable LLM client helpers for FactoryLens tools."""

from __future__ import annotations

from typing import Protocol

from factorylens.config import get_settings


class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
        json_mode: bool = True,
    ) -> str: ...


class OpenAIClient:
    def __init__(self, api_key: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
        json_mode: bool = True,
    ) -> str:
        request = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**request)
        return response.choices[0].message.content or ""


def get_llm_client() -> LLMClient | None:
    api_key = get_settings().openai_api_key
    if api_key is None or not api_key.strip():
        return None
    return OpenAIClient(api_key)
