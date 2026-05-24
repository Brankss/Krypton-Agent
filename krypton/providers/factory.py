"""Factory that materializes the active provider from `settings`."""
from __future__ import annotations

from krypton.config import ProviderName, settings
from krypton.providers.base import LLMProvider
from krypton.providers.nvidia import NvidiaProvider
from krypton.providers.ollama import OllamaProvider
from krypton.providers.openrouter import OpenRouterProvider


def build_provider(name: ProviderName | None = None) -> LLMProvider:
    chosen: ProviderName = name or settings.provider
    if chosen == "ollama_local":
        return OllamaProvider(
            name="ollama_local",
            host=settings.ollama_local_host,
            model=settings.ollama_local_model,
        )
    if chosen == "ollama_cloud":
        return OllamaProvider(
            name="ollama_cloud",
            host=settings.ollama_cloud_host,
            model=settings.ollama_cloud_model,
            api_key=settings.ollama_cloud_api_key,
        )
    if chosen == "openrouter":
        return OpenRouterProvider(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
    if chosen == "nvidia":
        eff = (settings.nvidia_reasoning_effort or "").strip().lower() or None
        return NvidiaProvider(
            model=settings.nvidia_model,
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            reasoning_effort=eff,  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown provider: {chosen}")
