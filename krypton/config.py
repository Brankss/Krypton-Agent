"""Centralized configuration loaded from .env.

Everything else in the codebase imports `settings` from here so we
have one source of truth and no scattered os.environ access.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["ollama_local", "ollama_cloud", "openrouter", "nvidia"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- provider ------------------------------------------------------
    provider: ProviderName = Field(default="ollama_local", alias="KRYPTON_PROVIDER")

    # ---- ollama local --------------------------------------------------
    ollama_local_host: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_LOCAL_HOST")
    ollama_local_model: str = Field(default="qwen2.5-coder:7b", alias="OLLAMA_LOCAL_MODEL")
    # one of: none | low | medium | high  (or empty for model default)
    ollama_local_reasoning_effort: str = Field(default="", alias="OLLAMA_LOCAL_REASONING_EFFORT")

    # ---- ollama cloud --------------------------------------------------
    ollama_cloud_host: str = Field(default="https://ollama.com", alias="OLLAMA_CLOUD_HOST")
    ollama_cloud_api_key: str = Field(default="", alias="OLLAMA_CLOUD_API_KEY")
    ollama_cloud_model: str = Field(default="gpt-oss:120b", alias="OLLAMA_CLOUD_MODEL")
    ollama_cloud_reasoning_effort: str = Field(default="", alias="OLLAMA_CLOUD_REASONING_EFFORT")

    # ---- openrouter ----------------------------------------------------
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="anthropic/claude-sonnet-4", alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")

    # ---- nvidia (NIM) --------------------------------------------------
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_model: str = Field(default="nvidia/nemotron-3-super-120b-a12b", alias="NVIDIA_MODEL")
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL")
    # one of: none | low | medium | high  (empty = OFF, max model compatibility)
    nvidia_reasoning_effort: str = Field(default="", alias="NVIDIA_REASONING_EFFORT")

    # ---- telegram ------------------------------------------------------
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_authorized_user_ids: str = Field(default="", alias="TELEGRAM_AUTHORIZED_USER_IDS")

    # ---- web search ----------------------------------------------------
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # ---- runtime -------------------------------------------------------
    workdir: Path = Field(default=Path.cwd(), alias="KRYPTON_WORKDIR")
    data_dir: Path = Field(default=Path.cwd() / "data", alias="KRYPTON_DATA_DIR")
    max_iterations: int = Field(default=40, alias="KRYPTON_MAX_ITERATIONS")
    context_budget: int = Field(default=24000, alias="KRYPTON_CONTEXT_BUDGET")
    # IANA tz (e.g. Europe/Rome) for daily-scheduled tasks; UTC if unset/invalid.
    timezone: str = Field(default="UTC", alias="KRYPTON_TIMEZONE")

    @field_validator("workdir", "data_dir", mode="before")
    @classmethod
    def _expand(cls, v):
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    @property
    def authorized_telegram_ids(self) -> set[int]:
        raw = self.telegram_authorized_user_ids.strip()
        if not raw:
            return set()
        out: set[int] = set()
        for chunk in raw.replace(";", ",").split(","):
            chunk = chunk.strip()
            if chunk.isdigit():
                out.add(int(chunk))
        return out

    def ensure_dirs(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
