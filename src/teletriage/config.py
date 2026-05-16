"""
Configuration system.

Why pydantic-settings instead of a plain YAML loader?
- Type-checked config (catches typos at startup, not runtime)
- Env vars override YAML cleanly (standard 12-factor pattern)
- Secrets (API keys) live in .env, never in YAML (never committed)
- IDE autocomplete on config.cache_tier.fuzzy_threshold etc.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ─── Sub-configs (mirror config.yaml structure) ───────────────────────


class AppConfig(BaseModel):
    name: str = "teletriage"
    version: str = "0.2.0"


class PathsConfig(BaseModel):
    cache_file: Path
    knowledge_base: Path
    faiss_index_dir: Path
    log_dir: Path


class CacheTierConfig(BaseModel):
    enabled: bool = True
    fuzzy_threshold: int = Field(90, ge=0, le=100)
    min_confidence: float = Field(0.85, ge=0.0, le=1.0)


class RetrievalTierConfig(BaseModel):
    enabled: bool = True
    embedder_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    bm25_top_k: int = 20
    dense_top_k: int = 20
    rerank_top_k: int = 10
    final_top_k: int = 1
    rrf_k: int = 60
    min_rerank_score: float = 0.5
    min_confidence: float = 0.70


class GenerativeTierConfig(BaseModel):
    enabled: bool = True
    backend: Literal["groq", "gemini", "local"] = "groq"
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.0-flash"
    local_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_tokens: int = 512
    temperature: float = 0.2


class ObservabilityConfig(BaseModel):
    log_level: str = "INFO"
    log_queries: bool = True
    metrics_file: Path = Path("logs/query_metrics.jsonl")


class YamlConfig(BaseModel):
    """Schema for config.yaml - this is what gets loaded from disk."""

    app: AppConfig
    paths: PathsConfig
    cache_tier: CacheTierConfig
    retrieval_tier: RetrievalTierConfig
    generative_tier: GenerativeTierConfig
    observability: ObservabilityConfig


# ─── Secrets (from .env, never YAML) ──────────────────────────────────


class Secrets(BaseSettings):
    """API keys and other secrets. Loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    google_api_key: str = ""
    hf_token: str = ""

    teletriage_env: str = "development"
    teletriage_log_level: str = "INFO"


# ─── Merged config object (what the rest of the app imports) ──────────


class Config:
    """
    Top-level config. Merges YAML config + .env secrets.

    Usage:
        from teletriage.config import get_config
        cfg = get_config()
        threshold = cfg.cache_tier.fuzzy_threshold
        api_key = cfg.secrets.groq_api_key
    """

    def __init__(self, yaml_path: Path | None = None):
        yaml_path = yaml_path or (PROJECT_ROOT / "config" / "config.yaml")
        with yaml_path.open() as f:
            raw = yaml.safe_load(f)
        self._yaml = YamlConfig.model_validate(raw)
        self.secrets = Secrets()

    # Expose YAML sections directly for clean access
    @property
    def app(self) -> AppConfig:
        return self._yaml.app

    @property
    def paths(self) -> PathsConfig:
        return self._yaml.paths

    @property
    def cache_tier(self) -> CacheTierConfig:
        return self._yaml.cache_tier

    @property
    def retrieval_tier(self) -> RetrievalTierConfig:
        return self._yaml.retrieval_tier

    @property
    def generative_tier(self) -> GenerativeTierConfig:
        return self._yaml.generative_tier

    @property
    def observability(self) -> ObservabilityConfig:
        return self._yaml.observability

    def resolve_path(self, p: Path) -> Path:
        """Resolve a config path relative to project root."""
        return p if p.is_absolute() else PROJECT_ROOT / p


# Singleton pattern - load once, reuse everywhere
_config: Config | None = None


def get_config(yaml_path: Path | None = None, *, reload: bool = False) -> Config:
    """Get the global config singleton. Pass reload=True to force re-read."""
    global _config
    if _config is None or reload:
        _config = Config(yaml_path=yaml_path)
    return _config
