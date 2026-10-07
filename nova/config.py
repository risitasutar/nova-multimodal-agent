"""
Single source of truth for Nova configuration.

Every tunable (model names, endpoints, limits, paths, observability) is read here from
environment variables / `.env`. Other modules call `get_settings()` and never read
`os.environ` themselves.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ---------------- Runtime ----------------
    environment: Literal["development", "production", "test"] = Field(
        "development", validation_alias="NOVA_ENV"
    )
    log_level: str = Field("INFO", validation_alias="NOVA_LOG_LEVEL")
    log_json: bool = Field(True, validation_alias="NOVA_LOG_JSON")

    # ---------------- LLM ----------------
    llm_provider: Literal["ollama", "openai_compatible"] = Field(
        "ollama", validation_alias="NOVA_LLM_PROVIDER"
    )
    ollama_model: str = Field("qwen3:8b", validation_alias="OLLAMA_MODEL")
    ollama_think: bool = Field(False, validation_alias="OLLAMA_THINK")
    ollama_embed_model: str = Field("nomic-embed-text", validation_alias="OLLAMA_EMBED_MODEL")
    ollama_base_url: str = Field("http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    # Hosted / OpenAI-compatible endpoint (OpenAI, Groq, Together, vLLM, ...)
    llm_base_url: str | None = Field(None, validation_alias="NOVA_LLM_BASE_URL")
    llm_api_key: SecretStr | None = Field(None, validation_alias="NOVA_LLM_API_KEY")
    llm_model: str | None = Field(None, validation_alias="NOVA_LLM_MODEL")
    embed_model: str | None = Field(None, validation_alias="NOVA_EMBED_MODEL")
    llm_temperature: float = Field(0.0, validation_alias="NOVA_LLM_TEMPERATURE")
    llm_seed: int = Field(42, validation_alias="NOVA_LLM_SEED")
    # CPU-only Ollama processes prompts at ~13 tok/s; keep generous. Lower it on GPU/hosted.
    llm_timeout_s: float = Field(600.0, validation_alias="NOVA_LLM_TIMEOUT_S")

    # ---------------- Market data ----------------
    alphavantage_api_key: SecretStr | None = Field(None, validation_alias="ALPHAVANTAGE_API_KEY")
    finance_provider: Literal["auto", "alphavantage", "yahoo"] = Field(
        "auto", validation_alias="NOVA_FINANCE_PROVIDER"
    )

    # ---------------- Web search ----------------
    search_enabled: bool = Field(True, validation_alias="NOVA_SEARCH_ENABLED")
    search_region: str = Field("us-en", validation_alias="NOVA_SEARCH_REGION")
    search_max_results: int = Field(4, validation_alias="NOVA_SEARCH_MAX_RESULTS")

    # ---------------- External calls ----------------
    http_timeout_s: float = Field(15.0, validation_alias="NOVA_HTTP_TIMEOUT_S")
    http_max_retries: int = Field(2, validation_alias="NOVA_HTTP_MAX_RETRIES")

    # ---------------- Storage ----------------
    data_dir: Path = Field(PROJECT_ROOT / "data", validation_alias="NOVA_DATA_DIR")
    log_dir: Path = Field(PROJECT_ROOT / "logs", validation_alias="NOVA_LOG_DIR")

    # ---------------- RAG ----------------
    # Smaller than the original 1000/200: tighter evidence per citation and ~40% fewer
    # prompt tokens for the answer step (prompt processing dominates CPU latency).
    chunk_size: int = Field(600, validation_alias="NOVA_CHUNK_SIZE")
    chunk_overlap: int = Field(100, validation_alias="NOVA_CHUNK_OVERLAP")
    retrieval_top_k: int = Field(4, validation_alias="NOVA_RETRIEVAL_TOP_K")
    retrieval_candidates: int = Field(12, validation_alias="NOVA_RETRIEVAL_CANDIDATES")
    # Cosine-similarity floor below which a chunk is not treated as evidence.
    # Calibrated for nomic-embed-text (+task prefixes) on a dev set disjoint from the eval
    # set: `python -m evaluation.calibrate_threshold` (see evaluation/README.md).
    min_relevance: float = Field(0.56, validation_alias="NOVA_MIN_RELEVANCE")

    # ---------------- Limits ----------------
    max_message_chars: int = Field(4000, validation_alias="NOVA_MAX_MESSAGE_CHARS")
    max_pdf_mb: float = Field(20.0, validation_alias="NOVA_MAX_PDF_MB")
    max_pdf_pages: int = Field(300, validation_alias="NOVA_MAX_PDF_PAGES")
    max_chunks_per_document: int = Field(3000, validation_alias="NOVA_MAX_CHUNKS")
    max_documents_per_thread: int = Field(10, validation_alias="NOVA_MAX_DOCUMENTS_PER_THREAD")
    max_plan_steps: int = Field(5, validation_alias="NOVA_MAX_PLAN_STEPS")
    max_verify_attempts: int = Field(2, validation_alias="NOVA_MAX_VERIFY_ATTEMPTS")
    history_turns: int = Field(3, validation_alias="NOVA_HISTORY_TURNS")
    history_message_chars: int = Field(600, validation_alias="NOVA_HISTORY_MESSAGE_CHARS")

    # ---------------- Human-in-the-loop ----------------
    approval_required_default: bool = Field(False, validation_alias="NOVA_APPROVAL_REQUIRED")

    # ---------------- API ----------------
    api_key: SecretStr | None = Field(None, validation_alias="NOVA_API_KEY")
    api_cors_origins: list[str] = Field(default_factory=list, validation_alias="NOVA_CORS_ORIGINS")

    # ---------------- Media (video / audio) ----------------
    whisper_backend: Literal["faster", "openai"] = Field("faster", validation_alias="NOVA_WHISPER_BACKEND")
    whisper_model: str = Field("small", validation_alias="WHISPER_MODEL")
    whisper_compute_type: str = Field("int8", validation_alias="NOVA_WHISPER_COMPUTE_TYPE")
    sarvam_api_key: SecretStr | None = Field(None, validation_alias="SARVAM_API_KEY")
    sarvam_stt_model: str = Field("saaras:v2.5", validation_alias="SARVAM_STT_MODEL")
    # LLM for meeting summaries/insights: "default" reuses Nova's chat model; "mistral" keeps the
    # original AI Video Assistant provider (needs langchain-mistralai + MISTRAL_API_KEY).
    media_llm_provider: Literal["default", "mistral"] = Field("default", validation_alias="NOVA_MEDIA_LLM_PROVIDER")
    mistral_api_key: SecretStr | None = Field(None, validation_alias="MISTRAL_API_KEY")
    mistral_model: str = Field("mistral-small-latest", validation_alias="MISTRAL_MODEL")
    media_max_size_mb: float = Field(500.0, validation_alias="MEDIA_MAX_SIZE_MB")
    media_max_duration_seconds: float = Field(7200.0, validation_alias="MEDIA_MAX_DURATION_SECONDS")
    media_max_transcript_chars: int = Field(400_000, validation_alias="MEDIA_MAX_TRANSCRIPT_CHARS")
    media_max_chunks: int = Field(3000, validation_alias="MEDIA_MAX_CHUNKS")
    max_media_per_thread: int = Field(10, validation_alias="NOVA_MAX_MEDIA_PER_THREAD")
    video_chunk_size: int = Field(500, validation_alias="VIDEO_CHUNK_SIZE")  # characters per transcript chunk
    # Max time span of one chunk: keeps timestamp citations tight even for slow speech.
    video_chunk_max_seconds: float = Field(120.0, validation_alias="VIDEO_CHUNK_MAX_SECONDS")
    video_chunk_overlap: int = Field(1, validation_alias="VIDEO_CHUNK_OVERLAP")  # segments shared by neighbours
    video_min_relevance: float = Field(0.55, validation_alias="NOVA_VIDEO_MIN_RELEVANCE")  # calibrate_threshold --video
    media_insight_window_chars: int = Field(6000, validation_alias="NOVA_MEDIA_INSIGHT_WINDOW_CHARS")
    youtube_enabled: bool = Field(True, validation_alias="NOVA_YOUTUBE_ENABLED")

    # ---------------- Observability (optional LangSmith) ----------------
    langchain_tracing_v2: bool = Field(False, validation_alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: SecretStr | None = Field(None, validation_alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field("nova", validation_alias="LANGCHAIN_PROJECT")

    @field_validator("data_dir", "log_dir", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("llm_api_key", "alphavantage_api_key", "langchain_api_key", "api_key", "sarvam_api_key",
                     "mistral_api_key", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    # ---------------- Derived paths ----------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "nova.db"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "vectorstores"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def media_cache_dir(self) -> Path:
        return self.data_dir / "media_cache"

    @property
    def media_vector_dir(self) -> Path:
        return self.data_dir / "media_vectors"

    @property
    def chat_model_name(self) -> str:
        if self.llm_provider == "openai_compatible":
            return self.llm_model or "gpt-4o-mini"
        return self.ollama_model

    @property
    def embedding_model_name(self) -> str:
        if self.llm_provider == "openai_compatible":
            return self.embed_model or "text-embedding-3-small"
        return self.ollama_embed_model

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.vector_dir, self.log_dir, self.media_dir, self.media_cache_dir,
                     self.media_vector_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
