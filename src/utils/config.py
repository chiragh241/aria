"""Configuration management for Aria using pydantic-settings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure .env is loaded so ${VAR} expansion and os.environ lookups work
load_dotenv(dotenv_path=Path(".") / ".env", override=False)


class AriaConfig(BaseModel):
    """Core Aria configuration."""

    name: str = "Aria"
    version: str = "0.1.0"
    data_dir: str = os.environ.get("DATA_DIR", "./data")
    deployment_mode: str = "local"  # "local" or "docker"


class LocalLLMConfig(BaseModel):
    """Local LLM (Ollama) configuration."""

    provider: str = "ollama"
    model: str = "llama3.2:8b"
    base_url: str = "http://localhost:11434"
    timeout: int = 60
    enabled: bool = True


class CloudLLMConfig(BaseModel):
    """Cloud LLM (Claude) configuration."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    timeout: int = 120
    enabled: bool = True


class LLMRoutingConfig(BaseModel):
    """LLM routing configuration."""

    simple_threshold: int = 50
    always_cloud: list[str] = Field(
        default_factory=lambda: [
            "code_generation",
            "complex_reasoning",
            "skill_creation",
            "multi_step_planning",
        ]
    )
    fallback_to_cloud: bool = True


class LLMConfig(BaseModel):
    """Combined LLM configuration."""

    local: LocalLLMConfig = Field(default_factory=LocalLLMConfig)
    cloud: CloudLLMConfig = Field(default_factory=CloudLLMConfig)
    routing: LLMRoutingConfig = Field(default_factory=LLMRoutingConfig)


class SlackChannelConfig(BaseModel):
    """Slack channel configuration."""

    enabled: bool = True
    bot_token: str = ""
    app_token: str = ""
    allowed_channels: list[str] = Field(default_factory=list)


class WhatsAppChannelConfig(BaseModel):
    """WhatsApp channel configuration."""

    enabled: bool = False
    session_path: str = "./data/whatsapp-session"
    bridge_port: int = 3001
    bridge_host: str = "localhost"
    allowed_numbers: list[str] = Field(default_factory=list)


class WebChannelConfig(BaseModel):
    """Web channel configuration."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    jwt_secret: str = ""
    jwt_expiry_hours: int = 24


class ChannelsConfig(BaseModel):
    """All channels configuration."""

    slack: SlackChannelConfig = Field(default_factory=SlackChannelConfig)
    whatsapp: WhatsAppChannelConfig = Field(default_factory=WhatsAppChannelConfig)
    web: WebChannelConfig = Field(default_factory=WebChannelConfig)


class SecurityConfig(BaseModel):
    """Security configuration."""

    active_profile: str = "balanced"
    approval_timeout: int = 300
    approval_channels: list[str] = Field(default_factory=lambda: ["slack", "whatsapp", "web"])
    require_all_approvals: bool = False


class DockerSandboxConfig(BaseModel):
    """Docker sandbox configuration."""

    image: str = "aria-sandbox:latest"
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    network_mode: str = "none"
    timeout: int = 300


class SandboxConfig(BaseModel):
    """Sandbox configuration."""

    default: str = "docker"
    docker: DockerSandboxConfig = Field(default_factory=DockerSandboxConfig)
    trusted_paths: list[str] = Field(
        default_factory=lambda: ["~/Documents", "~/Projects", "~/Downloads"]
    )
    safe_commands: list[str] = Field(
        default_factory=lambda: ["ls", "pwd", "whoami", "date", "echo"]
    )


class ShortTermMemoryConfig(BaseModel):
    """Short-term memory configuration."""

    max_messages: int = 50
    max_tokens: int = 8000


class LongTermMemoryConfig(BaseModel):
    """Long-term memory configuration."""

    provider: str = "chromadb"
    persist_directory: str = "./data/chromadb"
    collection_name: str = "aria_memory"
    embedding_model: str = "all-MiniLM-L6-v2"


class EpisodicMemoryConfig(BaseModel):
    """Episodic memory configuration."""

    max_episodes: int = 1000
    summary_threshold: int = 10


class KnowledgeGraphConfig(BaseModel):
    """Knowledge graph (Cognee) configuration."""

    enabled: bool = True
    provider: str = "cognee"
    auto_process_after_ingest: bool = False


class MemoryConfig(BaseModel):
    """Combined memory configuration."""

    short_term: ShortTermMemoryConfig = Field(default_factory=ShortTermMemoryConfig)
    long_term: LongTermMemoryConfig = Field(default_factory=LongTermMemoryConfig)
    episodic: EpisodicMemoryConfig = Field(default_factory=EpisodicMemoryConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    user_profiles_enabled: bool = True
    entity_extraction_enabled: bool = True
    auto_summarize: bool = True


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = "sqlite+aiosqlite:///./data/aria.db"
    echo: bool = False


class RedisConfig(BaseModel):
    """Redis configuration."""

    enabled: bool = False
    url: str = "redis://localhost:6379/0"
    queue_name: str = "aria:messages"
    result_ttl: int = 3600


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    file: str = "./data/logs/aria.log"
    max_size_mb: int = 100
    backup_count: int = 5
    audit_file: str = "./data/logs/audit.log"


class SkillConfig(BaseModel):
    """Individual skill configuration."""

    enabled: bool = True
    # Additional skill-specific settings can be added via extra fields


class BuiltinSkillsConfig(BaseModel):
    """Built-in skills configuration."""

    filesystem: dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "max_file_size_mb": 100})
    shell: dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "timeout": 60})
    browser: dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "headless": True, "timeout": 30})
    calendar: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    email: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    sms: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    tts: dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "provider": "edge-tts", "voice": "en-US-AriaNeural"})
    stt: dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "provider": "whisper", "model": "base"})
    image: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    video: dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "ffmpeg_path": "ffmpeg"})
    documents: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    memory: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    weather: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    news: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    finance: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    contacts: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    tracking: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    home: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    webhook: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    agent: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    research: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    notion: dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "api_key": ""})
    todoist: dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "api_key": ""})
    linear: dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "api_key": ""})
    spotify: dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "client_id": "", "client_secret": ""})


class LearnedSkillsConfig(BaseModel):
    """Learned skills configuration."""

    enabled: bool = True
    directory: str = "./data/learned_skills"
    auto_test: bool = True
    require_approval: bool = True


class SkillsConfig(BaseModel):
    """Combined skills configuration."""

    builtin: BuiltinSkillsConfig = Field(default_factory=BuiltinSkillsConfig)
    learned: LearnedSkillsConfig = Field(default_factory=LearnedSkillsConfig)


class DashboardConfig(BaseModel):
    """Dashboard configuration."""

    title: str = "Aria Control Center"
    theme: str = "dark"
    features: dict[str, bool] = Field(
        default_factory=lambda: {
            "chat": True,
            "approvals": True,
            "settings": True,
            "logs": True,
            "skills": True,
            "metrics": True,
        }
    )


class ProactiveConfig(BaseModel):
    """Proactive intelligence configuration."""

    enabled: bool = True
    morning_briefing: bool = True
    briefing_time: str = "08:00"
    briefing_channel: str = ""
    follow_up_tracking: bool = True
    suggestions_enabled: bool = True
    context_reminders: bool = True
    time_of_day_awareness: bool = True


class AgentsConfig(BaseModel):
    """Autonomous agents configuration."""

    enabled: bool = True
    max_concurrent: int = 3
    timeout_seconds: int = 300
    research_enabled: bool = True
    coding_enabled: bool = True
    data_enabled: bool = True


class Settings(BaseSettings):
    """Main settings class that loads from YAML and environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ARIA_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    aria: AriaConfig = Field(default_factory=AriaConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    feature_overrides: dict[str, bool] = Field(default_factory=dict)

    # API keys from environment
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    slack_bot_token: str = Field(default="", alias="SLACK_BOT_TOKEN")
    slack_app_token: str = Field(default="", alias="SLACK_APP_TOKEN")
    jwt_secret: str = Field(default="change-me-in-production", alias="JWT_SECRET")

    @classmethod
    def from_yaml(cls, config_path: str | Path | None = None) -> "Settings":
        """Load settings from YAML file with environment variable overrides."""
        if config_path is None:
            # Try to find config file
            possible_paths = [
                Path("config/settings.yaml"),
                Path("config/settings.local.yaml"),
                Path.home() / ".config/aria/settings.yaml",
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break

        config_data: dict[str, Any] = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                config_data = yaml.safe_load(f) or {}

        # Expand environment variables in the config
        config_data = cls._expand_env_vars(config_data)

        return cls(**config_data)

    @classmethod
    def _expand_env_vars(cls, data: Any) -> Any:
        """Recursively expand environment variables in config values."""
        if isinstance(data, dict):
            return {k: cls._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls._expand_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Expand ${VAR} patterns
            if data.startswith("${") and data.endswith("}"):
                env_var = data[2:-1]
                return os.environ.get(env_var, "")
            return data
        return data

    def get_data_path(self, *parts: str) -> Path:
        """Get a path within the data directory."""
        base = Path(self.aria.data_dir).expanduser()
        return base.joinpath(*parts)

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        dirs_to_create = [
            self.aria.data_dir,
            self.memory.long_term.persist_directory,
            self.skills.learned.directory,
            Path(self.logging.file).parent,
            Path(self.logging.audit_file).parent,
        ]
        for dir_path in dirs_to_create:
            Path(dir_path).expanduser().mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_yaml()


def reload_settings() -> Settings:
    """Reload settings (clears cache)."""
    get_settings.cache_clear()
    return get_settings()
