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


class GeminiLLMConfig(BaseModel):
    """Google Gemini (Nano, Flash, Pro, etc.) configuration. Uses GOOGLE_API_KEY."""

    provider: str = "gemini"
    model: str = "gemini-2.0-flash"
    max_tokens: int = 4096
    timeout: int = 120
    enabled: bool = False


class OpenRouterLLMConfig(BaseModel):
    """OpenRouter (400+ models via one API). Uses OPENROUTER_API_KEY."""

    provider: str = "openrouter"
    model: str = "anthropic/claude-3.5-sonnet"
    max_tokens: int = 4096
    timeout: int = 120
    enabled: bool = False
    use_free_models: bool = False  # Use :free variant (no cost, may have rate limits)


class NvidiaLLMConfig(BaseModel):
    """NVIDIA NIM (integrate.api.nvidia.com). Uses NVIDIA_API_KEY."""

    provider: str = "nvidia"
    model: str = "moonshotai/kimi-k2.5"
    max_tokens: int = 16384
    timeout: int = 120
    enabled: bool = False


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
    # Default context window (tokens) when model-specific value unknown; used for proportional tool truncation
    context_window_default: int = 200_000


class LLMConfig(BaseModel):
    """Combined LLM configuration."""

    local: LocalLLMConfig = Field(default_factory=LocalLLMConfig)
    cloud: CloudLLMConfig = Field(default_factory=CloudLLMConfig)
    gemini: GeminiLLMConfig = Field(default_factory=GeminiLLMConfig)
    openrouter: OpenRouterLLMConfig = Field(default_factory=OpenRouterLLMConfig)
    nvidia: NvidiaLLMConfig = Field(default_factory=NvidiaLLMConfig)
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
    """Short-term memory configuration (context efficiency)."""

    max_messages: int = 50
    max_tokens: int = 8000
    # Truncate tool results before adding to context
    max_tool_result_chars: int = 4000
    # If True, cap tool result to min(max_tool_result_chars, context_window * 0.3 * 4)
    use_proportional_tool_result: bool = True
    # Hard cap when persisting tool results to disk (single result never exceeds this)
    max_tool_result_chars_at_persist: int = 400_000
    # History limit by user turns (last N user messages + their replies); 0 = use max_messages only
    max_turns: int = 30
    # Reserve tokens for compaction summary generation (must stay free when trimming)
    compaction_reserve_tokens: int = 20_000
    # Token count above which we run compaction/trim before next LLM call (0 = disable token-based trim)
    compaction_trigger_tokens: int = 0  # 0 = use max_messages only; set e.g. 100_000 to enable


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
    auto_process_after_ingest: bool = True


class MemoryConfig(BaseModel):
    """Combined memory configuration."""

    short_term: ShortTermMemoryConfig = Field(default_factory=ShortTermMemoryConfig)
    long_term: LongTermMemoryConfig = Field(default_factory=LongTermMemoryConfig)
    episodic: EpisodicMemoryConfig = Field(default_factory=EpisodicMemoryConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    user_profiles_enabled: bool = True
    entity_extraction_enabled: bool = True
    auto_summarize: bool = True
    # Per-channel history turn limit (e.g. {"web": 50, "slack:dm": 20}). Overrides short_term.max_turns when key matches channel or "channel:user".
    channel_history_limits: dict[str, int] = Field(default_factory=dict)
    # RAG/injected context: max total chars, and head/tail ratio when trimming long content (0.7 = 70% head, 0.2 = 20% tail)
    rag_max_chars: int = 2500
    rag_head_ratio: float = 0.7
    rag_tail_ratio: float = 0.2


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
    camera: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    voice_call: dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "provider": "twilio", "account_sid": "", "auth_token": "", "from_number": "", "twiml_base_url": ""})
    agent: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    research: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    notion: dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "api_key": ""})
    todoist: dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "api_key": ""})
    linear: dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "api_key": ""})
    spotify: dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "client_id": "", "client_secret": ""})
    context: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    workflow: dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "workflows_dir": "./data/workflows", "retry_per_step": 1})


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
    self_healing_enabled: bool = False
    self_healing_check_interval_seconds: int = 300
    morning_briefing: bool = True
    briefing_time: str = "08:00"
    briefing_channel: str = ""
    follow_up_tracking: bool = True
    suggestions_enabled: bool = True
    context_reminders: bool = True
    time_of_day_awareness: bool = True
    meeting_prep: bool = True
    deadline_reminders: bool = True
    focus_mode: bool = True
    smart_summaries: bool = True


class AgentsConfig(BaseModel):
    """Autonomous agents configuration."""

    enabled: bool = True
    max_concurrent: int = 3
    timeout_seconds: int = 300
    research_enabled: bool = True
    coding_enabled: bool = True
    data_enabled: bool = True
    # Sub-agents: main assistant can delegate work to specialist agents (research, coding, data)
    subagents_enabled: bool = True


class OrchestratorConfig(BaseModel):
    """Orchestrator/chat loop configuration (more iterations = finish more tasks in one turn)."""

    max_tool_iterations: int = 15  # Tool-call rounds per user message; raise to let model complete complex tasks


class WorkspaceConfig(BaseModel):
    """Workspace: SOUL.md, USER.md, AGENTS.md, IDENTITY.md loaded into system prompt."""

    enabled: bool = True
    path: str = "./data/workspace"  # Directory containing SOUL.md, USER.md, AGENTS.md, IDENTITY.md
    max_injected_chars: int = 32_768  # Cap total workspace content injected into system prompt
    bootstrap_on_first_run: bool = True  # If True, run bootstrap when workspace is unconfigured (template-only, no LLM)


class OneContextConfig(BaseModel):
    """OneContext integration — unified context for all AI agents."""

    enabled: bool = True
    working_path: str = "./data/aria_workspace"
    sync_on_agent_complete: bool = True


class WebhooksConfig(BaseModel):
    """Inbound webhooks (e.g. Gmail Pub/Sub push)."""

    gmail: dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False, "token": "", "deliver_user_id": "webhook"}
    )


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
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    onecontext: OneContextConfig = Field(default_factory=OneContextConfig)
    webhooks: WebhooksConfig = Field(default_factory=WebhooksConfig)
    feature_overrides: dict[str, bool] = Field(default_factory=dict)

    # API keys from environment
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
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

        # Inject API keys from environment so providers get keys after .env is loaded
        env_keys = [
            ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            ("google_api_key", "GOOGLE_API_KEY"),
            ("openrouter_api_key", "OPENROUTER_API_KEY"),
            ("nvidia_api_key", "NVIDIA_API_KEY"),
            ("slack_bot_token", "SLACK_BOT_TOKEN"),
            ("slack_app_token", "SLACK_APP_TOKEN"),
            ("jwt_secret", "JWT_SECRET"),
        ]
        for field_name, env_var in env_keys:
            if field_name not in config_data:
                config_data[field_name] = os.environ.get(env_var, "")

        try:
            instance = cls(**config_data)
        except Exception as e:
            raise ValueError(f"Invalid config: {e}") from e
        instance.validate()
        return instance

    def validate(self) -> None:
        """Validate critical config. Raises ValueError on failure."""
        errors: list[str] = []
        if not (self.aria.name and isinstance(self.aria.name, str)):
            errors.append("aria.name must be a non-empty string")
        if getattr(self.llm, "local", None) and getattr(self.llm.local, "enabled", True):
            if not getattr(self.llm.local, "base_url", "").strip():
                errors.append("llm.local.base_url required when local LLM is enabled")
        if errors:
            raise ValueError("Config validation failed: " + "; ".join(errors))

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
