"""FastAPI backend for Aria web dashboard."""

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from pydantic import BaseModel

from ..utils.config import get_settings
from ..utils.crypto import hash_password, verify_password
from ..utils.logging import get_logger

logger = get_logger(__name__)

# ── Pydantic request/response models ──────────────────────────────────────────


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class MessageRequest(BaseModel):
    content: str
    channel: str = "web"


class MessageResponse(BaseModel):
    id: str
    content: str
    timestamp: str


class ApprovalRequest(BaseModel):
    approval_id: str
    approved: bool


class SettingsUpdate(BaseModel):
    security_profile: str | None = None
    llm_local_enabled: bool | None = None
    llm_cloud_enabled: bool | None = None


class SkillToggle(BaseModel):
    skill_name: str
    enabled: bool


class ConfigUpdate(BaseModel):
    """Generic configuration update payload."""
    data: dict[str, Any]


class KeyValidation(BaseModel):
    """API key validation request."""
    key_type: str  # "anthropic" | "brave" | "slack_bot" | "slack_app"
    key_value: str
    extra: dict[str, str] | None = None


# ── Security ──────────────────────────────────────────────────────────────────

security = HTTPBearer()

# ── Resolve frontend dist path ────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"


def create_app(
    orchestrator: Any = None,
    skill_registry: Any = None,
    security_guardian: Any = None,
    audit_logger: Any = None,
) -> FastAPI:
    """Create the FastAPI application with API routes and static frontend."""
    settings = get_settings()

    app = FastAPI(
        title="Aria Control Center",
        description="Web dashboard for Aria AI assistant",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store instances
    app.state.orchestrator = orchestrator
    app.state.skill_registry = skill_registry
    app.state.security_guardian = security_guardian
    app.state.audit_logger = audit_logger

    # User store
    admin_password = os.environ.get("ADMIN_PASSWORD", "admin")
    app.state.users = {
        "admin": hash_password(admin_password),
    }
    app.state.trace_log = []

    # JWT
    JWT_SECRET = settings.jwt_secret
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_HOURS = settings.channels.web.jwt_expiry_hours

    def create_token(user_id: str) -> tuple[str, int]:
        expires = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS)
        payload = {"sub": user_id, "exp": expires, "iat": datetime.now(timezone.utc)}
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        return token, JWT_EXPIRY_HOURS * 3600

    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> str:
        try:
            payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token")
            return user_id
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    # ── API Router (/api prefix) ──────────────────────────────────────────

    api = APIRouter(prefix="/api")

    # -- Auth --

    @api.post("/auth/login", response_model=LoginResponse)
    async def login(request: LoginRequest):
        stored_hash = app.state.users.get(request.username)
        if not stored_hash or not verify_password(request.password, stored_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token, expires_in = create_token(request.username)
        return LoginResponse(access_token=token, expires_in=expires_in)

    @api.get("/auth/me")
    async def get_me(user_id: str = Depends(get_current_user)):
        return {"user_id": user_id}

    # -- Chat --

    @api.post("/chat/message")
    async def send_message(
        request: MessageRequest,
        http_request: Request,
        user_id: str = Depends(get_current_user),
    ):
        if not app.state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        # Set client IP for IP-based location (e.g. weather without explicit location)
        client_ip = (
            http_request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or http_request.headers.get("x-real-ip", "")
            or (http_request.client.host if http_request.client else "")
        )
        from ..utils.request_context import set_client_ip
        set_client_ip(client_ip)
        from ..features.registry import is_feature_enabled
        trace_on = is_feature_enabled("debug_trace")
        if trace_on:
            app.state.trace_log.append({"ts": datetime.now(timezone.utc).isoformat(), "event": "chat_start", "user_id": user_id, "content_preview": request.content[:80]})
        response_content = await app.state.orchestrator.chat(
            channel="web", user_id=user_id, content=request.content
        )
        if trace_on:
            app.state.trace_log.append({"ts": datetime.now(timezone.utc).isoformat(), "event": "chat_done", "user_id": user_id, "response_preview": response_content[:80]})
        return {"response": response_content, "timestamp": datetime.now(timezone.utc).isoformat()}

    @api.get("/chat/history")
    async def get_chat_history(limit: int = 50, user_id: str = Depends(get_current_user)):
        if not app.state.orchestrator:
            return {"messages": []}
        context = await app.state.orchestrator.context_manager.get_context(
            channel="web", user_id=user_id, create_if_missing=False
        )
        if not context:
            return {"messages": []}
        messages = context.get_messages(limit)
        return {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat() if hasattr(m, "timestamp") else None,
                }
                for m in messages
                if m.role != "system"
            ]
        }

    @api.delete("/chat/history")
    async def clear_chat_history(user_id: str = Depends(get_current_user)):
        if app.state.orchestrator:
            await app.state.orchestrator.context_manager.clear_context(
                channel="web", user_id=user_id
            )
        return {"cleared": True}

    # -- Approvals --

    @api.get("/approvals/pending")
    async def get_pending_approvals(user_id: str = Depends(get_current_user)):
        if not app.state.security_guardian:
            return {"approvals": []}
        # Web admin sees ALL pending approvals regardless of originating channel/user
        requests = app.state.security_guardian.get_pending_requests()
        return {
            "approvals": [
                {
                    "id": r.id,
                    "action_type": r.action_type,
                    "description": r.description,
                    "user_id": r.user_id,
                    "channel": r.channel,
                    "created_at": r.created_at.isoformat(),
                    "timeout": r.timeout,
                }
                for r in requests
            ]
        }

    @api.post("/approvals/respond")
    async def respond_to_approval(request: ApprovalRequest, user_id: str = Depends(get_current_user)):
        if not app.state.security_guardian:
            raise HTTPException(status_code=503, detail="Security guardian not available")
        result = await app.state.security_guardian.handle_approval_response(
            request_id=request.approval_id,
            approved=request.approved,
            approved_by=user_id,
            channel="web",
        )
        return {"processed": result}

    # -- Skills --

    @api.get("/skills")
    async def list_skills(user_id: str = Depends(get_current_user)):
        if not app.state.skill_registry:
            return {"skills": []}
        return {"skills": app.state.skill_registry.list_skills()}

    @api.get("/skills/{skill_name}")
    async def get_skill(skill_name: str, user_id: str = Depends(get_current_user)):
        if not app.state.skill_registry:
            raise HTTPException(status_code=503, detail="Skill registry not available")
        skill = app.state.skill_registry.get_skill(skill_name)
        if not skill:
            raise HTTPException(status_code=404, detail="Skill not found")
        return skill.to_dict()

    @api.post("/skills/toggle")
    async def toggle_skill(request: SkillToggle, user_id: str = Depends(get_current_user)):
        if not app.state.skill_registry:
            raise HTTPException(status_code=503, detail="Skill registry not available")
        if request.enabled:
            result = app.state.skill_registry.enable_skill(request.skill_name)
        else:
            result = app.state.skill_registry.disable_skill(request.skill_name)
        # Persist the change to YAML config
        cfg = _load_yaml_config()
        if "skills" not in cfg:
            cfg["skills"] = {}
        if "builtin" not in cfg["skills"]:
            cfg["skills"]["builtin"] = {}
        if request.skill_name not in cfg["skills"]["builtin"]:
            cfg["skills"]["builtin"][request.skill_name] = {}
        if isinstance(cfg["skills"]["builtin"][request.skill_name], dict):
            cfg["skills"]["builtin"][request.skill_name]["enabled"] = request.enabled
        _save_yaml_config(cfg)
        return {"success": result}

    # -- Settings (legacy) --

    @api.get("/settings")
    async def get_settings_endpoint(user_id: str = Depends(get_current_user)):
        return {
            "security": {
                "active_profile": settings.security.active_profile,
                "approval_timeout": settings.security.approval_timeout,
            },
            "llm": {
                "local_enabled": settings.llm.local.enabled,
                "local_model": settings.llm.local.model,
                "cloud_enabled": settings.llm.cloud.enabled,
                "cloud_model": settings.llm.cloud.model,
            },
            "channels": {
                "slack_enabled": settings.channels.slack.enabled,
                "whatsapp_enabled": settings.channels.whatsapp.enabled,
                "web_enabled": settings.channels.web.enabled,
            },
        }

    @api.post("/settings")
    async def update_settings(request: SettingsUpdate, user_id: str = Depends(get_current_user)):
        if request.security_profile and app.state.security_guardian:
            app.state.security_guardian.profile_manager.set_active_profile(
                request.security_profile
            )
        return {"updated": True, "restart_required": True}

    @api.get("/settings/profiles")
    async def get_security_profiles(user_id: str = Depends(get_current_user)):
        if not app.state.security_guardian:
            return {"profiles": []}
        return {"profiles": app.state.security_guardian.profile_manager.list_profiles()}

    # ── Configuration Management Endpoints ────────────────────────────────

    @api.get("/config")
    async def get_full_config(user_id: str = Depends(get_current_user)):
        """Return the full configuration state for the Settings UI."""
        cfg = _load_yaml_config()
        env = _load_env_values()
        return {
            "llm": {
                "provider": _detect_provider_mode(cfg),
                "anthropic_model": cfg.get("llm", {}).get("cloud", {}).get("model", "claude-sonnet-4-20250514"),
                "anthropic_key_set": bool(env.get("ANTHROPIC_API_KEY")),
                "ollama_enabled": cfg.get("llm", {}).get("local", {}).get("enabled", False),
                "ollama_model": cfg.get("llm", {}).get("local", {}).get("model", "llama3.2:latest"),
                "ollama_base_url": cfg.get("llm", {}).get("local", {}).get("base_url", "http://localhost:11434"),
                "cloud_enabled": cfg.get("llm", {}).get("cloud", {}).get("enabled", False),
            },
            "channels": {
                "web_enabled": True,
                "slack_enabled": cfg.get("channels", {}).get("slack", {}).get("enabled", False),
                "slack_bot_token_set": bool(env.get("SLACK_BOT_TOKEN")),
                "slack_app_token_set": bool(env.get("SLACK_APP_TOKEN")),
                "whatsapp_enabled": cfg.get("channels", {}).get("whatsapp", {}).get("enabled", False),
                "whatsapp_bridge_port": cfg.get("channels", {}).get("whatsapp", {}).get("bridge_port", 3001),
                "whatsapp_allowed_numbers": cfg.get("channels", {}).get("whatsapp", {}).get("allowed_numbers", []),
            },
            "environment": {
                "deployment_mode": cfg.get("aria", {}).get("deployment_mode", "local"),
                "sandbox_mode": cfg.get("sandbox", {}).get("default", "docker"),
                "docker_memory": cfg.get("sandbox", {}).get("docker", {}).get("memory_limit", "512m"),
                "docker_cpu": cfg.get("sandbox", {}).get("docker", {}).get("cpu_limit", 1.0),
                "trusted_paths": cfg.get("sandbox", {}).get("trusted_paths", ["~/Documents", "~/Projects", "~/Downloads"]),
            },
            "security": {
                "active_profile": cfg.get("security", {}).get("active_profile", "balanced"),
                "approval_timeout": cfg.get("security", {}).get("approval_timeout", 300),
            },
            "browser": {
                "mode": _detect_browser_mode(cfg),
                "brave_key_set": bool(env.get("BRAVE_API_KEY")),
            },
            "skills": {
                name: skill_cfg.get("enabled", False)
                for name, skill_cfg in cfg.get("skills", {}).get("builtin", {}).items()
                if isinstance(skill_cfg, dict)
            },
            "integrations": {
                "notion": {"enabled": _skill_enabled(cfg, "notion"), "api_key_set": bool(env.get("NOTION_API_KEY"))},
                "todoist": {"enabled": _skill_enabled(cfg, "todoist"), "api_key_set": bool(env.get("TODOIST_API_KEY"))},
                "linear": {"enabled": _skill_enabled(cfg, "linear"), "api_key_set": bool(env.get("LINEAR_API_KEY"))},
                "spotify": {"enabled": _skill_enabled(cfg, "spotify"), "client_id_set": bool(env.get("SPOTIFY_CLIENT_ID")), "client_secret_set": bool(env.get("SPOTIFY_CLIENT_SECRET"))},
            },
            "dashboard": {
                "port": cfg.get("channels", {}).get("web", {}).get("port", 8080),
            },
            "memory": {
                "knowledge_graph_enabled": cfg.get("memory", {}).get("knowledge_graph", {}).get("enabled", True),
                "knowledge_graph_provider": cfg.get("memory", {}).get("knowledge_graph", {}).get("provider", "cognee"),
                "knowledge_graph_auto_process_after_ingest": cfg.get("memory", {}).get("knowledge_graph", {}).get("auto_process_after_ingest", False),
            },
        }

    @api.put("/config/llm")
    async def update_llm_config(body: ConfigUpdate, user_id: str = Depends(get_current_user)):
        """Update LLM configuration."""
        cfg = _load_yaml_config()
        d = body.data

        if "llm" not in cfg:
            cfg["llm"] = {}
        if "local" not in cfg["llm"]:
            cfg["llm"]["local"] = {}
        if "cloud" not in cfg["llm"]:
            cfg["llm"]["cloud"] = {}

        if "ollama_enabled" in d:
            cfg["llm"]["local"]["enabled"] = d["ollama_enabled"]
        if "ollama_model" in d:
            cfg["llm"]["local"]["model"] = d["ollama_model"]
        if "ollama_base_url" in d:
            cfg["llm"]["local"]["base_url"] = d["ollama_base_url"]
        if "cloud_enabled" in d:
            cfg["llm"]["cloud"]["enabled"] = d["cloud_enabled"]
        if "anthropic_model" in d:
            cfg["llm"]["cloud"]["model"] = d["anthropic_model"]
        if "anthropic_api_key" in d:
            _set_env_value("ANTHROPIC_API_KEY", d["anthropic_api_key"])

        _save_yaml_config(cfg)
        return {"updated": True, "restart_required": True}

    @api.put("/config/channels")
    async def update_channels_config(body: ConfigUpdate, user_id: str = Depends(get_current_user)):
        """Update channels configuration."""
        cfg = _load_yaml_config()
        d = body.data

        if "channels" not in cfg:
            cfg["channels"] = {}

        # Slack
        if "slack_enabled" in d:
            if "slack" not in cfg["channels"]:
                cfg["channels"]["slack"] = {}
            cfg["channels"]["slack"]["enabled"] = d["slack_enabled"]
        if "slack_bot_token" in d:
            _set_env_value("SLACK_BOT_TOKEN", d["slack_bot_token"])
        if "slack_app_token" in d:
            _set_env_value("SLACK_APP_TOKEN", d["slack_app_token"])

        # WhatsApp
        if "whatsapp_enabled" in d:
            if "whatsapp" not in cfg["channels"]:
                cfg["channels"]["whatsapp"] = {}
            cfg["channels"]["whatsapp"]["enabled"] = d["whatsapp_enabled"]
        if "whatsapp_bridge_port" in d:
            if "whatsapp" not in cfg["channels"]:
                cfg["channels"]["whatsapp"] = {}
            cfg["channels"]["whatsapp"]["bridge_port"] = d["whatsapp_bridge_port"]
        if "whatsapp_allowed_numbers" in d:
            if "whatsapp" not in cfg["channels"]:
                cfg["channels"]["whatsapp"] = {}
            cfg["channels"]["whatsapp"]["allowed_numbers"] = d["whatsapp_allowed_numbers"]

        _save_yaml_config(cfg)
        return {"updated": True, "restart_required": True}

    @api.put("/config/environment")
    async def update_environment_config(body: ConfigUpdate, user_id: str = Depends(get_current_user)):
        """Update execution environment configuration."""
        cfg = _load_yaml_config()
        d = body.data

        if "sandbox" not in cfg:
            cfg["sandbox"] = {}

        if "deployment_mode" in d:
            if "aria" not in cfg:
                cfg["aria"] = {}
            cfg["aria"]["deployment_mode"] = d["deployment_mode"]

        if "sandbox_mode" in d:
            cfg["sandbox"]["default"] = d["sandbox_mode"]
        if "docker_memory" in d:
            if "docker" not in cfg["sandbox"]:
                cfg["sandbox"]["docker"] = {}
            cfg["sandbox"]["docker"]["memory_limit"] = d["docker_memory"]
        if "docker_cpu" in d:
            if "docker" not in cfg["sandbox"]:
                cfg["sandbox"]["docker"] = {}
            cfg["sandbox"]["docker"]["cpu_limit"] = d["docker_cpu"]
        if "trusted_paths" in d:
            cfg["sandbox"]["trusted_paths"] = d["trusted_paths"]

        _save_yaml_config(cfg)
        return {"updated": True, "restart_required": True}

    @api.put("/config/security")
    async def update_security_config(body: ConfigUpdate, user_id: str = Depends(get_current_user)):
        """Update security configuration."""
        cfg = _load_yaml_config()
        d = body.data

        if "security" not in cfg:
            cfg["security"] = {}

        if "active_profile" in d:
            cfg["security"]["active_profile"] = d["active_profile"]
            if app.state.security_guardian:
                app.state.security_guardian.profile_manager.set_active_profile(d["active_profile"])
        if "approval_timeout" in d:
            cfg["security"]["approval_timeout"] = d["approval_timeout"]

        _save_yaml_config(cfg)
        return {"updated": True}

    @api.put("/config/browser")
    async def update_browser_config(body: ConfigUpdate, user_id: str = Depends(get_current_user)):
        """Update browser configuration."""
        cfg = _load_yaml_config()
        d = body.data

        if "skills" not in cfg:
            cfg["skills"] = {}
        if "builtin" not in cfg["skills"]:
            cfg["skills"]["builtin"] = {}

        if "mode" in d:
            mode = d["mode"]
            if mode == "playwright":
                cfg["skills"]["builtin"]["browser"] = cfg["skills"]["builtin"].get("browser", {})
                cfg["skills"]["builtin"]["browser"]["enabled"] = True
            elif mode == "none":
                cfg["skills"]["builtin"]["browser"] = cfg["skills"]["builtin"].get("browser", {})
                cfg["skills"]["builtin"]["browser"]["enabled"] = False

        if "brave_api_key" in d:
            _set_env_value("BRAVE_API_KEY", d["brave_api_key"])

        _save_yaml_config(cfg)
        return {"updated": True, "restart_required": True}

    @api.put("/config/skills")
    async def update_skills_config(body: ConfigUpdate, user_id: str = Depends(get_current_user)):
        """Update skills enable/disable configuration."""
        cfg = _load_yaml_config()
        d = body.data  # expects {"skill_name": true/false, ...}

        if "skills" not in cfg:
            cfg["skills"] = {}
        if "builtin" not in cfg["skills"]:
            cfg["skills"]["builtin"] = {}

        for skill_name, enabled in d.items():
            if skill_name not in cfg["skills"]["builtin"]:
                cfg["skills"]["builtin"][skill_name] = {}
            if isinstance(cfg["skills"]["builtin"][skill_name], dict):
                cfg["skills"]["builtin"][skill_name]["enabled"] = enabled

        _save_yaml_config(cfg)
        return {"updated": True, "restart_required": True}

    @api.put("/config/integrations")
    async def update_integrations_config(body: ConfigUpdate, user_id: str = Depends(get_current_user)):
        """Update integration API keys and enable/disable."""
        cfg = _load_yaml_config()
        d = body.data

        if "skills" not in cfg:
            cfg["skills"] = {}
        if "builtin" not in cfg["skills"]:
            cfg["skills"]["builtin"] = {}

        for name in ("notion", "todoist", "linear"):
            if f"{name}_enabled" in d:
                if name not in cfg["skills"]["builtin"]:
                    cfg["skills"]["builtin"][name] = {}
                cfg["skills"]["builtin"][name]["enabled"] = d[f"{name}_enabled"]
            if f"{name}_api_key" in d and d[f"{name}_api_key"]:
                _set_env_value(f"{name.upper()}_API_KEY", d[f"{name}_api_key"])

        if "spotify_enabled" in d:
            if "spotify" not in cfg["skills"]["builtin"]:
                cfg["skills"]["builtin"]["spotify"] = {}
            cfg["skills"]["builtin"]["spotify"]["enabled"] = d["spotify_enabled"]
        if d.get("spotify_client_id"):
            _set_env_value("SPOTIFY_CLIENT_ID", d["spotify_client_id"])
        if d.get("spotify_client_secret"):
            _set_env_value("SPOTIFY_CLIENT_SECRET", d["spotify_client_secret"])

        _save_yaml_config(cfg)
        return {"updated": True, "restart_required": True}

    @api.put("/config/dashboard")
    async def update_dashboard_config(body: ConfigUpdate, user_id: str = Depends(get_current_user)):
        """Update dashboard configuration (port, password)."""
        cfg = _load_yaml_config()
        d = body.data

        if "admin_password" in d:
            _set_env_value("ADMIN_PASSWORD", d["admin_password"])
            app.state.users["admin"] = hash_password(d["admin_password"])

        if "port" in d:
            if "channels" not in cfg:
                cfg["channels"] = {}
            if "web" not in cfg["channels"]:
                cfg["channels"]["web"] = {}
            cfg["channels"]["web"]["port"] = d["port"]

        _save_yaml_config(cfg)
        return {"updated": True, "restart_required": "port" in d}

    @api.post("/config/validate-key")
    async def validate_api_key(body: KeyValidation, user_id: str = Depends(get_current_user)):
        """Validate an API key before saving."""
        from src.cli.detection import SystemDetector

        if body.key_type == "anthropic":
            valid, msg = SystemDetector.validate_anthropic_key(body.key_value)
            return {"valid": valid, "message": msg}

        if body.key_type == "brave":
            valid, msg = SystemDetector.validate_brave_key(body.key_value)
            return {"valid": valid, "message": msg}

        if body.key_type == "slack_bot":
            app_token = (body.extra or {}).get("app_token", "")
            valid, msg = SystemDetector.check_slack_credentials(body.key_value, app_token)
            return {"valid": valid, "message": msg}

        return {"valid": False, "message": f"Unknown key type: {body.key_type}"}

    @api.get("/config/detection")
    async def run_detection(user_id: str = Depends(get_current_user)):
        """Run system detection and return results."""
        from src.cli.detection import SystemDetector

        detector = SystemDetector()
        results = detector.run_all()
        return {
            "ollama": {"installed": results.ollama.installed, "running": results.ollama.running, "version": results.ollama.version, "models": results.ollama.extra.get("models", [])},
            "docker": {"installed": results.docker.installed, "running": results.docker.running, "version": results.docker.version, "has_sandbox_image": results.docker.extra.get("has_sandbox_image", False)},
            "ffmpeg": {"installed": results.ffmpeg.installed, "version": results.ffmpeg.version},
            "playwright": {"installed": results.playwright.installed, "has_chromium": results.playwright.extra.get("has_chromium", False)},
            "node": {"installed": results.node.installed, "version": results.node.version},
            "anthropic_key": {"found": results.anthropic_key.installed, "source": results.anthropic_key.extra.get("source", "")},
            "brave_key": {"found": results.brave_key.installed},
        }

    # -- Audit --

    @api.get("/audit")
    async def get_audit_log(
        limit: int = 100, offset: int = 0, event: str | None = None,
        user_id: str = Depends(get_current_user),
    ):
        if not app.state.audit_logger:
            return {"entries": [], "total": 0}
        entries = await app.state.audit_logger.query(event=event, limit=limit, offset=offset)
        return {
            "entries": [
                {
                    "id": e.id, "timestamp": e.timestamp.isoformat(), "event": e.event,
                    "action_type": e.action_type, "user_id": e.user_id, "channel": e.channel,
                    "status": e.status, "details": e.details,
                }
                for e in entries
            ],
            "limit": limit, "offset": offset,
        }

    @api.get("/audit/stats")
    async def get_audit_stats(user_id: str = Depends(get_current_user)):
        if not app.state.audit_logger:
            return {}
        return await app.state.audit_logger.get_stats()

    # -- System --

    @api.get("/system/status")
    async def get_system_status(user_id: str = Depends(get_current_user)):
        status_data: dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat()}
        if app.state.orchestrator:
            status_data["llm"] = {
                "local_available": app.state.orchestrator.llm_router._local_available,
                "cloud_available": app.state.orchestrator.llm_router._cloud_available,
            }
            status_data["context"] = app.state.orchestrator.context_manager.get_stats()
            status_data["queue"] = app.state.orchestrator.message_router.get_stats()
            status_data["rag"] = app.state.orchestrator.rag_pipeline.get_stats()
        if app.state.skill_registry:
            status_data["skills"] = app.state.skill_registry.get_stats()
        if app.state.security_guardian:
            status_data["security"] = app.state.security_guardian.get_stats()
        # New subsystems
        if hasattr(app.state, "vector_memory") and app.state.vector_memory:
            status_data["vector_memory"] = app.state.vector_memory.get_stats()
        if hasattr(app.state, "process_manager") and app.state.process_manager:
            status_data["background_processes"] = len(app.state.process_manager.list_processes(running_only=True))
        if hasattr(app.state, "scheduler") and app.state.scheduler:
            status_data["scheduled_jobs"] = len(app.state.scheduler.list_jobs())
        if hasattr(app.state, "plugin_loader") and app.state.plugin_loader:
            status_data["plugins"] = len(app.state.plugin_loader.list_plugins())
        if hasattr(app.state, "device_manager") and app.state.device_manager:
            status_data["paired_devices"] = len(app.state.device_manager.list_devices())
        return status_data

    @api.post("/knowledge/process")
    async def process_knowledge_graph(user_id: str = Depends(get_current_user)):
        """Trigger cognee to process accumulated knowledge into the graph."""
        if not app.state.orchestrator or not app.state.orchestrator._rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
        success = await app.state.orchestrator._rag_pipeline.process_knowledge_graph()
        return {"success": success}

    @api.put("/config/memory")
    async def update_memory_config(body: ConfigUpdate, user_id: str = Depends(get_current_user)):
        """Update memory/knowledge graph configuration."""
        cfg = _load_yaml_config()
        d = body.data

        if "memory" not in cfg:
            cfg["memory"] = {}
        if "knowledge_graph" not in cfg["memory"]:
            cfg["memory"]["knowledge_graph"] = {}

        if "knowledge_graph_enabled" in d:
            cfg["memory"]["knowledge_graph"]["enabled"] = d["knowledge_graph_enabled"]
        if "knowledge_graph_provider" in d:
            cfg["memory"]["knowledge_graph"]["provider"] = d["knowledge_graph_provider"]
        if "knowledge_graph_auto_process_after_ingest" in d:
            cfg["memory"]["knowledge_graph"]["auto_process_after_ingest"] = d["knowledge_graph_auto_process_after_ingest"]

        _save_yaml_config(cfg)
        return {"updated": True, "restart_required": True}

    # -- HUD / Dashboard --

    @api.get("/features")
    async def list_features(user_id: str = Depends(get_current_user)):
        """List all features with status."""
        from ..features.registry import FEATURES, is_feature_enabled
        return {
            "features": [
                {
                    "id": f.id,
                    "name": f.name,
                    "description": f.description,
                    "category": f.category,
                    "enabled": is_feature_enabled(f.id),
                }
                for f in FEATURES.values()
            ]
        }

    @api.put("/features/{feature_id}")
    async def toggle_feature(
        feature_id: str,
        body: dict = Body(..., embed=True),
        user_id: str = Depends(get_current_user),
    ):
        """Toggle a feature on/off. Body: { "enabled": true|false }."""
        from ..features.registry import FEATURES
        if feature_id not in FEATURES:
            raise HTTPException(404, f"Unknown feature: {feature_id}")
        enabled = body.get("enabled", True)
        cfg = _load_yaml_config()
        if "feature_overrides" not in cfg:
            cfg["feature_overrides"] = {}
        cfg["feature_overrides"][feature_id] = bool(enabled)
        _save_yaml_config(cfg)
        from ..utils.config import reload_settings
        reload_settings()
        return {"feature_id": feature_id, "enabled": enabled}

    @api.get("/usage")
    async def get_usage_stats(user_id: str = Depends(get_current_user)):
        """LLM usage and cost tracking."""
        from ..core.usage_tracker import get_usage_tracker
        return get_usage_tracker().get_stats()

    @api.get("/export")
    async def export_data(
        user_id: str = Depends(get_current_user),
        type: str = Query("all", description="all | conversations | audit"),
    ):
        """Export data (conversations, audit logs)."""
        from fastapi.responses import StreamingResponse
        import io
        settings = get_settings()
        data_dir = Path(settings.aria.data_dir).expanduser()

        if type == "conversations":
            # Export from context manager if available
            if app.state.orchestrator:
                ctxs = await app.state.orchestrator.context_manager.get_active_contexts()
                export_data = {"conversations": [{"channel": c.channel, "user_id": c.user_id} for c in ctxs]}
            else:
                export_data = {"conversations": []}
            buf = io.BytesIO(json.dumps(export_data, indent=2).encode())
            return StreamingResponse(
                iter([buf.getvalue()]),
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename=aria-export-{type}.json'},
            )
        elif type == "audit":
            audit_path = Path(settings.logging.audit_file).expanduser()
            if audit_path.exists():
                return FileResponse(audit_path, filename="audit.log")
            raise HTTPException(404, "Audit log not found")
        else:
            # all
            export_data = {"exported_at": datetime.now(timezone.utc).isoformat(), "type": "full"}
            if app.state.orchestrator:
                ctxs = await app.state.orchestrator.context_manager.get_active_contexts()
                export_data["conversations"] = [{"channel": c.channel, "user_id": c.user_id} for c in ctxs]
            from ..core.usage_tracker import get_usage_tracker
            export_data["usage"] = get_usage_tracker().get_stats()
            buf = io.BytesIO(json.dumps(export_data, indent=2).encode())
            return StreamingResponse(
                iter([buf.getvalue()]),
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=aria-export-full.json"},
            )

    @api.post("/push/subscribe")
    async def push_subscribe(
        body: dict = Body(...),
        user_id: str = Depends(get_current_user),
    ):
        """Register a push subscription for notifications (Web Push API)."""
        if not hasattr(app.state, "push_subscriptions"):
            app.state.push_subscriptions = {}
        key = body.get("endpoint") or str(body)
        app.state.push_subscriptions[user_id] = body
        return {"success": True, "message": "Push subscription registered"}

    @api.get("/debug/trace")
    async def get_trace_log(user_id: str = Depends(get_current_user)):
        """Get recent trace log when debug_trace feature is enabled."""
        from ..features.registry import is_feature_enabled
        if not is_feature_enabled("debug_trace"):
            return {"traces": [], "message": "Debug trace disabled"}
        traces = getattr(app.state, "trace_log", [])[-100:]
        return {"traces": traces}

    @api.get("/skill-templates")
    async def list_skill_templates(user_id: str = Depends(get_current_user)):
        """List available skill templates for quick-start."""
        from ..features.registry import is_feature_enabled
        if not is_feature_enabled("skill_templates"):
            return {"templates": []}
        settings = get_settings()
        templates_dir = Path(settings.aria.data_dir).expanduser() / "skill_templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        templates = []
        for f in templates_dir.glob("*.py"):
            try:
                content = f.read_text(encoding="utf-8")
                # Extract docstring or first 200 chars as description
                desc = content.split('"""')[1][:150] if '"""' in content else f.name
                templates.append({"id": f.stem, "name": f.stem.replace("_", " ").title(), "description": desc, "file": f.name})
            except Exception:
                pass
        # Seed default templates if empty
        if not templates:
            for name, content in _default_skill_templates():
                path = templates_dir / f"{name}.py"
                if not path.exists():
                    path.write_text(content, encoding="utf-8")
                desc = content.split('"""')[1].split('"""')[0].strip()[:100] if '"""' in content else name
                templates.append({"id": name, "name": name.replace("_", " ").title(), "description": desc, "file": f"{name}.py"})
        return {"templates": templates}

    @api.get("/widgets")
    async def list_custom_widgets(user_id: str = Depends(get_current_user)):
        """List user-defined dashboard widgets."""
        from ..features.registry import is_feature_enabled
        if not is_feature_enabled("custom_widgets"):
            return {"widgets": []}
        cfg = _load_yaml_config()
        widgets = cfg.get("dashboard", {}).get("custom_widgets", [])
        return {"widgets": widgets}

    @api.put("/widgets")
    async def update_custom_widgets(
        body: dict = Body(...),
        user_id: str = Depends(get_current_user),
    ):
        """Update custom widget definitions. Body: { "widgets": [...] }."""
        from ..features.registry import is_feature_enabled
        if not is_feature_enabled("custom_widgets"):
            raise HTTPException(403, "Custom widgets disabled")
        cfg = _load_yaml_config()
        if "dashboard" not in cfg:
            cfg["dashboard"] = {}
        cfg["dashboard"]["custom_widgets"] = body.get("widgets", [])
        _save_yaml_config(cfg)
        return {"updated": True}

    @api.get("/hud/vitals")
    async def get_hud_vitals(user_id: str = Depends(get_current_user)):
        """System vitals for HUD dashboard."""
        vitals: dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat()}
        try:
            import psutil
            vitals["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            vitals["memory_percent"] = psutil.virtual_memory().percent
            vitals["memory_used_gb"] = round(psutil.virtual_memory().used / (1024**3), 2)
            vitals["memory_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
            vitals["disk_percent"] = psutil.disk_usage("/").percent
        except ImportError:
            vitals["cpu_percent"] = 0
            vitals["memory_percent"] = 0
            vitals["error"] = "psutil not installed"
        if app.state.orchestrator:
            vitals["llm"] = {
                "local": app.state.orchestrator.llm_router._local_available,
                "cloud": app.state.orchestrator.llm_router._cloud_available,
            }
        if hasattr(app.state, "orchestrator") and app.state.orchestrator and hasattr(app.state.orchestrator, "_channels"):
            vitals["channels"] = {n: c.is_connected for n, c in app.state.orchestrator._channels.items()}
        return vitals

    @api.get("/hud/timeline")
    async def get_hud_timeline(user_id: str = Depends(get_current_user)):
        """Today's conversations for timeline view."""
        if not app.state.orchestrator:
            return {"events": []}
        ctxs = await app.state.orchestrator.context_manager.get_active_contexts()
        today = datetime.now(timezone.utc).date()
        events = []
        for ctx in ctxs:
            if ctx.updated_at and ctx.updated_at.date() == today:
                events.append({
                    "channel": ctx.channel,
                    "user_id": ctx.user_id,
                    "updated_at": ctx.updated_at.isoformat(),
                    "message_count": len([m for m in ctx.messages if m.role != "system"]),
                })
        return {"events": events[:50]}

    @api.get("/hud/agents")
    async def get_hud_agents(user_id: str = Depends(get_current_user)):
        """Active agent tasks."""
        if not hasattr(app.state, "agent_coordinator") or not app.state.agent_coordinator:
            return {"agents": []}
        return {"agents": app.state.agent_coordinator.list_running_agents()}

    @api.get("/hud/agents/full")
    async def get_all_agents_full(user_id: str = Depends(get_current_user)):
        """All agent tasks with bot status for real-time dashboard."""
        if not hasattr(app.state, "agent_coordinator") or not app.state.agent_coordinator:
            return {"agents": []}
        return {"agents": app.state.agent_coordinator.list_all_agents(include_completed=True)}

    # -- Agents --

    @api.get("/agents")
    async def list_agents(user_id: str = Depends(get_current_user)):
        """List running/completed agent tasks."""
        if not hasattr(app.state, "agent_coordinator") or not app.state.agent_coordinator:
            return {"tasks": []}
        return {"tasks": app.state.agent_coordinator.list_running_agents()}

    @api.post("/agents/run")
    async def run_agent(
        request: dict = Body(...),
        user_id: str = Depends(get_current_user),
    ):
        """Start an agent task."""
        if not hasattr(app.state, "agent_coordinator") or not app.state.agent_coordinator:
            raise HTTPException(status_code=503, detail="Agent coordinator not available")
        task = request.get("task", "")
        agent_type = request.get("agent_type")
        if not task:
            raise HTTPException(status_code=400, detail="task required")
        # Multi-destination tasks (itinerary, etc.): decompose and run parallel bots
        if agent_type in ("itinerary", "automate") and hasattr(app.state.agent_coordinator, "delegate_parallel_subtasks"):
            result = await app.state.agent_coordinator.delegate_parallel_subtasks(
                task=task,
                user_id=user_id,
                channel="web",
            )
        # Research: multi-bot (Reddit + Web + X in parallel)
        elif agent_type == "research" and hasattr(app.state.agent_coordinator, "delegate_multi_bot"):
            result = await app.state.agent_coordinator.delegate_multi_bot(
                task=task,
                user_id=user_id,
                channel="web",
            )
        else:
            result = await app.state.agent_coordinator.delegate(
                task=task,
                agent_type=agent_type,
                user_id=user_id,
                channel="web",
            )
        return {"success": result.success, "output": result.output, "task_id": result.task_id, "error": result.error}

    @api.get("/system/health")
    async def health_check():
        """Health check with channel status for diagnostics."""
        channel_status = {}
        if orchestrator:
            for name, ch in orchestrator._channels.items():
                channel_status[name] = {
                    "connected": ch.is_connected,
                    "handlers": len(ch._message_handlers),
                }

        return {
            "status": "healthy",
            "channels": channel_status,
        }

    @api.post("/system/restart")
    async def restart_system(user_id: str = Depends(get_current_user)):
        """Restart the Aria process.

        In Docker deployment mode, rebuilds and restarts the containers
        via ``docker compose up -d --build`` (using the Docker socket
        mounted into the container).  Otherwise does a local process
        restart with ``os.execv``.
        """
        import asyncio
        import subprocess as sp
        import sys

        cfg = _load_yaml_config()
        deployment_mode = cfg.get("aria", {}).get("deployment_mode", "local")
        in_container = os.environ.get("ARIA_IN_CONTAINER")

        async def _do_restart():
            await asyncio.sleep(1)

            if in_container:
                # Inside Docker — use mounted socket to rebuild + restart
                host_dir = os.environ.get("HOST_PROJECT_DIR", "")
                has_socket = Path("/var/run/docker.sock").exists()

                if has_socket and host_dir:
                    # Find compose file on the host via HOST_PROJECT_DIR
                    compose_file = f"{host_dir}/docker-compose.yaml"
                    logger.info(
                        "Rebuilding Docker containers from inside container",
                        compose_file=compose_file,
                        project_dir=host_dir,
                    )
                    # Fire-and-forget: compose will restart this container
                    sp.Popen(
                        [
                            "docker", "compose",
                            "-f", compose_file,
                            "--project-directory", host_dir,
                            "up", "-d", "--build",
                        ],
                    )
                    await asyncio.sleep(2)
                    os._exit(0)
                else:
                    # No socket or path — just exit; compose restart policy restarts us
                    logger.info("No Docker socket — exiting for compose to restart")
                    os._exit(0)

            elif deployment_mode == "docker":
                # Host-side with Docker mode — rebuild + restart containers
                project_root = Path(__file__).resolve().parent.parent.parent
                compose_file = project_root / "docker-compose.yaml"
                if not compose_file.exists():
                    compose_file = project_root / "docker" / "docker-compose.yaml"
                compose_dir = compose_file.parent if compose_file.exists() else project_root

                logger.info("Restarting Docker containers with rebuild...")
                sp.Popen(
                    ["docker", "compose", "up", "-d", "--build"],
                    cwd=str(compose_dir),
                )
                await asyncio.sleep(2)
                os._exit(0)

            else:
                # Local mode — classic os.execv restart
                restart_argv = [sys.executable] + sys.argv
                if "--skip-setup" not in restart_argv:
                    restart_argv.append("--skip-setup")
                os.execv(sys.executable, restart_argv)

        asyncio.ensure_future(_do_restart())
        return {"restarting": True, "message": "Aria is restarting..."}

    @api.post("/system/reset")
    async def reset_all_settings(user_id: str = Depends(get_current_user)):
        """Reset all settings by removing config files and the configured marker.
        After reset, the next start will trigger the setup wizard."""
        import asyncio
        import sys

        removed = []
        try:
            config_file = Path("config/settings.yaml")
            if config_file.exists():
                config_file.unlink()
                removed.append("config/settings.yaml")

            env_file = Path(".env")
            if env_file.exists():
                env_file.unlink()
                removed.append(".env")

            marker = Path("data/.aria_configured")
            if marker.exists():
                marker.unlink()
                removed.append("data/.aria_configured")

            logger.info("Settings reset", removed=removed)

            # Schedule restart (will trigger setup wizard)
            async def _do_restart():
                await asyncio.sleep(1)
                in_container = os.environ.get("ARIA_IN_CONTAINER")

                if in_container:
                    # Exit container — compose restart policy brings it back
                    os._exit(0)
                else:
                    restart_argv = [sys.executable] + sys.argv
                    # Remove --skip-setup so wizard runs
                    restart_argv = [a for a in restart_argv if a != "--skip-setup"]
                    os.execv(sys.executable, restart_argv)

            asyncio.ensure_future(_do_restart())
            return {
                "success": True,
                "message": "All settings reset. Aria will restart and run the setup wizard.",
                "removed": removed,
            }
        except Exception as e:
            logger.error("Failed to reset settings", error=str(e))
            return {"success": False, "message": f"Reset failed: {e}"}

    @api.post("/channels/test-slack")
    async def test_slack_connection(user_id: str = Depends(get_current_user)):
        """Send a test message on Slack to verify connection."""
        env = _load_env_values()
        bot_token = env.get("SLACK_BOT_TOKEN") or os.environ.get("SLACK_BOT_TOKEN")
        if not bot_token:
            return {"success": False, "message": "Slack bot token not configured"}
        try:
            import aiohttp
            # Disable auto-decompress to avoid Brotli issues; set Accept-Encoding explicitly
            headers = {
                "Authorization": f"Bearer {bot_token}",
                "Accept-Encoding": "gzip, deflate",
            }
            async with aiohttp.ClientSession(auto_decompress=False) as session:
                # Use auth.test to find the bot user
                async with session.post(
                    "https://slack.com/api/auth.test",
                    headers=headers,
                ) as resp:
                    import gzip, zlib
                    raw = await resp.read()
                    ce = resp.headers.get("Content-Encoding", "")
                    if ce == "gzip":
                        raw = gzip.decompress(raw)
                    elif ce == "deflate":
                        raw = zlib.decompress(raw)
                    import json as _json
                    auth_data = _json.loads(raw)
                    if not auth_data.get("ok"):
                        return {"success": False, "message": f"Auth failed: {auth_data.get('error', 'unknown')}"}
                    bot_user_id = auth_data.get("user_id")

                # Post to the first allowed channel or the bot user DM
                cfg = _load_yaml_config()
                channels_list = cfg.get("channels", {}).get("slack", {}).get("allowed_channels", [])
                test_channel = channels_list[0] if channels_list else bot_user_id

                async with session.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={**headers, "Content-Type": "application/json"},
                    data=_json.dumps({"channel": test_channel, "text": "Aria test message — Slack connection is working!"}),
                ) as resp:
                    raw = await resp.read()
                    ce = resp.headers.get("Content-Encoding", "")
                    if ce == "gzip":
                        raw = gzip.decompress(raw)
                    elif ce == "deflate":
                        raw = zlib.decompress(raw)
                    result = _json.loads(raw)
                    if result.get("ok"):
                        return {"success": True, "message": f"Test message sent to {test_channel}"}
                    return {"success": False, "message": result.get("error", "Failed to send")}
        except Exception as e:
            return {"success": False, "message": str(e)}

    @api.post("/channels/test-whatsapp")
    async def test_whatsapp_connection(user_id: str = Depends(get_current_user)):
        """Send a test message via WhatsApp bridge."""
        cfg = _load_yaml_config()
        bridge_port = cfg.get("channels", {}).get("whatsapp", {}).get("bridge_port", 3001)
        bridge_url = f"http://localhost:{bridge_port}"
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # Check status first
                async with session.get(f"{bridge_url}/status", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    status = await resp.json()
                    if not status.get("ready"):
                        return {"success": False, "message": "WhatsApp not connected. Scan the QR code first."}

                # Get the list of chats to find a valid chatId to send to
                async with session.get(f"{bridge_url}/chats", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        chats = await resp.json()
                        # Find the first non-group chat (or any chat)
                        target_chat = None
                        for chat in chats:
                            if not chat.get("isGroup"):
                                target_chat = chat.get("id")
                                break
                        if not target_chat and chats:
                            target_chat = chats[0].get("id")

                        if not target_chat:
                            return {"success": True, "message": "WhatsApp is connected but no chats found. Send a message to the number first."}

                        # Send test message using the bridge's /send endpoint
                        async with session.post(
                            f"{bridge_url}/send",
                            json={"chatId": target_chat, "content": "Aria test message — WhatsApp connection is working!"},
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as send_resp:
                            if send_resp.status == 200:
                                result = await send_resp.json()
                                if result.get("success"):
                                    return {"success": True, "message": "Test message sent on WhatsApp"}
                            return {"success": False, "message": "Failed to send test message via bridge"}
                    return {"success": False, "message": "Could not get chat list from bridge"}
        except Exception as e:
            return {"success": False, "message": f"Could not reach WhatsApp bridge: {str(e)}"}

    @api.post("/chat/transcribe")
    async def transcribe_audio(request: Request, user_id: str = Depends(get_current_user)):
        """Transcribe uploaded audio using STT skill."""
        if not app.state.skill_registry:
            raise HTTPException(status_code=503, detail="Skill registry not available")
        import base64
        body = await request.json()
        audio_data = body.get("audio")  # base64 encoded
        audio_format = body.get("format", "webm")  # webm, mp4, ogg, wav
        if not audio_data:
            raise HTTPException(status_code=400, detail="No audio data provided")
        result = await app.state.skill_registry.execute(
            "stt", "transcribe_bytes", audio_data=audio_data, format=audio_format
        )
        if result.success:
            # STT skill returns {"text": "...", "language": "...", "segments": [...]}
            # Extract just the text string for the frontend
            text = result.output
            if isinstance(text, dict):
                text = text.get("text", str(text))
            elif not isinstance(text, str):
                text = str(text)
            return {"text": text, "success": True}
        return {"text": "", "success": False, "error": result.error}

    # -- WhatsApp Bridge --

    @api.get("/whatsapp/status")
    async def whatsapp_bridge_status(user_id: str = Depends(get_current_user)):
        """Get WhatsApp bridge status."""
        cfg = _load_yaml_config()
        bridge_port = cfg.get("channels", {}).get("whatsapp", {}).get("bridge_port", 3001)
        bridge_url = f"http://localhost:{bridge_port}"
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{bridge_url}/status", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        return await resp.json()
            return {"ready": False, "hasQR": False, "bridge_running": False}
        except Exception:
            return {"ready": False, "hasQR": False, "bridge_running": False}

    @api.post("/whatsapp/start")
    async def start_whatsapp_bridge(user_id: str = Depends(get_current_user)):
        """Start the WhatsApp bridge process."""
        import asyncio
        import subprocess

        cfg = _load_yaml_config()
        bridge_port = cfg.get("channels", {}).get("whatsapp", {}).get("bridge_port", 3001)
        bridge_dir = Path("whatsapp-bridge")
        if not bridge_dir.exists():
            return {"success": False, "message": "WhatsApp bridge directory not found. Run setup first."}
        try:
            # Check if already running
            import aiohttp
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:{bridge_port}/status",
                        timeout=aiohttp.ClientTimeout(total=2),
                    ) as resp:
                        if resp.status == 200:
                            return {"success": True, "message": "Bridge is already running"}
            except Exception:
                pass

            # Start the bridge
            env = os.environ.copy()
            env["PORT"] = str(bridge_port)
            subprocess.Popen(
                ["node", "index.js"],
                cwd=str(bridge_dir),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await asyncio.sleep(2)
            return {"success": True, "message": f"WhatsApp bridge starting on port {bridge_port}"}
        except FileNotFoundError:
            return {"success": False, "message": "Node.js not found. Install Node.js first."}
        except Exception as e:
            return {"success": False, "message": str(e)}

    @api.get("/whatsapp/qr")
    async def whatsapp_qr_code(user_id: str = Depends(get_current_user)):
        """Get the current WhatsApp QR code from the bridge."""
        cfg = _load_yaml_config()
        bridge_port = cfg.get("channels", {}).get("whatsapp", {}).get("bridge_port", 3001)
        bridge_url = f"http://localhost:{bridge_port}"
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{bridge_url}/qr", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data
                    elif resp.status == 404:
                        return {"qr": None, "message": "QR code not available yet. Bridge may still be initializing."}
            return {"qr": None, "message": "Could not reach WhatsApp bridge"}
        except Exception:
            return {"qr": None, "message": "WhatsApp bridge is not running. Start it first."}

    # ── Skill Credentials Management ─────────────────────────────────────

    # Map of skill name → required env vars
    SKILL_CREDENTIALS = {
        "calendar": [
            {"key": "GOOGLE_CREDENTIALS_FILE", "label": "Google Credentials File Path", "secret": False},
        ],
        "email": [
            {"key": "EMAIL_HOST", "label": "SMTP Host", "secret": False},
            {"key": "EMAIL_PORT", "label": "SMTP Port", "secret": False},
            {"key": "EMAIL_USER", "label": "Email Address", "secret": False},
            {"key": "EMAIL_PASSWORD", "label": "Email Password", "secret": True},
            {"key": "IMAP_HOST", "label": "IMAP Host", "secret": False},
            {"key": "IMAP_PORT", "label": "IMAP Port", "secret": False},
        ],
        "sms": [
            {"key": "TWILIO_ACCOUNT_SID", "label": "Twilio Account SID", "secret": True},
            {"key": "TWILIO_AUTH_TOKEN", "label": "Twilio Auth Token", "secret": True},
            {"key": "TWILIO_PHONE_NUMBER", "label": "Twilio Phone Number", "secret": False},
        ],
        "browser": [
            {"key": "BRAVE_API_KEY", "label": "Brave Search API Key (optional)", "secret": True},
        ],
        "weather": [
            {"key": "WEATHERAPI_KEY", "label": "WeatherAPI.com API Key", "secret": True},
        ],
    }

    @api.get("/skills/{skill_name}/credentials")
    async def get_skill_credentials(skill_name: str, user_id: str = Depends(get_current_user)):
        """Get credential fields for a skill and whether each is configured."""
        fields = SKILL_CREDENTIALS.get(skill_name, [])
        if not fields:
            return {"skill": skill_name, "fields": [], "has_credentials": False}

        env = _load_env_values()
        result_fields = []
        all_set = True
        for f in fields:
            val = env.get(f["key"], "") or os.environ.get(f["key"], "")
            is_set = bool(val)
            if not is_set and f["key"] not in ("BRAVE_API_KEY",):  # optional fields
                all_set = False
            result_fields.append({
                "key": f["key"],
                "label": f["label"],
                "secret": f["secret"],
                "is_set": is_set,
                "value": "" if f["secret"] else val,  # never expose secret values
            })

        return {"skill": skill_name, "fields": result_fields, "has_credentials": all_set}

    @api.post("/skills/{skill_name}/credentials")
    async def save_skill_credentials(skill_name: str, request: Request, user_id: str = Depends(get_current_user)):
        """Save credential values for a skill to .env file."""
        body = await request.json()
        credentials = body.get("credentials", {})
        fields = SKILL_CREDENTIALS.get(skill_name, [])
        valid_keys = {f["key"] for f in fields}

        saved = []
        for key, value in credentials.items():
            if key in valid_keys and value:  # only save non-empty values
                _set_env_value(key, value)
                saved.append(key)

        return {"success": True, "saved": saved, "message": f"Saved {len(saved)} credential(s) for {skill_name}"}

    @api.post("/skills/{skill_name}/test")
    async def test_skill_connection(skill_name: str, user_id: str = Depends(get_current_user)):
        """Test if a skill's credentials are working."""
        env = _load_env_values()

        if skill_name == "email":
            host = env.get("EMAIL_HOST") or os.environ.get("EMAIL_HOST", "")
            port = int(env.get("EMAIL_PORT") or os.environ.get("EMAIL_PORT", "587"))
            user = env.get("EMAIL_USER") or os.environ.get("EMAIL_USER", "")
            password = env.get("EMAIL_PASSWORD") or os.environ.get("EMAIL_PASSWORD", "")
            if not all([host, user, password]):
                return {"success": False, "message": "Missing email credentials"}
            try:
                import smtplib
                with smtplib.SMTP(host, port, timeout=10) as smtp:
                    smtp.ehlo()
                    smtp.starttls()
                    smtp.login(user, password)
                return {"success": True, "message": f"Connected to {host} as {user}"}
            except Exception as e:
                return {"success": False, "message": f"SMTP connection failed: {str(e)}"}

        elif skill_name == "sms":
            sid = env.get("TWILIO_ACCOUNT_SID") or os.environ.get("TWILIO_ACCOUNT_SID", "")
            token = env.get("TWILIO_AUTH_TOKEN") or os.environ.get("TWILIO_AUTH_TOKEN", "")
            if not sid or not token:
                return {"success": False, "message": "Missing Twilio credentials"}
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}.json"
                    auth = aiohttp.BasicAuth(sid, token)
                    async with session.get(url, auth=auth, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return {"success": True, "message": f"Twilio connected: {data.get('friendly_name', sid)}"}
                        return {"success": False, "message": f"Twilio auth failed (HTTP {resp.status})"}
            except Exception as e:
                return {"success": False, "message": f"Twilio test failed: {str(e)}"}

        elif skill_name == "calendar":
            cred_file = env.get("GOOGLE_CREDENTIALS_FILE") or os.environ.get("GOOGLE_CREDENTIALS_FILE", "")
            if not cred_file:
                return {"success": False, "message": "GOOGLE_CREDENTIALS_FILE not set"}
            if not Path(cred_file).exists():
                return {"success": False, "message": f"File not found: {cred_file}"}
            try:
                import json
                with open(cred_file) as f:
                    data = json.load(f)
                if "installed" in data or "web" in data or "type" in data:
                    return {"success": True, "message": "Google credentials file is valid JSON"}
                return {"success": False, "message": "File doesn't look like Google credentials"}
            except Exception as e:
                return {"success": False, "message": f"Invalid credentials file: {str(e)}"}

        elif skill_name == "browser":
            try:
                import subprocess
                result = subprocess.run(
                    ["python", "-m", "playwright", "install", "--dry-run", "chromium"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 or "already installed" in (result.stdout + result.stderr).lower():
                    return {"success": True, "message": "Playwright chromium is available"}
                return {"success": True, "message": "Playwright available (may need: playwright install chromium)"}
            except Exception:
                return {"success": True, "message": "Browser skill uses Playwright (run: playwright install chromium)"}

        return {"success": False, "message": f"No test available for skill: {skill_name}"}

    # ── Docker Management ─────────────────────────────────────────────────

    @api.post("/system/docker/start")
    async def start_docker(user_id: str = Depends(get_current_user)):
        """Build and start Aria in Docker containers."""
        docker_dir = Path(__file__).parent.parent.parent / "docker"
        if not docker_dir.exists():
            return {"success": False, "message": "Docker directory not found"}

        # Check if Docker is available
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return {"success": False, "message": "Docker is not running. Start Docker Desktop first."}
        except FileNotFoundError:
            return {"success": False, "message": "Docker is not installed. Install Docker Desktop first."}
        except Exception as e:
            return {"success": False, "message": f"Docker check failed: {str(e)}"}

        # Check for docker-compose
        try:
            # Try docker compose (v2) first, then docker-compose (v1)
            compose_cmd = None
            try:
                subprocess.run(
                    ["docker", "compose", "version"], capture_output=True, text=True, timeout=5
                )
                compose_cmd = ["docker", "compose"]
            except Exception:
                subprocess.run(
                    ["docker-compose", "version"], capture_output=True, text=True, timeout=5
                )
                compose_cmd = ["docker-compose"]

            if not compose_cmd:
                return {"success": False, "message": "docker-compose not found"}

            # Build and start in background
            process = subprocess.Popen(
                compose_cmd + ["up", "-d", "--build"],
                cwd=str(docker_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            return {
                "success": True,
                "message": "Docker containers are building and starting. This may take a few minutes.",
                "command": " ".join(compose_cmd + ["up", "-d", "--build"]),
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to start Docker: {str(e)}"}

    @api.get("/system/docker/status")
    async def docker_status(user_id: str = Depends(get_current_user)):
        """Check Docker container status."""
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=aria", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return {"running": False, "containers": [], "message": "Docker not available"}

            containers = []
            for line in result.stdout.strip().splitlines():
                if not line.strip():
                    continue
                parts = line.split("\t")
                containers.append({
                    "name": parts[0] if len(parts) > 0 else "",
                    "status": parts[1] if len(parts) > 1 else "",
                    "ports": parts[2] if len(parts) > 2 else "",
                })

            return {"running": len(containers) > 0, "containers": containers}
        except Exception as e:
            return {"running": False, "containers": [], "message": str(e)}

    @api.post("/system/docker/stop")
    async def stop_docker(user_id: str = Depends(get_current_user)):
        """Stop Docker containers."""
        docker_dir = Path(__file__).parent.parent.parent / "docker"
        try:
            import subprocess
            compose_cmd = ["docker", "compose"]
            try:
                subprocess.run(compose_cmd + ["version"], capture_output=True, timeout=5, check=True)
            except Exception:
                compose_cmd = ["docker-compose"]

            subprocess.Popen(
                compose_cmd + ["down"],
                cwd=str(docker_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return {"success": True, "message": "Docker containers stopping..."}
        except Exception as e:
            return {"success": False, "message": str(e)}

    # ── Event System ──────────────────────────────────────────────────────

    @api.get("/events/history")
    async def get_event_history(
        event_name: Optional[str] = Query(default=None),
        limit: int = Query(default=50),
        user_id: str = Depends(get_current_user),
    ):
        """Get recent event history."""
        from src.core.events import get_event_bus
        bus = get_event_bus()
        events = bus.get_history(event_name=event_name, limit=limit)
        return [
            {
                "name": e.name,
                "data": e.data,
                "timestamp": e.timestamp.isoformat(),
                "source": e.source,
            }
            for e in events
        ]

    # ── Background Processes ─────────────────────────────────────────────

    @api.get("/processes")
    async def list_processes(user_id: str = Depends(get_current_user)):
        """List background processes."""
        if hasattr(app.state, "process_manager") and app.state.process_manager:
            return app.state.process_manager.list_processes()
        return []

    @api.post("/processes/spawn")
    async def spawn_process(body: dict = Body(...), user_id: str = Depends(get_current_user)):
        """Spawn a background process."""
        if not hasattr(app.state, "process_manager") or not app.state.process_manager:
            raise HTTPException(status_code=503, detail="Process manager not available")
        command = body.get("command", "")
        if not command:
            raise HTTPException(status_code=400, detail="command is required")
        proc_id = await app.state.process_manager.spawn(command, cwd=body.get("cwd"))
        return {"id": proc_id}

    @api.get("/processes/{proc_id}/log")
    async def get_process_log(proc_id: str, tail: int = 100, user_id: str = Depends(get_current_user)):
        """Get log output from a background process."""
        if not hasattr(app.state, "process_manager") or not app.state.process_manager:
            raise HTTPException(status_code=503, detail="Process manager not available")
        log = app.state.process_manager.get_log(proc_id, limit=tail)
        return {"id": proc_id, "log": log}

    @api.post("/processes/{proc_id}/kill")
    async def kill_process(proc_id: str, user_id: str = Depends(get_current_user)):
        """Kill a background process."""
        if not hasattr(app.state, "process_manager") or not app.state.process_manager:
            raise HTTPException(status_code=503, detail="Process manager not available")
        success = await app.state.process_manager.kill(proc_id)
        return {"success": success}

    # ── Scheduler / Cron ─────────────────────────────────────────────────

    @api.get("/scheduler/jobs")
    async def list_jobs(user_id: str = Depends(get_current_user)):
        """List scheduled jobs."""
        if hasattr(app.state, "scheduler") and app.state.scheduler:
            return app.state.scheduler.list_jobs()
        return []

    @api.post("/scheduler/jobs")
    async def create_job(body: dict = Body(...), user_id: str = Depends(get_current_user)):
        """Create a scheduled job."""
        if not hasattr(app.state, "scheduler") or not app.state.scheduler:
            raise HTTPException(status_code=503, detail="Scheduler not available")
        job = app.state.scheduler.add_job(
            name=body.get("name", body.get("label", "")),
            schedule_type=body.get("schedule_type", "once"),
            schedule_value=body.get("schedule_value", body.get("schedule", "")),
            action=body.get("action", "reminder"),
            payload=body.get("payload", {}),
            channel=body.get("channel", ""),
            user_id=body.get("user_id", ""),
        )
        return {"job_id": job.id, "next_run_at": job.next_run_at}

    @api.delete("/scheduler/jobs/{job_id}")
    async def delete_job(job_id: str, user_id: str = Depends(get_current_user)):
        """Delete a scheduled job."""
        if not hasattr(app.state, "scheduler") or not app.state.scheduler:
            raise HTTPException(status_code=503, detail="Scheduler not available")
        success = app.state.scheduler.remove_job(job_id)
        return {"success": success}

    # ── Vector Memory ────────────────────────────────────────────────────

    @api.post("/memory/search")
    async def search_memory(body: dict = Body(...), user_id: str = Depends(get_current_user)):
        """Search vector memory."""
        if not hasattr(app.state, "vector_memory") or not app.state.vector_memory:
            raise HTTPException(status_code=503, detail="Vector memory not available")
        query = body.get("query", "")
        top_k = body.get("top_k", 5)
        results = await app.state.vector_memory.search(query, top_k=top_k)
        return {"results": results}

    @api.get("/memory/stats")
    async def memory_stats(user_id: str = Depends(get_current_user)):
        """Get vector memory statistics."""
        if not hasattr(app.state, "vector_memory") or not app.state.vector_memory:
            return {"available": False}
        return app.state.vector_memory.get_stats()

    # ── Plugins ──────────────────────────────────────────────────────────

    @api.get("/plugins")
    async def list_plugins(user_id: str = Depends(get_current_user)):
        """List loaded plugins."""
        if hasattr(app.state, "plugin_loader") and app.state.plugin_loader:
            return app.state.plugin_loader.list_plugins()
        return []

    # ── Device Pairing ───────────────────────────────────────────────────

    @api.post("/devices/pair/generate")
    async def generate_pairing_code(
        body: Optional[dict] = Body(default=None),
        user_id: str = Depends(get_current_user),
    ):
        """Generate a 6-digit pairing code."""
        if not hasattr(app.state, "device_manager") or not app.state.device_manager:
            raise HTTPException(status_code=503, detail="Device pairing not available")
        device_name = (body or {}).get("device_name", "")
        code = app.state.device_manager.generate_code(device_name)
        return {"code": code, "expires_in": 300}

    @api.post("/devices/pair/redeem")
    async def redeem_pairing_code(request: Request, body: dict = Body(...)):
        """Redeem a pairing code (no auth required — this IS the auth step)."""
        if not hasattr(app.state, "device_manager") or not app.state.device_manager:
            raise HTTPException(status_code=503, detail="Device pairing not available")
        code = body.get("code", "")
        device_name = body.get("device_name", "")
        user_agent = request.headers.get("user-agent", "")
        ip_address = request.client.host if request.client else ""
        result = app.state.device_manager.redeem_code(
            code=code,
            device_name=device_name,
            user_agent=user_agent,
            ip_address=ip_address,
        )
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    @api.get("/devices")
    async def list_devices(user_id: str = Depends(get_current_user)):
        """List paired devices."""
        if not hasattr(app.state, "device_manager") or not app.state.device_manager:
            return []
        return app.state.device_manager.list_devices()

    @api.delete("/devices/{device_id}")
    async def remove_device(device_id: str, user_id: str = Depends(get_current_user)):
        """Unpair a device."""
        if not hasattr(app.state, "device_manager") or not app.state.device_manager:
            raise HTTPException(status_code=503, detail="Device pairing not available")
        success = app.state.device_manager.remove_device(device_id)
        return {"success": success}

    # ── Daemon Status ────────────────────────────────────────────────────

    @api.get("/system/daemon")
    async def daemon_status(user_id: str = Depends(get_current_user)):
        """Get daemon/service status."""
        from src.core.daemon import get_status
        return get_status()

    @api.post("/system/daemon/{action}")
    async def daemon_action(action: str, user_id: str = Depends(get_current_user)):
        """Perform daemon action (install/uninstall/start/stop)."""
        from src.core.daemon import install_service, uninstall_service, start_service, stop_service
        actions = {
            "install": install_service,
            "uninstall": uninstall_service,
            "start": start_service,
            "stop": stop_service,
        }
        func = actions.get(action)
        if not func:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
        return func()

    # ── Register API router ───────────────────────────────────────────────

    app.include_router(api)

    # ── Serve built React frontend ────────────────────────────────────────

    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
        # Serve static assets (js, css, images) under /assets
        assets_dir = FRONTEND_DIR / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="static-assets")

        # Serve other static files at root level (favicon, manifest, etc.)
        @app.get("/vite.svg")
        @app.get("/favicon.ico")
        async def serve_static_root(request: Request):
            file_name = request.url.path.lstrip("/")
            file_path = FRONTEND_DIR / file_name
            if file_path.exists():
                return FileResponse(str(file_path))
            raise HTTPException(status_code=404)

        # SPA catch-all: serve index.html for all non-API, non-asset routes
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            # Don't intercept API or WebSocket paths
            if full_path.startswith("api/") or full_path.startswith("ws/"):
                raise HTTPException(status_code=404)
            # Try to serve a real file first
            file_path = FRONTEND_DIR / full_path
            if full_path and file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            # Fall back to index.html for SPA routing
            return FileResponse(str(FRONTEND_DIR / "index.html"))
    else:
        # No built frontend — show a helpful message at root
        @app.get("/")
        async def root():
            return HTMLResponse(
                "<html><body style='font-family:system-ui;background:#0f172a;color:#e2e8f0;display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>"
                "<div style='text-align:center'>"
                "<h1>Aria Control Center</h1>"
                "<p style='color:#94a3b8'>Frontend not built yet. Run:</p>"
                "<pre style='background:#1e293b;padding:16px;border-radius:8px;color:#38bdf8'>cd src/web/frontend &amp;&amp; npm install &amp;&amp; npm run build</pre>"
                "<p style='color:#94a3b8;margin-top:16px'>Then restart Aria.</p>"
                "</div></body></html>"
            )

    return app


# ── Config file helpers ───────────────────────────────────────────────────────


def _default_skill_templates() -> list[tuple[str, str]]:
    """Default skill templates for skill_templates feature."""
    return [
        (
            "hello_world",
            '"""A minimal skill that says hello."""\n'
            "from ..base import BaseSkill, SkillResult\n\n"
            "class HelloWorldSkill(BaseSkill):\n"
            '    name = "hello_world"\n'
            '    description = "Say hello"\n\n'
            "    def _register_capabilities(self):\n"
            '        self.register_capability("greet", "Return a greeting", {"type": "object"})\n\n'
            "    async def execute(self, capability, **kwargs):\n"
            '        return SkillResult(success=True, output="Hello from your custom skill!")',
        ),
        (
            "url_fetcher",
            '"""Fetch and summarize a URL."""\n'
            "import httpx\n"
            "from ..base import BaseSkill, SkillResult\n\n"
            "class UrlFetcherSkill(BaseSkill):\n"
            '    name = "url_fetcher"\n'
            '    description = "Fetch content from URLs"\n\n'
            "    def _register_capabilities(self):\n"
            '        self.register_capability("fetch", "Fetch URL content", '
            '{"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]})\n\n'
            "    async def execute(self, capability, **kwargs):\n"
            '        url = kwargs.get("url", "")\n'
            "        async with httpx.AsyncClient() as c:\n"
            "            r = await c.get(url)\n"
            '            return SkillResult(success=True, output=r.text[:2000])',
        ),
    ]


def _skill_enabled(cfg: dict[str, Any], skill_name: str) -> bool:
    """Check if a skill is enabled in config."""
    return cfg.get("skills", {}).get("builtin", {}).get(skill_name, {}).get("enabled", False) if isinstance(
        cfg.get("skills", {}).get("builtin", {}).get(skill_name), dict
    ) else False


def _load_yaml_config() -> dict[str, Any]:
    """Load the current settings.yaml as a dict."""
    config_path = Path("config/settings.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_yaml_config(cfg: dict[str, Any]) -> None:
    """Write the config dict back to settings.yaml with a backup."""
    config_path = Path("config/settings.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup
    if config_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = config_path.with_suffix(f".{ts}.bak")
        shutil.copy2(config_path, backup)

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _load_env_values() -> dict[str, str]:
    """Load key-value pairs from the .env file."""
    env_path = Path(".env")
    values: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                values[key.strip()] = val.strip()
    return values


def _set_env_value(key: str, value: str) -> None:
    """Set or update a key in the .env file."""
    env_path = Path(".env")
    lines: list[str] = []
    found = False

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k, _, _ = stripped.partition("=")
                if k.strip() == key:
                    lines.append(f"{key}={value}")
                    found = True
                    continue
            lines.append(line)

    if not found:
        lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n")

    # Also update the current process environment
    os.environ[key] = value


def _detect_provider_mode(cfg: dict) -> str:
    """Detect provider mode from config."""
    llm = cfg.get("llm", {})
    local_on = llm.get("local", {}).get("enabled", False)
    cloud_on = llm.get("cloud", {}).get("enabled", False)
    if local_on and cloud_on:
        return "hybrid"
    if cloud_on:
        return "anthropic"
    if local_on:
        return "ollama"
    return "hybrid"


def _detect_browser_mode(cfg: dict) -> str:
    """Detect browser mode from config."""
    browser_cfg = cfg.get("skills", {}).get("builtin", {}).get("browser", {})
    if isinstance(browser_cfg, dict) and browser_cfg.get("enabled"):
        return "playwright"
    env = _load_env_values()
    if env.get("BRAVE_API_KEY"):
        return "brave"
    return "none"


# ── WebSocket manager ─────────────────────────────────────────────────────────


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket, user_id: str) -> None:
        if user_id in self.active_connections:
            self.active_connections[user_id] = [
                ws for ws in self.active_connections[user_id] if ws != websocket
            ]

    async def send_to_user(self, user_id: str, message: dict[str, Any]) -> None:
        if user_id in self.active_connections:
            for websocket in self.active_connections[user_id]:
                try:
                    await websocket.send_json(message)
                except Exception:
                    pass

    async def broadcast(self, message: dict[str, Any]) -> None:
        for user_id in self.active_connections:
            await self.send_to_user(user_id, message)


ws_manager = WebSocketManager()


def add_websocket_routes(app: FastAPI) -> None:
    """Add WebSocket routes to the app."""

    @app.websocket("/ws/{token}")
    async def websocket_endpoint(websocket: WebSocket, token: str):
        settings = get_settings()
        try:
            payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
            user_id = payload.get("sub")
            if not user_id:
                await websocket.close(code=4001)
                return
        except JWTError:
            await websocket.close(code=4001)
            return

        await ws_manager.connect(websocket, user_id)
        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg_type == "message":
                    if app.state.orchestrator:
                        content = data.get("content", "")
                        message_id = await app.state.orchestrator.process_message(
                            channel="web", user_id=user_id, content=content
                        )
                        await websocket.send_json({"type": "message_queued", "message_id": message_id})
                elif msg_type == "approval_response":
                    if app.state.security_guardian:
                        await app.state.security_guardian.handle_approval_response(
                            request_id=data.get("approval_id"),
                            approved=data.get("approved", False),
                            approved_by=user_id,
                            channel="web",
                        )
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, user_id)
        except Exception as e:
            logger.error("WebSocket error", error=str(e))
            ws_manager.disconnect(websocket, user_id)
