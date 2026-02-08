"""System detection utilities for the Aria setup wizard.

Checks for installed tools, running services, and available resources.
"""

import importlib
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ToolStatus:
    """Status of a detected tool or service."""

    installed: bool = False
    version: str = ""
    running: bool = False
    path: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class DetectionResults:
    """Aggregated results from all system checks."""

    ollama: ToolStatus = field(default_factory=ToolStatus)
    docker: ToolStatus = field(default_factory=ToolStatus)
    ffmpeg: ToolStatus = field(default_factory=ToolStatus)
    playwright: ToolStatus = field(default_factory=ToolStatus)
    node: ToolStatus = field(default_factory=ToolStatus)
    anthropic_key: ToolStatus = field(default_factory=ToolStatus)
    brave_key: ToolStatus = field(default_factory=ToolStatus)


class SystemDetector:
    """Detects installed software, running services, and system capabilities."""

    def __init__(self):
        self.results = DetectionResults()

    def run_all(self) -> DetectionResults:
        """Run all detection checks synchronously."""
        self.check_ollama()
        self.check_docker()
        self.check_ffmpeg()
        self.check_playwright()
        self.check_node()
        self.check_anthropic_key()
        self.check_brave_key()
        return self.results

    # ── Individual checks ──────────────────────────────────────────────────

    def check_ollama(self) -> ToolStatus:
        """Check if Ollama is installed, its version, and available models."""
        status = self.results.ollama
        path = shutil.which("ollama")
        if not path:
            return status

        status.installed = True
        status.path = path

        # Get version
        version = self._run_cmd("ollama", "--version")
        if version:
            status.version = version.strip().split()[-1] if version.strip() else ""

        # Check if running by listing models
        models_out = self._run_cmd("ollama", "list")
        if models_out is not None:
            status.running = True
            models = []
            for line in models_out.strip().split("\n")[1:]:  # skip header
                parts = line.split()
                if parts:
                    models.append(parts[0])
            status.extra["models"] = models
        else:
            status.running = False
            status.extra["models"] = []

        self.results.ollama = status
        return status

    def check_docker(self) -> ToolStatus:
        """Check if Docker is installed and the daemon is running."""
        status = self.results.docker
        path = shutil.which("docker")
        if not path:
            return status

        status.installed = True
        status.path = path

        # Get version
        version = self._run_cmd("docker", "--version")
        if version:
            status.version = version.strip()

        # Check daemon - use 'docker ps' which is faster than 'docker info'
        ps_out = self._run_cmd("docker", "ps", "--format", "{{.ID}}")
        status.running = ps_out is not None

        # Check for sandbox image
        if status.running:
            images = self._run_cmd("docker", "images", "--format", "{{.Repository}}:{{.Tag}}")
            if images:
                status.extra["has_sandbox_image"] = "aria-sandbox:latest" in images
            else:
                status.extra["has_sandbox_image"] = False

        self.results.docker = status
        return status

    def check_ffmpeg(self) -> ToolStatus:
        """Check if ffmpeg is installed and its version."""
        status = self.results.ffmpeg
        path = shutil.which("ffmpeg")
        if not path:
            return status

        status.installed = True
        status.path = path

        version = self._run_cmd("ffmpeg", "-version")
        if version:
            first_line = version.strip().split("\n")[0]
            status.version = first_line

        self.results.ffmpeg = status
        return status

    def check_playwright(self) -> ToolStatus:
        """Check if playwright is installed and browsers are available."""
        status = self.results.playwright

        # Check if the Python package is installed
        try:
            import playwright  # noqa: F401
            status.installed = True
        except ImportError:
            self.results.playwright = status
            return status

        # Check if chromium browser binary exists
        pw_path = Path.home() / ".cache" / "ms-playwright"
        if pw_path.exists():
            chromium_dirs = list(pw_path.glob("chromium-*"))
            status.extra["has_chromium"] = len(chromium_dirs) > 0
        else:
            status.extra["has_chromium"] = False

        self.results.playwright = status
        return status

    def check_node(self) -> ToolStatus:
        """Check if Node.js and npm are installed."""
        status = self.results.node
        path = shutil.which("node")
        if not path:
            return status

        status.installed = True
        status.path = path

        version = self._run_cmd("node", "--version")
        if version:
            status.version = version.strip()

        # Check npm
        npm_path = shutil.which("npm")
        status.extra["npm_installed"] = npm_path is not None

        # Check if whatsapp-bridge deps are installed
        bridge_dir = Path("whatsapp-bridge")
        status.extra["bridge_deps_installed"] = (bridge_dir / "node_modules").exists()

        self.results.node = status
        return status

    def check_anthropic_key(self) -> ToolStatus:
        """Check for Anthropic API key in env, .env file, or Claude Code auth."""
        status = self.results.anthropic_key

        # Check environment variable
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if key:
            status.installed = True
            status.extra["source"] = "environment"
            self.results.anthropic_key = status
            return status

        # Check .env file
        env_path = Path(".env")
        if env_path.exists():
            content = env_path.read_text()
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY=") and len(line.split("=", 1)[1].strip()) > 0:
                    val = line.split("=", 1)[1].strip()
                    if val and not val.startswith("#"):
                        status.installed = True
                        status.extra["source"] = ".env"
                        self.results.anthropic_key = status
                        return status

        # Check Claude Code auth
        claude_dir = Path.home() / ".claude"
        if claude_dir.exists():
            status.installed = True
            status.extra["source"] = "claude_code"

        self.results.anthropic_key = status
        return status

    def check_brave_key(self) -> ToolStatus:
        """Check for Brave Search API key in environment."""
        status = self.results.brave_key
        key = os.environ.get("BRAVE_API_KEY", "")
        if key:
            status.installed = True
            status.extra["source"] = "environment"
        self.results.brave_key = status
        return status

    # ── Utility checks (called on-demand, not in run_all) ─────────────────

    @staticmethod
    def check_python_packages(packages: list[str]) -> dict[str, bool]:
        """Check which Python packages are importable."""
        results = {}
        for pkg in packages:
            # Handle packages with different import names
            import_name = {
                "python-docx": "docx",
                "pypdf": "pypdf",
                "pillow": "PIL",
                "ffmpeg-python": "ffmpeg",
                "google-api-python-client": "googleapiclient",
                "google-auth-oauthlib": "google_auth_oauthlib",
                "openai-whisper": "whisper",
                "edge-tts": "edge_tts",
            }.get(pkg, pkg.replace("-", "_"))

            try:
                importlib.import_module(import_name)
                results[pkg] = True
            except ImportError:
                results[pkg] = False
        return results

    @staticmethod
    def check_port_available(port: int) -> bool:
        """Check if a TCP port is available for binding."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return True
        except OSError:
            return False

    @staticmethod
    def check_slack_credentials(bot_token: str, app_token: str) -> tuple[bool, str]:
        """Validate Slack credentials by calling auth.test (synchronous)."""
        try:
            from slack_sdk import WebClient

            client = WebClient(token=bot_token)
            response = client.auth_test()
            if response["ok"]:
                return True, response.get("team", "Unknown workspace")
            return False, response.get("error", "Unknown error")
        except ImportError:
            return False, "slack_sdk not installed"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def validate_anthropic_key(api_key: str) -> tuple[bool, str]:
        """Validate an Anthropic API key with a minimal test call."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True, "Key is valid"
        except ImportError:
            return False, "anthropic package not installed"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def validate_brave_key(api_key: str) -> tuple[bool, str]:
        """Validate a Brave Search API key with a test query."""
        try:
            import httpx

            with httpx.Client() as client:
                resp = client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": "test", "count": 1},
                    headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    return True, "Key is valid"
                return False, f"HTTP {resp.status_code}"
        except ImportError:
            return False, "httpx not installed"
        except Exception as e:
            return False, str(e)

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _run_cmd(*args: str) -> Optional[str]:
        """Run a command and return stdout, or None on failure."""
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                return result.stdout
            return None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None
