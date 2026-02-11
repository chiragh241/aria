"""Self-healing service — automatically detects and fixes any error from logs.

Uses configurable patterns (config/self_healing.yaml) and LLM fallback for
unknown errors. Can remediate code crashes when the main loop catches them.
"""

import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

def _load_remediations() -> tuple[list[tuple[str, str, str]], dict[str, str], dict[str, Any]]:
    """Load patterns and pip packages from config/self_healing.yaml."""
    base = Path(__file__).resolve().parent.parent.parent
    config_path = base / "config" / "self_healing.yaml"
    patterns: list[tuple[str, str, str]] = []
    pip_packages: dict[str, str] = {
        "install_chromadb": "chromadb",
        "install_whisper": "openai-whisper",
        "check_stt": "openai-whisper",
    }
    llm_config: dict[str, Any] = {
        "enabled": True,
        "min_confidence": 0.7,
        "allowed_actions": ["pip_install", "run_command"],
    }

    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            for p in data.get("patterns", []):
                if isinstance(p, dict) and p.get("pattern"):
                    patterns.append((
                        p["pattern"],
                        p.get("action", "llm_analyze"),
                        p.get("description", ""),
                    ))
            pip_packages.update(data.get("pip_packages", {}))
            if "llm_fallback" in data:
                llm_config.update(data["llm_fallback"])
        except Exception as e:
            logger.warning("Failed to load self_healing config", path=str(config_path), error=str(e))

    if not patterns:
        # Fallback defaults
        patterns = [
            ("ChromaDB not installed", "install_chromadb", "Install ChromaDB"),
            ("No module named .chromadb.", "install_chromadb", "Install ChromaDB"),
            ("whisper|No module named .whisper.", "install_whisper", "Install Whisper"),
            ("STT not initialized|Transcription failed", "install_whisper", "Install Whisper"),
            ("vite\\.svg.*404|404.*vite", "fix_vite_svg", "Add vite.svg"),
        ]

    return patterns, pip_packages, llm_config


class SelfHealingService:
    """
    Monitors logs for any errors and attempts automatic remediation.

    - Uses configurable patterns from config/self_healing.yaml
    - LLM fallback for unknown errors and code crashes
    - Runs periodically on heartbeat and on-demand via check_logs_and_heal tool
    """

    def __init__(
        self,
        event_bus: Any = None,
        check_interval_seconds: int = 10,
        llm_router: Any = None,
        security_guardian: Any = None,
    ) -> None:
        self.settings = get_settings()
        self._event_bus = event_bus
        self._check_interval = check_interval_seconds
        self._llm_router = llm_router
        self._security_guardian = security_guardian
        self._attempted: set[str] = set()
        self._task: asyncio.Task | None = None
        self._running = False
        self._patterns, self._pip_packages, self._llm_config = _load_remediations()
        # Avoid calling LLM every interval when errors are unchanged (saves tokens/rate limits)
        self._last_error_signature: str | None = None
        self._last_llm_call_time: float = 0.0
        self._llm_backoff_until: float = 0.0
        self._same_error_skip_seconds = 300  # 5 min
        self._backoff_429_seconds = 300
        self._backoff_402_seconds = 900

    def set_llm_router(self, router: Any) -> None:
        """Set the LLM router for analyzing unknown errors."""
        self._llm_router = router

    def update_config(self, enabled: bool, interval_seconds: int) -> None:
        """Update interval and whether the periodic loop should run. Call start/stop from caller."""
        self._check_interval = max(60, min(86400, interval_seconds))  # clamp 1 min to 24h

    async def start(self) -> None:
        """Start the periodic check loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Self-healing service started", interval=f"{self._check_interval}s", patterns=len(self._patterns))

    async def stop(self) -> None:
        """Stop the periodic check loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        """Run check every N seconds."""
        await asyncio.sleep(min(self._check_interval, 30))  # First check after up to 30s
        while self._running:
            try:
                await self.check_and_heal()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Self-healing tick error", error=str(e))
            await asyncio.sleep(self._check_interval)

    async def check_and_heal(self) -> dict[str, Any]:
        """
        Read recent logs, detect issues, and attempt fixes.
        Uses pattern matching first, then LLM for unknown errors.
        """
        if not self.settings.proactive.self_healing_enabled:
            return {"detected": [], "fixed": [], "failed": [], "skipped": []}

        logger.info("Self-healing check started", interval_seconds=self._check_interval)

        result: dict[str, Any] = {
            "detected": [],
            "fixed": [],
            "failed": [],
            "skipped": [],
        }

        # Collect log file paths: configured + any from root logger (Cognee etc. may use stdlib)
        log_paths: list[Path] = []
        cfg_path = Path(self.settings.logging.file).expanduser().resolve()
        if cfg_path.exists():
            log_paths.append(cfg_path)
        import logging
        for h in getattr(logging.root, "handlers", []):
            if hasattr(h, "baseFilename"):
                p = Path(h.baseFilename).resolve()
                if p.exists() and p not in log_paths:
                    log_paths.append(p)

        if not log_paths:
            logger.info("Self-healing: no issues found", log_path=str(cfg_path))
            return result

        try:
            all_lines: list[str] = []
            for log_path in log_paths:
                with open(log_path, "rb") as f:
                    f.seek(max(0, f.seek(0, 2) - 200_000))
                    tail = f.read().decode("utf-8", errors="replace")
                all_lines.extend(tail.strip().split("\n"))
            # Dedupe and take most recent
            seen: set[str] = set()
            unique: list[str] = []
            for line in reversed(all_lines):
                if line not in seen:
                    seen.add(line)
                    unique.append(line)
            unique.reverse()
            recent = unique[-300:] if len(unique) > 300 else unique

            # Collect ALL error/warning lines (no per-error patterns; catch everything)
            error_lines: list[str] = []
            for line in recent:
                line_lower = line.lower()
                if (
                    "error" in line_lower
                    or "exception" in line_lower
                    or "traceback" in line_lower
                    or "warning" in line_lower
                    or '"level":"warning"' in line_lower
                    or '"level":"error"' in line_lower
                    or '"level": "warning"' in line_lower
                    or '"level": "error"' in line_lower
                ):
                    error_lines.append(line)

            if not error_lines:
                logger.info("Self-healing: no issues found", log_paths=[str(p) for p in log_paths], recent_lines=len(recent))
                return result

            logger.info("Self-healing: found error lines", count=len(error_lines))

            # Try pattern-based remediation first
            detected_by_pattern: dict[str, str] = {}
            matched_lines: set[int] = set()
            matched_lines_by_action: dict[str, list[int]] = {}
            for i, line in enumerate(error_lines):
                for pattern, action_key, desc in self._patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        detected_by_pattern[action_key] = desc or line[:200]
                        matched_lines.add(i)
                        matched_lines_by_action.setdefault(action_key, []).append(i)
                        break

            for action_key, desc in detected_by_pattern.items():
                if action_key in self._attempted:
                    result["skipped"].append(f"{action_key}: already attempted")
                    continue
                result["detected"].append(desc)
                # llm_analyze: pass matched lines to LLM for code_edit etc.
                if action_key == "llm_analyze":
                    indices = matched_lines_by_action.get("llm_analyze", [])
                    llm_lines = [error_lines[j] for j in indices][-10:]
                    if self._llm_config.get("enabled") and self._llm_router and llm_lines:
                        llm_result = await self._llm_analyze_and_fix(llm_lines)
                        if llm_result:
                            result["fixed"].extend(llm_result.get("fixed", []))
                            result["failed"].extend(llm_result.get("failed", []))
                            result["detected"].extend(llm_result.get("detected", []))
                        self._attempted.add(action_key)
                    continue
                success, msg = await self._run_remediation(action_key)
                if success:
                    result["fixed"].append(f"{action_key}: {msg}")
                    self._attempted.add(action_key)
                    logger.info("Self-healing: fixed", action=action_key, detail=msg)
                    if self._event_bus:
                        await self._event_bus.emit("self_healing_fixed", {
                            "action": action_key,
                            "message": f"Fixed: {msg}",
                        }, source="self_healing")
                else:
                    result["failed"].append(f"{action_key}: {msg}")
                    self._attempted.add(action_key)
                    logger.warning("Self-healing: could not fix", action=action_key, reason=msg)
                    if self._event_bus:
                        await self._event_bus.emit("self_healing_fixed", {
                            "action": action_key,
                            "message": f"Detected issue ({desc}) but couldn't fix: {msg}",
                        }, source="self_healing")

            # LLM fallback for error lines that didn't match any pattern
            unmatched_lines = [error_lines[i] for i in range(len(error_lines)) if i not in matched_lines]
            llm_fallback_result: dict[str, Any] | None = None
            if self._llm_config.get("enabled") and self._llm_router and unmatched_lines:
                # Skip LLM if same errors as last check (saves tokens and avoids rate limits)
                sig = hashlib.sha256("\n".join(unmatched_lines[-10:]).encode()).hexdigest()
                if (
                    self._last_error_signature == sig
                    and (time.monotonic() - self._last_llm_call_time) < self._same_error_skip_seconds
                ):
                    logger.debug("Self-healing: skipping LLM (same errors as last check, wait %ss)", self._same_error_skip_seconds)
                else:
                    llm_fallback_result = await self._llm_analyze_and_fix(unmatched_lines[-10:])
                    if llm_fallback_result:
                        self._last_error_signature = sig
                        self._last_llm_call_time = time.monotonic()
                if llm_fallback_result:
                    result["fixed"].extend(llm_fallback_result.get("fixed", []))
                    result["failed"].extend(llm_fallback_result.get("failed", []))
                    result["detected"].extend(llm_fallback_result.get("detected", []))

        except Exception as e:
            logger.error("Self-healing check failed", error=str(e), log_paths=[str(p) for p in log_paths])
            result["failed"].append(str(e))

        # Log outcome
        if result["fixed"]:
            logger.info("Self-healing: fixed", items=result["fixed"])
        elif result["failed"]:
            logger.warning("Self-healing: could not fix", items=result["failed"])
        elif result["detected"] or result["skipped"]:
            logger.info("Self-healing: issues detected/skipped", detected=result["detected"], skipped=result["skipped"])
        else:
            logger.info("Self-healing: no issues found")

        return result

    async def _llm_analyze_and_fix(self, error_lines: list[str]) -> dict[str, Any] | None:
        """Use LLM to analyze unknown errors and suggest fixes."""
        if not self._llm_router or not error_lines:
            return None
        if time.monotonic() < self._llm_backoff_until:
            logger.debug("Self-healing: skipping LLM (backoff after rate/spend limit)")
            return None

        error_context = "\n".join(error_lines[-10:])[:3000]

        prompt = f"""Analyze this error from a Python application (Aria). Suggest a fix.

Error log excerpt:
```
{error_context}
```

Respond with ONLY valid JSON (no markdown, no explanation):
{{"action": "pip_install"|"run_command"|"code_edit"|"skip", "payload": "...", "reason": "brief explanation", "confidence": 0.0-1.0}}

- pip_install: for missing Python packages (payload = package name only, e.g. "chromadb")
- run_command: for shell commands (only pip, brew install, npm install - keep payload minimal)
- code_edit: for code bugs. payload = JSON object: {{"edits": [{{"file_path": "...", "old_string": "...", "new_string": "..."}}]}} or single edit {{"file_path": "...", "old_string": "...", "new_string": "..."}}. file_path relative to project root. Multiple edits allowed.
- skip: if fix requires manual intervention or is unclear

Be conservative. Only suggest actions you are confident about (confidence >= 0.7)."""

        try:
            from .llm_router import LLMMessage

            response = await self._llm_router.generate(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.2,
                max_tokens=1024,
                prefer_local=True,
            )
            content = (response.content or "").strip()
            # Extract JSON from response (may be wrapped in markdown)
            if "```" in content:
                parts = content.split("```")
                for p in parts:
                    p = p.strip()
                    if p.startswith("json"):
                        p = p[4:].strip()
                    if p.startswith("{"):
                        content = p
                        break
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                return None

            action = data.get("action", "skip")
            payload_raw = data.get("payload")
            payload = (payload_raw or "").strip() if isinstance(payload_raw, str) else payload_raw
            confidence = float(data.get("confidence", 0))

            if action == "skip" or confidence < self._llm_config.get("min_confidence", 0.7):
                return None
            if action not in self._llm_config.get("allowed_actions", []):
                return None

            if action == "code_edit":
                return await self._handle_code_edit(data, payload)

            attempt_key = f"llm_{action}_{payload}"
            if attempt_key in self._attempted:
                return {"skipped": [attempt_key]}

            self._attempted.add(attempt_key)

            if action == "pip_install" and payload:
                success, msg = await self._pip_install(payload)
                if success:
                    logger.info("Self-healing (LLM): fixed", action=action, payload=payload)
                    return {"fixed": [f"{action}: {msg}"], "detected": [data.get("reason", payload)]}
                return {"failed": [f"{action}: {msg}"], "detected": [data.get("reason", payload)]}
            if action == "run_command" and payload:
                success, msg = await self._run_safe_command(payload)
                if success:
                    logger.info("Self-healing (LLM): fixed", action=action, payload=payload)
                    return {"fixed": [f"{action}: {msg}"], "detected": [data.get("reason", payload)]}
                return {"failed": [f"{action}: {msg}"], "detected": [data.get("reason", payload)]}
        except Exception as e:
            err_str = str(e)
            logger.warning("LLM self-healing failed", error=err_str)
            # Back off on rate limit (429) or spend limit (402) to avoid burning tokens
            if "429" in err_str or "Rate limit" in err_str:
                self._llm_backoff_until = time.monotonic() + self._backoff_429_seconds
                logger.info("Self-healing: LLM backoff %ss (rate limit)", self._backoff_429_seconds)
            elif "402" in err_str or "spend limit" in err_str.lower() or "USD" in err_str:
                self._llm_backoff_until = time.monotonic() + self._backoff_402_seconds
                logger.info("Self-healing: LLM backoff %ss (spend limit)", self._backoff_402_seconds)
        return None

    async def _handle_code_edit(self, data: dict[str, Any], payload: Any) -> dict[str, Any] | None:
        """Request user approval for code edit(s), then apply on approve."""
        try:
            raw = payload if isinstance(payload, dict) else json.loads(str(payload))
        except (json.JSONDecodeError, TypeError):
            return None
        edits: list[dict[str, Any]] = []
        if "edits" in raw and isinstance(raw["edits"], list):
            for e in raw["edits"]:
                if isinstance(e, dict) and (e.get("file_path") or "").strip():
                    edits.append({
                        "file_path": (e.get("file_path") or "").strip(),
                        "old_string": e.get("old_string", ""),
                        "new_string": e.get("new_string", ""),
                    })
        elif (raw.get("file_path") or "").strip():
            edits.append({
                "file_path": (raw.get("file_path") or "").strip(),
                "old_string": raw.get("old_string", ""),
                "new_string": raw.get("new_string", ""),
            })
        if not edits:
            return None

        reason = data.get("reason", "Code fix suggested by self-healing")
        attempt_key = f"llm_code_edit_{','.join(e['file_path'] for e in edits)}"
        if attempt_key in self._attempted:
            return {"skipped": [attempt_key]}

        if not self._security_guardian:
            logger.warning("Self-healing: code_edit requires security_guardian for approval")
            return {"detected": [reason]}

        asyncio.create_task(
            self._request_code_edit_approval_and_apply(
                edits=edits,
                reason=reason,
                attempt_key=attempt_key,
            )
        )
        files_str = ", ".join(e["file_path"] for e in edits)
        return {"detected": [f"Code fix proposed for {files_str} — awaiting your approval in Approvals"]}

    async def _request_code_edit_approval_and_apply(
        self,
        edits: list[dict[str, Any]],
        reason: str,
        attempt_key: str,
    ) -> None:
        """Request approval via SecurityGuardian; on approve, apply all edits."""
        files_str = ", ".join(e["file_path"] for e in edits)
        description = f"Self-healing proposes fix for {len(edits)} file(s): {reason}"
        details = {
            "reason": reason,
            "edits": edits,
        }
        try:
            result = await self._security_guardian.request_approval(
                action_type="code_edit",
                description=description,
                details=details,
                user_id="admin",
                channel="web",
                timeout=600,
            )
            self._attempted.add(attempt_key)
            if result.approved:
                applied, failed = [], []
                try:
                    for e in edits:
                        success, msg = await self._apply_code_edit(
                            e["file_path"], e.get("old_string", ""), e.get("new_string", "")
                        )
                        if success:
                            applied.append(e["file_path"])
                        else:
                            failed.append(f"{e['file_path']}: {msg}")
                    status = "success" if not failed else ("partial" if applied else "failed")
                    if result.request_id and self._security_guardian:
                        self._security_guardian.set_approval_outcome(
                            result.request_id,
                            {"status": status, "applied": applied, "failed": failed},
                        )
                    if applied:
                        logger.info("Self-healing (code_edit): applied", files=applied)
                        if self._event_bus:
                            await self._event_bus.emit("self_healing_fixed", {
                                "action": "code_edit",
                                "message": f"Applied fix to {', '.join(applied)}",
                            }, source="self_healing")
                    if failed:
                        logger.warning("Self-healing (code_edit): some failed", failed=failed)
                except Exception as apply_err:
                    logger.warning("Self-healing (code_edit): apply failed", error=str(apply_err))
                    if result.request_id and self._security_guardian:
                        self._security_guardian.set_approval_outcome(
                            result.request_id,
                            {"status": "error", "applied": [], "failed": [str(apply_err)]},
                        )
            else:
                logger.info("Self-healing (code_edit): user denied", files=files_str)
        except Exception as e:
            logger.warning("Self-healing (code_edit): approval flow failed", error=str(e))
            self._attempted.add(attempt_key)

    async def _apply_code_edit(self, file_path: str, old_string: str, new_string: str) -> tuple[bool, str]:
        """Apply a string replacement in a file. Returns (success, message)."""
        base = Path(__file__).resolve().parent.parent.parent
        # Normalize: strip leading segment if it equals project dir name (e.g. aria/auth.py -> auth.py)
        parts = file_path.replace("\\", "/").strip("/").split("/")
        if base.name and parts and parts[0].lower() == base.name.lower():
            parts = parts[1:]
        if not parts:
            return False, "Invalid file path"
        normalized = "/".join(parts)
        path = (base / normalized).resolve()
        if not path.exists():
            alt = (base / "src" / normalized).resolve()
            if alt.exists() and str(alt).startswith(str(base)):
                path = alt
            else:
                return False, f"File not found: {path}"
        if not str(path).startswith(str(base)):
            return False, f"File outside project: {path}"
        try:
            content = path.read_text(encoding="utf-8")
            if old_string in content:
                new_content = content.replace(old_string, new_string, 1)
                path.write_text(new_content, encoding="utf-8")
                return True, "Edit applied"
            # Fallback: try matching with flexible whitespace (LLM may have different spacing/newlines)
            import re
            escaped = re.escape(old_string)
            # Replace escaped spaces and newlines/tabs with \s+ so we match any run of whitespace
            pattern = re.sub(r"(\\\\ |\\\\n|\\\\t|\\\\r)+", r"\\\\s+", escaped)
            match = re.search(pattern, content, re.DOTALL)
            if match:
                new_content = content[: match.start()] + new_string + content[match.end() :]
                path.write_text(new_content, encoding="utf-8")
                return True, "Edit applied (whitespace relaxed)"
            return False, "old_string not found in file (content may have changed)"
        except Exception as e:
            return False, str(e)

    async def _run_safe_command(self, cmd: str) -> tuple[bool, str]:
        """Run a command if it's in the allowed safe set (pip, brew, npm)."""
        parts = cmd.strip().split()
        if not parts:
            return False, "Empty command"
        prog = parts[0].lower()
        if prog in ("pip", "pip3", "python", "python3") and "install" in cmd.lower():
            # Extract package name
            try:
                idx = [p.lower() for p in parts].index("install")
                pkg = parts[idx + 1] if idx + 1 < len(parts) else None
                if pkg:
                    return await self._pip_install(pkg)
            except (ValueError, IndexError):
                pass
        if prog == "brew" and "install" in cmd.lower():
            try:
                proc = await asyncio.create_subprocess_exec(
                    *parts,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                if proc.returncode == 0:
                    return True, f"Ran: {cmd}"
                return False, stderr.decode("utf-8", errors="replace")[:300]
            except Exception as e:
                return False, str(e)
        if prog == "touch" and len(parts) >= 2:
            # Allow touch <path> to create an empty file under project (path = first arg only)
            try:
                base = Path(__file__).resolve().parent.parent.parent
                path_arg = parts[1].lstrip("/")
                target = (base / path_arg).resolve()
                if not str(target).startswith(str(base)):
                    return False, "Path outside project"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
                return True, f"Created {target.name}"
            except Exception as e:
                return False, str(e)
        return False, f"Command not in allowed set: {prog}"

    async def _run_remediation(self, action_key: str) -> tuple[bool, str]:
        """Execute a remediation action. Returns (success, message)."""
        if action_key in self._pip_packages:
            return await self._pip_install(self._pip_packages[action_key])
        if action_key == "fix_vite_svg":
            return await self._ensure_vite_svg()
        return False, f"Unknown action: {action_key}"

    async def _pip_install(self, package: str) -> tuple[bool, str]:
        """Run pip install for a package."""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pip",
                "install",
                package,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode == 0:
                return True, f"Installed {package}"
            err = stderr.decode("utf-8", errors="replace")[:300]
            return False, f"pip install failed: {err}"
        except asyncio.TimeoutError:
            return False, "pip install timed out"
        except Exception as e:
            return False, str(e)

    async def _ensure_vite_svg(self) -> tuple[bool, str]:
        """Ensure vite.svg exists in frontend public and rebuild if needed."""
        base = Path(__file__).resolve().parent.parent
        public_dir = base / "web" / "frontend" / "public"
        svg_path = public_dir / "vite.svg"
        if svg_path.exists():
            return True, "vite.svg already exists"
        public_dir.mkdir(parents=True, exist_ok=True)
        minimal_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path d="M20 2L8 18h7l-5 12L24 14h-7l5-12z" fill="#38bdf8"/></svg>'
        svg_path.write_text(minimal_svg)
        frontend_dir = base / "web" / "frontend"
        if (frontend_dir / "package.json").exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "npm",
                    "run",
                    "build",
                    cwd=str(frontend_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(proc.communicate(), timeout=180)
                if proc.returncode == 0:
                    return True, "Created vite.svg and rebuilt frontend"
                return True, "Created vite.svg (rebuild may need manual run)"
            except Exception:
                return True, "Created vite.svg (run: cd src/web/frontend && npm run build)"
        return True, "Created vite.svg"

    async def handle_crash(self, exc: BaseException) -> dict[str, Any]:
        """
        Analyze and attempt to fix a code crash (uncaught exception).
        Call this from the main loop when catching a fatal exception.
        """
        import traceback

        error_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        logger.error("Crash detected, attempting self-heal", error=str(exc))

        # Try pattern match on traceback first
        error_text = "".join(error_lines)
        for pattern, action_key, _ in self._patterns:
            if re.search(pattern, error_text, re.IGNORECASE) and action_key not in self._attempted:
                self._attempted.add(action_key)
                if action_key == "llm_analyze":
                    if self._llm_config.get("enabled") and self._llm_router:
                        llm_result = await self._llm_analyze_and_fix(error_lines)
                        if llm_result and (llm_result.get("fixed") or llm_result.get("detected")):
                            return llm_result
                    continue
                success, msg = await self._run_remediation(action_key)
                if success:
                    return {"fixed": [f"{action_key}: {msg}"], "detected": [str(exc)]}

        # Try LLM analysis (crash may not be in log yet)
        if self._llm_config.get("enabled") and self._llm_router:
            result = await self._llm_analyze_and_fix(error_lines)
            if result and result.get("fixed"):
                return result

        # Also run normal check (in case crash was logged)
        return await self.check_and_heal()
