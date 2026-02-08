"""Filesystem operations skill."""

import asyncio
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..base import BaseSkill, SkillResult


class FilesystemSkill(BaseSkill):
    """
    Skill for file and directory operations.

    Capabilities:
    - Read/write files
    - List directories
    - Create/delete files and directories
    - Move/copy files
    - Search files
    """

    name = "filesystem"
    description = "File and directory operations"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.max_file_size = config.get("max_file_size_mb", 100) * 1024 * 1024

    def _register_capabilities(self) -> None:
        """Register filesystem capabilities."""
        self.register_capability(
            name="read_file",
            description="Read the contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "encoding": {"type": "string", "default": "utf-8"},
                },
                "required": ["path"],
            },
            security_action="read_files",
        )

        self.register_capability(
            name="write_file",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write"},
                    "append": {"type": "boolean", "default": False},
                },
                "required": ["path", "content"],
            },
            security_action="write_files",
        )

        self.register_capability(
            name="list_directory",
            description="List contents of a directory",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "recursive": {"type": "boolean", "default": False},
                    "pattern": {"type": "string", "description": "Glob pattern filter"},
                },
                "required": ["path"],
            },
            security_action="read_files",
        )

        self.register_capability(
            name="create_directory",
            description="Create a directory",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "parents": {"type": "boolean", "default": True},
                },
                "required": ["path"],
            },
            security_action="write_files",
        )

        self.register_capability(
            name="delete",
            description="Delete a file or directory",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to delete"},
                    "recursive": {"type": "boolean", "default": False},
                },
                "required": ["path"],
            },
            security_action="delete_files",
        )

        self.register_capability(
            name="move",
            description="Move a file or directory",
            parameters={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source path"},
                    "destination": {"type": "string", "description": "Destination path"},
                },
                "required": ["source", "destination"],
            },
            security_action="write_files",
        )

        self.register_capability(
            name="copy",
            description="Copy a file or directory",
            parameters={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source path"},
                    "destination": {"type": "string", "description": "Destination path"},
                },
                "required": ["source", "destination"],
            },
            security_action="write_files",
        )

        self.register_capability(
            name="get_info",
            description="Get information about a file or directory",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to inspect"},
                },
                "required": ["path"],
            },
            security_action="read_files",
        )

        self.register_capability(
            name="search",
            description="Search for files matching a pattern",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Starting directory"},
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "content": {"type": "string", "description": "Content to search for"},
                },
                "required": ["path"],
            },
            security_action="read_files",
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        """Execute a filesystem capability."""
        start_time = datetime.now(timezone.utc)

        handlers = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_directory": self._list_directory,
            "create_directory": self._create_directory,
            "delete": self._delete,
            "move": self._move,
            "copy": self._copy,
            "get_info": self._get_info,
            "search": self._search,
        }

        handler = handlers.get(capability)
        if not handler:
            return self._error_result(f"Unknown capability: {capability}", start_time)

        try:
            result = await handler(**kwargs)
            return self._success_result(result, start_time)
        except Exception as e:
            return self._error_result(str(e), start_time)

    async def _read_file(
        self,
        path: str,
        encoding: str = "utf-8",
    ) -> str:
        """Read file contents."""
        file_path = Path(path).expanduser()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if file_path.stat().st_size > self.max_file_size:
            raise ValueError(f"File too large: {file_path.stat().st_size} bytes")

        # Run in executor for non-blocking
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None,
            lambda: file_path.read_text(encoding=encoding),
        )
        return content

    async def _write_file(
        self,
        path: str,
        content: str,
        append: bool = False,
    ) -> dict[str, Any]:
        """Write content to file."""
        file_path = Path(path).expanduser()

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_event_loop()
        mode = "a" if append else "w"
        await loop.run_in_executor(
            None,
            lambda: file_path.open(mode).write(content),
        )

        return {
            "path": str(file_path),
            "bytes_written": len(content.encode()),
            "appended": append,
        }

    async def _list_directory(
        self,
        path: str,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        """List directory contents."""
        dir_path = Path(path).expanduser()

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        entries = []
        if recursive:
            items = dir_path.rglob(pattern or "*")
        else:
            items = dir_path.glob(pattern or "*")

        for item in items:
            try:
                stat = item.stat()
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "is_file": item.is_file(),
                    "is_dir": item.is_dir(),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
            except (OSError, PermissionError):
                continue

        return entries

    async def _create_directory(
        self,
        path: str,
        parents: bool = True,
    ) -> dict[str, Any]:
        """Create a directory."""
        dir_path = Path(path).expanduser()
        dir_path.mkdir(parents=parents, exist_ok=True)

        return {
            "path": str(dir_path),
            "created": True,
        }

    async def _delete(
        self,
        path: str,
        recursive: bool = False,
    ) -> dict[str, Any]:
        """Delete a file or directory."""
        target_path = Path(path).expanduser()

        if not target_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if target_path.is_dir():
            if recursive:
                shutil.rmtree(target_path)
            else:
                target_path.rmdir()
        else:
            target_path.unlink()

        return {
            "path": str(target_path),
            "deleted": True,
        }

    async def _move(
        self,
        source: str,
        destination: str,
    ) -> dict[str, Any]:
        """Move a file or directory."""
        src = Path(source).expanduser()
        dst = Path(destination).expanduser()

        if not src.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

        return {
            "source": str(src),
            "destination": str(dst),
            "moved": True,
        }

    async def _copy(
        self,
        source: str,
        destination: str,
    ) -> dict[str, Any]:
        """Copy a file or directory."""
        src = Path(source).expanduser()
        dst = Path(destination).expanduser()

        if not src.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

        return {
            "source": str(src),
            "destination": str(dst),
            "copied": True,
        }

    async def _get_info(self, path: str) -> dict[str, Any]:
        """Get file or directory information."""
        target_path = Path(path).expanduser()

        if not target_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        stat = target_path.stat()

        return {
            "path": str(target_path),
            "name": target_path.name,
            "is_file": target_path.is_file(),
            "is_dir": target_path.is_dir(),
            "is_symlink": target_path.is_symlink(),
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
        }

    async def _search(
        self,
        path: str,
        pattern: str | None = None,
        content: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for files."""
        dir_path = Path(path).expanduser()

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        results = []
        search_pattern = pattern or "**/*"

        for item in dir_path.glob(search_pattern):
            if not item.is_file():
                continue

            match_info: dict[str, Any] = {
                "path": str(item),
                "name": item.name,
            }

            # Content search if specified
            if content:
                try:
                    file_content = item.read_text(errors="ignore")
                    if content.lower() in file_content.lower():
                        # Find matching lines
                        lines = file_content.split("\n")
                        matches = []
                        for i, line in enumerate(lines, 1):
                            if content.lower() in line.lower():
                                matches.append({"line": i, "text": line.strip()[:200]})
                        match_info["content_matches"] = matches[:10]
                        results.append(match_info)
                except Exception:
                    continue
            else:
                results.append(match_info)

            if len(results) >= 100:  # Limit results
                break

        return results
