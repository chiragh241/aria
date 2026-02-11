"""LLM Router for intelligent routing between local and cloud models."""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

import tiktoken

from ..utils.config import get_settings
from ..utils.logging import get_logger
from .usage_tracker import get_usage_tracker

logger = get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    NVIDIA = "nvidia"

class TaskComplexity(str, Enum):
    """Task complexity levels for routing decisions."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class LLMMessage:
    """A message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str
    model: str
    provider: LLMProvider
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    raw_response: Any = None


@dataclass
class Tool:
    """Tool definition for function calling."""

    name: str
    description: str
    parameters: dict[str, Any]


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM service is available."""
        pass


class OllamaClient(BaseLLMClient):
    """Client for local Ollama models."""

    def __init__(self, base_url: str, model: str, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: Any = None

    async def _get_client(self) -> Any:
        """Get or create the Ollama client."""
        if self._client is None:
            import ollama

            self._client = ollama.AsyncClient(host=self.base_url)
        return self._client

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert LLMMessages to Ollama format, preserving tool_calls and tool results."""
        ollama_messages = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content or ""}

            # Include tool_calls on assistant messages so Ollama sees what was called
            if m.role == "assistant" and m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": tc.get("arguments", {}),
                        }
                    }
                    for tc in m.tool_calls
                ]

            ollama_messages.append(msg)
        return ollama_messages

    @staticmethod
    def _normalize_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]] | None:
        """Normalize Ollama tool_calls to flat [{id, name, arguments}] dicts."""
        if not raw_tool_calls:
            return None

        result = []
        for i, tc in enumerate(raw_tool_calls):
            if isinstance(tc, dict):
                # Ollama dict format: {"function": {"name": "...", "arguments": {...}}}
                func = tc.get("function", {})
                if isinstance(func, dict):
                    result.append({
                        "id": f"call_{i}",
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", {}),
                    })
                else:
                    # func might be an SDK object
                    result.append({
                        "id": f"call_{i}",
                        "name": getattr(func, "name", ""),
                        "arguments": dict(getattr(func, "arguments", {})),
                    })
            elif hasattr(tc, "function"):
                # SDK ToolCall object with .function attribute
                func = tc.function
                result.append({
                    "id": f"call_{i}",
                    "name": getattr(func, "name", ""),
                    "arguments": dict(getattr(func, "arguments", {})),
                })
            else:
                logger.warning("Unknown tool_call type, skipping", tc_type=type(tc).__name__)
                continue

        return result if result else None

    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using Ollama."""
        client = await self._get_client()

        # Convert messages to Ollama format (preserving tool data)
        ollama_messages = self._convert_messages(messages)

        # Convert tools to Ollama format if provided
        ollama_tools = None
        if tools:
            ollama_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        try:
            response = await client.chat(
                model=self.model,
                messages=ollama_messages,
                tools=ollama_tools,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or -1,
                },
            )

            # Extract and normalize tool calls
            raw_tool_calls = response.get("message", {}).get("tool_calls")
            tool_calls = self._normalize_tool_calls(raw_tool_calls)

            return LLMResponse(
                content=response.get("message", {}).get("content", ""),
                model=self.model,
                provider=LLMProvider.OLLAMA,
                usage={
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                },
                tool_calls=tool_calls,
                finish_reason=response.get("done_reason"),
                raw_response=response,
            )
        except Exception as e:
            logger.error("Ollama generation failed", error=str(e))
            raise

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Ollama."""
        client = await self._get_client()

        ollama_messages = self._convert_messages(messages)

        try:
            async for chunk in await client.chat(
                model=self.model,
                messages=ollama_messages,
                stream=True,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or -1,
                },
            ):
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]
        except Exception as e:
            logger.error("Ollama streaming failed", error=str(e))
            raise

    async def health_check(self) -> bool:
        """Check if Ollama is available."""
        try:
            client = await self._get_client()
            await client.list()
            return True
        except Exception:
            return False


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the Anthropic client."""
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using Claude."""
        client = self._get_client()

        # Separate system message from conversation
        system_message = None
        conversation_messages = []
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append({"role": msg.role, "content": msg.content})

        # Convert tools to Anthropic format
        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        try:
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": conversation_messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature,
            }
            if system_message:
                create_kwargs["system"] = system_message
            if anthropic_tools:
                create_kwargs["tools"] = anthropic_tools

            response = await client.messages.create(**create_kwargs)

            # Extract content and tool use
            content = ""
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "arguments": block.input,
                        }
                    )

            return LLMResponse(
                content=content,
                model=self.model,
                provider=LLMProvider.ANTHROPIC,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                },
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=response.stop_reason,
                raw_response=response,
            )
        except Exception as e:
            logger.error("Anthropic generation failed", error=str(e))
            raise

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Claude."""
        client = self._get_client()

        system_message = None
        conversation_messages = []
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append({"role": msg.role, "content": msg.content})

        try:
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": conversation_messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature,
            }
            if system_message:
                create_kwargs["system"] = system_message

            async with client.messages.stream(**create_kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error("Anthropic streaming failed", error=str(e))
            raise

    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        try:
            client = self._get_client()
            # Simple test with minimal tokens
            await client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return True
        except Exception:
            return False


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini (Nano, Flash, Pro, etc.) via GOOGLE_API_KEY."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._model_instance: Any = None

    def _get_model(self) -> Any:
        """Get or create the Gemini model instance."""
        if self._model_instance is None:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model_instance = genai.GenerativeModel(
                self.model,
                generation_config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": 0.7,
                },
            )
        return self._model_instance

    def _messages_to_content(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert LLMMessages to Gemini chat format (roles + parts)."""
        contents = []
        for m in messages:
            role = "user" if m.role == "user" else "model" if m.role == "assistant" else "user"
            if m.role == "system":
                contents.append({"role": "user", "parts": [f"System: {m.content}"]})
                contents.append({"role": "model", "parts": ["Understood."]})
            else:
                parts: list[Any] = [m.content or ""]
                if m.role == "assistant" and m.tool_calls:
                    for tc in m.tool_calls:
                        args = tc.get("arguments", {})
                        if isinstance(args, str):
                            import json
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        parts.append({
                            "function_call": {
                                "name": tc.get("name", ""),
                                "args": args,
                            },
                        })
                contents.append({"role": role, "parts": parts})
        return contents

    def _tools_to_gemini(self, tools: list[Tool]) -> list[Any]:
        """Convert Tool list to Gemini function declarations."""
        if not tools:
            return []
        import google.generativeai as genai

        decls = []
        for t in tools:
            decls.append({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            })
        try:
            return [genai.types.Tool(function_declarations=decls)]
        except AttributeError:
            return [genai.protos.Tool(function_declarations=decls)]

    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using Gemini."""
        model = self._get_model()
        contents = self._messages_to_content(messages)
        generation_config = {
            "max_output_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
        }
        kwargs_sync: dict[str, Any] = {
            "contents": contents,
            "generation_config": generation_config,
        }
        if tools:
            kwargs_sync["tools"] = self._tools_to_gemini(tools)

        def _run() -> Any:
            return model.generate_content(**kwargs_sync)

        response = await asyncio.to_thread(_run)
        text = ""
        tool_calls = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append({
                        "id": getattr(fc, "id", f"call_{len(tool_calls)}"),
                        "name": getattr(fc, "name", ""),
                        "arguments": getattr(fc, "args", {}) or {},
                    })
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
            }
        return LLMResponse(
            content=text,
            model=self.model,
            provider=LLMProvider.GEMINI,
            usage=usage,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=getattr(response.candidates[0], "finish_reason", None) if response.candidates else None,
            raw_response=response,
        )

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Gemini."""
        model = self._get_model()
        contents = self._messages_to_content(messages)
        generation_config = {
            "max_output_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
        }
        kwargs_sync = {"contents": contents, "generation_config": generation_config, "stream": True}
        if tools:
            kwargs_sync["tools"] = self._tools_to_gemini(tools)

        def _stream() -> Any:
            return model.generate_content(**kwargs_sync)

        stream = await asyncio.to_thread(_stream)
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    async def health_check(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            model = self._get_model()
            result = await asyncio.to_thread(
                model.generate_content,
                "Hi",
                generation_config={"max_output_tokens": 5},
            )
            return result is not None and (not result.candidates or len(result.candidates) > 0)
        except Exception:
            return False


class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter (400+ models via one API). OpenAI-compatible."""

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        max_tokens: int = 4096,
        timeout: int = 120,
        use_free_models: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.use_free_models = use_free_models
        self._client: Any = None

    def _model_for_request(self) -> str:
        """Model ID to send to API. Free tier uses openrouter/free (with reasoning)."""
        if self.use_free_models:
            return "openrouter/free"
        return self.model

    def _extra_body(self) -> dict[str, Any]:
        """Extra body for OpenRouter. Enable reasoning when using free tier."""
        if self.use_free_models:
            return {"reasoning": {"enabled": True}}
        return {}

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def _messages_to_openai(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            role = m.role if m.role in ("user", "assistant", "system") else "user"
            msg: dict[str, Any] = {"role": role, "content": m.content or ""}
            if m.role == "assistant" and m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": (
                                json.dumps(tc["arguments"])
                                if isinstance(tc.get("arguments"), dict)
                                else str(tc.get("arguments", "{}"))
                            ),
                        },
                    }
                    for i, tc in enumerate(m.tool_calls)
                ]
            out.append(msg)
        return out

    def _tools_to_openai(self, tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate using OpenRouter (OpenAI-compatible API)."""
        client = self._get_client()
        openai_messages = self._messages_to_openai(messages)
        kwargs_call: dict[str, Any] = {
            "model": self._model_for_request(),
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        extra = self._extra_body()
        if extra:
            kwargs_call["extra_body"] = extra
        if tools:
            kwargs_call["tools"] = self._tools_to_openai(tools)
            kwargs_call["tool_choice"] = "auto"

        response = await client.chat.completions.create(**kwargs_call)
        choice = response.choices[0] if response.choices else None
        if not choice:
            return LLMResponse(
                content="",
                model=self.model,
                provider=LLMProvider.OPENROUTER,
                usage={},
            )
        msg = choice.message
        content = msg.content or ""
        tool_calls = None
        if getattr(msg, "tool_calls", None):
            tool_calls = []
            for tc in msg.tool_calls:
                args = getattr(tc.function, "arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                tool_calls.append({
                    "id": getattr(tc, "id", ""),
                    "name": getattr(tc.function, "name", ""),
                    "arguments": args,
                })
        usage = {}
        if getattr(response, "usage", None):
            u = response.usage
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0),
                "completion_tokens": getattr(u, "completion_tokens", 0),
            }
        return LLMResponse(
            content=content,
            model=self.model,
            provider=LLMProvider.OPENROUTER,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            raw_response=response,
        )

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream from OpenRouter."""
        client = self._get_client()
        openai_messages = self._messages_to_openai(messages)
        kwargs_call: dict[str, Any] = {
            "model": self._model_for_request(),
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }
        extra = self._extra_body()
        if extra:
            kwargs_call["extra_body"] = extra
        if tools:
            kwargs_call["tools"] = self._tools_to_openai(tools)
            kwargs_call["tool_choice"] = "auto"

        stream = await client.chat.completions.create(**kwargs_call)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def health_check(self) -> bool:
        """Check if OpenRouter API is accessible."""
        try:
            client = self._get_client()
            kwargs: dict[str, Any] = {
                "model": self._model_for_request(),
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
            }
            extra = self._extra_body()
            if extra:
                kwargs["extra_body"] = extra
            await client.chat.completions.create(**kwargs)
            return True
        except Exception:
            return False


class NvidiaClient(BaseLLMClient):
    """Client for NVIDIA NIM (integrate.api.nvidia.com). OpenAI-compatible."""

    def __init__(
        self,
        api_key: str,
        model: str = "moonshotai/kimi-k2.5",
        max_tokens: int = 16384,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def _messages_to_openai(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            role = m.role if m.role in ("user", "assistant", "system") else "user"
            msg: dict[str, Any] = {"role": role, "content": m.content or ""}
            if m.role == "assistant" and m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": (
                                json.dumps(tc["arguments"])
                                if isinstance(tc.get("arguments"), dict)
                                else str(tc.get("arguments", "{}"))
                            ),
                        },
                    }
                    for i, tc in enumerate(m.tool_calls)
                ]
            out.append(msg)
        return out

    def _tools_to_openai(self, tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _extra_body(self) -> dict[str, Any]:
        """Optional body for thinking models (e.g. kimi-k2.5)."""
        return {"chat_template_kwargs": {"thinking": True}}

    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate using NVIDIA NIM (OpenAI-compatible API)."""
        client = self._get_client()
        openai_messages = self._messages_to_openai(messages)
        kwargs_call: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "extra_body": self._extra_body(),
        }
        if "thinking" in self.model.lower():
            kwargs_call["top_p"] = 0.9
        if tools:
            kwargs_call["tools"] = self._tools_to_openai(tools)
            kwargs_call["tool_choice"] = "auto"

        response = await client.chat.completions.create(**kwargs_call)
        choice = response.choices[0] if response.choices else None
        if not choice:
            return LLMResponse(
                content="",
                model=self.model,
                provider=LLMProvider.NVIDIA,
                usage={},
            )
        msg = choice.message
        content = msg.content or ""
        tool_calls = None
        if getattr(msg, "tool_calls", None):
            tool_calls = []
            for tc in msg.tool_calls:
                args = getattr(tc.function, "arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                tool_calls.append({
                    "id": getattr(tc, "id", ""),
                    "name": getattr(tc.function, "name", ""),
                    "arguments": args,
                })
        usage = {}
        if getattr(response, "usage", None):
            u = response.usage
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0),
                "completion_tokens": getattr(u, "completion_tokens", 0),
            }
        return LLMResponse(
            content=content,
            model=self.model,
            provider=LLMProvider.NVIDIA,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            raw_response=response,
        )

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream from NVIDIA NIM."""
        client = self._get_client()
        openai_messages = self._messages_to_openai(messages)
        kwargs_call: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
            "extra_body": self._extra_body(),
        }
        if "thinking" in self.model.lower():
            kwargs_call["top_p"] = 0.9
        if tools:
            kwargs_call["tools"] = self._tools_to_openai(tools)
            kwargs_call["tool_choice"] = "auto"

        stream = await client.chat.completions.create(**kwargs_call)
        async for chunk in stream:
            if not getattr(chunk, "choices", None) or not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # Thinking/reasoning models (e.g. kimi-k2-thinking) stream reasoning_content then content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield reasoning
            if getattr(delta, "content", None):
                yield delta.content

    async def health_check(self) -> bool:
        """Check if NVIDIA NIM API is accessible."""
        try:
            client = self._get_client()
            await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                extra_body=self._extra_body(),
            )
            return True
        except Exception:
            return False


class LLMRouter:
    """
    Intelligent router that decides whether to use local or cloud LLM.

    Routes based on:
    - Task complexity (simple tasks go to local)
    - Token count (short messages go to local)
    - Task type (code generation goes to cloud)
    - Model availability (fallback if local is down)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._local_client: OllamaClient | None = None
        self._cloud_client: AnthropicClient | None = None
        self._gemini_client: GeminiClient | None = None
        self._openrouter_client: OpenRouterClient | None = None
        self._nvidia_client: NvidiaClient | None = None
        self._local_available: bool | None = None
        self._cloud_available: bool | None = None
        self._gemini_available: bool | None = None
        self._openrouter_available: bool | None = None
        self._nvidia_available: bool | None = None

        # Token counter for routing decisions
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None

    async def initialize(self) -> None:
        """Initialize the LLM router and check provider availability."""
        # Check local availability
        if self.local_client:
            self._local_available = await self.local_client.health_check()
            logger.info("Local LLM", available=self._local_available, model=self.settings.llm.local.model)
        else:
            self._local_available = False
            logger.info("Local LLM disabled")

        # Check cloud (Anthropic) availability
        if self.cloud_client:
            self._cloud_available = await self.cloud_client.health_check()
            logger.info("Cloud LLM", available=self._cloud_available, model=self.settings.llm.cloud.model)
        else:
            self._cloud_available = False
            logger.info("Cloud LLM disabled or no API key")

        # Check Gemini availability
        if self.gemini_client:
            self._gemini_available = await self.gemini_client.health_check()
            logger.info("Gemini LLM", available=self._gemini_available, model=self.settings.llm.gemini.model)
        else:
            self._gemini_available = False
            logger.info("Gemini LLM disabled or no API key")

        # Check OpenRouter availability
        if self.openrouter_client:
            self._openrouter_available = await self.openrouter_client.health_check()
            logger.info("OpenRouter LLM", available=self._openrouter_available, model=self.settings.llm.openrouter.model)
        else:
            self._openrouter_available = False
            logger.info("OpenRouter LLM disabled or no API key")

        # Check NVIDIA NIM availability
        if self.nvidia_client:
            self._nvidia_available = await self.nvidia_client.health_check()
            logger.info("NVIDIA LLM", available=self._nvidia_available, model=self.settings.llm.nvidia.model)
        else:
            self._nvidia_available = False
            logger.info("NVIDIA LLM disabled or no API key")

    @property
    def local_client(self) -> OllamaClient | None:
        """Get or create the local LLM client."""
        if self._local_client is None and self.settings.llm.local.enabled:
            import os
            # Allow OLLAMA_HOST env var to override (needed for Docker networking)
            base_url = os.environ.get("OLLAMA_HOST") or self.settings.llm.local.base_url
            self._local_client = OllamaClient(
                base_url=base_url,
                model=self.settings.llm.local.model,
                timeout=self.settings.llm.local.timeout,
            )
        return self._local_client

    @property
    def cloud_client(self) -> AnthropicClient | None:
        """Get or create the cloud (Anthropic) LLM client."""
        if self._cloud_client is None and self.settings.llm.cloud.enabled:
            api_key = self.settings.anthropic_api_key
            if api_key:
                self._cloud_client = AnthropicClient(
                    api_key=api_key,
                    model=self.settings.llm.cloud.model,
                    max_tokens=self.settings.llm.cloud.max_tokens,
                    timeout=self.settings.llm.cloud.timeout,
                )
        return self._cloud_client

    @property
    def gemini_client(self) -> GeminiClient | None:
        """Get or create the Gemini LLM client."""
        if self._gemini_client is None and self.settings.llm.gemini.enabled:
            api_key = self.settings.google_api_key
            if api_key:
                self._gemini_client = GeminiClient(
                    api_key=api_key,
                    model=self.settings.llm.gemini.model,
                    max_tokens=self.settings.llm.gemini.max_tokens,
                    timeout=self.settings.llm.gemini.timeout,
                )
        return self._gemini_client

    @property
    def openrouter_client(self) -> OpenRouterClient | None:
        """Get or create the OpenRouter LLM client."""
        if self._openrouter_client is None and self.settings.llm.openrouter.enabled:
            api_key = self.settings.openrouter_api_key
            if api_key:
                self._openrouter_client = OpenRouterClient(
                    api_key=api_key,
                    model=self.settings.llm.openrouter.model,
                    max_tokens=self.settings.llm.openrouter.max_tokens,
                    timeout=self.settings.llm.openrouter.timeout,
                    use_free_models=getattr(
                        self.settings.llm.openrouter, "use_free_models", False
                    ),
                )
        return self._openrouter_client

    @property
    def nvidia_client(self) -> NvidiaClient | None:
        """Get or create the NVIDIA NIM LLM client."""
        if self._nvidia_client is None and self.settings.llm.nvidia.enabled:
            api_key = self.settings.nvidia_api_key
            if api_key:
                self._nvidia_client = NvidiaClient(
                    api_key=api_key,
                    model=self.settings.llm.nvidia.model,
                    max_tokens=self.settings.llm.nvidia.max_tokens,
                    timeout=self.settings.llm.nvidia.timeout,
                )
        return self._nvidia_client

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Fallback: rough estimate
        return len(text) // 4

    def get_context_window(self, provider: str | None = None, model_id: str | None = None) -> int:
        """Return effective context window in tokens (for proportional truncation, compaction)."""
        return getattr(
            self.settings.llm.routing,
            "context_window_default",
            200_000,
        )

    def classify_complexity(
        self,
        messages: list[LLMMessage],
        task_type: str | None = None,
    ) -> TaskComplexity:
        """
        Classify the complexity of a task for routing.

        Args:
            messages: The conversation messages
            task_type: Optional explicit task type

        Returns:
            TaskComplexity level
        """
        # Check for explicit complex task types
        if task_type and task_type in self.settings.llm.routing.always_cloud:
            return TaskComplexity.COMPLEX

        # Count total tokens
        total_text = " ".join(m.content for m in messages)
        token_count = self.count_tokens(total_text)

        if token_count < self.settings.llm.routing.simple_threshold:
            return TaskComplexity.SIMPLE

        # Check for complexity indicators in the content
        complex_indicators = [
            "code",
            "implement",
            "debug",
            "analyze",
            "explain",
            "compare",
            "design",
            "architecture",
            "optimize",
            "refactor",
            "plan",
            "strategy",
        ]

        last_message = messages[-1].content.lower() if messages else ""
        if any(indicator in last_message for indicator in complex_indicators):
            return TaskComplexity.COMPLEX

        return TaskComplexity.MODERATE

    async def check_availability(self) -> dict[str, bool]:
        """Check availability of all LLM providers."""
        results = {}

        if self.local_client:
            self._local_available = await self.local_client.health_check()
            results["local"] = self._local_available

        if self.cloud_client:
            self._cloud_available = await self.cloud_client.health_check()
            results["cloud"] = self._cloud_available

        if self.gemini_client:
            self._gemini_available = await self.gemini_client.health_check()
            results["gemini"] = self._gemini_available

        if self.openrouter_client:
            self._openrouter_available = await self.openrouter_client.health_check()
            results["openrouter"] = self._openrouter_available

        if self.nvidia_client:
            self._nvidia_available = await self.nvidia_client.health_check()
            results["nvidia"] = self._nvidia_available

        return results

    async def _recheck_providers(self) -> None:
        """Re-check providers that were marked unavailable at startup.

        Ollama or the cloud API might not have been ready during
        ``initialize()`` (e.g. container still booting).  This gives
        them a second chance on the first real request.
        """
        if self._local_available is False and self.local_client:
            self._local_available = await self.local_client.health_check()
            if self._local_available:
                logger.info("Local LLM became available on recheck")

        if self._cloud_available is False and self.cloud_client:
            self._cloud_available = await self.cloud_client.health_check()
            if self._cloud_available:
                logger.info("Cloud LLM became available on recheck")

        if self._gemini_available is False and self.gemini_client:
            self._gemini_available = await self.gemini_client.health_check()
            if self._gemini_available:
                logger.info("Gemini LLM became available on recheck")

        if self._openrouter_available is False and self.openrouter_client:
            self._openrouter_available = await self.openrouter_client.health_check()
            if self._openrouter_available:
                logger.info("OpenRouter LLM became available on recheck")

        if self._nvidia_available is False and self.nvidia_client:
            self._nvidia_available = await self.nvidia_client.health_check()
            if self._nvidia_available:
                logger.info("NVIDIA LLM became available on recheck")

    def select_provider(
        self,
        messages: list[LLMMessage],
        task_type: str | None = None,
        prefer_local: bool = False,
        require_tools: bool = False,
    ) -> tuple[BaseLLMClient, LLMProvider]:
        """
        Select the best LLM provider for a task.

        Args:
            messages: The conversation messages
            task_type: Optional explicit task type
            prefer_local: Whether to prefer local even for complex tasks
            require_tools: Whether tool calling is required

        Returns:
            Tuple of (client, provider)

        Raises:
            RuntimeError: If no suitable provider is available
        """
        complexity = self.classify_complexity(messages, task_type)

        # If tools are required and local doesn't support them well, use cloud
        if require_tools and complexity != TaskComplexity.SIMPLE:
            if self.gemini_client and self._gemini_available is not False:
                return self.gemini_client, LLMProvider.GEMINI
            if self.openrouter_client and self._openrouter_available is not False:
                return self.openrouter_client, LLMProvider.OPENROUTER
            if self.nvidia_client and self._nvidia_available is not False:
                return self.nvidia_client, LLMProvider.NVIDIA
            if self.cloud_client and self._cloud_available is not False:
                return self.cloud_client, LLMProvider.ANTHROPIC

        # Simple tasks go to local if available
        if complexity == TaskComplexity.SIMPLE or prefer_local:
            if self.local_client and self._local_available is not False:
                return self.local_client, LLMProvider.OLLAMA

        # Complex tasks: prefer Gemini, then OpenRouter, then NVIDIA, then Anthropic, then local
        if complexity == TaskComplexity.COMPLEX:
            if self.gemini_client and self._gemini_available is not False:
                return self.gemini_client, LLMProvider.GEMINI
            if self.openrouter_client and self._openrouter_available is not False:
                return self.openrouter_client, LLMProvider.OPENROUTER
            if self.nvidia_client and self._nvidia_available is not False:
                return self.nvidia_client, LLMProvider.NVIDIA
            if self.cloud_client and self._cloud_available is not False:
                return self.cloud_client, LLMProvider.ANTHROPIC

        # Moderate: prefer Gemini, then OpenRouter, then NVIDIA, then Anthropic, then local
        if self.gemini_client and self._gemini_available is not False:
            return self.gemini_client, LLMProvider.GEMINI
        if self.openrouter_client and self._openrouter_available is not False:
            return self.openrouter_client, LLMProvider.OPENROUTER
        if self.nvidia_client and self._nvidia_available is not False:
            return self.nvidia_client, LLMProvider.NVIDIA
        if self.cloud_client and self._cloud_available is not False:
            return self.cloud_client, LLMProvider.ANTHROPIC

        if self.local_client and self._local_available is not False:
            return self.local_client, LLMProvider.OLLAMA

        # Last resort: use any configured provider even if health check failed
        # (e.g. NVIDIA-only setup; health check may have been transient)
        if self.gemini_client:
            return self.gemini_client, LLMProvider.GEMINI
        if self.openrouter_client:
            return self.openrouter_client, LLMProvider.OPENROUTER
        if self.nvidia_client:
            return self.nvidia_client, LLMProvider.NVIDIA
        if self.cloud_client:
            return self.cloud_client, LLMProvider.ANTHROPIC
        if self.local_client:
            return self.local_client, LLMProvider.OLLAMA

        raise RuntimeError(
            "No LLM provider available. Check that your chosen provider (e.g. NVIDIA NIM) "
            "is enabled in config and its API key is set (e.g. NVIDIA_API_KEY in .env). "
            "Restart Aria after setup so config and env are loaded."
        )

    async def _generate_with_retry(
        self,
        client: Any,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_attempts: int = 3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call client.generate with retries on 429/502/503."""
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return await client.generate(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                if attempt < max_attempts - 1 and (
                    "429" in msg or "502" in msg or "503" in msg or "rate limit" in msg
                ):
                    delay = 2**attempt
                    logger.warning("LLM request failed, retrying", attempt=attempt + 1, delay_s=delay, error=msg[:100])
                    await asyncio.sleep(delay)
                else:
                    raise
        raise last_error or RuntimeError("generate failed")

    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        task_type: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        prefer_local: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response using the best available LLM.

        Args:
            messages: The conversation messages
            tools: Optional tools for function calling
            task_type: Optional task type for routing
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            prefer_local: Whether to prefer local LLM
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMResponse from the selected provider
        """
        try:
            client, provider = self.select_provider(
                messages,
                task_type=task_type,
                prefer_local=prefer_local,
                require_tools=bool(tools),
            )
        except RuntimeError:
            # All providers were unavailable — recheck in case they came up
            await self._recheck_providers()
            client, provider = self.select_provider(
                messages,
                task_type=task_type,
                prefer_local=prefer_local,
                require_tools=bool(tools),
            )

        logger.debug(
            "Routing request",
            provider=provider.value,
            task_type=task_type,
            message_count=len(messages),
        )

        try:
            import time
            t0 = time.perf_counter()
            response = await self._generate_with_retry(
                client,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            usage = response.usage or {}
            get_usage_tracker().record(
                provider=provider.value,
                model=response.model,
                input_tokens=usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                output_tokens=usage.get("completion_tokens", usage.get("output_tokens", 0)),
                latency_ms=latency_ms,
                task_type=task_type or "",
            )
            return response
        except Exception as e:
            # Try fallback if configured (Gemini, OpenRouter, then Anthropic)
            if self.settings.llm.routing.fallback_to_cloud and provider == LLMProvider.OLLAMA:
                self._local_available = False
                if self.gemini_client and self._gemini_available is not False:
                    logger.warning("Local LLM failed, falling back to Gemini", error=str(e))
                    return await self.gemini_client.generate(
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                if self.openrouter_client and self._openrouter_available is not False:
                    logger.warning("Local LLM failed, falling back to OpenRouter", error=str(e))
                    return await self.openrouter_client.generate(
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                if self.nvidia_client and self._nvidia_available is not False:
                    logger.warning("Local LLM failed, falling back to NVIDIA", error=str(e))
                    return await self.nvidia_client.generate(
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                if self.cloud_client:
                    logger.warning("Local LLM failed, falling back to cloud", error=str(e))
                    return await self.cloud_client.generate(
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
            raise

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None = None,
        task_type: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        prefer_local: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response using the best available LLM.

        Yields text chunks as they are generated.
        """
        client, provider = self.select_provider(
            messages,
            task_type=task_type,
            prefer_local=prefer_local,
            require_tools=bool(tools),
        )

        logger.debug(
            "Routing streaming request",
            provider=provider.value,
            task_type=task_type,
        )

        try:
            async for chunk in client.generate_stream(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            ):
                yield chunk
        except Exception as e:
            if self.settings.llm.routing.fallback_to_cloud and provider == LLMProvider.OLLAMA:
                self._local_available = False
                if self.gemini_client and self._gemini_available is not False:
                    logger.warning("Local LLM streaming failed, falling back to Gemini", error=str(e))
                    async for chunk in self.gemini_client.generate_stream(
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    ):
                        yield chunk
                    return
                if self.openrouter_client and self._openrouter_available is not False:
                    logger.warning("Local LLM streaming failed, falling back to OpenRouter", error=str(e))
                    async for chunk in self.openrouter_client.generate_stream(
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    ):
                        yield chunk
                    return
                if self.nvidia_client and self._nvidia_available is not False:
                    logger.warning("Local LLM streaming failed, falling back to NVIDIA", error=str(e))
                    async for chunk in self.nvidia_client.generate_stream(
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    ):
                        yield chunk
                    return
                if self.cloud_client:
                    logger.warning("Local LLM streaming failed, falling back to cloud", error=str(e))
                    async for chunk in self.cloud_client.generate_stream(
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    ):
                        yield chunk
                    return
            raise
