"""LLM Router for intelligent routing between local and cloud models."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

import tiktoken

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


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
        self._local_available: bool | None = None
        self._cloud_available: bool | None = None

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

        # Check cloud availability
        if self.cloud_client:
            self._cloud_available = await self.cloud_client.health_check()
            logger.info("Cloud LLM", available=self._cloud_available, model=self.settings.llm.cloud.model)
        else:
            self._cloud_available = False
            logger.info("Cloud LLM disabled or no API key")

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
        """Get or create the cloud LLM client."""
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

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Fallback: rough estimate
        return len(text) // 4

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
            if self.cloud_client and self._cloud_available is not False:
                return self.cloud_client, LLMProvider.ANTHROPIC

        # Simple tasks go to local if available
        if complexity == TaskComplexity.SIMPLE or prefer_local:
            if self.local_client and self._local_available is not False:
                return self.local_client, LLMProvider.OLLAMA

        # Complex tasks go to cloud
        if complexity == TaskComplexity.COMPLEX:
            if self.cloud_client and self._cloud_available is not False:
                return self.cloud_client, LLMProvider.ANTHROPIC

        # Moderate complexity: prefer cloud but fall back to local
        if self.cloud_client and self._cloud_available is not False:
            return self.cloud_client, LLMProvider.ANTHROPIC

        if self.local_client and self._local_available is not False:
            return self.local_client, LLMProvider.OLLAMA

        raise RuntimeError("No LLM provider available")

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
            response = await client.generate(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response
        except Exception as e:
            # Try fallback if configured
            if (
                self.settings.llm.routing.fallback_to_cloud
                and provider == LLMProvider.OLLAMA
                and self.cloud_client
            ):
                logger.warning("Local LLM failed, falling back to cloud", error=str(e))
                self._local_available = False
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
            if (
                self.settings.llm.routing.fallback_to_cloud
                and provider == LLMProvider.OLLAMA
                and self.cloud_client
            ):
                logger.warning("Local LLM streaming failed, falling back to cloud", error=str(e))
                self._local_available = False
                async for chunk in self.cloud_client.generate_stream(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ):
                    yield chunk
            else:
                raise
