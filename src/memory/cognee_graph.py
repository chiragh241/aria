"""Knowledge graph memory using cognee for structured knowledge extraction."""

import os
from typing import TYPE_CHECKING, Any

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .user_profile import UserProfile


def _configure_cognee_llm_from_aria() -> str:
    """
    Configure Cognee's LLM and embeddings from Aria's settings.
    Uses cloud (Anthropic/Claude) if enabled and key set, else Ollama.
    Returns provider name for logging.
    """
    settings = get_settings()
    llm_local = settings.llm.local
    llm_cloud = settings.llm.cloud
    anthropic_key = getattr(settings, "anthropic_api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")

    # Prefer cloud (Claude) if enabled and API key is set
    if llm_cloud.enabled and anthropic_key:
        os.environ["LLM_PROVIDER"] = "anthropic"
        os.environ["LLM_MODEL"] = llm_cloud.model
        os.environ["LLM_API_KEY"] = anthropic_key
        os.environ["LLM_MAX_TOKENS"] = str(llm_cloud.max_tokens)

        # Embeddings: Anthropic doesn't have embeddings; use fastembed (local, no key)
        # to avoid defaulting to OpenAI and requiring OPENAI_API_KEY
        os.environ["EMBEDDING_PROVIDER"] = "fastembed"
        os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
        os.environ["EMBEDDING_DIMENSIONS"] = "384"

        return "anthropic"
    else:
        # Use Ollama (local) - matches Aria's local LLM config
        base_url = llm_local.base_url.rstrip("/")
        # Cognee expects /v1 suffix for Ollama
        endpoint = f"{base_url}/v1" if "/v1" not in base_url else base_url
        model = llm_local.model

        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["LLM_MODEL"] = model
        os.environ["LLM_API_KEY"] = "ollama"
        os.environ["LLM_ENDPOINT"] = endpoint

        # Use Ollama for embeddings too (avoids NoDataError with mixed providers)
        ollama_base = base_url.replace("/v1", "").rstrip("/")
        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        os.environ["EMBEDDING_MODEL"] = "nomic-embed-text:latest"
        os.environ["EMBEDDING_ENDPOINT"] = f"{ollama_base}/api/embed"
        os.environ["EMBEDDING_DIMENSIONS"] = "768"
        os.environ["HUGGINGFACE_TOKENIZER"] = "nomic-ai/nomic-embed-text-v1.5"

        return "ollama"


class CogneeGraphMemory:
    """
    Knowledge graph memory layer powered by cognee.

    Enhances the RAG pipeline by extracting entities, relationships,
    and structured knowledge from ingested data. Provides graph-based
    search alongside vector search.

    Uses Aria's configured LLM: Anthropic (Claude) if enabled with API key,
    otherwise Ollama (local). See docs.cognee.ai for env configuration.
    """

    def __init__(self, data_dir: str | None = None) -> None:
        self.settings = get_settings()
        self._data_dir = data_dir or self.settings.aria.data_dir
        self._initialized = False
        self._available = False

    async def initialize(self) -> None:
        """Initialize cognee and configure storage."""
        if self._initialized:
            return

        try:
            import cognee

            # Configure cognee to use Aria's data directory
            db_path = os.path.join(self._data_dir, "cognee")
            os.makedirs(db_path, exist_ok=True)

            # Point ALL cognee storage to our data directory
            cognee.config.data_root_directory(db_path)
            cognee.config.set_relational_db_config({
                "db_path": db_path,
                "db_name": "cognee_db",
                "db_provider": "sqlite",
            })
            cognee.config.set_vector_db_config({
                "vector_db_url": os.path.join(db_path, "cognee.lancedb"),
                "vector_db_provider": "lancedb",
            })

            # Use Aria's LLM config (Ollama or Claude)
            provider = _configure_cognee_llm_from_aria()

            cognee.config.set_llm_config({
                "llm_provider": os.environ.get("LLM_PROVIDER", "ollama"),
                "llm_model": os.environ.get("LLM_MODEL", "llama3.1:8b"),
                "llm_api_key": os.environ.get("LLM_API_KEY", "ollama"),
                "llm_endpoint": os.environ.get("LLM_ENDPOINT", "http://localhost:11434/v1"),
                "llm_max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "4096")),
            })

            self._available = True
            self._initialized = True
            logger.info(
                "Cognee knowledge graph initialized",
                provider=provider,
                data_dir=db_path,
            )
        except ImportError:
            logger.warning("cognee not installed, knowledge graph disabled")
            self._available = False
            self._initialized = True
        except Exception as e:
            logger.error("Failed to initialize cognee", error=str(e))
            self._available = False
            self._initialized = True

    async def add_knowledge(
        self,
        content: str,
        dataset_name: str = "aria_knowledge",
    ) -> bool:
        """
        Add content to the knowledge graph for processing.

        Args:
            content: Text content to add
            dataset_name: Dataset to add to

        Returns:
            True if content was added
        """
        if not self._available:
            return False

        try:
            import cognee

            await cognee.add(content, dataset_name)
            logger.debug("Added content to cognee", dataset=dataset_name, length=len(content))
            return True
        except Exception as e:
            logger.error("Failed to add to cognee", error=str(e))
            return False

    async def add_user_profile_knowledge(
        self,
        user_id: str,
        profile: "UserProfile",
    ) -> bool:
        """
        Sync user profile into the knowledge graph for unified search.
        Formats preferences, facts, and relationships so Cognee can extract
        entities and relationships alongside ingested documents.

        Args:
            user_id: User identifier
            profile: User profile object

        Returns:
            True if content was added
        """
        if not self._available:
            return False

        parts = [f"User profile for {user_id}:"]
        if profile.preferred_name or profile.name:
            parts.append(f"Name: {profile.preferred_name or profile.name}")
        if profile.timezone:
            parts.append(f"Timezone: {profile.timezone}")
        if profile.language and profile.language != "en":
            parts.append(f"Language: {profile.language}")
        if profile.communication_style:
            parts.append(f"Communication style: {profile.communication_style}")
        if profile.interests:
            parts.append(f"Interests: {', '.join(profile.interests)}")
        if profile.important_people:
            for name, rel in profile.important_people.items():
                parts.append(f"Important person: {name} (relationship: {rel})")
        if profile.important_dates:
            for label, date in profile.important_dates.items():
                parts.append(f"Important date: {label} = {date}")
        if profile.preferences:
            for k, v in profile.preferences.items():
                parts.append(f"Preference: {k} = {v}")
        if profile.facts:
            parts.append("Facts about the user:")
            for fact in profile.facts[-20:]:
                parts.append(f"  - {fact}")

        if len(parts) <= 1:
            return False

        content = "\n".join(parts)
        return await self.add_knowledge(
            content=content,
            dataset_name="aria_user_profiles",
        )

    async def process_knowledge(self, dataset_name: str = "aria_knowledge") -> bool:
        """
        Process added content into the knowledge graph.
        Extracts entities, relationships, and builds graph structure.

        Args:
            dataset_name: Dataset to process

        Returns:
            True if processing succeeded
        """
        if not self._available:
            return False

        try:
            import cognee

            await cognee.cognify()
            logger.info("Knowledge graph processed", dataset=dataset_name)
            return True
        except Exception as e:
            logger.error("Failed to process knowledge graph", error=str(e))
            return False

    async def search(
        self,
        query: str,
        search_type: str = "GRAPH_COMPLETION",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search the knowledge graph.

        Args:
            query: Search query
            search_type: One of GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS, SUMMARIES
            top_k: Max results

        Returns:
            List of search results
        """
        if not self._available:
            return []

        try:
            import cognee
            from cognee.api.v1.search import SearchType

            type_map = {
                "GRAPH_COMPLETION": SearchType.GRAPH_COMPLETION,
                "RAG_COMPLETION": SearchType.RAG_COMPLETION,
                "CHUNKS": SearchType.CHUNKS,
                "SUMMARIES": SearchType.SUMMARIES,
                # Legacy alias
                "INSIGHTS": SearchType.GRAPH_COMPLETION,
            }
            st = type_map.get(search_type.upper(), SearchType.GRAPH_COMPLETION)

            results = await cognee.search(query_text=query, query_type=st)

            formatted = []
            if not results:
                return formatted

            for item in results[:top_k]:
                if isinstance(item, dict):
                    formatted.append({
                        "content": item.get("text", item.get("content", str(item))),
                        "source": "knowledge_graph",
                        "score": item.get("score", 0.8),
                        "metadata": {"search_type": search_type, "graph": True},
                    })
                elif hasattr(item, "text"):
                    formatted.append({
                        "content": item.text,
                        "source": "knowledge_graph",
                        "score": getattr(item, "score", 0.8),
                        "metadata": {"search_type": search_type, "graph": True},
                    })
                else:
                    formatted.append({
                        "content": str(item),
                        "source": "knowledge_graph",
                        "score": 0.8,
                        "metadata": {"search_type": search_type, "graph": True},
                    })

            return formatted
        except Exception as e:
            logger.error("Cognee search failed", error=str(e))
            return []

    async def delete_dataset(self, dataset_name: str = "aria_knowledge") -> bool:
        """Delete a dataset from the knowledge graph."""
        if not self._available:
            return False

        try:
            import cognee
            await cognee.prune.prune_data()
            logger.info("Pruned cognee data")
            return True
        except Exception as e:
            logger.error("Failed to prune cognee data", error=str(e))
            return False

    def is_available(self) -> bool:
        """Check if cognee is available."""
        return self._available

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics."""
        return {
            "available": self._available,
            "initialized": self._initialized,
        }
