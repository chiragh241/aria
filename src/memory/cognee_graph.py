"""Knowledge graph memory using cognee for structured knowledge extraction."""

import os
from typing import Any

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CogneeGraphMemory:
    """
    Knowledge graph memory layer powered by cognee.

    Enhances the RAG pipeline by extracting entities, relationships,
    and structured knowledge from ingested data. Provides graph-based
    search alongside vector search.

    Features:
    - Automatic entity and relationship extraction
    - Knowledge graph construction
    - Graph-based semantic search
    - Insights generation from connected knowledge
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
            # (defaults to site-packages/.cognee_system/ which is read-only in Docker)
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

            # Use Ollama for cognee's internal LLM calls (local, private, no cost)
            ollama_endpoint = os.environ.get("LLM_ENDPOINT", "http://localhost:11434/v1")
            ollama_model = os.environ.get("LLM_MODEL", "llama3.1:8b")

            # Set env vars cognee expects
            os.environ["LLM_API_KEY"] = "ollama"

            cognee.config.set_llm_config({
                "llm_api_key": "ollama",
                "llm_provider": "ollama",
                "llm_model": ollama_model,
                "llm_endpoint": ollama_endpoint,
            })

            self._available = True
            self._initialized = True
            logger.info(
                "Cognee knowledge graph initialized with Ollama",
                data_dir=db_path,
                model=ollama_model,
                endpoint=ollama_endpoint,
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
