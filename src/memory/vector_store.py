"""Persistent vector memory for Aria using ChromaDB.

Indexes conversation messages and files for semantic search.
Enables "what did we discuss about X?" queries across all past sessions.

Uses ChromaDB (already a project dependency) for embedding + retrieval.
Falls back to keyword search if embeddings aren't available.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class VectorMemory:
    """Persistent vector memory with hybrid search."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        self._collection = None
        self._available = False
        self._doc_count = 0

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            persist_dir = str(
                Path(self.settings.aria.data_dir).expanduser() / "chromadb"
            )
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            # Main collection for conversation memory
            self._collection = self._client.get_or_create_collection(
                name="aria_memory",
                metadata={"hnsw:space": "cosine"},
            )

            self._doc_count = self._collection.count()
            self._available = True
            logger.info(
                "Vector memory initialized",
                persist_dir=persist_dir,
                documents=self._doc_count,
            )
        except ImportError:
            logger.warning("chromadb not installed — vector memory disabled")
            self._available = False
        except Exception as e:
            logger.error("Failed to initialize vector memory", error=str(e))
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    # --- Indexing ---

    async def add_message(
        self,
        content: str,
        role: str,
        channel: str = "",
        user_id: str = "",
        conversation_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Index a conversation message for future recall."""
        if not self._available or not content or len(content.strip()) < 10:
            return None

        try:
            doc_id = self._make_id(content, channel, user_id)
            meta = {
                "role": role,
                "channel": channel or "",
                "user_id": user_id or "",
                "conversation_id": conversation_id or "",
                "timestamp": time.time(),
                "type": "message",
                **(metadata or {}),
            }
            # ChromaDB metadata values must be str, int, float, or bool
            meta = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}

            self._collection.upsert(
                ids=[doc_id],
                documents=[content],
                metadatas=[meta],
            )
            self._doc_count = self._collection.count()
            return doc_id
        except Exception as e:
            logger.warning("Failed to index message", error=str(e))
            return None

    async def add_document(
        self,
        content: str,
        source: str,
        doc_type: str = "file",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Index a document (file contents) by chunking it."""
        if not self._available or not content:
            return []

        chunks = self._chunk_text(content, max_chars=1000, overlap=200)
        ids = []

        for i, chunk in enumerate(chunks):
            try:
                doc_id = self._make_id(chunk, source, str(i))
                meta = {
                    "source": source,
                    "type": doc_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "timestamp": time.time(),
                    **(metadata or {}),
                }
                meta = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}

                self._collection.upsert(
                    ids=[doc_id],
                    documents=[chunk],
                    metadatas=[meta],
                )
                ids.append(doc_id)
            except Exception as e:
                logger.warning("Failed to index chunk", source=source, chunk=i, error=str(e))

        self._doc_count = self._collection.count()
        logger.debug("Indexed document", source=source, chunks=len(ids))
        return ids

    # --- Search ---

    async def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search memory by semantic similarity."""
        if not self._available or not query:
            return []

        try:
            # Refresh count to avoid requesting more results than exist
            count = self._collection.count()
            if count == 0:
                return []

            params: dict[str, Any] = {
                "query_texts": [query],
                "n_results": min(top_k, count),
            }
            if where:
                params["where"] = where

            results = self._collection.query(**params)

            if not results or not results["documents"] or not results["documents"][0]:
                return []

            formatted = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                formatted.append({
                    "content": doc,
                    "score": round(1 - distance, 3),  # cosine distance → similarity
                    "metadata": meta,
                    "id": results["ids"][0][i] if results["ids"] else "",
                })

            return formatted
        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            return []

    async def search_messages(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search only conversation messages."""
        return await self.search(query, top_k=top_k, where={"type": "message"})

    async def search_documents(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search only indexed documents."""
        return await self.search(query, top_k=top_k, where={"type": "file"})

    # --- Management ---

    async def delete(self, doc_ids: list[str]) -> None:
        """Delete documents by ID."""
        if not self._available or not doc_ids:
            return
        try:
            self._collection.delete(ids=doc_ids)
            self._doc_count = self._collection.count()
        except Exception as e:
            logger.warning("Failed to delete from vector store", error=str(e))

    async def clear(self) -> None:
        """Clear all indexed data."""
        if not self._available or not self._client:
            return
        try:
            self._client.delete_collection("aria_memory")
            self._collection = self._client.get_or_create_collection(
                name="aria_memory",
                metadata={"hnsw:space": "cosine"},
            )
            self._doc_count = 0
            logger.info("Vector memory cleared")
        except Exception as e:
            logger.warning("Failed to clear vector store", error=str(e))

    def get_stats(self) -> dict[str, Any]:
        return {
            "available": self._available,
            "document_count": self._doc_count,
        }

    # --- Helpers ---

    @staticmethod
    def _make_id(content: str, *extra: str) -> str:
        """Generate a deterministic document ID."""
        h = hashlib.sha256()
        h.update(content.encode())
        for e in extra:
            h.update(e.encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunk = text[start:end]

            # Try to break at sentence/paragraph boundary
            if end < len(text):
                for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                    last = chunk.rfind(sep)
                    if last > max_chars // 2:
                        chunk = chunk[: last + len(sep)]
                        end = start + len(chunk)
                        break

            chunks.append(chunk.strip())
            start = end - overlap

        return [c for c in chunks if c]
