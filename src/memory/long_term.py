"""Long-term memory using ChromaDB vector store."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryDocument:
    """A document stored in long-term memory."""

    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""  # conversation, file, web, etc.
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """Result from a memory search."""

    document: MemoryDocument
    score: float
    distance: float


class LongTermMemory:
    """
    Long-term memory using ChromaDB for vector storage.

    Features:
    - Semantic search over memories
    - Metadata filtering
    - Automatic embedding generation
    - Persistence to disk
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.settings = get_settings()
        self._persist_dir = persist_directory or self.settings.memory.long_term.persist_directory
        self._collection_name = collection_name or self.settings.memory.long_term.collection_name

        # Ensure directory exists
        Path(self._persist_dir).expanduser().mkdir(parents=True, exist_ok=True)

        self._client: Any = None
        self._collection: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            persist_path = str(Path(self._persist_dir).expanduser())

            # Use same ChromaSettings as vector_store.py to avoid "already exists with different settings"
            self._client = chromadb.PersistentClient(
                path=persist_path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            self._initialized = True
            logger.info(
                "Long-term memory initialized",
                collection=self._collection_name,
                persist_dir=self._persist_dir,
            )
        except ImportError:
            logger.warning("ChromaDB not installed, long-term memory disabled")
        except Exception as e:
            logger.error("Failed to initialize long-term memory", error=str(e))

    async def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        source: str = "conversation",
        doc_id: str | None = None,
    ) -> MemoryDocument:
        """
        Add a document to long-term memory.

        Args:
            content: Document content
            metadata: Optional metadata
            source: Source of the document
            doc_id: Optional document ID

        Returns:
            The created document
        """
        if not self._initialized:
            await self.initialize()

        if not self._collection:
            raise RuntimeError("Long-term memory not available")

        doc = MemoryDocument(
            id=doc_id or str(uuid4()),
            content=content,
            metadata=metadata or {},
            source=source,
        )

        # Add timestamp to metadata
        doc.metadata["created_at"] = doc.created_at.isoformat()
        doc.metadata["source"] = source

        try:
            self._collection.add(
                ids=[doc.id],
                documents=[content],
                metadatas=[doc.metadata],
            )
            logger.debug("Added document to long-term memory", doc_id=doc.id)
        except Exception as e:
            logger.error("Failed to add document", error=str(e))
            raise

        return doc

    async def add_many(
        self,
        documents: list[tuple[str, dict[str, Any] | None]],
        source: str = "batch",
    ) -> list[MemoryDocument]:
        """
        Add multiple documents at once.

        Args:
            documents: List of (content, metadata) tuples
            source: Source identifier

        Returns:
            List of created documents
        """
        if not self._initialized:
            await self.initialize()

        if not self._collection:
            raise RuntimeError("Long-term memory not available")

        docs = []
        ids = []
        contents = []
        metadatas = []

        for content, metadata in documents:
            doc = MemoryDocument(
                content=content,
                metadata=metadata or {},
                source=source,
            )
            doc.metadata["created_at"] = doc.created_at.isoformat()
            doc.metadata["source"] = source

            docs.append(doc)
            ids.append(doc.id)
            contents.append(content)
            metadatas.append(doc.metadata)

        try:
            self._collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
            )
            logger.debug("Added batch to long-term memory", count=len(docs))
        except Exception as e:
            logger.error("Failed to add batch", error=str(e))
            raise

        return docs

    async def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results
            where: Metadata filter
            where_document: Document content filter

        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()

        if not self._collection:
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"],
            )

            search_results = []
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = MemoryDocument(
                        id=doc_id,
                        content=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    )

                    distance = results["distances"][0][i] if results["distances"] else 0
                    # Convert distance to similarity score (cosine: 1 - distance)
                    score = 1 - distance

                    search_results.append(
                        SearchResult(
                            document=doc,
                            score=score,
                            distance=distance,
                        )
                    )

            return search_results

        except Exception as e:
            logger.error("Search failed", error=str(e))
            return []

    async def get(self, doc_id: str) -> MemoryDocument | None:
        """Get a document by ID."""
        if not self._initialized:
            await self.initialize()

        if not self._collection:
            return None

        try:
            results = self._collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"],
            )

            if results and results["ids"]:
                return MemoryDocument(
                    id=doc_id,
                    content=results["documents"][0] if results["documents"] else "",
                    metadata=results["metadatas"][0] if results["metadatas"] else {},
                )
        except Exception as e:
            logger.error("Failed to get document", error=str(e))

        return None

    async def update(
        self,
        doc_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update a document.

        Args:
            doc_id: Document ID
            content: New content (optional)
            metadata: New metadata (optional)

        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()

        if not self._collection:
            return False

        try:
            update_kwargs: dict[str, Any] = {"ids": [doc_id]}
            if content:
                update_kwargs["documents"] = [content]
            if metadata:
                update_kwargs["metadatas"] = [metadata]

            self._collection.update(**update_kwargs)
            return True
        except Exception as e:
            logger.error("Failed to update document", error=str(e))
            return False

    async def delete(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        if not self._initialized:
            await self.initialize()

        if not self._collection:
            return False

        try:
            self._collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            logger.error("Failed to delete document", error=str(e))
            return False

    async def delete_many(
        self,
        where: dict[str, Any] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        """
        Delete multiple documents.

        Args:
            where: Metadata filter
            ids: List of document IDs

        Returns:
            Number of documents deleted
        """
        if not self._initialized:
            await self.initialize()

        if not self._collection:
            return 0

        try:
            if ids:
                self._collection.delete(ids=ids)
                return len(ids)
            elif where:
                # Get matching IDs first
                results = self._collection.get(where=where, include=[])
                if results and results["ids"]:
                    self._collection.delete(ids=results["ids"])
                    return len(results["ids"])
            return 0
        except Exception as e:
            logger.error("Failed to delete documents", error=str(e))
            return 0

    async def count(self) -> int:
        """Get the total number of documents."""
        if not self._initialized:
            await self.initialize()

        if not self._collection:
            return 0

        return self._collection.count()

    def persist(self) -> None:
        """Persist the database to disk."""
        if self._client:
            self._client.persist()

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "initialized": self._initialized,
            "collection_name": self._collection_name,
            "persist_directory": self._persist_dir,
            "document_count": self._collection.count() if self._collection else 0,
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._client:
            # PersistentClient auto-persists, nothing special needed
            self._client = None
            self._collection = None
            self._initialized = False
            logger.info("Long-term memory cleaned up")
