import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from loguru import logger

from backend.config import settings
from backend.utils import JSONEncoder


class VectorMemory:
    """Vector-based memory system for agents"""

    def __init__(self, collection_name: str):
        self.client = chromadb.PersistentClient(
            path=f"{settings.data_dir}/chroma",
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"Memory for {collection_name}"}
        )

    def store(self, content: str, metadata: Dict[str, Any] = None,
              embedding: Optional[List[float]] = None):
        """Store information in memory"""
        if metadata is None:
            metadata = {}

        # Add timestamp
        metadata["timestamp"] = datetime.now().isoformat()

        # Generate ID from content
        from backend.utils import generate_id
        doc_id = generate_id(content)

        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id],
            embeddings=[embedding] if embedding else None
        )
        logger.debug(f"Stored memory: {doc_id}")

    def retrieve(self, query: str, n_results: int = 5,
                 where: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Retrieve similar memories"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

        memories = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                memories.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })

        return memories

    def search_by_metadata(self, metadata_filter: Dict[str, Any],
                           n_results: int = 10) -> List[Dict[str, Any]]:
        """Search memories by metadata"""
        results = self.collection.get(
            where=metadata_filter,
            limit=n_results
        )

        memories = []
        if results['documents']:
            for i, doc in enumerate(results['documents']):
                memories.append({
                    "content": doc,
                    "metadata": results['metadatas'][i],
                    "id": results['ids'][i]
                })

        return memories

    def clear(self):
        """Clear all memories"""
        self.collection.delete(where={})
        logger.info(f"Cleared memory collection: {self.collection.name}")


class SummaryMemory:
    """Maintains summary of important information"""

    def __init__(self):
        self.summaries = {}
        self.important_facts = []

    def add_summary(self, topic: str, summary: str, importance: float = 0.5):
        """Add or update summary for a topic"""
        self.summaries[topic] = {
            "summary": summary,
            "importance": importance,
            "updated_at": datetime.now().isoformat()
        }

    def get_summary(self, topic: str) -> Optional[str]:
        """Get summary for a topic"""
        return self.summaries.get(topic, {}).get("summary")

    def add_fact(self, fact: str, source: str, confidence: float = 1.0):
        """Add important fact"""
        self.important_facts.append({
            "fact": fact,
            "source": source,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

    def get_relevant_facts(self, query: str, threshold: float = 0.7) -> List[Dict]:
        """Get facts relevant to query (simple keyword matching)"""
        relevant = []
        query_words = set(query.lower().split())

        for fact in self.important_facts:
            fact_words = set(fact["fact"].lower().split())
            overlap = len(query_words.intersection(fact_words))
            similarity = overlap / max(len(query_words), 1)

            if similarity >= threshold:
                relevant.append(fact)

        return relevant[:5]  # Return top 5