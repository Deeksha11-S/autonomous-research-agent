from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
import re


class SummaryMemory:
    """Maintains summary of important information across research sessions"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.summaries = {}
        self.important_facts = []
        self.decisions = []
        self.lessons_learned = []
        self.timeline = []

    def add_summary(self, topic: str, summary: str, importance: float = 0.5,
                    metadata: Dict[str, Any] = None):
        """Add or update summary for a topic"""
        if metadata is None:
            metadata = {}

        self.summaries[topic] = {
            "summary": summary,
            "importance": importance,
            "metadata": metadata,
            "updated_at": datetime.now().isoformat(),
            "update_count": self.summaries.get(topic, {}).get("update_count", 0) + 1
        }

        logger.debug(f"Added summary for topic: {topic}")

    def get_summary(self, topic: str) -> Optional[str]:
        """Get summary for a topic"""
        return self.summaries.get(topic, {}).get("summary")

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get all summaries"""
        return self.summaries

    def add_fact(self, fact: str, source: str, confidence: float = 1.0,
                 category: str = "general", metadata: Dict[str, Any] = None):
        """Add important fact to memory"""
        if metadata is None:
            metadata = {}

        fact_entry = {
            "fact": fact,
            "source": source,
            "confidence": confidence,
            "category": category,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }

        self.important_facts.append(fact_entry)

        # Keep only recent facts (last 100)
        if len(self.important_facts) > 100:
            self.important_facts = self.important_facts[-100:]

        logger.debug(f"Added fact: {fact[:50]}...")

    def get_relevant_facts(self, query: str, threshold: float = 0.7,
                           limit: int = 5) -> List[Dict[str, Any]]:
        """Get facts relevant to query"""
        relevant = []
        query_words = set(query.lower().split())

        for fact in self.important_facts:
            fact_text = fact["fact"].lower()
            fact_words = set(re.findall(r'\b\w+\b', fact_text))

            # Calculate Jaccard similarity
            if query_words and fact_words:
                intersection = len(query_words.intersection(fact_words))
                union = len(query_words.union(fact_words))
                similarity = intersection / union if union > 0 else 0

                if similarity >= threshold:
                    relevant.append(fact)

        # Sort by confidence and recency
        relevant.sort(key=lambda x: (x["confidence"], x["timestamp"]), reverse=True)

        return relevant[:limit]

    def add_decision(self, decision: str, rationale: str, agent: str,
                     outcome: str = "pending", metadata: Dict[str, Any] = None):
        """Add decision to memory"""
        if metadata is None:
            metadata = {}

        decision_entry = {
            "decision": decision,
            "rationale": rationale,
            "agent": agent,
            "outcome": outcome,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }

        self.decisions.append(decision_entry)
        logger.debug(f"Added decision by {agent}: {decision[:50]}...")

    def get_decisions_by_agent(self, agent: str) -> List[Dict[str, Any]]:
        """Get decisions made by specific agent"""
        return [d for d in self.decisions if d["agent"] == agent]

    def add_lesson(self, lesson: str, context: str, agent: str,
                   importance: float = 0.5, metadata: Dict[str, Any] = None):
        """Add lesson learned"""
        if metadata is None:
            metadata = {}

        lesson_entry = {
            "lesson": lesson,
            "context": context,
            "agent": agent,
            "importance": importance,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }

        self.lessons_learned.append(lesson_entry)
        logger.debug(f"Added lesson from {agent}: {lesson[:50]}...")

    def get_lessons(self, agent: Optional[str] = None,
                    min_importance: float = 0.0) -> List[Dict[str, Any]]:
        """Get lessons learned, optionally filtered by agent and importance"""
        lessons = self.lessons_learned

        if agent:
            lessons = [l for l in lessons if l["agent"] == agent]

        if min_importance > 0:
            lessons = [l for l in lessons if l.get("importance", 0) >= min_importance]

        # Sort by importance and recency
        lessons.sort(key=lambda x: (x.get("importance", 0), x["timestamp"]), reverse=True)

        return lessons

    def add_timeline_event(self, event: str, stage: str, agent: str,
                           duration: Optional[float] = None,
                           metadata: Dict[str, Any] = None):
        """Add event to research timeline"""
        if metadata is None:
            metadata = {}

        event_entry = {
            "event": event,
            "stage": stage,
            "agent": agent,
            "duration": duration,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }

        self.timeline.append(event_entry)

        # Keep timeline manageable
        if len(self.timeline) > 50:
            self.timeline = self.timeline[-50:]

    def get_timeline(self, stage: Optional[str] = None,
                     agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get timeline events, optionally filtered"""
        events = self.timeline

        if stage:
            events = [e for e in events if e["stage"] == stage]

        if agent:
            events = [e for e in events if e["agent"] == agent]

        # Sort by timestamp
        events.sort(key=lambda x: x["timestamp"])

        return events

    def create_session_summary(self) -> Dict[str, Any]:
        """Create comprehensive session summary"""
        summary = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "summary_count": len(self.summaries),
            "fact_count": len(self.important_facts),
            "decision_count": len(self.decisions),
            "lesson_count": len(self.lessons_learned),
            "timeline_event_count": len(self.timeline),
            "key_topics": list(self.summaries.keys()),
            "recent_decisions": self.decisions[-5:] if self.decisions else [],
            "important_lessons": self.get_lessons(min_importance=0.7)[:3],
            "timeline_overview": self.get_timeline()[-10:] if self.timeline else []
        }

        # Add overall assessment
        summary["overall_assessment"] = self._create_overall_assessment()

        return summary

    def _create_overall_assessment(self) -> str:
        """Create overall assessment of research session"""
        if not self.summaries and not self.important_facts:
            return "Session just started, no significant findings yet."

        # Count successful facts (high confidence)
        high_confidence_facts = sum(1 for f in self.important_facts if f["confidence"] >= 0.8)

        # Count important decisions with outcomes
        decisions_with_outcomes = sum(1 for d in self.decisions if d["outcome"] != "pending")

        assessment_parts = []

        if self.summaries:
            assessment_parts.append(f"Explored {len(self.summaries)} key topics.")

        if high_confidence_facts > 0:
            assessment_parts.append(f"Established {high_confidence_facts} high-confidence facts.")

        if decisions_with_outcomes > 0:
            assessment_parts.append(f"Made {decisions_with_outcomes} decisions with known outcomes.")

        if self.lessons_learned:
            assessment_parts.append(f"Learned {len(self.lessons_learned)} important lessons.")

        if not assessment_parts:
            assessment_parts.append("Session in progress, collecting data and making initial observations.")

        return " ".join(assessment_parts)

    def clear(self):
        """Clear all memory (use with caution)"""
        self.summaries.clear()
        self.important_facts.clear()
        self.decisions.clear()
        self.lessons_learned.clear()
        self.timeline.clear()

        logger.info("Cleared all summary memory")

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "summaries": self.summaries,
            "important_facts": self.important_facts,
            "decisions": self.decisions,
            "lessons_learned": self.lessons_learned,
            "timeline": self.timeline,
            "exported_at": datetime.now().isoformat()
        }

    def from_dict(self, data: Dict[str, Any]):
        """Load memory from dictionary"""
        self.session_id = data.get("session_id", self.session_id)
        self.summaries = data.get("summaries", {})
        self.important_facts = data.get("important_facts", [])
        self.decisions = data.get("decisions", [])
        self.lessons_learned = data.get("lessons_learned", [])
        self.timeline = data.get("timeline", [])

        logger.info(f"Loaded summary memory for session {self.session_id}")