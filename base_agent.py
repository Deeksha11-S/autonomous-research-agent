from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
from loguru import logger
from pydantic import BaseModel, Field

from backend.config import settings
from backend.memory.vector_store import VectorMemory


class AgentMessage(BaseModel):
    """Standard message format for agent communication"""
    sender: str
    recipient: str
    content: Any
    message_type: str = "information"  # information, question, command, result
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, agent_id: str, role: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.memory = VectorMemory(f"{role}_{agent_id}")
        self.conversation_history: List[AgentMessage] = []
        self.tools = {}
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LLM based on agent role"""
        from langchain_groq import ChatGroq

        if self.role in ["critic", "uncertainty"] and settings.anthropic_api_key:
            # Use Claude for critique (if available)
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=settings.critique_model,
                temperature=0.2,
                max_tokens=4000
            )
        else:
            # Default to Groq
            return ChatGroq(
                groq_api_key=settings.groq_api_key,
                model_name=settings.groq_model,
                temperature=0.3,
                max_tokens=4000
            )

    @abstractmethod
    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process incoming message and return response"""
        pass

    async def send_message(self, recipient: 'BaseAgent', content: Any,
                           message_type: str = "information", **metadata) -> AgentMessage:
        """Send message to another agent"""
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient.agent_id,
            content=content,
            message_type=message_type,
            metadata=metadata
        )
        self.conversation_history.append(message)
        return message

    def add_tool(self, name: str, tool_func: callable, description: str):
        """Add tool to agent's toolkit"""
        self.tools[name] = {
            "function": tool_func,
            "description": description
        }

    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Use a tool from agent's toolkit"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not available")

        tool = self.tools[tool_name]
        logger.info(f"{self.agent_id} using tool: {tool_name}")

        try:
            if asyncio.iscoroutinefunction(tool["function"]):
                result = await tool["function"](**kwargs)
            else:
                result = tool["function"](**kwargs)
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            raise

    def get_capability_score(self, task_description: str) -> float:
        """Calculate agent's capability score for a task"""
        # Simple keyword matching for now
        keywords = task_description.lower().split()
        capability_keywords = " ".join(self.capabilities).lower()

        matches = sum(1 for keyword in keywords if keyword in capability_keywords)
        score = matches / max(len(keywords), 1)

        # Adjust based on role
        role_boost = {
            "domain_scout": 0.3 if "discover" in task_description else 0,
            "question_generator": 0.3 if "question" in task_description else 0,
            "data_alchemist": 0.3 if "data" in task_description else 0,
            "experiment_designer": 0.3 if "experiment" in task_description else 0,
            "critic": 0.3 if "critique" in task_description else 0,
        }

        return min(score + role_boost.get(self.role, 0), 1.0)

    def log(self, message: str, level: str = "info"):
        """Log agent activity"""
        log_method = getattr(logger, level)
        log_method(f"[{self.role.upper()}:{self.agent_id}] {message}")