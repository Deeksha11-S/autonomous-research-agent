from pydantic_settings import BaseSettings
from typing import List, Optional
from enum import Enum


class AgentRole(str, Enum):
    DOMAIN_SCOUT = "domain_scout"
    QUESTION_GENERATOR = "question_generator"
    DATA_ALCHEMIST = "data_alchemist"
    EXPERIMENT_DESIGNER = "experiment_designer"
    CRITIC = "critic"
    UNCERTAINTY = "uncertainty"
    ORCHESTRATOR = "orchestrator"


class Settings(BaseSettings):
    # API Keys (Free Tier)
    groq_api_key: str
    anthropic_api_key: Optional[str] = None
    serper_api_key: str
    tavily_api_key: Optional[str] = None

    # Application Settings
    max_iterations: int = 5
    min_confidence_threshold: float = 0.6
    research_timeout: int = 1800  # 30 minutes

    # Model Settings
    groq_model: str = "llama-3.1-70b-versatile"
    critique_model: str = "claude-3-haiku-20240307"  # Free tier

    # Agent Settings
    enable_agents: List[AgentRole] = [
        AgentRole.DOMAIN_SCOUT,
        AgentRole.QUESTION_GENERATOR,
        AgentRole.DATA_ALCHEMIST,
        AgentRole.EXPERIMENT_DESIGNER,
        AgentRole.CRITIC,
        AgentRole.UNCERTAINTY,
        AgentRole.ORCHESTRATOR
    ]

    # Search Settings
    search_results_limit: int = 10
    min_domain_sources: int = 3
    min_data_sources: int = 3

    # Data Settings
    max_papers_to_fetch: int = 20
    max_web_pages: int = 10

    # Paths
    data_dir: str = "data"
    logs_dir: str = "logs"
    cache_dir: str = "cache"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()