"""Pydantic models for data validation and serialization."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class Impact(BaseModel):
    """Model for tracking the impact of work."""
    scope: str = Field(default="individual", description="Scope of impact: individual, team, organization")
    significance: int = Field(default=3, description="Impact significance rating (1-5 scale)", ge=1, le=5)
    metrics: List[str] = Field(default=[], description="Quantifiable metrics of impact")
    business_value: str = Field(default="", description="Business value delivered")


class JiraTicket(BaseModel):
    """Model for JIRA ticket information."""
    key: str = Field(description="JIRA ticket key (e.g., PROJ-123)")
    title: str = Field(description="Ticket title")
    confidence: str = Field(description="Confidence level: high, medium, low")
    match_reason: str = Field(description="Reason for matching this ticket")


class ProcessedEntry(BaseModel):
    """Model for LLM-processed entry data."""
    summary: str = Field(description="Brief summary of the accomplishment")
    work: str = Field(description="Detailed description of work performed")
    collaborators: List[str] = Field(default=[], description="People who collaborated on this work")
    projects: List[str] = Field(default=[], description="Projects this work is associated with")
    impact: Impact = Field(description="Impact assessment of the work")
    technical_details: List[str] = Field(default=[], description="Technical approaches and tools used")
    tags: List[str] = Field(default=[], description="Categorization tags")


class WorkflowConfig(BaseModel):
    """Model for tracking which LLM workflow was used."""
    conversation: Dict[str, str] = Field(description="Provider and model used for conversation")
    processing: Dict[str, str] = Field(description="Provider and model used for processing")
    jira_matching: Optional[Dict[str, str]] = Field(default=None, description="Provider and model used for JIRA matching")


class JournalEntry(BaseModel):
    """Model for a complete journal entry."""
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the entry")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the entry was created")
    date: str = Field(description="Date this entry is for (YYYY-MM-DD)")
    raw_input: str = Field(description="Original user input")
    processed: ProcessedEntry = Field(description="LLM-processed structured data")
    jira_tickets: List[JiraTicket] = Field(default=[], description="Associated JIRA tickets")
    metadata: Dict[str, Any] = Field(default={}, description="Processing metadata")
    tombstoned: bool = Field(default=False, description="Soft delete flag")
    tombstoned_at: Optional[datetime] = Field(default=None, description="When entry was tombstoned")


class Provider(BaseModel):
    """Simple LLM provider connection definition."""
    name: str = Field(description="Unique provider identifier")
    service_name: str = Field(description="Human-readable display name")
    protocol: str = Field(description="API protocol: openai_compatible, ollama, anthropic")
    api_base: str = Field(description="API endpoint URL")
    auth_env: Optional[str] = Field(default=None, description="Environment variable for auth token")
    
    def __hash__(self):
        """Make Provider hashable based on name."""
        return hash(self.name)
    
    def __eq__(self, other):
        """Provider equality based on name."""
        if isinstance(other, Provider):
            return self.name == other.name
        return False


class ModelAssignment(BaseModel):
    """Assignment of a specific model from a provider."""
    provider: str = Field(description="Provider name")
    model: str = Field(description="Model name")


class Configuration(BaseModel):
    """Named configuration with model assignments for all 3 needs."""
    name: str = Field(description="Configuration display name")
    conversation: ModelAssignment = Field(description="Model for interactive chat and refinement")
    processing: ModelAssignment = Field(description="Model for structuring work entries")
    jira_matching: ModelAssignment = Field(description="Model for finding JIRA tickets")


class Settings(BaseModel):
    """Complete application settings."""
    providers: Dict[str, Provider] = Field(default={}, description="Available LLM providers")
    configurations: Dict[str, Configuration] = Field(default={}, description="Named configurations")
    current_config: Optional[str] = Field(default=None, description="Currently active configuration name")