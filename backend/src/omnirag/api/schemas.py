"""Typed product API contracts, separate from storage and provider details."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnirag.config.profiles import ProfileName
from omnirag.models.query import EscalationPreference
from omnirag.models.response import OmniRAGResponse
from omnirag.experimentation.runtime import ExperimentConfig


class PreferencesResponse(BaseModel):
    preferred_name: str | None = None
    onboarding_skipped: bool = False
    default_profile: ProfileName = ProfileName.CORE
    escalation_preference: EscalationPreference = EscalationPreference.WHEN_NEEDED


class PreferencesUpdate(BaseModel):
    preferred_name: str | None = Field(default=None, max_length=80)
    onboarding_skipped: bool | None = None
    default_profile: ProfileName | None = None
    escalation_preference: EscalationPreference | None = None

    @field_validator("preferred_name")
    @classmethod
    def validate_name(cls, value):
        if value is not None:
            value = " ".join(value.split())
            if not value:
                raise ValueError("preferred_name must be non-empty or null")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(min_length=1, max_length=12000)
    conversation_id: str | None = None
    selected_profile: ProfileName | None = None
    escalation_preference: EscalationPreference | None = None
    collection_id: str | None = None
    experiment: ExperimentConfig | None = None

    @field_validator("query_text")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query_text cannot be blank")
        return value


class ChatTurnResponse(BaseModel):
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    response: OmniRAGResponse


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    selected_profile: ProfileName
    created_at: str
    updated_at: str


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        value = " ".join(value.split())
        if not value:
            raise ValueError("Collection name cannot be blank")
        return value


class CollectionResponse(BaseModel):
    collection_id: str
    name: str
    description: str
    created_at: str
    document_count: int = 0


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    version: str | None = None
    collection_id: str | None = None
    source_type: str | None = None
    document_type: str | None = None
    source_hash: str | None = None
    chunk_count: int = 0
    size_bytes: int = 0
    error_message: str | None = None
    created_at: str
    updated_at: str


class DocumentCollectionUpdate(BaseModel):
    collection_id: str | None


class EvaluationStatus(BaseModel):
    available: bool
    dataset_available: bool
    suites: list[str] = Field(default_factory=list)
    message: str
