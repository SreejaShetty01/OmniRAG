"""Validated query request contract."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnirag.config.profiles import ProfileName


class EscalationPreference(StrEnum):
    NEVER = "never"
    WHEN_NEEDED = "when_needed"
    ALWAYS = "always"
    ASK = "ask"
    AUTO_WITH_NOTICE = "auto_with_notice"


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str | None = None
    conversation_id: str | None = None
    query_text: str = Field(min_length=1)
    selected_profile: ProfileName = ProfileName.CORE
    filters: dict[str, object] = Field(default_factory=dict)
    conversation_context: list[dict[str, str]] = Field(default_factory=list)
    escalation_preference: EscalationPreference = EscalationPreference.WHEN_NEEDED

    @field_validator("query_text")
    @classmethod
    def query_must_contain_non_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query_text must contain non-whitespace characters")
        return value
