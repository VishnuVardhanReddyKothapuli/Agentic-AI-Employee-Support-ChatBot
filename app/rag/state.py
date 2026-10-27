from typing import List, Literal, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RouteDecision(BaseModel):
    route: Literal["kb", "direct", "off_topic"] = Field(
        description="kb for company HR/policy questions; direct for greetings/thanks/identity questions; off_topic for general trivia, coding, external AI models, entertainment, or non-HR topics"
    )


class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description="Whether evidence is sufficient to answer"
    )


class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int
    trace: List[str]
    citations: List[Any]
