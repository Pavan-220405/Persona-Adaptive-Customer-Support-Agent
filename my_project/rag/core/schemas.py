from pydantic import BaseModel, Field
from typing import Annotated, Literal, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


# For model.with_structured_output
class PersonaClassification(BaseModel):
    persona : Annotated[Literal["technical_expert", "frustrated_user", "business_executive"],
                        Field(...,description="Identify the persona of the user and classify them as : technical_expert/frustrated_user/business_executive")
                        ]
    score : Annotated[float,Field(...,description="Score of the detected persona on a scale of 0-1",ge=0.0,le=1.0)]


# For model.with_structured_output
class Escalation(BaseModel):
    escalate : Annotated[bool,Field(description="Should the matter be escalated to humans?")]


# For model.with_structured_output
class WebSearch(BaseModel):
    web_search : Annotated[bool,Field(description="If the chatbot requires web search")]

class Retrieval(BaseModel): 
    retrieval : Annotated[bool,"Is retrieval necessary is it a general question"]


# Inner Langgraph state
class ChatbotState(BaseModel):
    query : Annotated[str,Field(...,description="Query of the user")]
    escalate : Annotated[bool,Field(description="Should the matter be escalated to humans?")] = False 
    # web_search : Annotated[bool,Field(description="If the chatbot requires web search")] 
    retrieval : Annotated[bool,"Is retrieval necessary is it a general question"] = False
    persona_classification : Optional[PersonaClassification] = None 
    answer: Optional[str] = None
    docs: List[Document] = Field(default_factory=list)
    summary : Optional[str] = None
    chat_history: Annotated[List[BaseMessage],add_messages]