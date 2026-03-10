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
class ChatbotStateSample(BaseModel):
    query : Annotated[str,Field(...,description="Query of the user")]
    escalate : Annotated[bool,Field(description="Should the matter be escalated to humans?")] = False 
    # web_search : Annotated[bool,Field(description="If the chatbot requires web search")] 
    retrieval : Annotated[bool,"Is retrieval necessary is it a general question"] = False
    persona_classification : Optional[PersonaClassification] = None 
    answer: Optional[str] = None
    docs: List[Document] = Field(default_factory=list)
    summary : Optional[str] = None
    chat_history: Annotated[List[BaseMessage],add_messages]


# ------------------------------
# For optimization of LLM Calls
# ------------------------------
class TriageResult(BaseModel):
    persona: Annotated[
        Literal["technical_expert", "frustrated_user", "business_executive"],
        Field(description="Detected persona of the user")
    ]
    confidence: Annotated[
        float,
        Field(description="Confidence score between 0 and 1", ge=0.0, le=1.0)
    ]
    escalate: Annotated[
        bool,
        Field(description="Whether the issue should be escalated to a human agent")
    ]
    retrieval_required: Annotated[
        bool,
        Field(description="Whether answering the question requires knowledge base retrieval")
    ]


class ChatbotState(BaseModel):

    query : Annotated[str,Field(...,description="Query of the user")]

    persona : Annotated[Literal["technical_expert", "frustrated_user", "business_executive"],"Persona of the user"] = None 
    persona_confidence : Annotated[float,Field(ge=0.0,le=1.0,description="Score of the predicted persona on the scale 0-1")] = 0.0

    escalate : Annotated[bool,Field(description="Should the matter be escalated to humans?")] = False 
    retrieval_required : Annotated[bool,"Is retrieval necessary is it a general question"] = False

    general_answer : Optional[str] = None
    answer: Optional[str] = None
    docs: List[Document] = Field(default_factory=list)

    chat_history: Annotated[List[BaseMessage],add_messages] = Field(default_factory=list)



# -------------------
# FastAPI
# ------------------
    

class RegisterInput(BaseModel):
    gmail: str
    password: str
    name: str


class LoginInput(BaseModel):
    gmail: str
    password: str


class ChatInput(BaseModel):
    gmail: str
    thread_id: str
    question: str