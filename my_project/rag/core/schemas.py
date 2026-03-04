from pydantic import BaseModel, Field
from typing import Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class persona_classification(BaseModel):
    customer_persona : Annotated[Literal["technical_expert", "frustrated_user", "business_executive"],Field(...,description="Identify the persona of the user")]
    score : Annotated[int,Field(...,description="Score of the detected persona")]