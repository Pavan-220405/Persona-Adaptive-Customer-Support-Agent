from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from my_project.rag.core.schemas import WebSearch, Escalation, PersonaClassification, ChatbotState, Retrieval
from my_project.rag.core.prompts import persona_prompt, escalation_prompt, retrieval_prompt
from my_project.rag.core.config import settings
from my_project.rag.load_retrieve.retriever import get_retriever
from dotenv import load_dotenv
load_dotenv()


model1 = ChatGoogleGenerativeAI(model = settings.GOOGLE_MODEL_LITE)
model2 = ChatGoogleGenerativeAI(model = settings.GOOGLE_MODEL_HEAVY)
hf_llm = HuggingFaceEndpoint(repo_id=settings.HF_LLM)
model3 = ChatHuggingFace(llm = hf_llm)
retriever = get_retriever()
websearch_decider = model1.with_structured_output(WebSearch)
escalation_decider = model1.with_structured_output(Escalation)
persona_detector = model1.with_structured_output(PersonaClassification)
retrieval_decider = model1.with_structured_output(Retrieval)


# Nodes/Functions
def PersonaDetection(state : ChatbotState):
    chain = persona_prompt | persona_detector
    persona = chain.invoke({'query':state.query})
    return {'persona_classification' : persona}

def EscalationDecision(state : ChatbotState):
    chain = escalation_prompt | escalation_decider
    result = chain.invoke({
        'persona' : state.persona_classification.persona,
        'query' : state.query,
        'chat_history' : state.chat_history[-4:] if state.chat_history else []
    })
    return {'escalate':result.escalate}

def EscalationRouter(state : ChatbotState):
    if state.escalate:
        return END
    else:
        return "RetrievalDecider"
    
def RetrievalDecider(state : ChatbotState):
    chain = retrieval_prompt | retrieval_decider
    result = chain.invoke({'query':state.query,'chat_history':state.chat_history[-2:]})
    return {'retrieval':result.retrieval}

def RetrievalRouter(state : ChatbotState):
    if state.retrieval:
        return "Retrieve"
    else:
        return "GeneralAnswer"
    
def Retrieve(state : ChatbotState):
    docs = retriever.invoke(state.query)
    return {'docs':docs}

def GeneralAnswer(state : ChatbotState):
    pass 