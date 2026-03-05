from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from my_project.rag.core.schemas import WebSearch, Escalation, PersonaClassification, ChatbotState, Retrieval
from my_project.rag.core.prompts import persona_prompt, escalation_prompt, retrieval_prompt, general_answer_prompt,answer_prompt
from my_project.rag.core.config import settings
from my_project.rag.load_retrieve.retriever import get_retriever
from dotenv import load_dotenv
load_dotenv()


model1 = ChatGoogleGenerativeAI(model = settings.GOOGLE_MODEL_LITE)
model2 = ChatGoogleGenerativeAI(model = settings.GOOGLE_MODEL_HEAVY)
hf_llm = HuggingFaceEndpoint(repo_id=settings.HF_LLM)
model3 = ChatHuggingFace(llm = hf_llm)
parser = StrOutputParser()
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
        return "HumanSupport"
    else:
        return "RetrievalDecider"
    
def HumanSupport(state: ChatbotState):
    return {"answer": "Your issue has been escalated to a human support agent."}

def RetrievalDecider(state : ChatbotState):
    chain = retrieval_prompt | retrieval_decider
    result = chain.invoke({
        'query':state.query,
        'chat_history':state.chat_history[-2:] if state.chat_history else []
        })
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
    chain = general_answer_prompt | model1 
    result = chain.invoke({
        'persona' : state.persona_classification.persona,
        'query' : state.query,
        'chat_history' : state.chat_history[-2:] if state.chat_history else []
    })
    return {'answer':result.content,'chat_history':[result]}

def Answer(state : ChatbotState):
    chain = answer_prompt | model2
    context = "\n\n".join(doc.page_content for doc in state.docs)
    result = chain.invoke({
        'context' : context,
        'persona' : state.persona_classification.persona,
        'query' : state.query,
        'chat_history' : state.chat_history[-4:] if state.chat_history else []
    }) 
    return {'answer':result.content,'chat_history':[result]}



# Buidling the graph
graph = StateGraph(ChatbotState)

graph.add_node("PersonaDetection",PersonaDetection)
graph.add_node("EscalationDecision",EscalationDecision)
graph.add_node("HumanSupport",HumanSupport)
graph.add_node("RetrievalDecider",RetrievalDecider)
graph.add_node("Retrieve",Retrieve)
graph.add_node("GeneralAnswer",GeneralAnswer)
graph.add_node("Answer",Answer)


graph.add_edge(START,"PersonaDetection")
graph.add_edge("PersonaDetection","EscalationDecision")
graph.add_conditional_edges("EscalationDecision",EscalationRouter)
graph.add_edge("EscalationDecision","HumanSupport")
graph.add_edge("EscalationDecision","RetrievalDecider")
graph.add_edge("HumanSupport",END)
graph.add_conditional_edges("RetrievalDecider",RetrievalRouter)
graph.add_edge("RetrievalDecider","Retrieve")
graph.add_edge("RetrievalDecider","GeneralAnswer")
graph.add_edge("GeneralAnswer",END)
graph.add_edge("Retrieve","Answer")
graph.add_edge("Answer",END)

chatbot = graph.compile()