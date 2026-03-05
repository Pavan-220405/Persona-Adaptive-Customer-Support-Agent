from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from my_project.rag.core.schemas import ChatbotState, TriageResult
from my_project.rag.core.prompts import general_answer_prompt,answer_prompt, triage_prompt
from my_project.rag.core.config import settings
from my_project.rag.load_retrieve.retriever import get_retriever
from my_project.rag.core.sqlite_checkpointer import get_checkpointer
from dotenv import load_dotenv
load_dotenv()


model1 = ChatGoogleGenerativeAI(model = settings.GOOGLE_MODEL_LITE)
model2 = ChatGoogleGenerativeAI(model = settings.GOOGLE_MODEL_HEAVY)
retriever = get_retriever()
triage_model = model1.with_structured_output(TriageResult)


# -----------------
# Functions/ Nodes 
# -----------------
def TriageNode(state: ChatbotState):
    chain = triage_prompt | triage_model
    result = chain.invoke({
        "query": state.query,
        "chat_history": state.chat_history[-4:] if state.chat_history else []
    })

    return {
        "persona": result.persona,
        "persona_confidence": result.confidence,
        "escalate": result.escalate,
        "retrieval_required": result.retrieval_required
    }

def HumanSupport(state: ChatbotState):
    return {"answer": "Your issue has been escalated to a human support agent."}

def MainRouter(state : ChatbotState):
    if state.escalate:
        return "HumanSupport"
    elif state.retrieval_required:
        return "Retrieve"
    else:
        return "GeneralAnswer"
    
def Retrieve(state : ChatbotState):
    docs = retriever.invoke(state.query)
    return {'docs':docs}

def GeneralAnswer(state : ChatbotState):
    chain = general_answer_prompt | model1 
    result = chain.invoke({
        'persona' : state.persona,
        'query' : state.query,
        'chat_history' : state.chat_history[-2:] if state.chat_history else []
    })
    return {'general_answer':result.content,'chat_history':[result]}

def Answer(state : ChatbotState):
    chain = answer_prompt | model2
    context = "\n\n".join(doc.page_content for doc in state.docs)
    result = chain.invoke({
        'context' : context,
        'persona' : state.persona,
        'query' : state.query,
        'chat_history' : state.chat_history[-4:] if state.chat_history else []
    }) 
    return {'answer':result.content,'chat_history':[result]}



# ----------------------
# Building the chatbot 
# ---------------------
graph = StateGraph(ChatbotState)

graph.add_node("TriageNode",TriageNode)
graph.add_node("HumanSupport",HumanSupport)
graph.add_node("Retrieve",Retrieve)
graph.add_node("GeneralAnswer",GeneralAnswer)
graph.add_node("Answer",Answer)


graph.add_edge(START,"TriageNode")
graph.add_conditional_edges("TriageNode",MainRouter,{"HumanSupport":"HumanSupport","Retrieve":"Retrieve","GeneralAnswer":"GeneralAnswer"})
graph.add_edge("Retrieve","Answer")
graph.add_edge("Answer",END)
graph.add_edge("GeneralAnswer",END)
graph.add_edge("HumanSupport",END)

checkpointer = get_checkpointer()
chatbot = graph.compile(checkpointer=checkpointer)