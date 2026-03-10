from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
from my_project.rag.graph.agent import chatbot
from my_project.rag.core.schemas import ChatInput


app = FastAPI()

@app.post("/chat")
def chat(input : ChatInput):
    config = {'configurable':{'thread_id':input.thread_id}}
    initial_state = {'query':input.question,'chatbot_history':[input.question]}

    try:
        final_state = chatbot.invoke(initial_state,config=config)
        answer = final_state.get("answer") or final_state.get("general_answer")
        return {"final_state" : final_state,"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    