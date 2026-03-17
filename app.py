from fastapi import FastAPI,HTTPException, Depends
from my_project.rag.graph.agent import chatbot
from my_project.rag.core.schemas import ChatInput, RegisterInput, LoginInput
from my_project.database.crud import create_user, get_user
from my_project.database.utils import hash_password, verify_password
from my_project.database.database import get_connection



app = FastAPI(title="Persona Adaptive Customer Support Agent")

@app.get("/")
def root():
    return {"status": "Customer Support Chatbot API running"}


def get_db_connection():
    try:
        conn = get_connection()
        yield conn
    finally:
        conn.close()


@app.post("/register")
def register_user(user: RegisterInput, conn = Depends(get_db_connection)):
    existing_user = get_user(conn=conn,gmail=user.gmail)
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")


    hashed_password = hash_password(user.password)
    create_user(
        conn=conn,
        gmail=user.gmail,
        password=hashed_password,
        name=user.name
    )

    return {"message": "User registered successfully"}


@app.post("/login")
def login_user(user: LoginInput, conn = Depends(get_db_connection)):
    db_user = get_user(conn=conn,gmail=user.gmail)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "message": "Login successful",
        "name": db_user["name"],
        "gmail": db_user["gmail"]
    }


@app.post("/chat")
def chat(input: ChatInput, conn = Depends(get_db_connection)):
    user = get_user(conn=conn,gmail=input.gmail)
    if not user:
        raise HTTPException(status_code=401, detail="User not registered")


    config = {"configurable": {"thread_id": input.thread_id}}
    initial_state = {
        "query": input.question,
        "chatbot_history": [input.question]
    }


    try:
        final_state = chatbot.invoke(initial_state, config=config)
        answer = final_state.get("answer") or final_state.get("general_answer")
        return {
            "user": user["name"],
            "question": input.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    