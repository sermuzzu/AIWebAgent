from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from Example import Agent
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory agent instance (for demo; for multi-user, use sessions or DB)
agent_instance = None

class Message(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/start")
def start_chat(msg: Message):
    global agent_instance
    agent_instance = Agent(msg.message)
    return {"status": "started", "goal": msg.message}

@app.post("/chat")
def chat(msg: Message):
    global agent_instance
    if agent_instance is None:
        return JSONResponse(status_code=400, content={"error": "Chat not started."})

    user_input = msg.message
    # Add user input to agent memory
    agent_instance.memory.add(f"User: {user_input}")

    # Build context using Example.py logic
    context = agent_instance.context_mgr.build(
        goal=agent_instance.goal,
        last_action=agent_instance.last_action
    )

    # Compose the prompt for the agent to reply only (no thoughts)
    prompt = context + "\nUser: " + user_input

    # Use the Planner to get the agent's reply (as in Example.py)
    agent_reply = agent_instance.planner.plan(prompt)

    # Add agent reply to memory
    agent_instance.memory.add(f"Agent: {agent_reply}")
    agent_instance.last_action = user_input

    return {"reply": agent_reply}

if __name__ == "__main__":
    uvicorn.run("web_chat:app", host="0.0.0.0", port=8000, reload=True)
