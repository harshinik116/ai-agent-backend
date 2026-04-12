from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from datetime import datetime

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AI Agent is running 🚀"}

class Request(BaseModel):
    message: str
    history: list = []

# ✅ Load family data
def load_data():
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""

# ✅ Real-time date & time
def get_realtime_info():
    now = datetime.now()
    return f"Current date: {now.strftime('%Y-%m-%d')}, Current time: {now.strftime('%H:%M:%S')}"

@app.post("/chat")
def chat(req: Request):
    try:
        context = load_data()
        realtime = get_realtime_info()

        history_text = ""
        for msg in req.history:
            role = "User" if msg["role"] == "user" else "AI"
            history_text += f"{role}: {msg['content']}\n"

        prompt = f"""
You are a smart AI assistant.

You have access to:
1. Family data
2. General knowledge
3. Real-time system info (date/time)

Rules:
- If question is about family → use family data
- If question is about current date/time → use real-time info
- Otherwise → answer normally

Real-Time Info:
{realtime}

Family Data:
{context}

Conversation:
{history_text}

User: {req.message}

AI:
"""

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ]
            }
        )

        data = response.json()

        if "choices" in data:
            reply = data["choices"][0]["message"]["content"]
        elif "error" in data:
            reply = f"Groq Error: {data['error']['message']}"
        else:
            reply = str(data)

        return {"reply": reply}

    except Exception as e:
        return {"reply": f"Error: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )