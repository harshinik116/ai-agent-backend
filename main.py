from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

app = FastAPI()

# CORS
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

documents = [
    "My name is Harshini",
    "I am a software engineer",
    "I love spiritual topics and meditation",
    "I am learning AI and building agents"
]

def get_context(query):
    for doc in documents:
        if any(word.lower() in doc.lower() for word in query.split()):
            return doc
    return ""

@app.post("/chat")
def chat(req: Request):
    try:
        context = get_context(req.message)

        history_text = ""
        for msg in req.history:
            role = "User" if msg["role"] == "user" else "AI"
            history_text += f"{role}: {msg['content']}\n"

        prompt = f"""
You are a helpful assistant.

Context:
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
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ]
            }
        )

        data = response.json()

        reply = data["choices"][0]["message"]["content"]

        return {"reply": reply}

    except Exception as e:
        return {"reply": f"Error: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))