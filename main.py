from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Home route
@app.get("/")
def home():
    return {"message": "AI Agent is running 🚀"}

# ✅ Request model
class Request(BaseModel):
    message: str
    history: list = []

# ✅ Simple RAG (your personal data)
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

# ✅ Chat endpoint
@app.post("/chat")
def chat(req: Request):
    try:
        context = get_context(req.message)

        # Build conversation history
        history_text = ""
        for msg in req.history:
            role = "User" if msg["role"] == "user" else "AI"
            history_text += f"{role}: {msg['content']}\n"

        # Prompt
        prompt = f"""
You are a helpful assistant.

Context:
{context}

Conversation:
{history_text}

User: {req.message}
AI:
"""

        # ✅ HuggingFace NEW API (FIXED)
        response = requests.post(
            "https://router.huggingface.co/hf-inference/models/google/flan-t5-large",
            headers={
                "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
            },
            json={
                "inputs": prompt,
                "options": {
                    "wait_for_model": True
                }
            }
        )

        print("HF RESPONSE:", response.text)

        data = response.json()

        # ✅ Safe response handling
        if isinstance(data, list) and len(data) > 0:
            reply = data[0].get("generated_text", "")
        elif isinstance(data, dict) and "error" in data:
            reply = f"HF Error: {data['error']}"
        else:
            reply = str(data)

        return {"reply": reply}

    except Exception as e:
        return {"reply": f"Error: {str(e)}"}