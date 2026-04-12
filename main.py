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

        # Build history
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

        # ✅ GROQ API CALL
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

        print("GROQ RESPONSE:", response.text)

        data = response.json()

        # ✅ Safe handling
        if "choices" in data:
            reply = data["choices"][0]["message"]["content"]
        elif "error" in data:
            reply = f"Groq Error: {data['error']['message']}"
        else:
            reply = str(data)

        return {"reply": reply}

    except Exception as e:
        return {"reply": f"Error: {str(e)}"}


# ✅ IMPORTANT: Port binding (Render + local)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )