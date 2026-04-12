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

# ✅ Simple RAG
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

        # ✅ HuggingFace working model
        response = requests.post(
            "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2",
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

        # ✅ Handle empty response
        if response.text.strip() == "":
            return {"reply": "Model is loading, please try again..."}

        try:
            data = response.json()
        except Exception:
            return {"reply": f"Invalid response from HF: {response.text}"}

        if isinstance(data, list) and len(data) > 0:
            reply = data[0].get("generated_text", "")
        elif isinstance(data, dict) and "error" in data:
            reply = f"HF Error: {data['error']}"
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
        port=int(os.environ.get("PORT", 8000))  # Render uses PORT, local uses 8000
    )