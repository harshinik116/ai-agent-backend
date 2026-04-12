from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

app = FastAPI()

# ✅ CORS
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

# ✅ Load family data from file
def load_data():
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return ""

# ✅ Chat endpoint
@app.post("/chat")
def chat(req: Request):
    try:
        # 🧠 Load your personal data
        context = load_data()

        # 🧠 Build conversation history
        history_text = ""
        for msg in req.history:
            role = "User" if msg["role"] == "user" else "AI"
            history_text += f"{role}: {msg['content']}\n"

        # 🧠 Final prompt
        prompt = f"""
You are a helpful family assistant.

Use ONLY the provided family data to answer.

Family Data:
{context}

Conversation:
{history_text}

User: {req.message}

If the answer is not in the family data, say:
"I don't know from family data."

AI:
"""

        # 🚀 GROQ API CALL
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

        print("GROQ RESPONSE:", response.text)

        data = response.json()

        # ✅ Safe response handling
        if "choices" in data:
            reply = data["choices"][0]["message"]["content"]
        elif "error" in data:
            reply = f"Groq Error: {data['error']['message']}"
        else:
            reply = str(data)

        return {"reply": reply}

    except Exception as e:
        return {"reply": f"Error: {str(e)}"}


# ✅ Port binding (Render + local)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )