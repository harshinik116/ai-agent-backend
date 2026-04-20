from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from datetime import datetime
import base64

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Home route
@app.get("/")
def home():
    return {"message": "AI Agent is running"}

# Request model
class Request(BaseModel):
    message: str
    history: list = []

# Load personal data
def load_data():
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""

# Real-time info
def get_realtime_info():
    now = datetime.now()
    return f"Date: {now.strftime('%Y-%m-%d')} Time: {now.strftime('%H:%M:%S')}"

# Chat endpoint (Groq)
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

Use:
- Family data
- Real-time info
- General knowledge

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
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        data = response.json()

        if "choices" in data:
            reply = data["choices"][0]["message"]["content"]
        else:
            reply = str(data)

        return {"reply": reply}

    except Exception as e:
        return {"reply": str(e)}

# Image endpoint (Gemini – FINAL FIXED)
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        response = requests.post(
            "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large",
            headers={
                "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
            },
            files={"file": contents}
        )

        # 🔥 DEBUG (IMPORTANT)
        print("STATUS:", response.status_code)
        print("RAW RESPONSE:", response.text)

        # ✅ Safe handling
        if response.status_code != 200:
            return {"reply": f"HF Error: {response.text}"}

        try:
            data = response.json()
        except:
            return {"reply": "Model is loading... try again in few seconds ⏳"}

        if isinstance(data, list) and len(data) > 0:
            reply = data[0].get("generated_text", "")
        else:
            reply = str(data)

        return {"reply": reply}

    except Exception as e:
        return {"reply": str(e)}
# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )