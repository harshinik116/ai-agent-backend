from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from datetime import datetime

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Home
@app.get("/")
def home():
    return {"message": "AI Agent is running"}

# Request model
class Request(BaseModel):
    message: str
    history: list = []

# Load data
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

# Chat endpoint
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
                    {
                        "role": "user",
                        "content": prompt
                    }
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

# Image endpoint (TEMP SAFE VERSION)
import base64

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}",
            headers={
                "Content-Type": "application/json"
            },
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": "Describe this image in detail"},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            }
        )

        data = response.json()

        # ✅ Extract Gemini response
        if "candidates" in data:
            reply = data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            reply = str(data)

        return {"reply": reply}

    except Exception as e:
        return {"reply": str(e)}