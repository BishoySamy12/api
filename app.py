from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# تحميل الموديل من Hugging Face
MODEL_NAME = "bishoy1/swimming_coach"  # استبدل باسم الموديل الخاص بك
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# إنشاء تطبيق FastAPI
app = FastAPI()

# تعريف نموذج البيانات المستقبلة
class ChatRequest(BaseModel):
    message: str

# API Endpoint لمعالجة الأسئلة
@app.post("/chat")
async def chat(request: ChatRequest):
    inputs = tokenizer(request.message, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_length=150, temperature=0.7, top_p=0.9, repetition_penalty=1.2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
