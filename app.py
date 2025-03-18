from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API is running!"}

@app.get("/ping")
async def ping():
    return {"message": "I'm alive!"}

# 🔹 تحميل الموديل من Hugging Face مع تقليل استهلاك الذاكرة
MODEL_NAME = "bishoy1/swimming_coach"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)  

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"
)

# 🔹 تعريف نموذج البيانات المستقبلة
class ChatRequest(BaseModel):
    message: str

# 🔹 API Endpoint لمعالجة الأسئلة
@app.post("/chat")
async def chat(request: ChatRequest):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(request.message, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs, 
        max_length=150, 
        temperature=0.7, 
        top_p=0.9, 
        repetition_penalty=1.2
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
