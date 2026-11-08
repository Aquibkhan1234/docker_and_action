import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-20b") 

app = FastAPI(title="LAngchain AI Chatbot")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Give the answer of {question} in short and concise manner.")
])

chain = prompt | llm

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Langchain API is running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = chain.invoke({"question": request.question})  
        return {
            "question": request.question,
            "answer": response.content
        } 
    except Exception as e:
        raise HTTPException(status_code=500, detail= "LLM request failed") 