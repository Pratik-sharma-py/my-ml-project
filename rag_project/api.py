from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import query

app = FastAPI(title="RAG API", description="Local RAG System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list
    cached: bool = False

@app.get("/")
def home():
    return {"message": "RAG API is running!"}

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    result = query(req.question)
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"]
    )

@app.get("/health")
def health():
    return {"status": "ok"}