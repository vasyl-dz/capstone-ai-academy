"""
FastAPI Server for RAG Chatbot - OpenAI Compatible API
Exposes the RAG chatbot as an OpenAI-compatible API for Open WebUI integration
"""

import os
import time
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from steps.chatbot import RAGChatbot

# Load environment variables
load_dotenv()

# Configuration
DB_PATH = "chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "llama3.1:8b"

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="OpenAI-compatible API for RAG Chatbot with reasoning and tool calling",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot (lazy loading)
chatbot: Optional[RAGChatbot] = None


def get_chatbot() -> RAGChatbot:
    """Get or initialize the chatbot instance."""
    global chatbot
    if chatbot is None:
        print("Initializing RAG Chatbot...")
        chatbot = RAGChatbot(
            db_path=DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_model=EMBEDDING_MODEL,
            llm_model=LLM_MODEL
        )
        print("Chatbot initialized successfully!")
    return chatbot


# Pydantic models for OpenAI-compatible API
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "rag-chatbot"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "rag-system"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "RAG Chatbot API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible endpoint)."""
    return ModelsResponse(
        data=[
            ModelInfo(
                id="rag-chatbot",
                created=int(time.time())
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    This is the main endpoint that Open WebUI will call.
    """
    try:
        # Get chatbot instance
        bot = get_chatbot()
        
        # Extract the last user message
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        query = user_messages[-1].content
        
        # Get answer from RAG chatbot
        answer = bot.answer_question(query)
        
        # Format response in OpenAI style
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": len(query.split()),
                "completion_tokens": len(answer.split()),
                "total_tokens": len(query.split()) + len(answer.split())
            }
        )
        
        return response
        
    except Exception as e:
        print(f"Error in chat_completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_chatbot(query: str):
    """
    Simple query endpoint for testing.
    Send a query and get back an answer.
    """
    try:
        bot = get_chatbot()
        answer = bot.answer_question(query)
        return {
            "query": query,
            "answer": answer,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("API_PORT", "8000"))
    
    print("\n" + "=" * 80)
    print("STARTING RAG CHATBOT API SERVER")
    print("=" * 80)
    print(f"Server will start at: http://localhost:{port}")
    print("OpenAI-compatible endpoint: http://localhost:{port}/v1/chat/completions")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
