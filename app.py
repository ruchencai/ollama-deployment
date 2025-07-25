from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import time
import logging
import os
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen3 API",
    description="API for Qwen3 model inference using Ollama",
    version="1.0.0"
)

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b")

class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    model: str
    created_at: str
    done: bool

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL):
        self.base_url = base_url
        self.model_name = MODEL_NAME
        
    def health_check(self) -> bool:
        """Check if Ollama is running and responsive"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    def pull_model(self) -> bool:
        """Pull the model if not already available"""
        try:
            logger.info(f"Pulling model {self.model_name}...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=600  # 10 minutes timeout for model download
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
    
    def list_models(self) -> list:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return response.json().get("models", [])
            return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> dict:
        """Generate response from the model"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stop": ["<|im_end|>", "<|endoftext|>"]
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # 2 minutes timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Generation failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return None

# Initialize Ollama client
ollama_client = OllamaClient()

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Qwen3 API...")
    
    # Wait for Ollama to be ready
    max_retries = 30
    for i in range(max_retries):
        if ollama_client.health_check():
            logger.info("Ollama is ready!")
            break
        logger.info(f"Waiting for Ollama... ({i+1}/{max_retries})")
        time.sleep(10)
    else:
        logger.error("Ollama failed to start within timeout")
        return
    
    # Check if model is available, pull if not
    models = ollama_client.list_models()
    model_names = [model.get("name", "") for model in models]
    
    if not any(MODEL_NAME in name for name in model_names):
        logger.info(f"Model {MODEL_NAME} not found, pulling...")
        if not ollama_client.pull_model():
            logger.error("Failed to pull model")
            return
    
    logger.info("Application ready!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Qwen3 API is running",
        "model": MODEL_NAME,
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "models": "/models"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    ollama_healthy = ollama_client.health_check()
    models = ollama_client.list_models()
    
    return {
        "status": "healthy" if ollama_healthy else "unhealthy",
        "ollama_status": "running" if ollama_healthy else "down",
        "model": MODEL_NAME,
        "available_models": len(models),
        "timestamp": time.time()
    }

@app.get("/models")
async def list_models():
    """List available models"""
    models = ollama_client.list_models()
    return {
        "models": models,
        "current_model": MODEL_NAME
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with Qwen3 model"""
    if not ollama_client.health_check():
        raise HTTPException(status_code=503, detail="Ollama service is not available")
    
    # Generate response
    result = ollama_client.generate(
        prompt=request.message,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to generate response")
    
    return ChatResponse(
        response=result.get("response", ""),
        model=result.get("model", MODEL_NAME),
        created_at=result.get("created_at", ""),
        done=result.get("done", True)
    )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    if not ollama_client.health_check():
        raise HTTPException(status_code=503, detail="Ollama service is not available")
    
    # For streaming, you'd implement Server-Sent Events
    # This is a simplified version
    result = ollama_client.generate(
        prompt=request.message,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to generate response")
    
    return {"response": result.get("response", "")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)