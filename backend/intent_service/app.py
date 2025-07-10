from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Optional
import os
import logging
from prometheus_client import make_wsgi_app, Counter, Histogram
from fastapi.middleware.wsgi import WSGIMiddleware
import time

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intent Classification Service",
    description="Service for classifying user intents in technical support queries",
    version="0.1.0"
)

# Define Prometheus metrics
REQUEST_COUNT = Counter('intent_requests_total', 'Total Intent Classification Requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('intent_request_duration_seconds', 'Intent Classification Request Latency', ['method', 'endpoint'])
INTENT_COUNTER = Counter('intent_classifications_total', 'Number of intent classifications', ['intent'])
CONFIDENCE_HISTOGRAM = Histogram('intent_confidence_score', 'Confidence scores for intent classifications')

# Add middleware to record metrics
@app.middleware("http")
async def monitor_requests(request, call_next):
    method = request.method
    path = request.url.path
    
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)
    REQUEST_COUNT.labels(method=method, endpoint=path, status=response.status_code).inc()
    
    return response

# Mount metrics endpoint
app.mount("/metrics", WSGIMiddleware(make_wsgi_app()))

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request/response models
class IntentRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    context: Optional[Dict] = None

class IntentResponse(BaseModel):
    intent: str
    confidence: float
    entities: List[Dict] = []

# Load pre-trained models (will be replaced with fine-tuned versions)
MODEL_NAME = "distilbert-base-uncased"
tokenizer = None
model = None

# Intent classes from AskUbuntu dataset
INTENT_CLASSES = [
    "MakeUpdate",
    "SetupPrinter",
    "ShutdownComputer",
    "SoftwareRecommendation",
    "None"
]

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=len(INTENT_CLASSES)
        )
        logger.info(f"Model '{MODEL_NAME}' loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # For now, we'll continue without the model for the initial setup

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/classify", response_model=IntentResponse)
async def classify_intent(request: IntentRequest):
    if model is None:
        # For MVP, we'll use a simple rule-based approach if model isn't loaded
        intent, confidence = rule_based_intent(request.text)
        entities = extract_entities(request.text, intent)
        
        # Record metrics
        INTENT_COUNTER.labels(intent=intent).inc()
        CONFIDENCE_HISTOGRAM.observe(confidence)
        
        return IntentResponse(
            intent=intent,
            confidence=confidence,
            entities=entities
        )
    
    # Tokenize input text
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
    
    # Get the highest probability intent
    max_idx = np.argmax(probs)
    intent = INTENT_CLASSES[max_idx]
    confidence = probs[max_idx]
    
    # Extract entities
    entities = extract_entities(request.text, intent)
    
    # Record metrics
    INTENT_COUNTER.labels(intent=intent).inc()
    CONFIDENCE_HISTOGRAM.observe(confidence)
    
    logger.info(f"Classified intent: {intent} with confidence: {confidence:.4f}")
    
    return IntentResponse(
        intent=intent,
        confidence=confidence,
        entities=entities
    )

def rule_based_intent(text):
    """Simple rule-based intent classification for MVP"""
    text = text.lower()
    
    if "update" in text or "upgrade" in text or "install" in text:
        return "MakeUpdate", 0.7
    elif "print" in text or "printer" in text:
        return "SetupPrinter", 0.7
    elif "shutdown" in text or "turn off" in text or "restart" in text:
        return "ShutdownComputer", 0.7
    elif "recommend" in text or "alternative" in text or "suggest" in text:
        return "SoftwareRecommendation", 0.7
    else:
        return "None", 0.5

def extract_entities(text, intent):
    """Extract entities based on the intent"""
    entities = []
    text = text.lower()
    
    if intent == "MakeUpdate":
        # Extract package names
        for pkg in ["ubuntu", "firefox", "chrome", "python", "apt"]:
            if pkg in text:
                entities.append({
                    "type": "package",
                    "value": pkg,
                    "confidence": 0.8
                })
    
    elif intent == "SetupPrinter":
        # Extract printer models
        printer_models = ["hp", "canon", "epson", "brother"]
        for model in printer_models:
            if model in text:
                entities.append({
                    "type": "printer_model",
                    "value": model,
                    "confidence": 0.8
                })
    
    return entities

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)