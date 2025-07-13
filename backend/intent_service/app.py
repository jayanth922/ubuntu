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

# Import our advanced entity extractor
from entity_extractor import UbuntuEntityExtractor

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

# Define additional request models for new endpoints
class ComplexityRequest(BaseModel):
    text: str

class ComplexityResponse(BaseModel):
    complexity_score: float
    complexity_level: str
    technical_indicators: Dict
    total_entities: int
    entities: List[Dict] = []

# Load pre-trained models (will be replaced with fine-tuned versions)
MODEL_NAME = "distilbert-base-uncased"
tokenizer = None
model = None

# Initialize advanced entity extractor
entity_extractor = None

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
    global tokenizer, model, entity_extractor
    try:
        # Initialize entity extractor
        entity_extractor = UbuntuEntityExtractor(use_spacy=False)  # Start without spacy for deployment simplicity
        logger.info("Advanced entity extractor initialized")
        
        # Load intent classification model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=len(INTENT_CLASSES)
        )
        logger.info(f"Model '{MODEL_NAME}' loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Initialize entity extractor even if model fails
        if entity_extractor is None:
            entity_extractor = UbuntuEntityExtractor(use_spacy=False)
            logger.info("Entity extractor initialized as fallback")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "entity_extractor_loaded": entity_extractor is not None
    }

@app.post("/classify", response_model=IntentResponse)
async def classify_intent(request: IntentRequest):
    if model is None:
        # For MVP, we'll use a simple rule-based approach if model isn't loaded
        intent, confidence = rule_based_intent(request.text)
        entities = extract_entities_advanced(request.text, intent, request.context)
        
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
    
    # Extract entities using advanced extractor
    entities = extract_entities_advanced(request.text, intent, request.context)
    
    logger.info(f"Classified intent: {intent} with confidence: {confidence:.4f}, entities: {len(entities)}")
    
    return IntentResponse(
        intent=intent,
        confidence=confidence,
        entities=entities
    )

@app.post("/analyze-complexity", response_model=ComplexityResponse)
async def analyze_complexity(request: ComplexityRequest):
    """
    Analyze the technical complexity of user input
    This helps determine if advanced features like multi-hop reasoning are needed
    """
    if entity_extractor is None:
        raise HTTPException(status_code=503, detail="Entity extractor not available")
    
    try:
        # Get complexity analysis
        complexity_analysis = entity_extractor.analyze_technical_complexity(request.text)
        
        # Get detailed entities for debugging
        entities = entity_extractor.extract_flat_entities(request.text)
        
        return ComplexityResponse(
            complexity_score=complexity_analysis["complexity_score"],
            complexity_level=complexity_analysis["complexity_level"],
            technical_indicators=complexity_analysis["technical_indicators"],
            total_entities=complexity_analysis["total_entities"],
            entities=entities[:10]  # Limit to first 10 for response size
        )
        
    except Exception as e:
        logger.error(f"Error analyzing complexity: {e}")
        raise HTTPException(status_code=500, detail="Error analyzing text complexity")

@app.post("/extract-entities")
async def extract_entities_endpoint(request: ComplexityRequest):
    """
    Extract entities from text for debugging and analysis
    """
    if entity_extractor is None:
        raise HTTPException(status_code=503, detail="Entity extractor not available")
    
    try:
        # Get detailed entity breakdown
        entities = entity_extractor.extract_entities(request.text)
        flat_entities = entity_extractor.extract_flat_entities(request.text)
        
        return {
            "text": request.text,
            "entities_by_category": entities,
            "flat_entities": flat_entities,
            "total_entities": sum(len(entities[cat]) for cat in entities)
        }
        
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        raise HTTPException(status_code=500, detail="Error extracting entities")

def rule_based_intent(text):
    """Enhanced rule-based intent classification for MVP"""
    text = text.lower()
    
    # More sophisticated patterns
    update_keywords = ["update", "upgrade", "install", "apt", "package", "download"]
    printer_keywords = ["print", "printer", "printing", "cups", "driver"]
    shutdown_keywords = ["shutdown", "turn off", "restart", "reboot", "power"]
    recommendation_keywords = ["recommend", "alternative", "suggest", "best", "which"]
    troubleshoot_keywords = ["error", "problem", "issue", "fix", "trouble", "not working"]
    
    if any(word in text for word in update_keywords):
        return "MakeUpdate", 0.8
    elif any(word in text for word in printer_keywords):
        return "SetupPrinter", 0.8
    elif any(word in text for word in shutdown_keywords):
        return "ShutdownComputer", 0.8
    elif any(word in text for word in recommendation_keywords):
        return "SoftwareRecommendation", 0.8
    elif any(word in text for word in troubleshoot_keywords):
        return "Troubleshooting", 0.7
    else:
        return "None", 0.5

def extract_entities_advanced(text: str, intent: str, context: Optional[Dict] = None):
    """Extract entities using the advanced Ubuntu entity extractor"""
    if entity_extractor is None:
        # Fallback to simple extraction
        return extract_entities_simple(text, intent)
    
    try:
        # Use the advanced extractor
        entities = entity_extractor.extract_for_intent_service(text)
        
        # Add context-based enhancements
        if context and "recentTopics" in context:
            # If we have context about recent topics, we can enhance entity extraction
            for topic in context["recentTopics"]:
                if topic == "MakeUpdate" and intent == "MakeUpdate":
                    # Look for version numbers or update-related entities
                    pass
        
        # Filter and enhance based on intent
        filtered_entities = []
        for entity in entities:
            # Apply intent-specific filtering
            if intent == "MakeUpdate" and entity["type"] in ["package", "software", "version", "error_code"]:
                filtered_entities.append(entity)
            elif intent == "SetupPrinter" and entity["type"] in ["software", "service", "file_path"]:
                filtered_entities.append(entity)
            elif intent == "ShutdownComputer" and entity["type"] in ["command", "service"]:
                filtered_entities.append(entity)
            elif entity["type"] in ["software", "ubuntu_concept", "error_code"]:
                # Always include these high-value entities
                filtered_entities.append(entity)
        
        # Limit to top 5 entities to avoid overwhelming
        return filtered_entities[:5]
        
    except Exception as e:
        logger.error(f"Error in advanced entity extraction: {e}")
        return extract_entities_simple(text, intent)

def extract_entities_simple(text, intent):
    """Simple fallback entity extraction"""
    entities = []
    text = text.lower()
    
    if intent == "MakeUpdate":
        # Extract package names
        common_packages = ["ubuntu", "firefox", "chrome", "python", "apt", "vlc", "nginx", "docker"]
        for pkg in common_packages:
            if pkg in text:
                entities.append({
                    "type": "package",
                    "value": pkg,
                    "confidence": 0.8
                })
    
    elif intent == "SetupPrinter":
        # Extract printer models
        printer_models = ["hp", "canon", "epson", "brother", "samsung"]
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