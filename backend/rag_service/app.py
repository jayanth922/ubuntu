from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import os
import logging
from typing import List, Dict, Optional
import json
import random

# Import our new components
from utils.document_chunking import DocumentChunker
from utils.query_rewriter import QueryRewriter
from search_engine import HybridSearchEngine
from data_processor import UbuntuCorpusProcessor
from cache import ResponseCache

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation for technical support queries",
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
class RAGRequest(BaseModel):
    query: str
    intent: Optional[str] = None
    top_k: int = 3
    session_id: Optional[str] = None
    context: Optional[Dict] = None

class RAGResponse(BaseModel):
    response: str
    sources: List[Dict] = []
    confidence: float = 0.0
    rewritten_query: Optional[str] = None

# Global variables
search_engine = None
query_rewriter = None
document_chunker = None
documents = []
response_cache = None

@app.on_event("startup")
async def initialize_services():
    global search_engine, query_rewriter, document_chunker, documents, response_cache
    try:
        # Initialize cache with Redis URL from environment
        redis_url = os.environ.get('REDIS_URL')
        response_cache = ResponseCache(redis_url=redis_url)
        
        # Initialize components
        search_engine = HybridSearchEngine()
        query_rewriter = QueryRewriter()
        document_chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)
        
        # Process Ubuntu corpus data
        data_processor = UbuntuCorpusProcessor(
            input_dir=os.environ.get('DATA_RAW_DIR', '/data/raw'),
            output_dir=os.environ.get('DATA_PROCESSED_DIR', '/data/processed')
        )
        
        # Try to process real data or fall back to samples
        data_path = data_processor.process_dialogs()
        
        # Load document corpus
        with open(data_path, 'r') as f:
            documents = json.load(f)
        
        logger.info(f"Loaded {len(documents)} documents from {data_path}")
        
        # Process documents into chunks
        chunked_data_path = os.path.join(os.path.dirname(data_path), 'chunked_documents.json')
        chunked_documents = document_chunker.chunk_collection(documents)
        
        # Save chunked documents
        with open(chunked_data_path, 'w') as f:
            json.dump(chunked_documents, f, indent=2)
        logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
        
        # Index documents in search engine
        search_engine.index_documents(chunked_documents)
        logger.info(f"Indexed {len(chunked_documents)} documents in search engine")
        
    except Exception as e:
        logger.error(f"Error initializing RAG service: {e}")
        # Create fallback components
        if not search_engine:
            search_engine = HybridSearchEngine()
        if not query_rewriter:
            query_rewriter = QueryRewriter()
        if not document_chunker:
            document_chunker = DocumentChunker()
        if not documents:
            documents = []

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "search_engine": search_engine is not None,
        "query_rewriter": query_rewriter is not None,
        "document_chunker": document_chunker is not None,
        "documents_count": len(documents),
        "cache": {
            "enabled": response_cache.enabled if response_cache else False,
            "redis_connected": response_cache.redis_client is not None if response_cache else False
        }
    }

@app.post("/retrieve", response_model=RAGResponse)
async def retrieve(request: RAGRequest):
    if search_engine is None or query_rewriter is None:
        # For MVP without initialized components, return a default response
        return fallback_response(request.query, request.intent)
    
    try:
        # Check cache first if available
        if response_cache and response_cache.enabled:
            cached_response = response_cache.get(request.query, request.intent)
            if cached_response:
                logger.info(f"Cache hit for query: {request.query}")
                return RAGResponse(**cached_response)
        
        # Step 1: Rewrite the query for better retrieval
        original_query = request.query
        rewritten_query = query_rewriter.rewrite_query(original_query, request.context)
        
        if rewritten_query != original_query:
            logger.info(f"Rewritten query: '{original_query}' -> '{rewritten_query}'")
        
        # Step 2: Search for relevant documents
        results = search_engine.search(
            rewritten_query, 
            top_k=request.top_k, 
            alpha=0.7  # Favor dense retrieval slightly
        )
        
        # Step 3: Process results and build response
        if not results:
            return fallback_response(request.query, request.intent)
            
        # Get the best result
        best_result = results[0]
        response_text = ""
        
        # Check if the chunk has a response or if we need to use the parent document
        if "response" in best_result:
            response_text = best_result["response"]
        elif "parent_id" in best_result:
            # Find the parent document
            parent_doc = next((doc for doc in documents if doc.get("id") == best_result["parent_id"]), None)
            if parent_doc and "response" in parent_doc:
                response_text = parent_doc["response"]
        
        if not response_text:
            return fallback_response(request.query, request.intent)
        
        # Calculate confidence
        confidence = best_result.get("similarity_score", 0.0)
        
        # Format the sources
        sources = []
        for result in results:
            content = result.get("content", "")
            if len(content) > 150:
                content = content[:150] + "..."
            
            sources.append({
                "id": result.get("id") or result.get("chunk_id", "unknown"),
                "content": content,
                "similarity": result.get("similarity_score", 0.0),
                "source": result.get("source", "Unknown")
            })

        # Create response object
        response = RAGResponse(
            response=response_text,
            sources=sources,
            confidence=confidence,
            rewritten_query=rewritten_query if rewritten_query != original_query else None
        )
        
        # Cache successful responses with good confidence
        if response_cache and confidence > 0.5:
            response_cache.set(request.query, request.intent, response.dict())
        
        return response
    
    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        return fallback_response(request.query, request.intent)

def fallback_response(query, intent=None):
    """Generate a fallback response when retrieval fails"""
    generic_responses = [
        "I'm not sure how to help with that specific Ubuntu issue. Could you provide more details?",
        "I don't have enough information about that topic. Could you rephrase your question?",
        "That's a good question about Ubuntu. Let me check the documentation and get back to you.",
        "I'm still learning about Ubuntu support. Could you ask in a different way?",
        "I don't have the answer to that question yet. Have you tried searching the Ubuntu forums?"
    ]
    
    intent_responses = {
        "MakeUpdate": "It seems you're trying to update or install software. The basic command for updating Ubuntu is 'sudo apt update && sudo apt upgrade'. Could you tell me more about what you're trying to update?",
        "SetupPrinter": "For printer setup issues, first make sure your printer is connected and powered on. Then go to Settings > Printers to add or configure your printer.",
        "ShutdownComputer": "To shut down your Ubuntu computer, you can use the command 'sudo shutdown now' or click on the power icon in the top-right menu and select 'Power Off'.",
        "SoftwareRecommendation": "I can help recommend software for Ubuntu. Could you tell me more about what type of application you're looking for?"
    }
    
    if intent and intent in intent_responses:
        response = intent_responses[intent]
    else:
        response = random.choice(generic_responses)
    
    return RAGResponse(
        response=response,
        sources=[],
        confidence=0.3  # Low confidence for fallback responses
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)