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
import time
from prometheus_client import make_wsgi_app, Counter, Histogram
from fastapi.middleware.wsgi import WSGIMiddleware

# Import our new components
from utils.document_chunking import DocumentChunker
from utils.query_rewriter import QueryRewriter
from search_engine import HybridSearchEngine

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

# Define Prometheus metrics
REQUEST_COUNT = Counter('rag_requests_total', 'Total RAG Service Requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('rag_request_duration_seconds', 'RAG Service Request Latency', ['method', 'endpoint'])
RETRIEVAL_COUNT = Counter('rag_retrievals_total', 'Number of retrievals performed', ['status'])
CONFIDENCE_HISTOGRAM = Histogram('rag_confidence_score', 'Confidence scores for retrievals')
QUERY_REWRITE_COUNT = Counter('rag_query_rewrites_total', 'Number of query rewrites performed')

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
data_path = "/data/ubuntu_samples.json"
chunked_data_path = "/data/ubuntu_samples_chunked.json"

@app.on_event("startup")
async def initialize_services():
    global search_engine, query_rewriter, document_chunker, documents
    try:
        # Initialize components
        search_engine = HybridSearchEngine()
        query_rewriter = QueryRewriter()
        document_chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)
        
        # For MVP, create some sample data if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            create_sample_data()
        
        # Load document corpus
        with open(data_path, 'r') as f:
            documents = json.load(f)
        
        # Process documents into chunks
        if not os.path.exists(chunked_data_path):
            chunked_documents = document_chunker.chunk_collection(documents)
            # Save chunked documents
            with open(chunked_data_path, 'w') as f:
                json.dump(chunked_documents, f, indent=2)
            logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
        else:
            # Load pre-chunked documents
            with open(chunked_data_path, 'r') as f:
                chunked_documents = json.load(f)
            logger.info(f"Loaded {len(chunked_documents)} pre-chunked documents")
        
        # Index documents in search engine
        search_engine.index_documents(chunked_documents)
        logger.info(f"Indexed {len(chunked_documents)} documents in search engine")
        
    except Exception as e:
        logger.error(f"Error initializing RAG service: {e}")

def create_sample_data():
    """Create sample data from Ubuntu Dialogue Corpus for MVP"""
    sample_data = [
        {
            "id": "1",
            "content": "How do I update my system to the latest Ubuntu version?",
            "response": "To update your Ubuntu system to the latest version, you can use the following commands in terminal:\n\n```\nsudo apt update\nsudo apt upgrade\nsudo do-release-upgrade\n```\n\nThe first command refreshes your package lists, the second updates installed packages, and the third initiates the release upgrade process.",
            "source": "Ubuntu Dialogue Corpus"
        },
        {
            "id": "2",
            "content": "My printer isn't working with Ubuntu 22.04",
            "response": "To troubleshoot printer issues on Ubuntu 22.04:\n\n1. Check if the printer is properly connected and powered on\n2. Open System Settings > Printers to see if your printer is listed\n3. If not, click 'Add' to install a new printer\n4. You may need to install drivers using:\n   ```\n   sudo apt install cups printer-driver-all\n   sudo systemctl restart cups\n   ```\n5. For specific printer models, you might need to download drivers from the manufacturer's website",
            "source": "Ubuntu Dialogue Corpus"
        },
        {
            "id": "3",
            "content": "How do I install software from a PPA?",
            "response": "To install software from a PPA (Personal Package Archive) on Ubuntu:\n\n1. Add the PPA using:\n   ```\n   sudo add-apt-repository ppa:repository-name/ppa\n   ```\n\n2. Update package lists:\n   ```\n   sudo apt update\n   ```\n\n3. Install the software:\n   ```\n   sudo apt install package-name\n   ```\n\nReplace 'repository-name/ppa' and 'package-name' with the specific PPA and package you want to install.",
            "source": "Ubuntu Dialogue Corpus"
        },
        {
            "id": "4",
            "content": "My Ubuntu system is running slow after recent updates",
            "response": "If your Ubuntu system is running slow after updates, try these troubleshooting steps:\n\n1. Check system resources: Open System Monitor (gnome-system-monitor) to see which processes are consuming resources\n\n2. Clear package cache: `sudo apt clean`\n\n3. Remove old kernels: `sudo apt autoremove`\n\n4. Check startup applications: Open 'Startup Applications' and disable unnecessary programs\n\n5. Consider lighter desktop environments if you're on older hardware: `sudo apt install xubuntu-desktop` or `sudo apt install lubuntu-desktop`\n\n6. If the issue persists, try booting with an older kernel from the GRUB menu at startup.",
            "source": "Ubuntu Dialogue Corpus"
        },
        {
            "id": "5",
            "content": "How do I setup dual monitors on Ubuntu?",
            "response": "To set up dual monitors on Ubuntu:\n\n1. Connect your second monitor to your computer\n\n2. Go to Settings > Displays (or type 'Displays' in the Activities search)\n\n3. You should see both monitors represented in the configuration screen\n\n4. Arrange the monitors by dragging them to match your physical setup\n\n5. Choose whether to mirror displays or extend them (typically you want 'extend')\n\n6. Configure resolution, refresh rate, and scaling as needed for each display\n\n7. Click 'Apply' to save your changes\n\nIf your second monitor isn't detected, try:\n- Different connection ports/cables\n- Installing proprietary drivers for your graphics card: System Settings > Additional Drivers",
            "source": "Ubuntu Dialogue Corpus"
        }
    ]
    
    # Add more sample documents for better testing
    sample_data.extend([
        {
            "id": "6",
            "content": "How to install Google Chrome on Ubuntu 22.04?",
            "response": "To install Google Chrome on Ubuntu 22.04:\n\n1. Download the Chrome .deb package from the official website:\n   ```\n   wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb\n   ```\n\n2. Install the package using dpkg:\n   ```\n   sudo dpkg -i google-chrome-stable_current_amd64.deb\n   ```\n\n3. If there are any dependency issues, run:\n   ```\n   sudo apt install -f\n   ```\n\n4. You can now launch Chrome from your applications menu or by running `google-chrome` in the terminal.",
            "source": "Ubuntu Dialogue Corpus"
        },
        {
            "id": "7",
            "content": "How to fix 'Unable to locate package' error in Ubuntu?",
            "response": "When you encounter the 'Unable to locate package' error in Ubuntu, try these solutions:\n\n1. Update your package lists:\n   ```\n   sudo apt update\n   ```\n\n2. Make sure the Universe and Multiverse repositories are enabled:\n   ```\n   sudo add-apt-repository universe\n   sudo add-apt-repository multiverse\n   sudo apt update\n   ```\n\n3. Check if you've typed the package name correctly\n\n4. The package might be available under a different name; use apt search to find it:\n   ```\n   apt search keyword\n   ```\n\n5. If you're looking for a specific software that's not in the repositories, you may need to add a PPA or download it from the developer's website.",
            "source": "Ubuntu Dialogue Corpus"
        },
        {
            "id": "8",
            "content": "How to check Ubuntu version in terminal?",
            "response": "To check your Ubuntu version in the terminal, you have several options:\n\n1. Use the lsb_release command:\n   ```\n   lsb_release -a\n   ```\n\n2. Check the /etc/os-release file:\n   ```\n   cat /etc/os-release\n   ```\n\n3. Use the hostnamectl command:\n   ```\n   hostnamectl\n   ```\n\n4. Check the issue file:\n   ```\n   cat /etc/issue\n   ```\n\nThe lsb_release command is the most commonly used and provides detailed information about your Ubuntu version.",
            "source": "Ubuntu Dialogue Corpus"
        },
        {
            "id": "9",
            "content": "How to install and configure a firewall on Ubuntu?",
            "response": "Ubuntu comes with a built-in firewall called UFW (Uncomplicated Firewall). Here's how to set it up:\n\n1. Install UFW if it's not already installed:\n   ```\n   sudo apt install ufw\n   ```\n\n2. Check the status:\n   ```\n   sudo ufw status\n   ```\n\n3. Enable common services (optional):\n   ```\n   sudo ufw allow ssh     # Allow SSH (port 22)\n   sudo ufw allow http    # Allow HTTP (port 80)\n   sudo ufw allow https   # Allow HTTPS (port 443)\n   ```\n\n4. Enable the firewall:\n   ```\n   sudo ufw enable\n   ```\n\n5. To deny a specific port:\n   ```\n   sudo ufw deny 3306     # Deny MySQL connections\n   ```\n\n6. To allow a specific IP address:\n   ```\n   sudo ufw allow from 192.168.1.100\n   ```\n\nAfter making changes, check the status again with `sudo ufw status verbose`.",
            "source": "Ubuntu Dialogue Corpus"
        },
        {
            "id": "10",
            "content": "How to fix 'broken packages' error in Ubuntu?",
            "response": "To fix 'broken packages' errors in Ubuntu, try these steps in order:\n\n1. Update package lists and try to fix broken dependencies:\n   ```\n   sudo apt update\n   sudo apt --fix-broken install\n   ```\n\n2. Force package reconfiguration:\n   ```\n   sudo dpkg --configure -a\n   ```\n\n3. Clear the local repository of retrieved package files:\n   ```\n   sudo apt clean\n   sudo apt update\n   ```\n\n4. Try to fix missing dependencies:\n   ```\n   sudo apt-get -f install\n   ```\n\n5. Remove problematic packages and reinstall them if needed\n\n6. As a last resort, you can try to manually download and install the package with dpkg:\n   ```\n   sudo dpkg -i /path/to/package.deb\n   sudo apt-get -f install\n   ```\n\nThese steps resolve most broken package issues in Ubuntu.",
            "source": "Ubuntu Dialogue Corpus"
        }
    ])
    
    with open(data_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Created sample data at {data_path}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "search_engine": search_engine is not None,
        "query_rewriter": query_rewriter is not None,
        "document_chunker": document_chunker is not None,
        "documents_count": len(documents)
    }

@app.post("/retrieve", response_model=RAGResponse)
async def retrieve(request: RAGRequest):
    if search_engine is None or query_rewriter is None:
        # For MVP without initialized components, return a default response
        RETRIEVAL_COUNT.labels(status="fallback").inc()
        return fallback_response(request.query, request.intent)
    
    try:
        # Step 1: Rewrite the query for better retrieval
        original_query = request.query
        rewritten_query = query_rewriter.rewrite_query(original_query, request.context)
        
        if rewritten_query != original_query:
            QUERY_REWRITE_COUNT.inc()
            logger.info(f"Rewritten query: '{original_query}' -> '{rewritten_query}'")
        
        # Step 2: Search for relevant documents
        results = search_engine.search(
            rewritten_query, 
            top_k=request.top_k, 
            alpha=0.7  # Favor dense retrieval slightly
        )
        
        # Step 3: Process results and build response
        if not results:
            RETRIEVAL_COUNT.labels(status="no_results").inc()
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
            RETRIEVAL_COUNT.labels(status="no_response").inc()
            return fallback_response(request.query, request.intent)
        
        # Calculate confidence
        confidence = best_result.get("similarity_score", 0.0)
        CONFIDENCE_HISTOGRAM.observe(confidence)
        
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
        
        RETRIEVAL_COUNT.labels(status="success").inc()
        return RAGResponse(
            response=response_text,
            sources=sources,
            confidence=confidence,
            rewritten_query=rewritten_query if rewritten_query != original_query else None
        )
    
    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        RETRIEVAL_COUNT.labels(status="error").inc()
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