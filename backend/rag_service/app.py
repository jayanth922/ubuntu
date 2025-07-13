from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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
import asyncio
import re

# Import our new components
from utils.document_chunking import DocumentChunker
from utils.query_rewriter import QueryRewriter, ContextualQueryRewriter
from search_engine import HybridSearchEngine
from data_pipeline import UbuntuCorpusProcessor
from cache import ResponseCache
from answer_synthesizer import AnswerSynthesizer
from query_transformer import UbuntuQueryTransformer, QueryOptimizer
from multi_hop import MultiHopReasoner

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
contextual_rewriter = None
answer_synthesizer = None
document_chunker = None
documents = []
data_processor = None
response_cache = None
query_transformer = None
query_optimizer = None
multi_hop_reasoner = None
service_start_time = time.time()  # Track service startup time

# Add middleware for timing requests
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.on_event("startup")
async def initialize_services():
    global search_engine, query_rewriter, contextual_rewriter, answer_synthesizer, document_chunker, documents, data_processor, response_cache, query_transformer, query_optimizer, multi_hop_reasoner
    
    try:
        # Initialize cache with Redis URL from environment
        redis_url = os.environ.get('REDIS_URL')
        response_cache = ResponseCache(
            redis_url=redis_url,
            ttl=int(os.environ.get('CACHE_TTL', '3600')),
            namespace='rag'
        )
        
        # Initialize data processor with directories from environment
        data_processor = UbuntuCorpusProcessor(
            raw_data_dir=os.environ.get('DATA_RAW_DIR', '/data/raw'),
            processed_data_dir=os.environ.get('DATA_PROCESSED_DIR', '/data/processed'),
            index_data_dir=os.environ.get('DATA_INDEX_DIR', '/data/index'),
            chunk_size=int(os.environ.get('CHUNK_SIZE', '512')),
            chunk_overlap=int(os.environ.get('CHUNK_OVERLAP', '128'))
        )
        
        # Initialize core components
        search_engine = HybridSearchEngine()
        query_rewriter = QueryRewriter()
        contextual_rewriter = ContextualQueryRewriter()
        answer_synthesizer = AnswerSynthesizer()
        document_chunker = DocumentChunker(
            chunk_size=int(os.environ.get('CHUNK_SIZE', '512')),
            chunk_overlap=int(os.environ.get('CHUNK_OVERLAP', '128'))
        )
        
        # Initialize advanced components
        query_transformer = UbuntuQueryTransformer()
        query_optimizer = QueryOptimizer(query_transformer)
        multi_hop_reasoner = MultiHopReasoner(search_engine)
        
        # Check if we need to run the data pipeline
        chunked_file = os.path.join(
            os.environ.get('DATA_PROCESSED_DIR', '/data/processed'),
            'ubuntu_chunked.json'
        )
        
        if not os.path.exists(chunked_file):
            logger.info("Processed data not found, running data pipeline")
            data_processor.run_pipeline()
        
        # Load the chunked documents
        with open(chunked_file, 'r') as f:
            chunked_documents = json.load(f)
            
        logger.info(f"Loaded {len(chunked_documents)} chunked documents")
        
        # Index the documents
        search_engine.index_documents(chunked_documents)
        logger.info(f"Indexed {len(chunked_documents)} documents in search engine")
        
        # Keep a reference to the documents
        documents = chunked_documents
        
    except Exception as e:
        logger.error(f"Error initializing RAG service: {e}", exc_info=True)
        # Create fallback components
        if not search_engine:
            search_engine = HybridSearchEngine()
        if not query_rewriter:
            query_rewriter = QueryRewriter()
        if not document_chunker:
            document_chunker = DocumentChunker()
        if not documents:
            documents = []
        if not response_cache:
            response_cache = ResponseCache(disabled=True)

@app.get("/health", status_code=200)
async def health_check():
    """Enhanced health check for kubernetes readiness/liveness probes"""
    checks = {
        "status": "healthy",
        "checks": {
            "search_engine": check_search_engine(),
            "database": check_database_connection(),
            "embeddings": check_embedding_model(),
            "cache": check_cache_connection()
        },
        "version": os.environ.get("SERVICE_VERSION", "unknown"),
        "uptime_seconds": get_uptime(),
        "components": {
            "search_engine_initialized": search_engine is not None,
            "query_rewriter_initialized": query_rewriter is not None,
            "contextual_rewriter_initialized": contextual_rewriter is not None,
            "answer_synthesizer_initialized": answer_synthesizer is not None,
            "document_chunker_initialized": document_chunker is not None,
            "documents_loaded": len(documents) > 0,
            "documents_count": len(documents)
        }
    }
    
    # If any critical check fails, return unhealthy
    critical_checks = [
        checks["checks"]["search_engine"], 
        checks["checks"]["database"]
    ]
    
    if not all(critical_checks):
        checks["status"] = "unhealthy"
        return JSONResponse(status_code=503, content=checks)
        
    return checks

def check_search_engine():
    """Check if search engine is functional"""
    try:
        if search_engine is None:
            return False
        # Test search with a simple query
        results = search_engine.search("test", top_k=1)
        return True
    except Exception as e:
        logger.error(f"Search engine check failed: {e}")
        return False

def check_database_connection():
    """Check database/document store connection"""
    try:
        # For this implementation, we check if documents are loaded
        return len(documents) > 0
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return False

def check_embedding_model():
    """Check if embedding model is accessible"""
    try:
        if search_engine is None:
            return False
        # Test embedding generation if the search engine has this capability
        if hasattr(search_engine, 'dense_retriever') and search_engine.dense_retriever:
            # Try to encode a simple text
            test_result = search_engine.dense_retriever.encode(["test query"])
            return test_result is not None and len(test_result) > 0
        return True  # If no dense retriever, consider it healthy
    except Exception as e:
        logger.error(f"Embedding model check failed: {e}")
        return False

def check_cache_connection():
    """Check Redis cache connection"""
    try:
        if response_cache is None:
            return True  # Cache is optional
        if not response_cache.enabled:
            return True  # Cache disabled is not an error
        # Test cache with a simple operation
        test_key = "health_check_test"
        response_cache.redis_client.set(test_key, "test", ex=10)
        result = response_cache.redis_client.get(test_key)
        response_cache.redis_client.delete(test_key)
        return result is not None
    except Exception as e:
        logger.error(f"Cache check failed: {e}")
        return False

def get_uptime():
    """Get service uptime in seconds"""
    return int(time.time() - service_start_time)

# Add endpoint to view cache stats
@app.get("/cache/stats")
async def cache_stats():
    if response_cache:
        return response_cache.get_stats()
    return {"status": "unavailable"}

# Add endpoint to clear cache
@app.post("/cache/clear")
async def clear_cache():
    if response_cache:
        count = response_cache.flush()
        return {"status": "success", "cleared_items": count}
    return {"status": "unavailable"}

# Kubernetes readiness probe endpoint
@app.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes - checks if service can handle requests"""
    ready = (
        search_engine is not None and
        query_rewriter is not None and
        len(documents) > 0
    )
    
    if ready:
        return {"status": "ready"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "Service components not fully initialized"}
        )

# Kubernetes liveness probe endpoint  
@app.get("/live")
async def liveness_check():
    """Liveness probe for Kubernetes - checks if service is alive"""
    # Simple check - if we can respond, we're alive
    return {"status": "alive", "uptime_seconds": get_uptime()}

# Detailed system metrics endpoint
@app.get("/metrics")
async def system_metrics():
    """System metrics for monitoring and observability"""
    metrics = {
        "uptime_seconds": get_uptime(),
        "documents_indexed": len(documents),
        "search_engine_status": check_search_engine(),
        "cache_enabled": response_cache.enabled if response_cache else False,
        "memory_usage": get_memory_usage(),
        "environment": {
            "log_level": os.environ.get("LOG_LEVEL", "INFO"),
            "chunk_size": os.environ.get("CHUNK_SIZE", "512"),
            "chunk_overlap": os.environ.get("CHUNK_OVERLAP", "128"),
            "cache_ttl": os.environ.get("CACHE_TTL", "3600")
        }
    }
    
    if response_cache and response_cache.enabled:
        cache_stats = response_cache.get_stats()
        metrics["cache_stats"] = cache_stats
    
    return metrics

def get_memory_usage():
    """Get basic memory usage information"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent()
        }
    except ImportError:
        return {"status": "psutil not available"}
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}")
        return {"status": "error"}

@app.post("/retrieve", response_model=RAGResponse)
async def retrieve(request: RAGRequest):
    if search_engine is None or query_rewriter is None:
        # For MVP without initialized components, return a default response
        return fallback_response(request.query, request.intent)
    
    try:
        # Check cache first
        if response_cache:
            cache_key_extras = {
                "top_k": request.top_k
            }
            cached_response = response_cache.get(
                query=request.query, 
                intent=request.intent,
                **cache_key_extras
            )
            
            if cached_response:
                logger.info(f"Cache hit for query: {request.query}")
                return RAGResponse(**cached_response)
        
        # Step 1: Determine if multi-hop reasoning is needed
        use_multihop = False
        if multi_hop_reasoner:
            use_multihop = multi_hop_reasoner.should_use_multihop(request.query, request.context or {})
        
        if use_multihop:
            logger.info(f"Using multi-hop reasoning for: {request.query}")
            multihop_result = multi_hop_reasoner.reason(request.query, request.context or {})
            
            # Create response object from multi-hop result
            sources = []
            for evidence in multihop_result.get("evidence", []):
                content = evidence.get("content", "")
                if len(content) > 150:
                    content = content[:150] + "..."
                
                sources.append({
                    "id": evidence.get("id") or evidence.get("chunk_id", "unknown"),
                    "content": content,
                    "similarity": evidence.get("similarity_score", 0.0),
                    "source": evidence.get("source", "Multi-hop reasoning")
                })
            
            response = RAGResponse(
                response=multihop_result["answer"],
                sources=sources,
                confidence=multihop_result["confidence"],
                rewritten_query=f"Multi-hop queries: {', '.join(multihop_result.get('queries_used', []))}"
            )
            
            # Cache the response
            if response_cache:
                response_cache.set(
                    query=request.query,
                    intent=request.intent,
                    response=response.dict(),
                    **cache_key_extras
                )
            
            return response
        
        # Step 2: Advanced query transformation
        original_query = request.query
        rewritten_query = original_query
        optimization_result = None
        
        if query_optimizer:
            optimization_result = await query_optimizer.optimize_for_retrieval(
                request.query, request.context
            )
            logger.info(f"Query optimization: {optimization_result['transformations_applied']} transformations")
        
        # Step 3: Enhanced contextual query rewriting (fallback)
        if not optimization_result and request.context and contextual_rewriter:
            rewritten_query = contextual_rewriter.rewrite(original_query, request.context)
            logger.info(f"Contextual rewrite: '{original_query}' -> '{rewritten_query}'")
        elif not optimization_result and request.context:
            rewritten_query = query_rewriter.rewrite_query(original_query, request.context)
            logger.info(f"Enhanced rewrite: '{original_query}' -> '{rewritten_query}'")
        elif not optimization_result:
            rewritten_query = query_rewriter.expand_query(original_query)
            if rewritten_query != original_query:
                logger.info(f"Basic expansion: '{original_query}' -> '{rewritten_query}'")
        
        # Step 4: Search for relevant documents
        all_results = []
        
        if optimization_result:
            # Use optimized queries
            queries_to_search = optimization_result["optimized_queries"]
            search_strategy = optimization_result["search_strategy"]
            
            if search_strategy == "parallel" and len(queries_to_search) > 1:
                # Search all queries in parallel
                search_tasks = []
                for query in queries_to_search:
                    task = asyncio.create_task(
                        asyncio.to_thread(search_engine.search, query, request.top_k, 0.7)
                    )
                    search_tasks.append((query, task))
                
                for query, task in search_tasks:
                    try:
                        results = await task
                        for result in results:
                            result["search_query"] = query
                        all_results.extend(results)
                    except Exception as e:
                        logger.warning(f"Search failed for query '{query}': {e}")
            else:
                # Sequential search
                for query in queries_to_search:
                    try:
                        results = search_engine.search(query, request.top_k, 0.7)
                        for result in results:
                            result["search_query"] = query
                        all_results.extend(results)
                        
                        # Stop early if we have good results
                        if results and results[0].get("similarity_score", 0) > 0.8:
                            break
                    except Exception as e:
                        logger.warning(f"Search failed for query '{query}': {e}")
        else:
            # Fallback to single query search
            all_results = search_engine.search(rewritten_query, request.top_k, 0.7)
        
        # Remove duplicates and sort by score
        seen_ids = set()
        unique_results = []
        for result in all_results:
            result_id = result.get("id") or result.get("chunk_id")
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        # Sort by similarity score and take top_k
        unique_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        results = unique_results[:request.top_k]
        
        # Step 5: Synthesize answer using the answer synthesizer
        if not results:
            if answer_synthesizer:
                synthesized_response = answer_synthesizer.synthesize_answer(
                    request.query, 
                    [], 
                    request.context
                )
                return RAGResponse(
                    response=synthesized_response,
                    sources=[],
                    confidence=0.3,
                    rewritten_query=rewritten_query if rewritten_query != original_query else None
                )
            else:
                return fallback_response(request.query, request.intent)
        
        # Use answer synthesizer to create a well-formatted response
        if answer_synthesizer:
            synthesized_response = answer_synthesizer.synthesize_answer(
                request.query,
                results,
                request.context
            )
            
            # Calculate confidence from best result
            confidence = results[0].get("similarity_score", 0.0)
            
            # Format the sources for API response
            sources = []
            for result in results:
                content = result.get("content", "")
                if len(content) > 150:
                    content = content[:150] + "..."
                
                sources.append({
                    "id": result.get("id") or result.get("chunk_id", "unknown"),
                    "content": content,
                    "similarity": result.get("similarity_score", 0.0),
                    "source": result.get("source", "Unknown"),
                    "search_query": result.get("search_query", rewritten_query)
                })

            # Create response object with synthesized answer
            response = RAGResponse(
                response=synthesized_response,
                sources=sources,
                confidence=confidence,
                rewritten_query=rewritten_query if rewritten_query != original_query else None
            )
        else:
            # Fallback to old method if synthesizer not available
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
        
        # At the end, before returning, store in cache if it's a good response
        if response_cache and confidence > 0.5:
            response_dict = response.dict()
            response_cache.set(
                query=request.query,
                intent=request.intent,
                data=response_dict,
                top_k=request.top_k
            )
        
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

# New endpoint for query transformation analysis
@app.post("/analyze-query")
async def analyze_query(request: RAGRequest):
    """Analyze query transformations without performing retrieval"""
    if not query_transformer:
        raise HTTPException(status_code=503, detail="Query transformer not available")
    
    try:
        transformations = await query_transformer.transform_query(
            request.query, request.context
        )
        
        # Also get optimization result
        optimization_result = None
        if query_optimizer:
            optimization_result = await query_optimizer.optimize_for_retrieval(
                request.query, request.context
            )
        
        return {
            "original_query": request.query,
            "transformations": [
                {
                    "type": t.transformation_type,
                    "transformed_query": t.transformed_query,
                    "confidence": t.confidence,
                    "reasoning": t.reasoning
                }
                for t in transformations
            ],
            "optimization": optimization_result
        }
        
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query analysis failed: {str(e)}")

# New endpoint for multi-hop reasoning analysis
@app.post("/multi-hop-analysis")
async def multi_hop_analysis(request: RAGRequest):
    """Analyze if query needs multi-hop reasoning and preview the approach"""
    if not multi_hop_reasoner:
        raise HTTPException(status_code=503, detail="Multi-hop reasoner not available")
    
    try:
        context = request.context or {}
        should_use_multihop = multi_hop_reasoner.should_use_multihop(request.query, context)
        
        analysis_result = {
            "query": request.query,
            "should_use_multihop": should_use_multihop,
            "reasoning": "Multi-hop reasoning recommended" if should_use_multihop else "Single-hop sufficient",
            "complexity_indicators": []
        }
        
        # Add complexity indicators
        query_lower = request.query.lower()
        for pattern in multi_hop_reasoner.complex_patterns:
            if re.search(pattern, query_lower):
                analysis_result["complexity_indicators"].append(f"Pattern match: {pattern}")
        
        if context.get("previous_confidence", 1.0) < 0.5:
            analysis_result["complexity_indicators"].append("Low previous confidence")
        
        if context.get("conversation_depth", 0) > 2 and "error" in query_lower:
            analysis_result["complexity_indicators"].append("Deep conversation with error mention")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error in multi-hop analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-hop analysis failed: {str(e)}")

# New endpoint for advanced search with full feature showcase
@app.post("/advanced-search")
async def advanced_search(request: RAGRequest):
    """
    Advanced search endpoint that showcases all implemented features:
    - Entity extraction integration
    - Query transformation
    - Multi-hop reasoning
    - Comprehensive feedback
    """
    try:
        # Step 1: Analyze query complexity
        context = request.context or {}
        
        # Check if multi-hop is needed
        use_multihop = False
        multihop_analysis = None
        if multi_hop_reasoner:
            use_multihop = multi_hop_reasoner.should_use_multihop(request.query, context)
            multihop_analysis = {
                "recommended": use_multihop,
                "reasoning": "Complex query detected" if use_multihop else "Standard retrieval sufficient"
            }
        
        # Step 2: Query transformation analysis
        transformation_analysis = None
        if query_transformer:
            transformations = await query_transformer.transform_query(request.query, context)
            transformation_analysis = {
                "transformations_count": len(transformations),
                "best_transformations": [
                    {
                        "type": t.transformation_type,
                        "query": t.transformed_query,
                        "confidence": t.confidence
                    }
                    for t in transformations[:3]
                ]
            }
        
        # Step 3: Perform the actual retrieval using the enhanced endpoint
        retrieval_response = await retrieve(request)
        
        # Step 4: Compile comprehensive response
        advanced_response = {
            "standard_response": retrieval_response.dict(),
            "analysis": {
                "multihop_analysis": multihop_analysis,
                "transformation_analysis": transformation_analysis,
                "processing_metadata": {
                    "features_used": [],
                    "fallback_reasons": []
                }
            }
        }
        
        # Add metadata about which features were used
        if use_multihop:
            advanced_response["analysis"]["processing_metadata"]["features_used"].append("multi_hop_reasoning")
        if transformation_analysis and transformation_analysis["transformations_count"] > 0:
            advanced_response["analysis"]["processing_metadata"]["features_used"].append("query_transformation")
        
        return advanced_response
        
    except Exception as e:
        logger.error(f"Error in advanced search: {e}")
        raise HTTPException(status_code=500, detail=f"Advanced search failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)