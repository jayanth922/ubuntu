const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const redis = require('redis');
const circuitBreaker = require('opossum');
const contextManager = require('./context_manager');

// Environment variables
const PORT = process.env.PORT || 8000;
const INTENT_SERVICE_URL = process.env.INTENT_SERVICE_URL || 'http://localhost:8001';
const RAG_SERVICE_URL = process.env.RAG_SERVICE_URL || 'http://localhost:8002';
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';

// Initialize Express app
const app = express();

// Configure middleware
app.use(cors());
app.use(bodyParser.json());

// Initialize Redis client
let redisClient;

async function initRedis() {
  try {
    redisClient = redis.createClient({ url: REDIS_URL });
    
    await redisClient.connect();
    
    console.log('Connected to Redis');
    
    // Test the connection
    await redisClient.set('test', 'connected');
    const testResult = await redisClient.get('test');
    console.log('Redis test:', testResult);
    
  } catch (error) {
    console.error('Redis connection error:', error);
    console.log('Continuing without Redis...');
  }
}

// Configure circuit breaker options
const circuitBreakerOptions = {
  timeout: 5000,                // Time in ms to wait for a response
  errorThresholdPercentage: 50, // Error percentage threshold to trip the circuit
  resetTimeout: 30000           // Time in ms to wait before trying again
};

// Create circuit breakers for service calls
const intentServiceBreaker = new circuitBreaker(
  async (data) => {
    return await axios.post(`${INTENT_SERVICE_URL}/classify`, data);
  }, 
  circuitBreakerOptions
);

const ragServiceBreaker = new circuitBreaker(
  async (data) => {
    return await axios.post(`${RAG_SERVICE_URL}/retrieve`, data);
  },
  circuitBreakerOptions
);

// Add fallback handlers
intentServiceBreaker.fallback(async (data) => {
  console.log('Intent service circuit is open, using fallback');
  return {
    data: { 
      intent: "None", 
      confidence: 0.0, 
      entities: [] 
    }
  };
});

ragServiceBreaker.fallback(async (data) => {
  console.log('RAG service circuit is open, using fallback');
  return {
    data: {
      response: "I'm sorry, but I'm having trouble accessing my knowledge base right now. Please try again in a few moments.",
      confidence: 0.0,
      sources: []
    }
  };
});

// Add circuit breaker event listeners for monitoring
intentServiceBreaker.on('open', () => console.log('Intent service circuit breaker opened'));
intentServiceBreaker.on('halfOpen', () => console.log('Intent service circuit breaker half-open'));
intentServiceBreaker.on('close', () => console.log('Intent service circuit breaker closed'));

ragServiceBreaker.on('open', () => console.log('RAG service circuit breaker opened'));
ragServiceBreaker.on('halfOpen', () => console.log('RAG service circuit breaker half-open'));
ragServiceBreaker.on('close', () => console.log('RAG service circuit breaker closed'));

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    services: {
      intent: `${INTENT_SERVICE_URL}/health`,
      rag: `${RAG_SERVICE_URL}/health`,
      redis: redisClient ? 'connected' : 'not connected'
    },
    circuitBreakers: {
      intent: intentServiceBreaker.opened ? 'open' : intentServiceBreaker.halfOpen ? 'half-open' : 'closed',
      rag: ragServiceBreaker.opened ? 'open' : ragServiceBreaker.halfOpen ? 'half-open' : 'closed'
    }
  });
});

// Chat endpoint
app.post('/chat', async (req, res) => {
  try {
    const { message, session_id = uuidv4(), context = {} } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    console.log(`Received message: "${message}" for session ${session_id}`);
    
    // Get conversation history if available
    let history = [];
    try {
      if (redisClient) {
        const historyJson = await redisClient.get(`chat:${session_id}:history`);
        if (historyJson) {
          history = JSON.parse(historyJson);
        }
      }
    } catch (redisError) {
      console.error('Redis error retrieving history:', redisError);
      // Continue with empty history
    }
    
    // Extract context from history using our enhanced context manager
    const extractedContext = contextManager.extractContext(history, session_id);
    
    // Step 1: Classify intent with circuit breaker
    let intentResponse;
    try {
      const intentResult = await intentServiceBreaker.fire({
        text: message,
        session_id,
        context: {
          ...context,
          ...extractedContext
        }
      });
      
      intentResponse = intentResult.data;
      console.log('Intent classified:', intentResponse.intent);
      
      // Update topic memory
      contextManager.updateTopicMemory(session_id, intentResponse.intent, message);
      
    } catch (error) {
      console.error('Intent service error:', error.message);
      intentResponse = { intent: 'None', confidence: 0.0, entities: [] };
    }
    
    // Step 2: Get response from RAG service with circuit breaker
    let ragResponse;
    try {
      const ragResult = await ragServiceBreaker.fire({
        query: message,
        intent: intentResponse.intent,
        top_k: 3,
        session_id,
        context: {
          ...context,
          ...extractedContext,
          intent: intentResponse.intent,
          entities: intentResponse.entities
        }
      });
      
      ragResponse = ragResult.data;
      console.log('RAG response received with confidence:', ragResponse.confidence);
      
      // If query was rewritten, log it
      if (ragResponse.rewritten_query) {
        console.log(`Query rewritten: "${message}" -> "${ragResponse.rewritten_query}"`);
      }
    } catch (error) {
      console.error('RAG service error:', error.message);
      ragResponse = { 
        response: "I'm sorry, I couldn't find an answer to your question right now. Could you try again in a moment?",
        confidence: 0.0,
        sources: []
      };
    }
    
    // Update entity memory with any detected entities
    if (intentResponse.entities && intentResponse.entities.length > 0) {
      contextManager.updateEntityMemory(session_id, intentResponse.entities);
    }
    
    // Enhance response with context using our advanced context manager
    let enhancedResponse = contextManager.enhanceResponse(
      ragResponse.response, 
      {
        ...extractedContext,
        intent: intentResponse.intent,
        entities: intentResponse.entities
      },
      session_id
    );
    
    // Generate suggested follow-up questions
    const suggestions = contextManager.generateSuggestions(
      intentResponse.intent,
      intentResponse.entities,
      session_id
    );
    
    // Step 3: Update conversation history
    const newHistoryEntry = {
      role: 'user',
      content: message,
      timestamp: new Date().toISOString()
    };
    
    const botResponse = {
      role: 'assistant',
      content: enhancedResponse,
      timestamp: new Date().toISOString(),
      metadata: {
        intent: intentResponse.intent,
        confidence: ragResponse.confidence,
        entities: intentResponse.entities,
        suggestions: suggestions,
        rewritten_query: ragResponse.rewritten_query
      }
    };
    
    history.push(newHistoryEntry, botResponse);
    
    // Keep only the last 20 messages in history
    if (history.length > 20) {
      history = history.slice(history.length - 20);
    }
    
    // Save updated history to Redis
    try {
      if (redisClient) {
        await redisClient.set(`chat:${session_id}:history`, JSON.stringify(history));
        // Set expiration to 24 hours
        await redisClient.expire(`chat:${session_id}:history`, 24 * 60 * 60);
      }
    } catch (redisError) {
      console.error('Redis error saving history:', redisError);
      // Continue without saving history
    }
    
    // Clean up old sessions periodically (1% chance per request)
    if (Math.random() < 0.01) {
      contextManager.cleanupOldSessions();
    }
    
    // Step 4: Return response
    res.json({
      response: enhancedResponse,
      session_id,
      intent: intentResponse.intent,
      confidence: ragResponse.confidence,
      entities: intentResponse.entities,
      sources: ragResponse.sources,
      suggestions: suggestions,
      rewritten_query: ragResponse.rewritten_query,
      history: history.slice(-10) // Return the last 10 messages only
    });
    
  } catch (error) {
    console.error('Chat processing error:', error);
    res.status(500).json({
      error: 'An error occurred processing your request',
      message: error.message
    });
  }
});

// History endpoint
app.get('/history/:session_id', async (req, res) => {
  try {
    const { session_id } = req.params;
    
    if (!redisClient) {
      return res.status(503).json({ error: 'History service unavailable - Redis not connected' });
    }
    
    try {
      const historyJson = await redisClient.get(`chat:${session_id}:history`);
      
      if (!historyJson) {
        return res.json({ history: [] });
      }
      
      const history = JSON.parse(historyJson);
      res.json({ history });
      
    } catch (redisError) {
      console.error('Redis error retrieving history:', redisError);
      return res.status(503).json({ 
        error: 'History service temporarily unavailable',
        message: 'Could not retrieve chat history'
      });
    }
    
  } catch (error) {
    console.error('History retrieval error:', error);
    res.status(500).json({
      error: 'An error occurred retrieving chat history',
      message: error.message
    });
  }
});

// Start the server
app.listen(PORT, async () => {
  console.log(`Dialog Manager service running on port ${PORT}`);
  await initRedis();
});

// Handle process termination
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down...');
  if (redisClient) {
    await redisClient.quit();
  }
  process.exit(0);
});