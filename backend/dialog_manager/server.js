const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const redis = require('redis');
const promClient = require('prom-client');
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

// Setup Prometheus metrics
const register = new promClient.Registry();
promClient.collectDefaultMetrics({ register });

// Custom metrics
const httpRequestDurationMicroseconds = new promClient.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'code'],
  buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10]
});

const chatRequestCounter = new promClient.Counter({
  name: 'chat_requests_total',
  help: 'Total number of chat requests',
  labelNames: ['intent']
});

const chatResponseTime = new promClient.Histogram({
  name: 'chat_response_time_seconds',
  help: 'Response time for chat requests in seconds',
  buckets: [0.1, 0.5, 1, 2, 5, 10]
});

const userFeedbackCounter = new promClient.Counter({
  name: 'user_feedback_total',
  help: 'User feedback on responses',
  labelNames: ['type']
});

const contextEnhancementCounter = new promClient.Counter({
  name: 'context_enhancements_total',
  help: 'Number of responses enhanced with context',
  labelNames: ['type']
});

register.registerMetric(httpRequestDurationMicroseconds);
register.registerMetric(chatRequestCounter);
register.registerMetric(chatResponseTime);
register.registerMetric(userFeedbackCounter);
register.registerMetric(contextEnhancementCounter);

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

// Middleware to measure request duration
app.use((req, res, next) => {
  const end = httpRequestDurationMicroseconds.startTimer();
  res.on('finish', () => {
    end({ method: req.method, route: req.path, code: res.statusCode });
  });
  next();
});

// Metrics endpoint
app.get('/metrics', async (req, res) => {
  try {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
  } catch (error) {
    res.status(500).end(error);
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    services: {
      intent: `${INTENT_SERVICE_URL}/health`,
      rag: `${RAG_SERVICE_URL}/health`,
      redis: redisClient ? 'connected' : 'not connected'
    }
  });
});

// Chat endpoint
app.post('/chat', async (req, res) => {
  const timer = chatResponseTime.startTimer();
  
  try {
    const { message, session_id = uuidv4(), context = {} } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    console.log(`Received message: "${message}" for session ${session_id}`);
    
    // Get conversation history if available
    let history = [];
    if (redisClient) {
      const historyJson = await redisClient.get(`chat:${session_id}:history`);
      if (historyJson) {
        history = JSON.parse(historyJson);
      }
    }
    
    // Extract context from history using our enhanced context manager
    const extractedContext = contextManager.extractContext(history, session_id);
    
    // Step 1: Classify intent
    let intentResponse;
    try {
      const intentResult = await axios.post(`${INTENT_SERVICE_URL}/classify`, {
        text: message,
        session_id,
        context: {
          ...context,
          ...extractedContext
        }
      });
      intentResponse = intentResult.data;
      console.log('Intent classified:', intentResponse.intent);
      
      // Record intent metric
      chatRequestCounter.inc({ intent: intentResponse.intent });
      
      // Update topic memory
      contextManager.updateTopicMemory(session_id, intentResponse.intent, message);
      
    } catch (error) {
      console.error('Intent service error:', error.message);
      intentResponse = { intent: 'None', confidence: 0.0, entities: [] };
    }
    
    // Step 2: Get response from RAG service
    let ragResponse;
    try {
      const ragResult = await axios.post(`${RAG_SERVICE_URL}/retrieve`, {
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
        response: "I'm sorry, I couldn't find an answer to your question. Could you try rephrasing it?",
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
    
    if (enhancedResponse !== ragResponse.response) {
      contextEnhancementCounter.inc({ type: 'context_added' });
    }
    
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
    if (redisClient) {
      await redisClient.set(`chat:${session_id}:history`, JSON.stringify(history));
      // Set expiration to 24 hours
      await redisClient.expire(`chat:${session_id}:history`, 24 * 60 * 60);
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
  } finally {
    timer.end();
  }
});

// History endpoint
app.get('/history/:session_id', async (req, res) => {
  try {
    const { session_id } = req.params;
    
    if (!redisClient) {
      return res.status(503).json({ error: 'History service unavailable - Redis not connected' });
    }
    
    const historyJson = await redisClient.get(`chat:${session_id}:history`);
    
    if (!historyJson) {
      return res.json({ history: [] });
    }
    
    const history = JSON.parse(historyJson);
    res.json({ history });
    
  } catch (error) {
    console.error('History retrieval error:', error);
    res.status(500).json({
      error: 'An error occurred retrieving chat history',
      message: error.message
    });
  }
});

// Feedback endpoint
app.post('/feedback', async (req, res) => {
  try {
    const { session_id, message_id, feedback_type, message_content } = req.body;
    
    if (!session_id || !feedback_type) {
      return res.status(400).json({ error: 'Session ID and feedback type are required' });
    }
    
    // Record feedback metric
    userFeedbackCounter.inc({ type: feedback_type });
    
    // Store feedback for future analysis
    if (redisClient) {
      const feedbackData = {
        session_id,
        message_id,
        feedback_type,
        message_content,
        timestamp: new Date().toISOString()
      };
      
      await redisClient.rPush('feedback:logs', JSON.stringify(feedbackData));
    }
    
    res.json({ success: true });
    
  } catch (error) {
    console.error('Feedback processing error:', error);
    res.status(500).json({
      error: 'An error occurred processing feedback',
      message: error.message
    });
  }
});

// Suggestion usage endpoint
app.post('/track/suggestion', async (req, res) => {
  try {
    const { session_id, suggestion } = req.body;
    
    if (!session_id || !suggestion) {
      return res.status(400).json({ error: 'Session ID and suggestion are required' });
    }
    
    // Record usage of suggestions
    if (redisClient) {
      const trackData = {
        session_id,
        suggestion,
        timestamp: new Date().toISOString()
      };
      
      await redisClient.rPush('suggestion:usage', JSON.stringify(trackData));
    }
    
    res.json({ success: true });
    
  } catch (error) {
    console.error('Suggestion tracking error:', error);
    res.status(500).json({
      error: 'An error occurred tracking suggestion usage',
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