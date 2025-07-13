const CircuitBreaker = require('./circuit_breaker');
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const redis = require('redis');
const contextManager = require('./context_manager');

// Environment variables
const PORT = process.env.PORT || 8000;
const INTENT_SERVICE_URL = process.env.INTENT_SERVICE_URL || 'http://localhost:8001';
const RAG_SERVICE_URL = process.env.RAG_SERVICE_URL || 'http://localhost:8002';
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';

// Initialize circuit breaker
const circuitBreaker = new CircuitBreaker();

// Register services
const intentService = circuitBreaker.registerService('intent', INTENT_SERVICE_URL, {
  failureThreshold: 3,
  resetTimeout: 30000,
  timeout: 5000
});

const ragService = circuitBreaker.registerService('rag', RAG_SERVICE_URL, {
  failureThreshold: 3,
  resetTimeout: 30000,
  timeout: 10000  // RAG might take longer
});

// Set fallbacks
circuitBreaker.setFallback('intent', (data) => {
  console.log('Intent service fallback triggered');
  return {
    data: {
      intent: 'None',
      confidence: 0.0,
      entities: []
    }
  };
});

circuitBreaker.setFallback('rag', (data) => {
  console.log('RAG service fallback triggered');
  return {
    data: {
      response: "I'm sorry, but I'm having trouble accessing my knowledge base right now. Could you try again in a moment?",
      confidence: 0.0,
      sources: []
    }
  };
});

// Initialize Express app
const app = express();

// Configure middleware
app.use(cors());
app.use(bodyParser.json());

// Add request logging middleware
app.use((req, res, next) => {
  console.log(`${req.method} ${req.path} at ${new Date().toISOString()}`);
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`${req.method} ${req.path} ${res.statusCode} - ${duration}ms`);
  });
  
  next();
});

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

// Health check endpoint
app.get('/health', async (req, res) => {
  // Get circuit breaker status
  const circuitStatus = circuitBreaker.getStatus();
  
  // Check Redis status
  let redisStatus = 'disconnected';
  if (redisClient) {
    try {
      await redisClient.ping();
      redisStatus = 'connected';
    } catch (error) {
      redisStatus = 'error';
    }
  }
  
  res.json({
    status: 'healthy',
    services: {
      intent: {
        url: INTENT_SERVICE_URL,
        circuit: circuitStatus.intent
      },
      rag: {
        url: RAG_SERVICE_URL,
        circuit: circuitStatus.rag
      },
      redis: redisStatus
    },
    uptime: process.uptime()
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
    if (redisClient) {
      try {
        const historyJson = await redisClient.get(`chat:${session_id}:history`);
        if (historyJson) {
          history = JSON.parse(historyJson);
        }
      } catch (redisError) {
        console.error('Error retrieving history from Redis:', redisError);
      }
    }
    
    // Step 1: Extract context from history using our enhanced context manager
    const extractedContext = contextManager.extractContext(history, session_id);
    console.log(`Context extracted for session ${session_id}:`, {
      recentTopics: extractedContext.recentTopics,
      entityCount: extractedContext.mentionedEntities.length,
      conversationDepth: extractedContext.conversationDepth
    });
    
    // Step 2: Perform contextual query rewriting if this appears to be a follow-up
    const queryRewriteResult = contextManager.rewriteQueryWithContext(session_id, message);
    const queryToUse = queryRewriteResult.rewrittenQuery;
    
    console.log(`Query processing for session ${session_id}:`, {
      original: queryRewriteResult.originalQuery,
      resolved: queryRewriteResult.resolvedQuery,
      rewritten: queryRewriteResult.rewrittenQuery,
      isFollowUp: queryRewriteResult.context.isFollowUp
    });
    
    // Step 3: Classify intent with circuit breaker using the rewritten query
    let intentResponse;
    try {
      const intentResult = await circuitBreaker.exec('intent', '/classify', {
        method: 'post',
        data: {
          text: queryToUse,
          session_id,
          context: {
            ...context,
            ...extractedContext,
            originalQuery: message,
            rewrittenQuery: queryToUse,
            isFollowUp: queryRewriteResult.context.isFollowUp
          }
        }
      }, { text: queryToUse });  // Fallback args
      
      intentResponse = intentResult.data;
      console.log('Intent classified:', intentResponse.intent, 'for query:', queryToUse);
      
      // Update topic memory
      contextManager.updateTopicMemory(session_id, intentResponse.intent, message);
      
    } catch (error) {
      console.error('Intent service error:', error.message);
      intentResponse = { intent: 'None', confidence: 0.0, entities: [] };
    }
    
    // Step 4: Get response from RAG service with enhanced context
    let ragResponse;
    try {
      // Prepare comprehensive context for RAG service
      const ragContext = {
        ...context,
        ...extractedContext,
        intent: intentResponse.intent,
        entities: intentResponse.entities,
        originalQuery: message,
        rewrittenQuery: queryToUse,
        isFollowUp: queryRewriteResult.context.isFollowUp,
        sessionHistory: extractedContext.sessionHistory,
        recentSessionEntities: extractedContext.recentSessionEntities,
        lastSessionIntent: extractedContext.lastSessionIntent,
        conversationFlow: {
          turnCount: extractedContext.conversationDepth,
          recentTopics: extractedContext.recentTopics.slice(-3),
          mentionedEntities: extractedContext.mentionedEntities.slice(-5)
        }
      };
      
      const ragResult = await circuitBreaker.exec('rag', '/retrieve', {
        method: 'post',
        data: {
          query: queryToUse,
          intent: intentResponse.intent,
          top_k: 3,
          session_id,
          context: ragContext
        }
      }, { query: queryToUse });  // Fallback args
      
      ragResponse = ragResult.data;
      console.log('RAG response received:', {
        confidence: ragResponse.confidence,
        hasRewrittenQuery: !!ragResponse.rewritten_query,
        sourceCount: ragResponse.sources?.length || 0
      });
      
    } catch (error) {
      console.error('RAG service error:', error.message);
      ragResponse = { 
        response: "I'm sorry, I couldn't find an answer to your question. Could you try rephrasing it?",
        confidence: 0.0,
        sources: []
      };
    }
    
    // Step 5: Update entity memory with any detected entities
    if (intentResponse.entities && intentResponse.entities.length > 0) {
      contextManager.updateEntityMemory(session_id, intentResponse.entities);
    }
    
    // Step 6: Enhance response with context using the original user message for personalization
    let enhancedResponse = contextManager.enhanceResponse(
      ragResponse.response, 
      {
        ...extractedContext,
        intent: intentResponse.intent,
        entities: intentResponse.entities,
        originalQuery: message,
        rewrittenQuery: queryToUse,
        isFollowUp: queryRewriteResult.context.isFollowUp
      },
      session_id
    );
    
    // Step 7: Generate contextually aware follow-up suggestions
    const suggestions = contextManager.generateSuggestions(
      intentResponse.intent,
      intentResponse.entities,
      session_id
    );
    
    // Step 8: Create comprehensive conversation history entries
    const userHistoryEntry = {
      role: 'user',
      content: message,
      timestamp: new Date().toISOString(),
      metadata: {
        originalQuery: message,
        rewrittenQuery: queryToUse !== message ? queryToUse : undefined,
        isFollowUp: queryRewriteResult.context.isFollowUp,
        sessionContext: {
          turnCount: extractedContext.conversationDepth,
          recentTopics: extractedContext.recentTopics
        }
      }
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
        sources: ragResponse.sources?.length || 0,
        rewritten_query: ragResponse.rewritten_query || (queryToUse !== message ? queryToUse : undefined),
        processingInfo: {
          queryRewritten: queryToUse !== message,
          contextUsed: queryRewriteResult.context.isFollowUp,
          enhancementApplied: enhancedResponse !== ragResponse.response
        }
      }
    };
    
    history.push(userHistoryEntry, botResponse);
    
    // Keep only the last 20 messages in history
    if (history.length > 20) {
      history = history.slice(history.length - 20);
    }
    
    // Save updated history to Redis
    if (redisClient) {
      try {
        await redisClient.set(`chat:${session_id}:history`, JSON.stringify(history));
        // Set expiration to 24 hours
        await redisClient.expire(`chat:${session_id}:history`, 24 * 60 * 60);
      } catch (redisError) {
        console.error('Error saving history to Redis:', redisError);
      }
    }
    
    // Clean up old sessions periodically (1% chance per request)
    if (Math.random() < 0.01) {
      const cleanupResults = contextManager.cleanupOldSessions();
      console.log('Session cleanup performed:', cleanupResults);
    }
    
    // Return comprehensive response with enhanced information
    res.json({
      response: enhancedResponse,
      session_id,
      intent: intentResponse.intent,
      confidence: ragResponse.confidence,
      entities: intentResponse.entities,
      sources: ragResponse.sources || [],
      suggestions: suggestions,
      rewritten_query: ragResponse.rewritten_query || (queryToUse !== message ? queryToUse : undefined),
      context_info: {
        original_query: message,
        query_rewritten: queryToUse !== message,
        is_follow_up: queryRewriteResult.context.isFollowUp,
        conversation_depth: extractedContext.conversationDepth,
        recent_topics: extractedContext.recentTopics,
        response_enhanced: enhancedResponse !== ragResponse.response
      },
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
      console.error('Error retrieving history from Redis:', redisError);
      res.status(500).json({ error: 'Failed to retrieve history' });
    }
    
  } catch (error) {
    console.error('History retrieval error:', error);
    res.status(500).json({
      error: 'An error occurred retrieving chat history',
      message: error.message
    });
  }
});

// Service status endpoint
app.get('/status', (req, res) => {
  const status = circuitBreaker.getStatus();
  res.json({ services: status });
});

// Reset circuits endpoint
app.post('/reset-circuits', (req, res) => {
  circuitBreaker.resetAll();
  res.json({ status: 'success', message: 'All circuits reset' });
});

// Simple feedback system integration (without external dependencies)
class SimpleFeedbackSystem {
  constructor() {
    this.feedbackStore = new Map(); // In-memory store for development
    this.sessionFeedback = new Map();
    this.analytics = {
      total: 0,
      positive: 0,
      negative: 0,
      byType: {},
      byIntent: {}
    };
  }

  recordFeedback(sessionId, feedbackType, messageId, context = {}) {
    const feedbackId = uuidv4();
    const timestamp = Date.now();
    
    const feedback = {
      id: feedbackId,
      sessionId,
      feedbackType,
      messageId,
      timestamp,
      context,
      weight: this.getFeedbackWeight(feedbackType)
    };

    // Store feedback
    this.feedbackStore.set(feedbackId, feedback);
    
    // Index by session
    if (!this.sessionFeedback.has(sessionId)) {
      this.sessionFeedback.set(sessionId, []);
    }
    this.sessionFeedback.get(sessionId).push(feedbackId);

    // Update analytics
    this.updateAnalytics(feedback);

    console.log(`Recorded feedback: ${feedbackType} for session ${sessionId}`);
    return feedbackId;
  }

  getFeedbackWeight(feedbackType) {
    const weights = {
      'thumbs_up': 1.0,
      'thumbs_down': -2.0,
      'helpful': 1.5,
      'not_helpful': -2.5,
      'follow_up_clicked': 0.5,
      'problem_solved': 2.0,
      'problem_unsolved': -3.0,
      'suggestion_used': 1.0
    };
    return weights[feedbackType] || 0;
  }

  updateAnalytics(feedback) {
    this.analytics.total++;
    
    if (feedback.weight > 0) {
      this.analytics.positive++;
    } else if (feedback.weight < 0) {
      this.analytics.negative++;
    }

    this.analytics.byType[feedback.feedbackType] = 
      (this.analytics.byType[feedback.feedbackType] || 0) + 1;

    if (feedback.context.intent) {
      this.analytics.byIntent[feedback.context.intent] = 
        (this.analytics.byIntent[feedback.context.intent] || 0) + 1;
    }
  }

  getSessionFeedback(sessionId) {
    const feedbackIds = this.sessionFeedback.get(sessionId) || [];
    return feedbackIds.map(id => this.feedbackStore.get(id)).filter(Boolean);
  }

  getAnalytics() {
    const satisfaction = this.analytics.total > 0 ? 
      (this.analytics.positive / this.analytics.total) * 100 : 0;
    
    return {
      ...this.analytics,
      satisfactionRate: satisfaction.toFixed(1) + '%'
    };
  }
}

// Initialize feedback system
const feedbackSystem = new SimpleFeedbackSystem();

// Feedback endpoints
app.post('/feedback', async (req, res) => {
  try {
    const { 
      session_id, 
      feedback_type, 
      message_id, 
      intent, 
      confidence, 
      response_time,
      metadata = {} 
    } = req.body;

    if (!session_id || !feedback_type) {
      return res.status(400).json({ 
        error: 'session_id and feedback_type are required' 
      });
    }

    console.log(`Received feedback: ${feedback_type} for session ${session_id}`);

    // Record feedback with context
    const feedbackId = feedbackSystem.recordFeedback(
      session_id,
      feedback_type,
      message_id,
      {
        intent,
        confidence,
        response_time,
        timestamp: new Date().toISOString(),
        ...metadata
      }
    );

    // Get updated analytics
    const analytics = feedbackSystem.getAnalytics();

    res.json({
      feedback_id: feedbackId,
      status: 'recorded',
      analytics: {
        total_feedback: analytics.total,
        satisfaction_rate: analytics.satisfactionRate
      }
    });

  } catch (error) {
    console.error('Feedback recording error:', error);
    res.status(500).json({
      error: 'Failed to record feedback',
      message: error.message
    });
  }
});

app.get('/feedback/session/:session_id', async (req, res) => {
  try {
    const { session_id } = req.params;
    
    const sessionFeedback = feedbackSystem.getSessionFeedback(session_id);
    
    res.json({
      session_id,
      feedback: sessionFeedback,
      count: sessionFeedback.length
    });

  } catch (error) {
    console.error('Session feedback retrieval error:', error);
    res.status(500).json({
      error: 'Failed to retrieve session feedback',
      message: error.message
    });
  }
});

app.get('/feedback/analytics', async (req, res) => {
  try {
    const analytics = feedbackSystem.getAnalytics();
    
    res.json({
      analytics,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Analytics retrieval error:', error);
    res.status(500).json({
      error: 'Failed to retrieve analytics',
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