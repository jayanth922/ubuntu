/**
 * Enhanced Context Manager for Dialog Manager Service
 * Integrates features for multi-turn conversation management
 */
class ContextManager {
  constructor(historyWindow = 10, entityExpiry = 1200000) { // 20 minutes in ms
    // Entity memory store
    this.entityMemory = new Map();
    
    // Topic tracking
    this.topicMemory = new Map();
    
    // Conversation complexity tracking
    this.turnCounts = new Map();
    
    // Current session tracking
    this.activeContexts = new Map();
    
    // Configuration
    this.historyWindow = historyWindow;
    this.entityExpiry = entityExpiry; // milliseconds
    
    // Session-based conversation history
    this.conversationHistory = new Map();
  }
  
  /**
   * Update entity memory for a specific session
   * @param {string} sessionId - Session identifier
   * @param {Array} entities - New entities to add to memory
   */
  updateEntityMemory(sessionId, entities) {
    if (!this.entityMemory.has(sessionId)) {
      this.entityMemory.set(sessionId, new Map());
    }
    
    const sessionMemory = this.entityMemory.get(sessionId);
    
    if (entities && entities.length > 0) {
      entities.forEach(entity => {
        if (entity.type && entity.value) {
          // Store entity with timestamp
          sessionMemory.set(entity.type, {
            value: entity.value,
            timestamp: Date.now(),
            confidence: entity.confidence || 0.8,
            mentions: sessionMemory.has(entity.type) ? 
                      sessionMemory.get(entity.type).mentions + 1 : 1
          });
        }
      });
    }
  }
  
  /**
   * Get entity from memory for a session
   * @param {string} sessionId - Session identifier
   * @param {string} entityType - Type of entity to retrieve
   * @param {number} maxAgeMs - Maximum age of entity to be considered valid
   * @returns {Object|null} - The entity or null if not found
   */
  getEntity(sessionId, entityType, maxAgeMs = 3600000) {
    if (!this.entityMemory.has(sessionId)) {
      return null;
    }
    
    const sessionMemory = this.entityMemory.get(sessionId);
    
    if (!sessionMemory.has(entityType)) {
      return null;
    }
    
    const entity = sessionMemory.get(entityType);
    const now = Date.now();
    
    // Check if entity is too old
    if (now - entity.timestamp > maxAgeMs) {
      return null;
    }
    
    return entity;
  }
  
  /**
   * Update topic tracking for a session
   * @param {string} sessionId - Session identifier
   * @param {string} intent - Current intent
   * @param {string} query - User query
   */
  updateTopicMemory(sessionId, intent, query) {
    if (!this.topicMemory.has(sessionId)) {
      this.topicMemory.set(sessionId, []);
    }
    
    const topics = this.topicMemory.get(sessionId);
    
    // Add new topic
    topics.push({
      intent,
      query,
      timestamp: Date.now()
    });
    
    // Limit to last 10 topics
    if (topics.length > 10) {
      topics.shift();
    }
  }
  
  /**
   * Check if a topic was recently discussed
   * @param {string} sessionId - Session identifier
   * @param {string} intent - Intent to check
   * @param {number} maxAgeMs - Maximum age to consider
   * @returns {boolean} - True if topic was recently discussed
   */
  wasTopicRecent(sessionId, intent, maxAgeMs = 600000) {
    if (!this.topicMemory.has(sessionId)) {
      return false;
    }
    
    const topics = this.topicMemory.get(sessionId);
    const now = Date.now();
    
    return topics.some(topic => 
      topic.intent === intent && (now - topic.timestamp) < maxAgeMs
    );
  }
  
  /**
   * Track conversation turns
   * @param {string} sessionId - Session identifier
   */
  incrementTurnCount(sessionId) {
    if (!this.turnCounts.has(sessionId)) {
      this.turnCounts.set(sessionId, 0);
    }
    
    this.turnCounts.set(sessionId, this.turnCounts.get(sessionId) + 1);
  }
  
  /**
   * Get conversation turn count
   * @param {string} sessionId - Session identifier
   * @returns {number} - Number of turns
   */
  getTurnCount(sessionId) {
    return this.turnCounts.get(sessionId) || 0;
  }
  
  /**
   * Extract relevant context from conversation history
   * @param {Array} history - Array of conversation messages
   * @param {string} sessionId - Session identifier
   * @returns {Object} - Context object with extracted information
   */
  extractContext(history, sessionId) {
    if (!history || history.length === 0) {
      return {
        recentTopics: [],
        mentionedEntities: [],
        userPreferences: {},
        previousProblems: [],
        conversationDepth: this.getTurnCount(sessionId) || 0
      };
    }

    const context = {
      recentTopics: [],
      mentionedEntities: new Set(),
      userPreferences: {},
      previousProblems: [],
      sessionEntities: []
    };
    
    // Process the conversation history
    const recentMessages = history.slice(-5);
    const extractedEntities = [];
    
    for (const message of recentMessages) {
      // Store conversation history in session context
      if (sessionId) {
        this.updateConversationHistory(sessionId, message.role, message.content, message.metadata || {});
      }
      
      // Extract topics
      if (message.metadata && message.metadata.intent) {
        context.recentTopics.push(message.metadata.intent);
        
        // Update session topic
        if (sessionId) {
          this.updateSessionTopic(sessionId, message.metadata.intent);
          this.updateSessionIntent(sessionId, message.metadata.intent);
        }
      }
      
      // Extract entities
      if (message.metadata && message.metadata.entities) {
        for (const entity of message.metadata.entities) {
          context.mentionedEntities.add(entity.value);
          extractedEntities.push(entity.value);
          
          // Update entity memory (legacy)
          if (sessionId) {
            this.updateEntityMemory(sessionId, [entity]);
          }
        }
      }
      
      // Extract problems/issues
      if (message.role === 'user' && 
          (message.content.includes('error') || 
           message.content.includes('issue') || 
           message.content.includes('problem'))) {
        context.previousProblems.push(message.content);
      }
      
      // Extract potential preferences
      if (message.role === 'user') {
        if (message.content.includes('prefer ')) {
          const preferenceMatch = message.content.match(/prefer\s+(\w+)/i);
          if (preferenceMatch && preferenceMatch[1]) {
            context.userPreferences.preference = preferenceMatch[1].toLowerCase();
          }
        }
      }
    }
    
    // Update session entities
    if (sessionId && extractedEntities.length > 0) {
      this.updateSessionEntities(sessionId, extractedEntities);
    }
    
    // Get entities from memory (legacy support)
    if (sessionId && this.entityMemory.has(sessionId)) {
      const sessionMemory = this.entityMemory.get(sessionId);
      sessionMemory.forEach((value, key) => {
        context.sessionEntities.push({
          type: key,
          value: value.value,
          mentions: value.mentions
        });
      });
    }
    
    // Update active context
    if (sessionId) {
      this.activeContexts.set(sessionId, {
        lastUpdated: Date.now(),
        topics: context.recentTopics,
        entities: Array.from(context.mentionedEntities)
      });
      
      // Increment turn count
      this.incrementTurnCount(sessionId);
    }
    
    return {
      ...context,
      mentionedEntities: Array.from(context.mentionedEntities),
      lastMessage: recentMessages.length > 0 ? recentMessages[recentMessages.length - 1] : null,
      conversationDepth: this.getTurnCount(sessionId) || 0,
      // Add new session-based context
      sessionHistory: sessionId ? this.getConversationHistory(sessionId, 5) : [],
      recentSessionEntities: sessionId ? this.getRecentEntities(sessionId) : [],
      lastSessionIntent: sessionId ? this.getLastIntent(sessionId) : null
    };
  }
  
  /**
   * Enhance the response based on context and conversation history
   * @param {String} response - Original response
   * @param {Object} context - Context object
   * @param {String} sessionId - Session identifier
   * @returns {String} - Enhanced response
   */
  enhanceResponse(response, context, sessionId) {
    if (!context) {
      return response;
    }
    
    let enhancedResponse = response;
    
    // Add personalization based on session entities
    if (sessionId && this.entityMemory.has(sessionId)) {
      const sessionMemory = this.entityMemory.get(sessionId);
      
      // Find the most mentioned entity
      let topEntity = null;
      let maxMentions = 0;
      
      sessionMemory.forEach((value, key) => {
        if (value.mentions > maxMentions) {
          maxMentions = value.mentions;
          topEntity = {type: key, value: value.value};
        }
      });
      
      // Reference the entity if appropriate
      if (topEntity && maxMentions > 1 && !enhancedResponse.includes(topEntity.value)) {
        if (topEntity.type === 'package') {
          enhancedResponse = `For ${topEntity.value}, ${enhancedResponse.charAt(0).toLowerCase() + enhancedResponse.slice(1)}`;
        } else if (topEntity.type === 'printer_model') {
          enhancedResponse = `For your ${topEntity.value} printer, ${enhancedResponse.charAt(0).toLowerCase() + enhancedResponse.slice(1)}`;
        }
      }
    }
    
    // Add conversational continuity based on turns
    const turnCount = this.getTurnCount(sessionId) || 0;
    
    if (turnCount > 3) {
      // For deeper conversations, add more connecting phrases
      const continuityPhrases = [
        "As we discussed earlier, ",
        "Following up on this, ",
        "To continue our troubleshooting, ",
        "Building on what we've covered, "
      ];
      
      if (turnCount > 5 && !enhancedResponse.includes("earlier") && !enhancedResponse.includes("mentioned")) {
        const randomPhrase = continuityPhrases[Math.floor(Math.random() * continuityPhrases.length)];
        enhancedResponse = randomPhrase + enhancedResponse.charAt(0).toLowerCase() + enhancedResponse.slice(1);
      }
    }
    
    // Add follow-up suggestions based on context and conversation depth
    if (context.recentTopics && context.conversationDepth < 3) {
      // For early conversation, add more guidance
      if (context.recentTopics.includes('MakeUpdate')) {
        enhancedResponse += "\n\nWould you like to know how to verify the update was successful?";
      } else if (context.recentTopics.includes('SetupPrinter')) {
        enhancedResponse += "\n\nWould you like to know how to print a test page?";
      }
    } else if (context.conversationDepth >= 3) {
      // For deeper conversations, check if we've resolved the issue
      enhancedResponse += "\n\nDid this information help solve your problem?";
    }
    
    return enhancedResponse;
  }
  
  /**
   * Generate suggested follow-up questions based on the current context
   * @param {String} intent - The current intent
   * @param {Array} entities - Detected entities
   * @param {String} sessionId - Session identifier
   * @returns {Array} - List of suggested follow-up questions
   */
  generateSuggestions(intent, entities, sessionId) {
    const suggestions = [];
    const turnCount = this.getTurnCount(sessionId) || 0;
    
    // Get recent topics for this session
    const recentTopics = sessionId && this.activeContexts.has(sessionId) 
      ? this.activeContexts.get(sessionId).topics || []
      : [];
    
    // Base suggestions on current intent
    switch(intent) {
      case 'MakeUpdate':
        if (turnCount < 2) {
          suggestions.push("How do I fix broken packages?");
          suggestions.push("What's the difference between apt update and apt upgrade?");
        } else {
          suggestions.push("How do I fix failed updates?");
          suggestions.push("How can I see what packages were updated?");
        }
        
        if (entities && entities.some(e => e.type === 'package')) {
          const pkg = entities.find(e => e.type === 'package').value;
          suggestions.push(`How do I check the version of ${pkg}?`);
          suggestions.push(`How do I downgrade ${pkg} if there's a problem?`);
        }
        break;
        
      case 'SetupPrinter':
        if (turnCount < 2) {
          suggestions.push("Where can I download printer drivers?");
          suggestions.push("How do I set a default printer?");
        } else {
          suggestions.push("My printer shows offline status");
          suggestions.push("How do I share my printer on the network?");
        }
        
        if (entities && entities.some(e => e.type === 'printer_model')) {
          const model = entities.find(e => e.type === 'printer_model').value;
          suggestions.push(`Does ${model} work with Ubuntu automatically?`);
        }
        break;
        
      default:
        // Check if previous topics can guide suggestions
        if (recentTopics.includes('MakeUpdate')) {
          suggestions.push("How to fix broken packages after update?");
        } else if (recentTopics.includes('SetupPrinter')) {
          suggestions.push("Why is my printer printing blank pages?");
        } else {
          suggestions.push("How do I install Ubuntu?");
          suggestions.push("What are Ubuntu's system requirements?");
        }
    }
    
    // Add follow-up about previous problems
    if (sessionId && this.entityMemory.has(sessionId)) {
      const problemEntities = Array.from(this.entityMemory.get(sessionId).values())
        .filter(e => e.mentions > 1);
      
      if (problemEntities.length > 0) {
        const entity = problemEntities[0];
        suggestions.push(`Is your issue with ${entity.value} resolved now?`);
      }
    }
    
    // For deeper conversations, suggest resolution questions
    if (turnCount > 3) {
      suggestions.push("Did this solve your problem?");
      suggestions.push("Do you have any other Ubuntu questions?");
    }
    
    // Return 2-4 suggestions depending on conversation depth
    const maxSuggestions = Math.min(4, Math.max(2, 2 + Math.floor(turnCount / 2)));
    return suggestions.slice(0, maxSuggestions);
  }
  
  /**
   * Clean up old sessions (enhanced version)
   * @param {number} maxAgeMs - Maximum session age to retain
   */
  cleanupOldSessions(maxAgeMs = 86400000) { // 24 hours default
    const now = Date.now();
    const expiredFromActiveContexts = [];
    const expiredFromConversationHistory = this.clearExpiredSessions(maxAgeMs);
    
    // Clean up active contexts (legacy cleanup)
    this.activeContexts.forEach((context, sessionId) => {
      if (now - context.lastUpdated > maxAgeMs) {
        this.activeContexts.delete(sessionId);
        this.entityMemory.delete(sessionId);
        this.topicMemory.delete(sessionId);
        this.turnCounts.delete(sessionId);
        expiredFromActiveContexts.push(sessionId);
      }
    });
    
    return {
      expiredSessions: expiredFromConversationHistory,
      expiredActiveContexts: expiredFromActiveContexts.length,
      totalCleaned: expiredFromConversationHistory + expiredFromActiveContexts.length
    };
  }

  /**
   * Handle multi-turn reasoning by connecting current query with previous context
   * @param {Array} history - Conversation history
   * @param {string} currentQuery - Current user query
   * @returns {Object} - Object with rewritten query and reference context
   */
  handleMultiTurnReasoning(history, currentQuery) {
    // Identify if this is a follow-up question
    const lastBotMessage = history.filter(msg => msg.role === 'assistant').pop();
    const lastUserMessage = history.filter(msg => msg.role === 'user').pop();
    
    if (this.isFollowUpQuestion(currentQuery, lastUserMessage?.content)) {
      // Connect the current query with previous context
      return {
        rewrittenQuery: `Given that we were discussing ${this.extractTopics(lastBotMessage)}, ${currentQuery}`,
        referenceContext: this.extractKeyInformation(history.slice(-4))
      };
    }
    return { rewrittenQuery: currentQuery };
  }

  /**
   * Check if current query is a follow-up question
   * @param {string} query - Current query
   * @param {string} previousQuery - Previous user query
   * @returns {boolean} - True if this appears to be a follow-up question
   */
  isFollowUpQuestion(query, previousQuery) {
    // Check for pronouns, references, or very short queries
    const followUpIndicators = ['it', 'that', 'this', 'they', 'those', 'the same', 'what about'];
    return followUpIndicators.some(indicator => 
      query.toLowerCase().includes(indicator)) || 
      query.split(' ').length <= 5;
  }

  /**
   * Extract topics from a bot message
   * @param {Object} message - Bot message object
   * @returns {string} - Extracted topics as string
   */
  extractTopics(message) {
    if (!message || !message.content) return 'general Ubuntu topics';
    
    // Extract key topics from the message content
    const content = message.content.toLowerCase();
    const topics = [];
    
    if (content.includes('update') || content.includes('upgrade')) {
      topics.push('Ubuntu updates');
    }
    if (content.includes('printer') || content.includes('printing')) {
      topics.push('printer setup');
    }
    if (content.includes('install') || content.includes('package')) {
      topics.push('software installation');
    }
    if (content.includes('network') || content.includes('wifi')) {
      topics.push('network configuration');
    }
    if (content.includes('driver') || content.includes('hardware')) {
      topics.push('hardware drivers');
    }
    
    return topics.length > 0 ? topics.join(' and ') : 'Ubuntu support topics';
  }

  /**
   * Extract key information from recent conversation history
   * @param {Array} recentHistory - Recent conversation messages
   * @returns {Object} - Key information extracted from history
   */
  extractKeyInformation(recentHistory) {
    if (!recentHistory || recentHistory.length === 0) {
      return { entities: [], topics: [], problems: [] };
    }

    const context = {
      entities: [],
      topics: [],
      problems: [],
      solutions: []
    };

    recentHistory.forEach(message => {
      if (message.metadata) {
        // Extract entities
        if (message.metadata.entities) {
          context.entities.push(...message.metadata.entities.map(e => e.value));
        }
        
        // Extract intent as topic
        if (message.metadata.intent) {
          context.topics.push(message.metadata.intent);
        }
      }

      // Extract problems from user messages
      if (message.role === 'user') {
        const content = message.content.toLowerCase();
        if (content.includes('error') || content.includes('problem') || 
            content.includes('issue') || content.includes('not working')) {
          context.problems.push(message.content);
        }
      }

      // Extract solutions from assistant messages
      if (message.role === 'assistant') {
        const content = message.content.toLowerCase();
        if (content.includes('try') || content.includes('run') || 
            content.includes('install') || content.includes('configure')) {
          context.solutions.push(message.content);
        }
      }
    });

    // Remove duplicates
    context.entities = [...new Set(context.entities)];
    context.topics = [...new Set(context.topics)];

    return context;
  }

  /**
   * Get or create session context
   * @param {string} sessionId - Session identifier
   * @returns {Object} - Session context object
   */
  getSessionContext(sessionId) {
    if (!this.conversationHistory.has(sessionId)) {
      this.conversationHistory.set(sessionId, {
        history: [],
        entities: new Map(),
        topics: [],
        lastIntent: null,
        createdAt: Date.now()
      });
    }
    return this.conversationHistory.get(sessionId);
  }

  /**
   * Update conversation history for a session
   * @param {string} sessionId - Session identifier
   * @param {string} role - Role (user/assistant)
   * @param {string} content - Message content
   * @param {Object} metadata - Additional metadata
   */
  updateConversationHistory(sessionId, role, content, metadata = {}) {
    const context = this.getSessionContext(sessionId);
    
    const entry = {
      role,
      content,
      timestamp: Date.now(),
      ...metadata
    };
    
    context.history.push(entry);
    
    // Maintain history window
    if (context.history.length > this.historyWindow) {
      context.history = context.history.slice(-this.historyWindow);
    }
  }

  /**
   * Update entities for a session with expiry tracking
   * @param {string} sessionId - Session identifier
   * @param {Array} entities - Array of entity strings
   */
  updateSessionEntities(sessionId, entities) {
    const context = this.getSessionContext(sessionId);
    const now = Date.now();
    
    entities.forEach(entity => {
      const key = entity.toLowerCase();
      context.entities.set(key, {
        value: entity,
        lastSeen: now
      });
    });
  }

  /**
   * Get recent entities that haven't expired
   * @param {string} sessionId - Session identifier
   * @returns {Array} - Array of recent entity values
   */
  getRecentEntities(sessionId) {
    const context = this.getSessionContext(sessionId);
    const now = Date.now();
    const recentEntities = [];
    
    context.entities.forEach((entityData, key) => {
      if (now - entityData.lastSeen < this.entityExpiry) {
        recentEntities.push(entityData.value);
      }
    });
    
    return recentEntities;
  }

  /**
   * Update topic for a session
   * @param {string} sessionId - Session identifier
   * @param {string} topic - Topic to add
   */
  updateSessionTopic(sessionId, topic) {
    const context = this.getSessionContext(sessionId);
    context.topics.push(topic);
    
    // Keep only last 3 topics
    if (context.topics.length > 3) {
      context.topics = context.topics.slice(-3);
    }
  }

  /**
   * Get last N topics for a session
   * @param {string} sessionId - Session identifier
   * @param {number} n - Number of topics to retrieve
   * @returns {Array} - Array of recent topics
   */
  getLastTopics(sessionId, n = 1) {
    const context = this.getSessionContext(sessionId);
    return context.topics.slice(-n);
  }

  /**
   * Update last intent for a session
   * @param {string} sessionId - Session identifier
   * @param {string} intent - Intent to store
   */
  updateSessionIntent(sessionId, intent) {
    const context = this.getSessionContext(sessionId);
    context.lastIntent = intent;
  }

  /**
   * Get last intent for a session
   * @param {string} sessionId - Session identifier
   * @returns {string|null} - Last intent or null
   */
  getLastIntent(sessionId) {
    const context = this.getSessionContext(sessionId);
    return context.lastIntent;
  }

  /**
   * Resolve pronouns in user query using recent entities
   * @param {string} sessionId - Session identifier
   * @param {string} query - User query to process
   * @returns {string} - Query with pronouns resolved
   */
  resolvePronouns(sessionId, query) {
    const pronouns = ['it', 'this', 'that', 'they', 'those'];
    const recentEntities = this.getRecentEntities(sessionId);
    
    if (recentEntities.length === 0) {
      return query;
    }
    
    let resolvedQuery = query;
    const mostRecentEntity = recentEntities[recentEntities.length - 1];
    
    pronouns.forEach(pronoun => {
      const regex = new RegExp(`\\b${pronoun}\\b`, 'gi');
      if (regex.test(resolvedQuery)) {
        resolvedQuery = resolvedQuery.replace(regex, mostRecentEntity);
      }
    });
    
    return resolvedQuery;
  }

  /**
   * Get conversation history for a session
   * @param {string} sessionId - Session identifier
   * @param {number} limit - Maximum number of messages to return
   * @returns {Array} - Array of conversation messages
   */
  getConversationHistory(sessionId, limit = null) {
    const context = this.getSessionContext(sessionId);
    const history = context.history;
    
    if (limit && limit < history.length) {
      return history.slice(-limit);
    }
    
    return [...history];
  }

  /**
   * Clear expired sessions based on creation time
   * @param {number} expiry - Session expiry time in milliseconds (default: 2 hours)
   */
  clearExpiredSessions(expiry = 7200000) {
    const now = Date.now();
    const expiredSessions = [];
    
    this.conversationHistory.forEach((context, sessionId) => {
      if (now - context.createdAt > expiry) {
        expiredSessions.push(sessionId);
      }
    });
    
    expiredSessions.forEach(sessionId => {
      this.conversationHistory.delete(sessionId);
      this.entityMemory.delete(sessionId);
      this.topicMemory.delete(sessionId);
      this.turnCounts.delete(sessionId);
      this.activeContexts.delete(sessionId);
    });
    
    return expiredSessions.length;
  }

  /**
   * Enhanced query rewriting with context
   * @param {string} sessionId - Session identifier
   * @param {string} query - Original user query
   * @returns {Object} - Object with rewritten query and context
   */
  rewriteQueryWithContext(sessionId, query) {
    // Resolve pronouns first
    const resolvedQuery = this.resolvePronouns(sessionId, query);
    
    // Get recent context
    const recentTopics = this.getLastTopics(sessionId, 2);
    const recentEntities = this.getRecentEntities(sessionId);
    const lastIntent = this.getLastIntent(sessionId);
    
    // Check if this appears to be a follow-up question
    const isFollowUp = this.isFollowUpQuestion(resolvedQuery);
    
    let rewrittenQuery = resolvedQuery;
    let contextInfo = {
      recentTopics,
      recentEntities,
      lastIntent,
      isFollowUp
    };
    
    // If it's a follow-up and we have context, enhance the query
    if (isFollowUp && (recentTopics.length > 0 || recentEntities.length > 0)) {
      const contextParts = [];
      
      if (recentEntities.length > 0) {
        contextParts.push(`regarding ${recentEntities.slice(-2).join(' and ')}`);
      }
      
      if (recentTopics.length > 0) {
        contextParts.push(`about ${recentTopics.join(' and ')}`);
      }
      
      if (contextParts.length > 0) {
        rewrittenQuery = `Given our discussion ${contextParts.join(' ')}, ${resolvedQuery}`;
      }
    }
    
    return {
      originalQuery: query,
      resolvedQuery,
      rewrittenQuery,
      context: contextInfo
    };
  }

  /**
   * Check if a query appears to be a follow-up question
   * @param {string} query - User query
   * @returns {boolean} - True if appears to be follow-up
   */
  isFollowUpQuestion(query) {
    const followUpIndicators = [
      'it', 'this', 'that', 'they', 'those', 'the same', 'what about',
      'how about', 'and then', 'after that', 'next', 'also'
    ];
    
    const queryLower = query.toLowerCase();
    const hasIndicator = followUpIndicators.some(indicator => 
      queryLower.includes(indicator)
    );
    
    // Also check if query is very short (likely referencing previous context)
    const isShort = query.split(' ').length <= 5;
    
    return hasIndicator || isShort;
  }
}

module.exports = new ContextManager();