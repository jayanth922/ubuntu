/**
 * Enhanced Context Manager for Dialog Manager Service
 */
class ContextManager {
  constructor() {
    // Entity memory store
    this.entityMemory = new Map();
    
    // Topic tracking
    this.topicMemory = new Map();
    
    // Conversation complexity tracking
    this.turnCounts = new Map();
    
    // Current session tracking
    this.activeContexts = new Map();
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
    
    for (const message of recentMessages) {
      // Extract topics
      if (message.metadata && message.metadata.intent) {
        context.recentTopics.push(message.metadata.intent);
      }
      
      // Extract entities
      if (message.metadata && message.metadata.entities) {
        for (const entity of message.metadata.entities) {
          context.mentionedEntities.add(entity.value);
          
          // Update entity memory
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
    
    // Get entities from memory
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
      conversationDepth: this.getTurnCount(sessionId) || 0
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
   * Clean up old sessions
   * @param {number} maxAgeMs - Maximum session age to retain
   */
  cleanupOldSessions(maxAgeMs = 86400000) { // 24 hours default
    const now = Date.now();
    
    // Clean up active contexts
    this.activeContexts.forEach((context, sessionId) => {
      if (now - context.lastUpdated > maxAgeMs) {
        this.activeContexts.delete(sessionId);
        this.entityMemory.delete(sessionId);
        this.topicMemory.delete(sessionId);
        this.turnCounts.delete(sessionId);
      }
    });
  }
}

module.exports = new ContextManager();