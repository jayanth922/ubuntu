import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const messagesEndRef = useRef(null);

  // Initialize session
  useEffect(() => {
    const storedSessionId = localStorage.getItem('chatSessionId');
    if (storedSessionId) {
      setSessionId(storedSessionId);
      // Load chat history
      fetchHistory(storedSessionId);
    } else {
      const newSessionId = generateSessionId();
      setSessionId(newSessionId);
      localStorage.setItem('chatSessionId', newSessionId);
    }
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const generateSessionId = () => {
    return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  };

  const fetchHistory = async (sessionId) => {
    try {
      const response = await fetch(`${API_URL}/history/${sessionId}`);
      const data = await response.json();
      
      if (data.history && data.history.length > 0) {
        setMessages(data.history);
        // Set suggestions from the last bot message if available
        const lastBotMessage = [...data.history].reverse().find(msg => msg.role === 'assistant');
        if (lastBotMessage?.metadata?.suggestions) {
          setSuggestions(lastBotMessage.metadata.suggestions);
        }
      } else {
        // Add a welcome message if no history
        addBotMessage("Hi! I'm your Ubuntu support assistant. How can I help you today?", [
          "How do I update Ubuntu?",
          "Setting up a printer in Ubuntu",
          "How to install software from a PPA"
        ]);
      }
    } catch (error) {
      console.error('Error fetching history:', error);
      // Add a welcome message on error
      addBotMessage("Hi! I'm your Ubuntu support assistant. How can I help you today?", [
        "How do I update Ubuntu?",
        "Setting up a printer in Ubuntu",
        "How to install software from a PPA"
      ]);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    await sendMessage(input);
  };
  
  const handleSuggestionClick = async (suggestion) => {
    await sendMessage(suggestion);
  };

  const sendMessage = async (messageText) => {
    // Add user message to chat
    const userMessage = {
      role: 'user',
      content: messageText,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInput('');
    setIsLoading(true);
    setSuggestions([]);
    
    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageText,
          session_id: sessionId
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      
      const data = await response.json();
      
      // Add bot message to chat
      const botMessage = {
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString(),
        metadata: {
          intent: data.intent,
          confidence: data.confidence,
          sources: data.sources,
          suggestions: data.suggestions || []
        }
      };
      
      setMessages(prevMessages => [...prevMessages, botMessage]);
      
      // Update suggestions
      if (data.suggestions && data.suggestions.length > 0) {
        setSuggestions(data.suggestions);
      }
      
    } catch (error) {
      console.error('Error sending message:', error);
      addBotMessage("I'm sorry, I'm having trouble connecting to the server. Please try again later.");
    } finally {
      setIsLoading(false);
    }
  };

  const addBotMessage = (text, suggestionsArray = []) => {
    const botMessage = {
      role: 'assistant',
      content: text,
      timestamp: new Date().toISOString(),
      metadata: {
        suggestions: suggestionsArray
      }
    };
    
    setMessages(prevMessages => [...prevMessages, botMessage]);
    setSuggestions(suggestionsArray);
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const renderMessageContent = (content) => {
    // Simple markdown-like rendering for code blocks
    if (!content.includes('```')) {
      return <p>{content}</p>;
    }
    
    const parts = [];
    const segments = content.split(/```(\w*)\n?/);
    
    for (let i = 0; i < segments.length; i++) {
      if (i % 3 === 0) {
        // Regular text
        if (segments[i]) {
          parts.push(<p key={`text-${i}`}>{segments[i]}</p>);
        }
      } else if (i % 3 === 1) {
        // Code language (ignored for now)
        continue;
      } else {
        // Code content
        parts.push(
          <pre key={`code-${i}`} className="code-block">
            <code>{segments[i]}</code>
          </pre>
        );
      }
    }
    
    return <>{parts}</>;
  };
  
  const handleFeedback = (messageIndex, isPositive) => {
    // Clone messages array to avoid direct state mutation
    const updatedMessages = [...messages];
    
    // Update the message metadata with feedback
    if (updatedMessages[messageIndex] && updatedMessages[messageIndex].metadata) {
      updatedMessages[messageIndex].metadata.feedback = isPositive ? 'positive' : 'negative';
      setMessages(updatedMessages);
      
      // In a real app, you'd send this feedback to your backend
      console.log(`Feedback for message ${messageIndex}: ${isPositive ? 'positive' : 'negative'}`);
      
      // TODO: Send feedback to backend API
    }
  };

  const startNewConversation = () => {
    // Generate new session ID
    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    localStorage.setItem('chatSessionId', newSessionId);
    
    // Clear messages and add welcome message
    setMessages([]);
    addBotMessage("Hi! I'm your Ubuntu support assistant. How can I help you today?", [
      "How do I update Ubuntu?",
      "Setting up a printer in Ubuntu",
      "How to install software from a PPA"
    ]);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Ubuntu Support Assistant</h1>
        <button className="new-chat-btn" onClick={startNewConversation}>
          New Conversation
        </button>
      </header>
      
      <div className="chat-container">
        <div className="messages">
          {messages.length === 0 && !isLoading && (
            <div className="empty-state">
              <p>No messages yet. Ask a question about Ubuntu!</p>
            </div>
          )}
          
          {messages.map((message, index) => (
            <div 
              key={index} 
              className={`message ${message.role === 'user' ? 'user-message' : 'bot-message'}`}
            >
              <div className="message-content">
                {renderMessageContent(message.content)}
                
                {message.metadata?.sources?.length > 0 && (
                  <div className="message-sources">
                    <details>
                      <summary>Sources ({message.metadata.sources.length})</summary>
                      <ul>
                        {message.metadata.sources.map((source, idx) => (
                          <li key={idx}>
                            <small>{source.content}</small>
                          </li>
                        ))}
                      </ul>
                    </details>
                  </div>
                )}
                
                <div className="message-timestamp">
                  {formatTimestamp(message.timestamp)}
                  
                  {message.role === 'assistant' && (
                    <div className="message-feedback">
                      <button 
                        className={`feedback-btn ${message.metadata?.feedback === 'positive' ? 'active' : ''}`}
                        onClick={() => handleFeedback(index, true)}
                        aria-label="Helpful"
                      >
                        👍
                      </button>
                      <button 
                        className={`feedback-btn ${message.metadata?.feedback === 'negative' ? 'active' : ''}`}
                        onClick={() => handleFeedback(index, false)}
                        aria-label="Not helpful"
                      >
                        👎
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message bot-message">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
        
        {suggestions.length > 0 && (
          <div className="suggestions">
            <p>Suggested questions:</p>
            <div className="suggestion-buttons">
              {suggestions.map((suggestion, index) => (
                <button 
                  key={index}
                  className="suggestion-btn"
                  onClick={() => handleSuggestionClick(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
        
        <form className="input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            value={input}
            onChange={handleInputChange}
            placeholder="Ask about Ubuntu..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;