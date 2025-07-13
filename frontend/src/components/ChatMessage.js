import React, { useState, useEffect } from 'react';
import './ChatMessage.css';

function ChatMessage({ message, isTyping }) {
  const [displayText, setDisplayText] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  
  useEffect(() => {
    if (message.role === 'assistant' && message.content && !isTyping) {
      // Implement progressive rendering
      let index = 0;
      const content = message.content;
      setDisplayText('');
      
      const typingInterval = setInterval(() => {
        if (index < content.length) {
          setDisplayText(prev => prev + content.charAt(index));
          index++;
        } else {
          clearInterval(typingInterval);
          setIsComplete(true);
        }
      }, 10); // Speed of typing
      
      return () => clearInterval(typingInterval);
    } else {
      setDisplayText(message.content || '');
      setIsComplete(true);
    }
  }, [message, isTyping]);
  
  return (
    <div className={`message ${message.role}-message`}>
      <div className="message-content">
        {message.role === 'assistant' && !isComplete ? (
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        ) : null}
        <div className="message-text">
          {displayText}
        </div>
      </div>
    </div>
  );
}

export default ChatMessage;
