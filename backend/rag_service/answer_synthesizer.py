"""
Answer Synthesizer for RAG Service
Provides template-based response synthesis with citations and follow-up suggestions
"""

import re
from typing import List, Dict, Optional, Any

class AnswerSynthesizer:
    """Synthesize responses from retrieved chunks with citations and follow-ups"""
    
    def __init__(self):
        # Template patterns for different types of responses
        self.response_templates = {
            "how_to": "To {action}, follow these steps:\n\n{content}\n\n(Source: {source})",
            "troubleshooting": "For this issue: {content}\n\n(Source: {source})",
            "definition": "{content}\n\n(Source: {source})",
            "default": "{content}\n\n(Source: {source})"
        }
        
        # Follow-up question templates based on query patterns
        self.followup_patterns = {
            "update": [
                "Do you want to schedule automatic updates?",
                "Would you like to know how to check for package dependencies?",
                "Need help with fixing broken packages after update?"
            ],
            "install": [
                "Do you need help verifying the installation?",
                "Would you like to know how to uninstall if needed?",
                "Need assistance with package dependencies?"
            ],
            "printer": [
                "Is your printer detected in system settings?",
                "Do you need help with printer driver installation?",
                "Would you like to print a test page?"
            ],
            "network": [
                "Do you need help with WiFi password issues?",
                "Would you like to check network adapter settings?",
                "Need assistance with firewall configuration?"
            ],
            "error": [
                "Would you like help interpreting error logs?",
                "Do you need assistance with debugging steps?",
                "Should we check system resource usage?"
            ],
            "permission": [
                "Do you need help with sudo configuration?",
                "Would you like to learn about file ownership?",
                "Need assistance with user group management?"
            ]
        }

    def synthesize_answer(self, user_query: str, retrieved_chunks: List[Dict], context: Optional[Dict] = None) -> str:
        """
        Synthesize a response from retrieved chunks with citations and follow-ups
        
        Args:
            user_query: The original user query
            retrieved_chunks: List of retrieved document chunks
            context: Optional conversation context
            
        Returns:
            Synthesized response with citations and follow-ups
        """
        if not retrieved_chunks:
            return self._generate_fallback_response(user_query, context)
        
        # Get the most relevant chunk
        top_chunk = retrieved_chunks[0]
        
        # Determine response template based on query type
        template_type = self._classify_query_type(user_query)
        
        # Extract content and metadata
        content = self._extract_content(top_chunk)
        source = self._extract_source(top_chunk)
        
        # Format the main response using template
        main_response = self._format_response(template_type, user_query, content, source)
        
        # Add multiple source citations if available
        citations = self._generate_citations(retrieved_chunks[:3])  # Top 3 sources
        if citations:
            main_response += f"\n\nAdditional sources: {citations}"
        
        # Generate follow-up suggestions
        followups = self._generate_followups(user_query, context, retrieved_chunks)
        if followups:
            main_response += f"\n\nSuggested follow-up: {' / '.join(followups)}"
        
        return main_response

    def _classify_query_type(self, query: str) -> str:
        """Classify the query type to select appropriate template"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["how to", "how do i", "how can i", "steps to"]):
            return "how_to"
        elif any(word in query_lower for word in ["error", "problem", "issue", "not working", "failed"]):
            return "troubleshooting"
        elif any(word in query_lower for word in ["what is", "what are", "define", "meaning of"]):
            return "definition"
        else:
            return "default"

    def _extract_content(self, chunk: Dict) -> str:
        """Extract the main content from a chunk"""
        # Priority: response > content > any text field
        content = chunk.get('response') or chunk.get('content') or chunk.get('text', '')
        
        # Clean up content
        if len(content) > 500:
            # Truncate very long content but try to end at a sentence
            truncated = content[:500]
            last_period = truncated.rfind('.')
            if last_period > 300:  # If we have a reasonable sentence ending
                content = truncated[:last_period + 1]
            else:
                content = truncated + "..."
        
        return content.strip()

    def _extract_source(self, chunk: Dict) -> str:
        """Extract source information from a chunk"""
        source = chunk.get('source', 'Ubuntu Documentation')
        
        # Clean up source name
        if source.startswith('http'):
            # Extract domain from URL
            import re
            domain_match = re.search(r'https?://([^/]+)', source)
            if domain_match:
                source = domain_match.group(1)
        
        return source

    def _format_response(self, template_type: str, query: str, content: str, source: str) -> str:
        """Format the response using the appropriate template"""
        template = self.response_templates.get(template_type, self.response_templates["default"])
        
        if template_type == "how_to":
            # Extract action from query for how-to responses
            action = self._extract_action(query)
            return template.format(action=action, content=content, source=source)
        else:
            return template.format(content=content, source=source)

    def _extract_action(self, query: str) -> str:
        """Extract the main action from a how-to query"""
        query_lower = query.lower()
        
        # Remove common prefixes
        for prefix in ["how do i ", "how to ", "how can i ", "steps to "]:
            if query_lower.startswith(prefix):
                return query[len(prefix):].strip()
        
        return query.strip()

    def _generate_citations(self, chunks: List[Dict]) -> str:
        """Generate citation list from multiple chunks"""
        citations = []
        seen_sources = set()
        
        for chunk in chunks[1:]:  # Skip first chunk (already cited)
            source = self._extract_source(chunk)
            if source not in seen_sources:
                citations.append(source)
                seen_sources.add(source)
        
        return ", ".join(citations) if citations else ""

    def _generate_followups(self, query: str, context: Optional[Dict], chunks: List[Dict]) -> List[str]:
        """Generate relevant follow-up questions"""
        followups = []
        query_lower = query.lower()
        
        # Pattern-based follow-ups
        for pattern, suggestions in self.followup_patterns.items():
            if pattern in query_lower:
                # Select 1-2 relevant suggestions
                followups.extend(suggestions[:2])
                break
        
        # Context-based follow-ups
        if context:
            followups.extend(self._generate_context_followups(context, query))
        
        # Content-based follow-ups
        if chunks:
            followups.extend(self._generate_content_followups(chunks[0], query))
        
        # Remove duplicates and limit to 3
        unique_followups = list(dict.fromkeys(followups))[:3]
        return unique_followups

    def _generate_context_followups(self, context: Dict, query: str) -> List[str]:
        """Generate follow-ups based on conversation context"""
        followups = []
        
        # Check recent entities for specific follow-ups
        entities = context.get('recentSessionEntities', []) or context.get('mentionedEntities', [])
        
        for entity in entities:
            entity_lower = entity.lower()
            if 'printer' in entity_lower:
                followups.append("Need help troubleshooting your printer?")
            elif any(soft in entity_lower for soft in ['firefox', 'chrome', 'vlc', 'libreoffice']):
                followups.append(f"Do you need help configuring {entity}?")
            elif any(sys in entity_lower for sys in ['apache', 'mysql', 'nginx']):
                followups.append(f"Would you like help with {entity} configuration?")
        
        # Check conversation depth for different suggestions
        depth = context.get('conversationDepth', 0)
        if depth > 3:
            followups.append("Did this solve your problem?")
        
        return followups[:2]  # Limit context-based follow-ups

    def _generate_content_followups(self, chunk: Dict, query: str) -> List[str]:
        """Generate follow-ups based on retrieved content"""
        followups = []
        content = self._extract_content(chunk).lower()
        
        # Content-specific follow-ups
        if 'command' in content or 'terminal' in content:
            followups.append("Do you need help running terminal commands?")
        
        if 'configuration' in content or 'config' in content:
            followups.append("Would you like help with configuration files?")
        
        if 'package' in content and 'install' in content:
            followups.append("Need help with package dependencies?")
        
        return followups[:1]  # Limit content-based follow-ups

    def _generate_fallback_response(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate a helpful fallback response when no chunks are found"""
        fallback_responses = {
            "update": "I don't have specific information about that update. Try 'sudo apt update && sudo apt upgrade' for general updates, or provide more details about what you're trying to update.",
            "install": "I couldn't find installation instructions for that. Could you specify what software you're trying to install? You can also try 'apt search [software-name]' to find packages.",
            "printer": "For printer setup issues, first ensure your printer is connected and powered on. Then go to Settings > Printers to add or configure your printer. What specific printer model are you using?",
            "network": "For network issues, try checking your connection with 'ping google.com'. Could you describe the specific network problem you're experiencing?",
            "error": "I need more details about the error. Could you share the exact error message or describe what happens when you encounter this issue?",
            "default": "I couldn't find specific information about that. Could you provide more details or rephrase your question? You can also try the Ubuntu documentation or community forums."
        }
        
        query_lower = query.lower()
        
        # Select appropriate fallback based on query content
        for key, response in fallback_responses.items():
            if key in query_lower:
                fallback = response
                break
        else:
            fallback = fallback_responses["default"]
        
        # Add context-aware suggestions if available
        if context and context.get('recentSessionEntities'):
            entities = context['recentSessionEntities'][:2]
            fallback += f"\n\nBased on our conversation about {', '.join(entities)}, you might also want to ask about related configuration or troubleshooting steps."
        
        return fallback


# Convenience function for direct use
def synthesize_answer(user_query: str, retrieved_chunks: List[Dict], context: Optional[Dict] = None) -> str:
    """
    Convenience function to synthesize an answer
    
    Args:
        user_query: The original user query
        retrieved_chunks: List of retrieved document chunks
        context: Optional conversation context
        
    Returns:
        Synthesized response with citations and follow-ups
    """
    synthesizer = AnswerSynthesizer()
    return synthesizer.synthesize_answer(user_query, retrieved_chunks, context)
