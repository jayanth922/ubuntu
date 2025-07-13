import re
import random
from typing import List, Dict, Optional

class QueryRewriter:
    """Improve queries for better retrieval results"""
    
    def __init__(self):
        # Common technical terms in Ubuntu context
        self.tech_terms = {
            "ubuntu": ["ubuntu", "linux", "debian", "operating system", "os"],
            "apt": ["apt", "apt-get", "package manager", "packages", "install software"],
            "terminal": ["terminal", "command line", "bash", "shell", "console"],
            "update": ["update", "upgrade", "patch", "newer version"],
            "printer": ["printer", "printing", "cups", "print job"],
            "network": ["network", "wifi", "ethernet", "connection", "internet"],
            "driver": ["driver", "hardware support", "device driver"],
            "error": ["error", "issue", "problem", "bug", "crash"],
            "permission": ["permission", "access rights", "sudo", "root", "administrator"],
            "file": ["file", "directory", "folder", "path", "filesystem"]
        }
    
    def expand_query(self, query: str) -> str:
        """Expand the query with relevant terms"""
        expanded_terms = []
        
        # Look for key terms to expand
        for term, synonyms in self.tech_terms.items():
            if term in query.lower():
                # Add 1-2 related terms that aren't already in the query
                candidates = [s for s in synonyms if s != term and s not in query.lower()]
                if candidates:
                    # Add 1-2 relevant terms
                    num_to_add = min(2, len(candidates))
                    expanded_terms.extend(random.sample(candidates, num_to_add))
        
        # Only add a reasonable number of expansions
        if len(expanded_terms) > 3:
            expanded_terms = expanded_terms[:3]
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        return query
    
    def add_context(self, query: str, context: Dict) -> str:
        """Enhance query with conversation context"""
        enhanced_query = query
        
        # Add relevant entities from context
        if context and "mentionedEntities" in context and context["mentionedEntities"]:
            entities = context["mentionedEntities"]
            if isinstance(entities, list) and len(entities) > 0:
                # Add up to 2 entities that aren't already in the query
                entities_to_add = []
                for entity in entities[:2]:
                    if entity and entity not in query.lower():
                        entities_to_add.append(entity)
                
                if entities_to_add:
                    enhanced_query = f"{query} {' '.join(entities_to_add)}"
        
        # Add intent information if available
        if context and "intent" in context and context["intent"]:
            intent = context["intent"]
            intent_terms = {
                "MakeUpdate": "update install",
                "SetupPrinter": "printer setup",
                "ShutdownComputer": "shutdown restart",
                "SoftwareRecommendation": "software application recommendation"
            }
            
            if intent in intent_terms and intent_terms[intent] not in enhanced_query.lower():
                enhanced_query += f" {intent_terms[intent]}"
        
        return enhanced_query
    
    def rewrite_query(self, query: str, context: Optional[Dict] = None) -> str:
        """Perform full query rewriting with enhanced context support"""
        if context:
            # Use enhanced contextual rewriting
            return self.rewrite_with_context(query, context)
        
        # Fallback to simple expansion if no context
        return self.expand_query(query)
    
    def rewrite_with_context(self, user_query: str, context: Optional[Dict]) -> str:
        """Enhanced contextual query rewriting using conversation context"""
        enriched_query = user_query
        
        if not context:
            return self.expand_query(user_query)
        
        # Extract entities from context (from dialog manager)
        entities = []
        
        # Check for recent entities from session context
        if "recentSessionEntities" in context:
            entities.extend(context["recentSessionEntities"])
        elif "mentionedEntities" in context:
            entities.extend(context["mentionedEntities"])
        elif "entities" in context:
            # Handle both list of strings and list of dicts
            entity_values = []
            for entity in context["entities"]:
                if isinstance(entity, dict):
                    if "value" in entity:
                        entity_values.append(entity["value"])
                    elif "name" in entity:
                        entity_values.append(entity["name"])
                else:
                    entity_values.append(str(entity))
            entities.extend(entity_values)
        
        # Add last mentioned entities to query (max 2)
        if entities:
            # Filter out entities already in the query
            new_entities = []
            for entity in entities[-2:]:  # Take last 2
                if entity and entity.lower() not in user_query.lower():
                    new_entities.append(entity)
            
            if new_entities:
                enriched_query += " about " + ", ".join(new_entities)
        
        # Add intent context if available
        last_intent = context.get("last_intent") or context.get("lastSessionIntent") or context.get("intent")
        if last_intent and last_intent not in enriched_query:
            # Convert intent to readable terms
            intent_terms = {
                "MakeUpdate": "update install",
                "SetupPrinter": "printer setup configuration",
                "ShutdownComputer": "shutdown restart power",
                "SoftwareRecommendation": "software application recommendation",
                "InstallSoftware": "install software package",
                "FileManagement": "file directory management",
                "NetworkTroubleshooting": "network wifi connection",
                "PermissionIssues": "permission sudo access"
            }
            
            if last_intent in intent_terms:
                enriched_query += f" {intent_terms[last_intent]}"
            else:
                enriched_query += f" intent:{last_intent}"
        
        # Add topic context from recent topics
        recent_topics = context.get("recent_topics", []) or context.get("recentTopics", [])
        if recent_topics:
            topic_terms = {
                "MakeUpdate": "update upgrade",
                "SetupPrinter": "printer setup",
                "NetworkTroubleshooting": "network connection",
                "FileManagement": "file management"
            }
            
            for topic in recent_topics[-1:]:  # Just the most recent topic
                if topic in topic_terms and topic_terms[topic] not in enriched_query.lower():
                    enriched_query += f" {topic_terms[topic]}"
        
        # Finally, expand with synonyms
        enriched_query = self.expand_query(enriched_query)
        
        return enriched_query

class ContextualQueryRewriter:
    """Simplified contextual query rewriter focused on entities and intents"""
    
    def __init__(self):
        pass

    def rewrite(self, user_query: str, context: Optional[Dict]) -> str:
        """Rewrite query using conversation context with entities and intents"""
        enriched_query = user_query
        
        if not context:
            return enriched_query
        
        # Add last mentioned entities
        entities = context.get("entities", [])
        if entities:
            # Take last 2 entities that aren't already in the query
            recent_entities = []
            for entity in entities[-2:]:
                entity_value = entity if isinstance(entity, str) else entity.get("value", str(entity))
                if entity_value and entity_value.lower() not in user_query.lower():
                    recent_entities.append(entity_value)
            
            if recent_entities:
                enriched_query += " about " + ", ".join(recent_entities)
        
        # Add intent context
        last_intent = context.get("last_intent")
        if last_intent and last_intent not in enriched_query:
            enriched_query += f" intent:{last_intent}"
        
        return enriched_query