"""
Advanced Query Transformation System for Ubuntu Support Chatbot
Implements sophisticated query expansion, reformulation, and optimization
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class QueryTransformation:
    """Represents a query transformation with metadata"""
    original_query: str
    transformed_query: str
    transformation_type: str
    confidence: float
    reasoning: str

class UbuntuQueryTransformer:
    """
    Advanced query transformation system specifically designed for Ubuntu support
    Handles query expansion, reformulation, and context-aware optimization
    """
    
    def __init__(self):
        """Initialize the query transformer with Ubuntu-specific patterns"""
        
        # Ubuntu command synonyms and aliases
        self.command_synonyms = {
            "update": ["upgrade", "refresh", "sync"],
            "install": ["add", "setup", "get"],
            "remove": ["uninstall", "delete", "purge"],
            "search": ["find", "look for", "locate"],
            "configure": ["setup", "set up", "config"],
            "restart": ["reboot", "reload", "refresh"],
            "start": ["run", "launch", "execute"],
            "stop": ["kill", "terminate", "end"]
        }
        
        # Ubuntu-specific expansions
        self.ubuntu_expansions = {
            "apt": ["apt-get", "aptitude", "package manager"],
            "sudo": ["superuser", "administrator", "root access"],
            "systemctl": ["service", "daemon", "systemd"],
            "snap": ["snapd", "snap package", "universal package"],
            "ppa": ["personal package archive", "repository"],
            "kernel": ["linux kernel", "system kernel"],
            "grub": ["bootloader", "boot menu"],
            "unity": ["desktop environment", "ubuntu desktop"],
            "gnome": ["desktop environment", "gnome shell"]
        }
        
        # Error pattern transformations
        self.error_patterns = {
            r"error\s+\d+": "ubuntu error code",
            r"permission denied": "ubuntu permission error",
            r"command not found": "ubuntu command missing",
            r"package.*not.*found": "ubuntu package error",
            r"broken.*package": "ubuntu package dependency error",
            r"repository.*error": "ubuntu repository error"
        }
        
        # Context-aware transformations
        self.context_patterns = {
            "installation": ["setup", "dependency", "configure", "repository"],
            "networking": ["connection", "interface", "firewall", "dns"],
            "hardware": ["driver", "device", "compatibility", "detection"],
            "security": ["permission", "authentication", "certificate", "firewall"],
            "performance": ["slow", "optimization", "memory", "cpu", "disk"],
            "boot": ["grub", "kernel", "startup", "recovery"]
        }
        
        # Version-specific transformations
        self.version_patterns = {
            r"ubuntu\s+(\d+\.\d+)": lambda m: f"ubuntu {m.group(1)} LTS" if m.group(1) in ["18.04", "20.04", "22.04"] else f"ubuntu {m.group(1)}",
            r"(\d+\.\d+)\s+lts": lambda m: f"ubuntu {m.group(1)} long term support"
        }
    
    async def transform_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[QueryTransformation]:
        """
        Transform a query using multiple strategies
        
        Args:
            query: The original query to transform
            context: Optional context information
            
        Returns:
            List of QueryTransformation objects
        """
        transformations = []
        
        try:
            # 1. Command synonym expansion
            synonym_transform = self._expand_command_synonyms(query)
            if synonym_transform != query:
                transformations.append(QueryTransformation(
                    original_query=query,
                    transformed_query=synonym_transform,
                    transformation_type="command_synonym",
                    confidence=0.8,
                    reasoning="Expanded command synonyms for better matching"
                ))
            
            # 2. Ubuntu-specific term expansion
            expanded_transform = self._expand_ubuntu_terms(query)
            if expanded_transform != query:
                transformations.append(QueryTransformation(
                    original_query=query,
                    transformed_query=expanded_transform,
                    transformation_type="ubuntu_expansion",
                    confidence=0.9,
                    reasoning="Added Ubuntu-specific terminology"
                ))
            
            # 3. Error pattern normalization
            error_transform = self._normalize_error_patterns(query)
            if error_transform != query:
                transformations.append(QueryTransformation(
                    original_query=query,
                    transformed_query=error_transform,
                    transformation_type="error_normalization",
                    confidence=0.85,
                    reasoning="Normalized error patterns for better retrieval"
                ))
            
            # 4. Context-aware expansion
            if context:
                context_transform = self._expand_with_context(query, context)
                if context_transform != query:
                    transformations.append(QueryTransformation(
                        original_query=query,
                        transformed_query=context_transform,
                        transformation_type="context_expansion",
                        confidence=0.75,
                        reasoning="Added context-relevant terms"
                    ))
            
            # 5. Version-specific transformation
            version_transform = self._transform_version_references(query)
            if version_transform != query:
                transformations.append(QueryTransformation(
                    original_query=query,
                    transformed_query=version_transform,
                    transformation_type="version_specific",
                    confidence=0.8,
                    reasoning="Enhanced version-specific references"
                ))
            
            # 6. Question reformulation
            reformulated = self._reformulate_question(query)
            if reformulated != query:
                transformations.append(QueryTransformation(
                    original_query=query,
                    transformed_query=reformulated,
                    transformation_type="question_reformulation",
                    confidence=0.7,
                    reasoning="Reformulated as declarative statement"
                ))
            
            # 7. Technical term simplification
            simplified = self._simplify_technical_terms(query)
            if simplified != query:
                transformations.append(QueryTransformation(
                    original_query=query,
                    transformed_query=simplified,
                    transformation_type="simplification",
                    confidence=0.6,
                    reasoning="Simplified technical terminology"
                ))
            
            # Sort by confidence
            transformations.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"Generated {len(transformations)} query transformations")
            return transformations
            
        except Exception as e:
            logger.error(f"Error in query transformation: {e}")
            return []
    
    def _expand_command_synonyms(self, query: str) -> str:
        """Expand command synonyms in the query"""
        expanded = query.lower()
        
        for command, synonyms in self.command_synonyms.items():
            # Check if any synonym is in the query
            for synonym in synonyms:
                if synonym in expanded:
                    # Add the primary command as an alternative
                    expanded = expanded.replace(synonym, f"{synonym} {command}")
                    break
        
        return expanded
    
    def _expand_ubuntu_terms(self, query: str) -> str:
        """Expand Ubuntu-specific terms for better matching"""
        expanded = query.lower()
        
        for term, expansions in self.ubuntu_expansions.items():
            if term in expanded:
                # Add expansions after the original term
                expansion_text = " ".join(expansions[:2])  # Limit to avoid too much expansion
                expanded = expanded.replace(term, f"{term} {expansion_text}")
        
        return expanded
    
    def _normalize_error_patterns(self, query: str) -> str:
        """Normalize error patterns for better retrieval"""
        normalized = query.lower()
        
        for pattern, replacement in self.error_patterns.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
    
    def _expand_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """Expand query with context-relevant terms"""
        expanded = query.lower()
        
        # Get context category
        context_category = self._identify_context_category(query, context)
        
        if context_category and context_category in self.context_patterns:
            context_terms = self.context_patterns[context_category]
            # Add relevant context terms
            context_expansion = " ".join(context_terms[:2])
            expanded = f"{expanded} {context_expansion}"
        
        # Add recent session entities if available
        if context.get("recentSessionEntities"):
            recent_entities = context["recentSessionEntities"][-3:]
            if recent_entities:
                entity_text = " ".join(recent_entities)
                expanded = f"{expanded} {entity_text}"
        
        return expanded
    
    def _identify_context_category(self, query: str, context: Dict[str, Any]) -> Optional[str]:
        """Identify the context category for the query"""
        query_lower = query.lower()
        
        # Check explicit category keywords
        category_keywords = {
            "installation": ["install", "setup", "add", "get"],
            "networking": ["network", "wifi", "ethernet", "internet", "connection"],
            "hardware": ["driver", "device", "hardware", "usb", "graphics"],
            "security": ["security", "permission", "sudo", "auth", "certificate"],
            "performance": ["slow", "fast", "performance", "memory", "cpu"],
            "boot": ["boot", "grub", "startup", "recovery", "kernel"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        # Check context history
        if context.get("recentTopics"):
            recent_topics = " ".join(context["recentTopics"]).lower()
            for category, keywords in category_keywords.items():
                if any(keyword in recent_topics for keyword in keywords):
                    return category
        
        return None
    
    def _transform_version_references(self, query: str) -> str:
        """Transform version references for better matching"""
        transformed = query
        
        for pattern, transform_func in self.version_patterns.items():
            matches = re.finditer(pattern, transformed, re.IGNORECASE)
            for match in matches:
                if callable(transform_func):
                    replacement = transform_func(match)
                    transformed = transformed.replace(match.group(), replacement)
        
        return transformed
    
    def _reformulate_question(self, query: str) -> str:
        """Reformulate questions as declarative statements"""
        query_lower = query.lower().strip()
        
        # Question word mappings
        question_reformulations = {
            r"how do i (.+?)": r"\1 tutorial",
            r"how to (.+?)": r"\1 guide",
            r"what is (.+?)": r"\1 explanation",
            r"why (.+?)": r"\1 reason",
            r"when (.+?)": r"\1 timing",
            r"where (.+?)": r"\1 location",
            r"can i (.+?)": r"\1 possibility",
            r"should i (.+?)": r"\1 recommendation"
        }
        
        reformulated = query_lower
        
        for pattern, replacement in question_reformulations.items():
            reformulated = re.sub(pattern, replacement, reformulated, flags=re.IGNORECASE)
            if reformulated != query_lower:
                break
        
        return reformulated
    
    def _simplify_technical_terms(self, query: str) -> str:
        """Simplify technical terms for broader matching"""
        simplified = query.lower()
        
        # Technical term simplifications
        simplifications = {
            "authentication": "login",
            "configuration": "setup",
            "repository": "repo",
            "dependencies": "requirements",
            "executable": "program",
            "terminal": "command line",
            "directory": "folder",
            "compilation": "build"
        }
        
        for technical, simple in simplifications.items():
            if technical in simplified:
                simplified = simplified.replace(technical, f"{technical} {simple}")
        
        return simplified
    
    async def get_best_transformations(self, query: str, context: Optional[Dict[str, Any]] = None, 
                                     max_transforms: int = 3) -> List[QueryTransformation]:
        """
        Get the best query transformations for optimal retrieval
        
        Args:
            query: Original query
            context: Optional context
            max_transforms: Maximum number of transformations to return
            
        Returns:
            List of best transformations
        """
        all_transformations = await self.transform_query(query, context)
        
        # Filter out low-confidence transformations
        high_confidence = [t for t in all_transformations if t.confidence >= 0.7]
        
        # If we have enough high-confidence transformations, use those
        if len(high_confidence) >= max_transforms:
            return high_confidence[:max_transforms]
        
        # Otherwise, include some medium-confidence ones
        medium_confidence = [t for t in all_transformations if 0.5 <= t.confidence < 0.7]
        
        result = high_confidence + medium_confidence
        return result[:max_transforms]
    
    def explain_transformation(self, transformation: QueryTransformation) -> str:
        """
        Provide a human-readable explanation of a transformation
        """
        return f"Transformation: {transformation.transformation_type}\n" \
               f"Original: {transformation.original_query}\n" \
               f"Transformed: {transformation.transformed_query}\n" \
               f"Confidence: {transformation.confidence:.2f}\n" \
               f"Reasoning: {transformation.reasoning}"

class QueryOptimizer:
    """
    Optimizes queries for the Ubuntu support domain
    """
    
    def __init__(self, transformer: UbuntuQueryTransformer):
        self.transformer = transformer
    
    async def optimize_for_retrieval(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize a query for retrieval performance
        
        Returns:
            Dict with optimized queries and metadata
        """
        # Get best transformations
        transformations = await self.transformer.get_best_transformations(query, context)
        
        # Create optimized query set
        optimized_queries = [query]  # Always include original
        
        for transform in transformations:
            optimized_queries.append(transform.transformed_query)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in optimized_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return {
            "original_query": query,
            "optimized_queries": unique_queries,
            "transformations_applied": len(transformations),
            "transformation_details": transformations,
            "search_strategy": "parallel" if len(unique_queries) > 2 else "sequential"
        }
