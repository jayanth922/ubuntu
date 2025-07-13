"""
Query Processing Module for RAG Service
Handles query decomposition, analysis, and preprocessing
"""

import spacy
from typing import List, Dict, Any, Optional
import re


class QueryDecomposer:
    """
    Decomposes complex queries into simpler subqueries for better retrieval
    """
    
    def __init__(self, nlp_model=None):
        # Use spaCy or similar for syntactic analysis
        try:
            self.nlp = nlp_model or spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if spaCy model is not available
            print("Warning: spaCy model 'en_core_web_sm' not found. Using fallback decomposition.")
            self.nlp = None
    
    def decompose(self, query: str) -> List[str]:
        """
        Decompose a complex query into simpler subqueries
        
        Args:
            query (str): Input query to decompose
            
        Returns:
            List[str]: List of subqueries
        """
        if not self.nlp:
            return self._fallback_decompose(query)
            
        doc = self.nlp(query)
        
        # Identify if query has multiple parts
        subqueries = []
        
        # Check for multiple questions
        for sent in doc.sents:
            if '?' in sent.text:
                subqueries.append(sent.text.strip())
                
        # Check for conjunctions between key phrases
        if not subqueries:
            for token in doc:
                if token.dep_ == "conj" and token.head.pos_ == "VERB":
                    left_part = self._extract_span_around(token.head, doc)
                    right_part = self._extract_span_around(token, doc)
                    if left_part and right_part:
                        subqueries.extend([left_part, right_part])
        
        # Check for compound sentences with "and"
        if not subqueries:
            subqueries = self._split_on_conjunctions(query, doc)
        
        # If no decomposition found, return original query
        return subqueries if subqueries else [query]
    
    def _extract_span_around(self, token, doc) -> str:
        """
        Extract the phrase around a token
        
        Args:
            token: spaCy token
            doc: spaCy document
            
        Returns:
            str: Extracted phrase
        """
        start = token.i
        end = token.i + 1
        
        # Expand left to include modifiers
        while start > 0 and doc[start-1].dep_ in ["aux", "neg", "det", "amod", "prep", "pobj", "advmod"]:
            start -= 1
            
        # Expand right to include objects and complements
        while end < len(doc) and doc[end].dep_ in ["dobj", "pobj", "attr", "acomp", "prep"]:
            end += 1
            
        return doc[start:end].text.strip()
    
    def _split_on_conjunctions(self, query: str, doc) -> List[str]:
        """
        Split query on coordinating conjunctions
        
        Args:
            query (str): Original query
            doc: spaCy document
            
        Returns:
            List[str]: List of split queries
        """
        subqueries = []
        
        # Look for coordinating conjunctions
        conjunction_positions = []
        for token in doc:
            if token.text.lower() in ["and", "or", "but"] and token.pos_ == "CCONJ":
                conjunction_positions.append(token.i)
        
        if conjunction_positions:
            # Split at conjunction points
            start = 0
            for pos in conjunction_positions:
                if start < pos:
                    subquery = doc[start:pos].text.strip()
                    if subquery:
                        subqueries.append(subquery)
                start = pos + 1
            
            # Add remaining part
            if start < len(doc):
                remaining = doc[start:].text.strip()
                if remaining:
                    subqueries.append(remaining)
        
        return subqueries
    
    def _fallback_decompose(self, query: str) -> List[str]:
        """
        Fallback decomposition method when spaCy is not available
        
        Args:
            query (str): Input query
            
        Returns:
            List[str]: List of subqueries
        """
        # Simple rule-based decomposition
        subqueries = []
        
        # Split on question marks
        if '?' in query:
            parts = query.split('?')
            for part in parts:
                part = part.strip()
                if part:
                    # Add question mark back if it's a question
                    if not part.endswith('?'):
                        part += '?'
                    subqueries.append(part)
        
        # Split on coordinating conjunctions
        elif any(conj in query.lower() for conj in [' and ', ' or ', ' but ']):
            # Split on these conjunctions
            for conj in [' and ', ' or ', ' but ']:
                if conj in query.lower():
                    parts = re.split(re.escape(conj), query, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        subqueries = [part.strip() for part in parts if part.strip()]
                        break
        
        return subqueries if subqueries else [query]


class QueryClassifier:
    """
    Classifies queries by type and complexity
    """
    
    def __init__(self):
        self.question_types = {
            'how': ['how', 'how to', 'how can', 'how do'],
            'what': ['what', 'what is', 'what are', 'what does'],
            'why': ['why', 'why does', 'why is'],
            'where': ['where', 'where is', 'where can'],
            'when': ['when', 'when does', 'when is'],
            'which': ['which', 'which one', 'which is'],
            'who': ['who', 'who is', 'who does'],
            'troubleshooting': ['error', 'problem', 'issue', 'broken', 'not working', 'failed']
        }
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify query by type and characteristics
        
        Args:
            query (str): Input query
            
        Returns:
            Dict[str, Any]: Classification results
        """
        query_lower = query.lower().strip()
        
        classification = {
            'type': 'general',
            'complexity': 'simple',
            'question_words': [],
            'is_question': query.strip().endswith('?'),
            'contains_technical_terms': False,
            'requires_step_by_step': False
        }
        
        # Identify question type
        for q_type, patterns in self.question_types.items():
            for pattern in patterns:
                if pattern in query_lower:
                    classification['type'] = q_type
                    classification['question_words'].append(pattern)
                    break
        
        # Check complexity
        word_count = len(query.split())
        if word_count > 15:
            classification['complexity'] = 'complex'
        elif word_count > 8:
            classification['complexity'] = 'medium'
        
        # Check for technical terms
        technical_indicators = [
            'install', 'configure', 'setup', 'driver', 'package', 'repository',
            'terminal', 'command', 'sudo', 'apt', 'ubuntu', 'linux'
        ]
        
        if any(term in query_lower for term in technical_indicators):
            classification['contains_technical_terms'] = True
        
        # Check if requires step-by-step answer
        step_indicators = ['how to', 'how can', 'step by step', 'tutorial', 'guide']
        if any(indicator in query_lower for indicator in step_indicators):
            classification['requires_step_by_step'] = True
        
        return classification


class QueryExpander:
    """
    Expands queries with synonyms and related terms for better retrieval
    """
    
    def __init__(self):
        # Ubuntu-specific term mappings
        self.ubuntu_synonyms = {
            'install': ['setup', 'add', 'get', 'download'],
            'remove': ['uninstall', 'delete', 'purge'],
            'update': ['upgrade', 'refresh'],
            'fix': ['repair', 'solve', 'resolve'],
            'printer': ['printing', 'print'],
            'wifi': ['wireless', 'internet', 'network'],
            'driver': ['module', 'firmware'],
            'software': ['application', 'program', 'app'],
            'terminal': ['command line', 'shell', 'console'],
            'package': ['software package', 'deb'],
            'repository': ['repo', 'source'],
            'error': ['problem', 'issue', 'trouble']
        }
    
    def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand query with synonyms and related terms
        
        Args:
            query (str): Original query
            max_expansions (int): Maximum number of expanded versions
            
        Returns:
            List[str]: List of expanded queries including original
        """
        expanded_queries = [query]  # Always include original
        
        query_lower = query.lower()
        words = query.split()
        
        # Generate expanded versions
        for original_term, synonyms in self.ubuntu_synonyms.items():
            if original_term in query_lower:
                # Create expanded queries by replacing with synonyms
                for synonym in synonyms[:max_expansions-1]:  # Limit expansions
                    expanded_query = query
                    # Replace whole word only
                    pattern = r'\b' + re.escape(original_term) + r'\b'
                    expanded_query = re.sub(pattern, synonym, expanded_query, flags=re.IGNORECASE)
                    
                    if expanded_query != query and expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
                        
                    if len(expanded_queries) >= max_expansions + 1:  # +1 for original
                        break
                        
            if len(expanded_queries) >= max_expansions + 1:
                break
        
        return expanded_queries


class QueryProcessor:
    """
    Main query processing class that combines decomposition, classification, and expansion
    """
    
    def __init__(self, nlp_model=None):
        self.decomposer = QueryDecomposer(nlp_model)
        self.classifier = QueryClassifier()
        self.expander = QueryExpander()
    
    def process(self, query: str, include_expansions: bool = True) -> Dict[str, Any]:
        """
        Process query through decomposition, classification, and expansion
        
        Args:
            query (str): Input query
            include_expansions (bool): Whether to include query expansions
            
        Returns:
            Dict[str, Any]: Processed query information
        """
        # Decompose query
        subqueries = self.decomposer.decompose(query)
        
        # Classify original query
        classification = self.classifier.classify(query)
        
        # Expand queries if requested
        expanded_queries = []
        if include_expansions:
            for subquery in subqueries:
                expanded = self.expander.expand(subquery, max_expansions=2)
                expanded_queries.extend(expanded)
        else:
            expanded_queries = subqueries
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in expanded_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return {
            'original_query': query,
            'subqueries': subqueries,
            'classification': classification,
            'expanded_queries': unique_queries,
            'processing_metadata': {
                'num_subqueries': len(subqueries),
                'num_expanded': len(unique_queries),
                'complexity': classification['complexity'],
                'type': classification['type']
            }
        }
