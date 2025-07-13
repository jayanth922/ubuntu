"""
Advanced Multi-Hop Reasoning Module for RAG Service
Handles complex Ubuntu troubleshooting questions requiring multiple information sources
"""

import re
import logging
from typing import List, Dict, Any, Optional
import asyncio

logger = logging.getLogger(__name__)


class MultiHopReasoner:
    """
    Advanced multi-hop reasoning for complex Ubuntu troubleshooting questions
    Handles questions that require connecting multiple pieces of information
    """
    
    def __init__(self, search_engine, max_hops=3, confidence_threshold=0.6):
        """
        Initialize the multi-hop reasoner
        
        Args:
            search_engine: The search engine instance to use for retrieval
            max_hops: Maximum number of reasoning hops to perform
            confidence_threshold: Minimum confidence to continue reasoning
        """
        self.search_engine = search_engine
        self.max_hops = max_hops
        self.confidence_threshold = confidence_threshold
        
        # Patterns that indicate multi-hop reasoning is needed
        self.complex_patterns = [
            r"what if.*fail",
            r"after.*then.*how", 
            r"but.*still.*problem",
            r"tried.*but.*not work",
            r"followed.*step.*error",
            r"installed.*but.*can't",
            r"updated.*now.*issue"
        ]
        
        # Ubuntu-specific follow-up concepts
        self.ubuntu_follow_up_concepts = {
            "package_manager": ["dependencies", "repository", "ppa", "apt", "dpkg"],
            "system_update": ["kernel", "driver", "compatibility", "rollback"],
            "network": ["firewall", "dns", "routing", "interface"],
            "security": ["permissions", "sudo", "authentication", "certificates"],
            "hardware": ["driver", "firmware", "compatibility", "detection"],
            "services": ["systemctl", "daemon", "startup", "logs"]
        }
    
    def should_use_multihop(self, query: str, context: Dict[str, Any]) -> bool:
        """
        Determine if multi-hop reasoning should be used for this query
        """
        query_lower = query.lower()
        
        # Check for complex query patterns
        for pattern in self.complex_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"Multi-hop triggered by pattern: {pattern}")
                return True
        
        # Check if previous responses had low confidence
        if context.get("previous_confidence", 1.0) < 0.5:
            logger.info("Multi-hop triggered by low previous confidence")
            return True
        
        # Check conversation depth for complex troubleshooting
        if context.get("conversation_depth", 0) > 2 and "error" in query_lower:
            logger.info("Multi-hop triggered by conversation depth and error mention")
            return True
        
        return False
    
    def reason(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform multi-hop reasoning for complex questions
        Returns a dict with synthesized answer, evidence, and reasoning path
        """
        logger.info(f"Starting multi-hop reasoning for: {query}")
        
        reasoning_state = {
            "original_query": query,
            "hops": [],
            "evidence": [],
            "queries": [query],
            "confidence_scores": []
        }
        
        try:
            # First hop: Enhanced initial search
            enhanced_query = self._enhance_initial_query(query, context)
            reasoning_state["queries"][0] = enhanced_query
            
            initial_results = self.search_engine.search(enhanced_query, top_k=3)
            
            if not initial_results:
                return self._create_failure_response(query, "No initial results found")
            
            # Process first hop
            hop_result = self._process_hop_results(initial_results, query, context, 1)
            reasoning_state["hops"].append(hop_result)
            reasoning_state["evidence"].extend(hop_result["evidence"])
            reasoning_state["confidence_scores"].append(hop_result["confidence"])
            
            # Extract follow-up concepts
            follow_up_concepts = self._extract_follow_up_concepts(hop_result, query, context)
            
            # Perform additional hops if needed
            current_hop = 2
            while (current_hop <= self.max_hops and 
                   follow_up_concepts and 
                   self._should_continue_reasoning(reasoning_state)):
                
                # Generate and execute follow-up query
                follow_up_query = self._generate_follow_up_query(
                    follow_up_concepts[0], query, context, reasoning_state
                )
                reasoning_state["queries"].append(follow_up_query)
                
                hop_results = self.search_engine.search(follow_up_query, top_k=2)
                
                if hop_results:
                    hop_result = self._process_hop_results(
                        hop_results, follow_up_query, context, current_hop
                    )
                    reasoning_state["hops"].append(hop_result)
                    reasoning_state["evidence"].extend(hop_result["evidence"])
                    reasoning_state["confidence_scores"].append(hop_result["confidence"])
                    
                    # Update follow-up concepts
                    follow_up_concepts = follow_up_concepts[1:]
                
                current_hop += 1
            
            # Synthesize final answer
            final_answer = self._synthesize_multihop_answer(query, reasoning_state, context)
            overall_confidence = self._calculate_overall_confidence(reasoning_state)
            
            logger.info(f"Multi-hop reasoning complete: {len(reasoning_state['hops'])} hops, confidence: {overall_confidence:.3f}")
            
            return {
                "answer": final_answer,
                "evidence": reasoning_state["evidence"],
                "confidence": overall_confidence,
                "queries_used": reasoning_state["queries"],
                "hops_performed": len(reasoning_state["hops"]),
                "is_multihop": True
            }
            
        except Exception as e:
            logger.error(f"Error in multi-hop reasoning: {e}")
            return self._create_failure_response(query, f"Reasoning error: {str(e)}")
    
    def _enhance_initial_query(self, query: str, context: Dict[str, Any]) -> str:
        """Enhance the initial query with context information"""
        enhanced_query = query
        
        # Add context from recent topics
        if context.get("recentTopics"):
            recent_topics = context["recentTopics"][-2:]
            if recent_topics:
                topic_context = " ".join(recent_topics)
                enhanced_query = f"{query} (context: {topic_context})"
        
        # Add entity context
        if context.get("recentSessionEntities"):
            entities = context["recentSessionEntities"][-3:]
            if entities:
                entity_context = " ".join(entities)
                enhanced_query = f"{enhanced_query} regarding {entity_context}"
        
        return enhanced_query
    
    def _process_hop_results(self, results: List[Dict[str, Any]], query: str, 
                           context: Dict[str, Any], hop_number: int) -> Dict[str, Any]:
        """Process results from a single hop"""
        if not results:
            return {
                "hop_number": hop_number,
                "query": query,
                "evidence": [],
                "confidence": 0.0,
                "concepts_found": []
            }
        
        # Filter results based on relevance threshold
        threshold = 0.3 if hop_number == 1 else 0.2
        relevant_results = [r for r in results if r.get("similarity_score", 0) >= threshold]
        
        # Extract concepts for potential follow-up
        concepts_found = []
        for result in relevant_results:
            content = result.get("content", "").lower()
            for category, concepts in self.ubuntu_follow_up_concepts.items():
                for concept in concepts:
                    if concept in content and concept not in concepts_found:
                        concepts_found.append(concept)
        
        # Calculate hop confidence with diminishing returns
        hop_confidence = 0.0
        if relevant_results:
            hop_confidence = sum(r.get("similarity_score", 0) for r in relevant_results) / len(relevant_results)
            hop_confidence *= (0.9 ** (hop_number - 1))
        
        return {
            "hop_number": hop_number,
            "query": query,
            "evidence": relevant_results,
            "confidence": hop_confidence,
            "concepts_found": concepts_found
        }
    
    def _extract_follow_up_concepts(self, hop_result: Dict[str, Any], 
                                  original_query: str, context: Dict[str, Any]) -> List[str]:
        """Extract concepts that warrant follow-up investigation"""
        concepts_found = hop_result.get("concepts_found", [])
        query_lower = original_query.lower()
        
        # Prioritize concepts based on query content
        prioritized_concepts = []
        
        for concept in concepts_found:
            priority = self._calculate_concept_priority(concept, query_lower, context)
            if priority > 0.5:
                prioritized_concepts.append((concept, priority))
        
        # Sort by priority and return top concepts
        prioritized_concepts.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in prioritized_concepts[:3]]
    
    def _calculate_concept_priority(self, concept: str, query: str, context: Dict[str, Any]) -> float:
        """Calculate priority score for following up on a concept"""
        priority = 0.5
        
        # Higher priority for error-related concepts
        if "error" in query and concept in ["dependencies", "permissions", "driver"]:
            priority += 0.3
        
        # Higher priority for installation-related concepts  
        if "install" in query and concept in ["repository", "ppa", "dependencies"]:
            priority += 0.3
        
        # Higher priority for update-related concepts
        if "update" in query and concept in ["kernel", "driver", "compatibility"]:
            priority += 0.3
        
        # Lower priority if recently discussed
        if context.get("recentTopics") and concept in str(context["recentTopics"]).lower():
            priority -= 0.2
        
        return min(1.0, max(0.0, priority))
    
    def _generate_follow_up_query(self, concept: str, original_query: str, 
                                context: Dict[str, Any], reasoning_state: Dict[str, Any]) -> str:
        """Generate a follow-up query to explore a specific concept"""
        # Template-based query generation
        query_templates = {
            "dependencies": f"Ubuntu {concept} issues with",
            "repository": f"Ubuntu {concept} configuration for", 
            "driver": f"Ubuntu {concept} installation troubleshooting",
            "permissions": f"Ubuntu {concept} fix for",
            "kernel": f"Ubuntu {concept} compatibility issues"
        }
        
        # Extract key terms from original query
        key_terms = self._extract_key_terms(original_query)
        
        if concept in query_templates:
            base_query = query_templates[concept]
            if key_terms:
                follow_up_query = f"{base_query} {' '.join(key_terms[:2])}"
            else:
                follow_up_query = base_query
        else:
            follow_up_query = f"Ubuntu {concept} {' '.join(key_terms[:2])}"
        
        return follow_up_query
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from a query"""
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "how", "what", "when", "where", "why", "is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would"}
        
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return key_terms[:5]
    
    def _should_continue_reasoning(self, reasoning_state: Dict[str, Any]) -> bool:
        """Determine if reasoning should continue to another hop"""
        # Don't continue if we have high confidence
        if reasoning_state["confidence_scores"]:
            avg_confidence = sum(reasoning_state["confidence_scores"]) / len(reasoning_state["confidence_scores"])
            if avg_confidence > 0.8:
                return False
        
        # Don't continue if last hop had very low confidence
        if reasoning_state["confidence_scores"] and reasoning_state["confidence_scores"][-1] < 0.2:
            return False
        
        return True
    
    def _synthesize_multihop_answer(self, query: str, reasoning_state: Dict[str, Any], 
                                  context: Dict[str, Any]) -> str:
        """Synthesize a comprehensive answer from multi-hop evidence"""
        all_evidence = reasoning_state["evidence"]
        
        if not all_evidence:
            return "I couldn't find sufficient information to answer your question."
        
        # Start with the best evidence from the first hop
        main_answer = ""
        if reasoning_state["hops"] and reasoning_state["hops"][0]["evidence"]:
            main_evidence = reasoning_state["hops"][0]["evidence"][0]
            main_answer = main_evidence.get("content", "")
        
        # Add complementary information from other hops
        additional_info = []
        for hop in reasoning_state["hops"][1:]:
            if hop["evidence"]:
                for evidence in hop["evidence"][:1]:
                    content = evidence.get("content", "")
                    if content and self._is_valuable_addition(content, main_answer):
                        additional_info.append(content)
        
        # Construct final answer
        if main_answer:
            final_answer = main_answer
            
            if additional_info:
                final_answer += "\n\nAdditional considerations:\n"
                for i, info in enumerate(additional_info[:2], 1):
                    final_answer += f"{i}. {info}\n"
        else:
            final_answer = "Based on the available information: " + "; ".join(additional_info[:3])
        
        return final_answer
    
    def _is_valuable_addition(self, new_content: str, existing_content: str) -> bool:
        """Check if new content adds value to existing content"""
        new_words = set(new_content.lower().split())
        existing_words = set(existing_content.lower().split())
        
        overlap_ratio = len(new_words & existing_words) / len(new_words) if new_words else 0
        return overlap_ratio < 0.7 and len(new_content) > 20
    
    def _calculate_overall_confidence(self, reasoning_state: Dict[str, Any]) -> float:
        """Calculate overall confidence for the multi-hop reasoning result"""
        if not reasoning_state["confidence_scores"]:
            return 0.0
        
        # Weighted average with more weight on earlier hops
        weights = [1.0, 0.8, 0.6, 0.4]
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for i, confidence in enumerate(reasoning_state["confidence_scores"]):
            weight = weights[min(i, len(weights)-1)]
            weighted_sum += confidence * weight
            weight_sum += weight
        
        overall_confidence = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        # Boost for multiple successful hops
        hop_bonus = min(0.1, len(reasoning_state["hops"]) * 0.03)
        
        return min(1.0, overall_confidence + hop_bonus)
    
    def _create_failure_response(self, query: str, reason: str) -> Dict[str, Any]:
        """Create a response for when multi-hop reasoning fails"""
        return {
            "answer": f"I apologize, but I couldn't fully process your complex question: {reason}. Could you try breaking it down into simpler parts?",
            "evidence": [],
            "confidence": 0.1,
            "queries_used": [query],
            "hops_performed": 0,
            "is_multihop": True,
            "failure_reason": reason
        }
        
    def retrieve(self, query: str, max_hops: int = 3, top_k_per_hop: int = 3) -> List[Dict[str, Any]]:
        """
        Perform iterative retrieval for complex questions
        that require multiple pieces of information
        
        Args:
            query (str): Original query
            max_hops (int): Maximum number of retrieval iterations
            top_k_per_hop (int): Number of documents to retrieve per hop
            
        Returns:
            List[Dict[str, Any]]: Merged and deduplicated results
        """
        all_docs = []
        current_query = query
        hop_metadata = []
        
        for hop in range(max_hops):
            print(f"Hop {hop + 1}: Searching for '{current_query}'")
            
            # Get documents for current query
            hop_docs = self.search_engine.search(current_query, top_k=top_k_per_hop)
            
            if not hop_docs:
                print(f"No documents found for hop {hop + 1}, stopping")
                break
                
            # Add hop metadata
            hop_info = {
                "hop_number": hop + 1,
                "query": current_query,
                "docs_found": len(hop_docs),
                "doc_ids": [doc.get("id") or doc.get("chunk_id") for doc in hop_docs]
            }
            hop_metadata.append(hop_info)
            
            # Add to our collection with hop information
            for doc in hop_docs:
                doc["hop_number"] = hop + 1
                doc["hop_query"] = current_query
                
            all_docs.extend(hop_docs)
            
            # Generate follow-up query based on what we found
            if hop < max_hops - 1:
                follow_up_query = self._generate_follow_up(query, current_query, hop_docs)
                
                # If follow-up is too similar to current or original, stop
                if self._is_query_similar(follow_up_query, [query, current_query]):
                    print(f"Follow-up query too similar, stopping at hop {hop + 1}")
                    break
                    
                current_query = follow_up_query
            
        # Merge and deduplicate results
        unique_docs = self._merge_results(all_docs)
        
        # Add retrieval metadata
        retrieval_info = {
            "total_hops": len(hop_metadata),
            "total_docs_before_dedup": len(all_docs),
            "total_docs_after_dedup": len(unique_docs),
            "hop_details": hop_metadata
        }
        
        # Add metadata to first document if exists
        if unique_docs:
            unique_docs[0]["multi_hop_metadata"] = retrieval_info
        
        return unique_docs
        
    def _generate_follow_up(self, original_query: str, current_query: str, docs: List[Dict[str, Any]]) -> str:
        """
        Generate a follow-up query based on retrieved documents
        
        Args:
            original_query (str): The original user query
            current_query (str): Current iteration query
            docs (List[Dict]): Documents retrieved in current hop
            
        Returns:
            str: Follow-up query for next hop
        """
        # Extract key entities from the documents
        entities = self._extract_entities([doc.get("content", "") for doc in docs])
        
        # Extract technical terms that might need more context
        technical_terms = self._extract_technical_terms([doc.get("content", "") for doc in docs])
        
        # Determine what type of follow-up to generate
        follow_up_strategies = [
            self._generate_dependency_query,
            self._generate_context_query,
            self._generate_troubleshooting_query,
            self._generate_detail_query
        ]
        
        # Try each strategy and return the first valid one
        for strategy in follow_up_strategies:
            follow_up = strategy(original_query, entities, technical_terms, docs)
            if follow_up and len(follow_up.strip()) > 0:
                return follow_up
        
        # Fallback: create a query focused on missing information
        if entities:
            return f"more information about {entities[0]} related to {original_query}"
        elif technical_terms:
            return f"details about {technical_terms[0]} configuration"
        else:
            return f"additional context for {original_query}"
    
    def _generate_dependency_query(self, original_query: str, entities: List[str], 
                                 technical_terms: List[str], docs: List[Dict]) -> Optional[str]:
        """Generate query to find dependencies or prerequisites"""
        dependency_keywords = ["requires", "depends", "prerequisite", "before", "first", "install"]
        
        for doc in docs:
            content = doc.get("content", "").lower()
            if any(keyword in content for keyword in dependency_keywords):
                if entities:
                    return f"prerequisites for {entities[0]} installation"
                elif technical_terms:
                    return f"dependencies for {technical_terms[0]}"
        
        return None
    
    def _generate_context_query(self, original_query: str, entities: List[str], 
                              technical_terms: List[str], docs: List[Dict]) -> Optional[str]:
        """Generate query to find broader context or related components"""
        if "install" in original_query.lower():
            if entities:
                return f"configuration after installing {entities[0]}"
            elif technical_terms:
                return f"setup {technical_terms[0]} after installation"
        
        if "error" in original_query.lower() or "problem" in original_query.lower():
            if entities:
                return f"common issues with {entities[0]}"
            elif technical_terms:
                return f"troubleshooting {technical_terms[0]} problems"
        
        return None
    
    def _generate_troubleshooting_query(self, original_query: str, entities: List[str], 
                                      technical_terms: List[str], docs: List[Dict]) -> Optional[str]:
        """Generate query for troubleshooting steps"""
        error_indicators = ["error", "fail", "broken", "not working", "issue"]
        
        if any(indicator in original_query.lower() for indicator in error_indicators):
            if entities:
                return f"fix {entities[0]} errors"
            elif technical_terms:
                return f"resolve {technical_terms[0]} issues"
        
        return None
    
    def _generate_detail_query(self, original_query: str, entities: List[str], 
                             technical_terms: List[str], docs: List[Dict]) -> Optional[str]:
        """Generate query for more detailed information"""
        if entities:
            return f"detailed guide for {entities[0]}"
        elif technical_terms:
            return f"advanced {technical_terms[0]} configuration"
        
        return None
        
    def _extract_entities(self, texts: List[str]) -> List[str]:
        """
        Extract potential entities from text content
        
        Args:
            texts (List[str]): List of text content
            
        Returns:
            List[str]: Extracted entities
        """
        entities = set()
        
        for text in texts:
            if not text:
                continue
                
            # Extract Ubuntu-specific entities
            ubuntu_patterns = [
                r'\b(Ubuntu)\s+(\d+\.\d+)\b',  # Ubuntu versions
                r'\b(apt|snap|flatpak|dpkg)\b',  # Package managers
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:package|service|driver)\b',  # Service names
                r'\b[A-Z][A-Z0-9_]+\b',  # Environment variables or constants
            ]
            
            for pattern in ubuntu_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.add(match.group().strip())
            
            # Extract capitalized words that might be software names
            words = text.split()
            for word in words:
                # Skip common words and focus on potential software/package names
                if (len(word) > 3 and 
                    word[0].isupper() and 
                    word.lower() not in ['the', 'this', 'that', 'with', 'from', 'when', 'where']):
                    entities.add(word)
        
        # Return most relevant entities (limit to avoid too many)
        return list(entities)[:10]
        
    def _extract_technical_terms(self, texts: List[str]) -> List[str]:
        """
        Extract technical terms that might need more context
        
        Args:
            texts (List[str]): List of text content
            
        Returns:
            List[str]: Extracted technical terms
        """
        technical_terms = set()
        
        # Common Ubuntu/Linux technical terms
        tech_patterns = [
            r'\b(?:systemctl|service|daemon)\b',
            r'\b(?:repository|repo|ppa)\b',
            r'\b(?:kernel|module|driver)\b',
            r'\b(?:config|configuration|conf)\b',
            r'\b(?:terminal|shell|bash|zsh)\b',
            r'\b(?:permission|chmod|chown)\b',
            r'\b(?:mount|umount|filesystem)\b',
            r'\b(?:network|networking|wifi)\b'
        ]
        
        for text in texts:
            if not text:
                continue
                
            for pattern in tech_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    technical_terms.add(match.group().strip().lower())
        
        return list(technical_terms)[:5]
        
    def _is_query_similar(self, new_query: str, existing_queries: List[str], threshold: float = 0.7) -> bool:
        """
        Check if a new query is too similar to existing queries
        
        Args:
            new_query (str): New query to check
            existing_queries (List[str]): List of existing queries
            threshold (float): Similarity threshold (0-1)
            
        Returns:
            bool: True if query is too similar
        """
        new_words = set(new_query.lower().split())
        
        for existing in existing_queries:
            existing_words = set(existing.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(new_words.intersection(existing_words))
            union = len(new_words.union(existing_words))
            
            if union > 0:
                similarity = intersection / union
                if similarity > threshold:
                    return True
        
        return False
        
    def _merge_results(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate retrieval results
        
        Args:
            docs (List[Dict]): List of documents from all hops
            
        Returns:
            List[Dict]: Deduplicated and merged results
        """
        seen_ids = set()
        unique_docs = []
        
        # Sort by hop number to prioritize earlier findings
        sorted_docs = sorted(docs, key=lambda x: x.get("hop_number", 0))
        
        for doc in sorted_docs:
            # Try multiple ID fields for deduplication
            doc_id = (doc.get("id") or 
                     doc.get("chunk_id") or 
                     doc.get("document_id") or
                     hash(doc.get("content", "")[:100]))  # Fallback to content hash
            
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        
        # Sort final results by relevance score if available
        try:
            unique_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
        except (TypeError, KeyError):
            pass  # Keep original order if no scores available
                
        return unique_docs
