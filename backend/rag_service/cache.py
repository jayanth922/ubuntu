import redis
import json
import hashlib
import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class ResponseCache:
    """Cache for RAG responses to improve performance"""
    
    def __init__(self, redis_url=None, ttl=3600):
        """
        Initialize cache
        
        Args:
            redis_url: Redis connection URL
            ttl: Time to live for cache entries (seconds)
        """
        self.enabled = False
        self.ttl = ttl
        self.hit_count = 0
        self.miss_count = 0
        
        # Try to connect to Redis if URL is provided
        if redis_url:
            try:
                self.redis = redis.from_url(redis_url)
                self.enabled = True
                logger.info(f"Response cache initialized with Redis at {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for caching: {e}")
                self.redis = None
        else:
            logger.info("Response cache disabled (no Redis URL provided)")
            self.redis = None
    
    def _generate_cache_key(self, query: str, intent: Optional[str] = None) -> str:
        """Generate a cache key from query and intent"""
        key_parts = [query.lower()]
        if intent:
            key_parts.append(intent.lower())
        
        # Create a stable hash for the key
        key_str = ":".join(key_parts)
        return f"rag:response:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def get(self, query: str, intent: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a response from cache"""
        if not self.enabled or not self.redis:
            return None
            
        try:
            cache_key = self._generate_cache_key(query, intent)
            cached_data = self.redis.get(cache_key)
            
            if cached_data:
                self.hit_count += 1
                logger.debug(f"Cache hit for query: {query}")
                return json.loads(cached_data)
            else:
                self.miss_count += 1
                logger.debug(f"Cache miss for query: {query}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    def set(self, query: str, intent: Optional[str], response_data: Dict[str, Any]) -> bool:
        """Store a response in cache"""
        if not self.enabled or not self.redis:
            return False
            
        try:
            cache_key = self._generate_cache_key(query, intent)
            self.redis.setex(
                cache_key,
                self.ttl,
                json.dumps(response_data)
            )
            logger.debug(f"Cached response for query: {query}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "enabled": self.enabled,
            "hits": self.hit_count,
            "misses": self.miss_count,
            "total": total_requests,
            "hit_rate": hit_rate
        }
    
    def clear(self) -> bool:
        """Clear the cache"""
        if not self.enabled or not self.redis:
            return False
            
        try:
            # Find all keys matching the pattern
            pattern = "rag:response:*"
            keys = self.redis.keys(pattern)
            
            if keys:
                self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} entries from response cache")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False