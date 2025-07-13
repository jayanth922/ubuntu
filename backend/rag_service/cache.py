import redis
import json
import hashlib
import logging
import time
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class ResponseCache:
    """Cache for RAG responses to improve performance"""
    
    def __init__(
        self, 
        redis_url: Optional[str] = None, 
        ttl: int = 3600,
        namespace: str = "rag",
        disabled: bool = False
    ):
        """
        Initialize the response cache
        
        Args:
            redis_url: Redis connection URL (None for in-memory cache)
            ttl: Time to live in seconds
            namespace: Cache namespace
            disabled: Disable cache entirely
        """
        self.ttl = ttl
        self.namespace = namespace
        self.disabled = disabled
        self.stats = {"hits": 0, "misses": 0, "errors": 0}
        
        # In-memory cache as fallback
        self.memory_cache = {}
        
        if disabled:
            logger.info("Response cache is disabled")
            return
            
        # Try to connect to Redis if URL is provided
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                # Test connection
                self.redis_client.ping()
                logger.info(f"Connected to Redis cache at {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis, using in-memory cache: {e}")
        else:
            logger.info("No Redis URL provided, using in-memory cache")
    
    def _generate_key(self, query: str, intent: Optional[str] = None, **kwargs) -> str:
        """Generate a cache key from query parameters"""
        if self.disabled:
            return ""
            
        # Normalize query
        query = query.lower().strip()
        
        # Build key components
        key_parts = [query]
        
        # Add intent if provided
        if intent:
            key_parts.append(f"intent:{intent.lower()}")
        
        # Add any additional key components
        for k, v in kwargs.items():
            if v is not None:
                key_parts.append(f"{k}:{v}")
        
        # Create a stable hash
        key_str = ":".join(key_parts)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"{self.namespace}:{key_hash}"
    
    def get(self, query: str, intent: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get a cached response
        
        Args:
            query: The user query
            intent: The classified intent
            **kwargs: Additional parameters for the cache key
            
        Returns:
            Dict or None: The cached response or None if not found
        """
        if self.disabled:
            return None
            
        try:
            key = self._generate_key(query, intent, **kwargs)
            
            # Try Redis first if available
            if self.redis_client:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    self.stats["hits"] += 1
                    return json.loads(cached_data)
            
            # Fall back to in-memory cache
            if key in self.memory_cache:
                item = self.memory_cache[key]
                
                # Check if item is expired
                if time.time() < item["expires_at"]:
                    self.stats["hits"] += 1
                    return item["data"]
                else:
                    # Remove expired item
                    del self.memory_cache[key]
            
            # Cache miss
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            self.stats["errors"] += 1
            return None
    
    def set(
        self, 
        query: str, 
        data: Dict[str, Any], 
        intent: Optional[str] = None, 
        ttl: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Store a response in the cache
        
        Args:
            query: The user query
            data: The data to cache
            intent: The classified intent
            ttl: Override the default TTL
            **kwargs: Additional parameters for the cache key
            
        Returns:
            bool: True if successful
        """
        if self.disabled:
            return False
            
        try:
            key = self._generate_key(query, intent, **kwargs)
            cache_ttl = ttl if ttl is not None else self.ttl
            
            # Try Redis first if available
            if self.redis_client:
                serialized = json.dumps(data)
                return bool(self.redis_client.setex(key, cache_ttl, serialized))
            
            # Fall back to in-memory cache
            self.memory_cache[key] = {
                "data": data,
                "expires_at": time.time() + cache_ttl
            }
            
            # Simple cache size management - if too many items, remove oldest
            if len(self.memory_cache) > 1000:  # Arbitrary limit
                oldest_key = None
                oldest_time = float('inf')
                
                for k, v in self.memory_cache.items():
                    if v["expires_at"] < oldest_time:
                        oldest_time = v["expires_at"]
                        oldest_key = k
                
                if oldest_key:
                    del self.memory_cache[oldest_key]
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            self.stats["errors"] += 1
            return False
    
    def delete(self, query: str, intent: Optional[str] = None, **kwargs) -> bool:
        """
        Delete a specific item from the cache
        
        Args:
            query: The user query
            intent: The classified intent
            **kwargs: Additional parameters for the cache key
            
        Returns:
            bool: True if successful
        """
        if self.disabled:
            return False
            
        try:
            key = self._generate_key(query, intent, **kwargs)
            
            # Try Redis first if available
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            
            # Fall back to in-memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            self.stats["errors"] += 1
            return False
    
    def flush(self, pattern: Optional[str] = None) -> int:
        """
        Flush all items from the cache matching a pattern
        
        Args:
            pattern: Optional pattern to match keys (e.g., "rag:*")
            
        Returns:
            int: Number of items removed
        """
        if self.disabled:
            return 0
            
        try:
            count = 0
            
            # Set default pattern to namespace
            if pattern is None:
                pattern = f"{self.namespace}:*"
            
            # Try Redis first if available
            if self.redis_client:
                # Get matching keys
                keys = self.redis_client.keys(pattern)
                
                if keys:
                    count = self.redis_client.delete(*keys)
            
            # Fall back to in-memory cache
            else:
                # For in-memory, we'll just clear everything if pattern contains namespace
                if self.namespace in pattern:
                    count = len(self.memory_cache)
                    self.memory_cache.clear()
            
            return count
            
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            self.stats["errors"] += 1
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.disabled:
            return {"status": "disabled"}
            
        stats = self.stats.copy()
        total = stats["hits"] + stats["misses"]
        stats["total"] = total
        stats["hit_ratio"] = stats["hits"] / total if total > 0 else 0
        stats["type"] = "redis" if self.redis_client else "memory"
        stats["size"] = len(self.memory_cache) if not self.redis_client else None
        
        # Get additional Redis stats if available
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats["redis_used_memory"] = info.get("used_memory_human")
                stats["redis_clients"] = info.get("connected_clients")
            except Exception:
                pass
        
        return stats