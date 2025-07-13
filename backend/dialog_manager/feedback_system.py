import time
import json
import uuid
import asyncio
import aioredis
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class FeedbackSystem:
    """
    Enterprise-grade feedback collection and analysis system
    Captures user reactions, conversation outcomes, and enables continuous improvement
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", enable_analytics=True):
        self.redis_url = redis_url
        self.enable_analytics = enable_analytics
        self.redis = None
        self.analytics_queue = asyncio.Queue() if enable_analytics else None
        self.analytics_task = None
        
        # Feedback types and their importance scores
        self.feedback_weights = {
            "thumbs_up": 1.0,
            "thumbs_down": -2.0,  # Negative feedback weighted more heavily
            "helpful": 1.5,
            "not_helpful": -2.5,
            "follow_up_clicked": 0.5,
            "conversation_abandoned": -1.0,
            "problem_solved": 2.0,
            "problem_unsolved": -3.0,
            "suggestion_used": 1.0,
            "suggestion_ignored": -0.2
        }
        
        # Analytics counters for real-time monitoring
        self.session_counters = defaultdict(int)
        self.recent_feedback = deque(maxlen=1000)  # Keep last 1000 feedback items in memory
    
    async def initialize(self):
        """Initialize Redis connection and start analytics worker"""
        try:
            self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            logger.info("Feedback system connected to Redis")
            
            if self.enable_analytics:
                self.analytics_task = asyncio.create_task(self._analytics_worker())
                logger.info("Analytics worker started")
                
        except Exception as e:
            logger.error(f"Failed to initialize feedback system: {e}")
            # Continue without Redis for development
            self.redis = None
    
    async def record_feedback(self, 
                             session_id: str, 
                             feedback_type: str,
                             message_id: Optional[str] = None,
                             intent: Optional[str] = None,
                             entities: Optional[List[str]] = None,
                             confidence: Optional[float] = None,
                             response_time: Optional[float] = None,
                             context: Optional[Dict] = None,
                             metadata: Optional[Dict] = None) -> str:
        """
        Record comprehensive user feedback with full context for analysis
        Returns: feedback_id
        """
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        feedback_data = {
            "id": feedback_id,
            "session_id": session_id,
            "feedback_type": feedback_type,
            "message_id": message_id,
            "intent": intent,
            "entities": entities or [],
            "confidence": confidence,
            "response_time": response_time,
            "timestamp": timestamp.isoformat(),
            "context": context or {},
            "metadata": metadata or {},
            "weight": self.feedback_weights.get(feedback_type, 0)
        }
        
        try:
            # Store the feedback in Redis if available
            if self.redis:
                await self._store_in_redis(feedback_data)
            else:
                # Fallback to local storage
                await self._store_locally(feedback_data)
            
            # Add to real-time analytics
            self.recent_feedback.append(feedback_data)
            self.session_counters[feedback_type] += 1
            
            # Queue for analytics processing
            if self.analytics_queue:
                await self.analytics_queue.put(feedback_data)
            
            # Handle special feedback types
            await self._handle_special_feedback(feedback_data)
            
            logger.info(f"Recorded feedback: {feedback_type} for session {session_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            # Always try to save locally as last resort
            await self._store_locally(feedback_data)
            return feedback_id
    
    async def _store_in_redis(self, feedback_data: Dict):
        """Store feedback data in Redis with proper indexing"""
        feedback_id = feedback_data["id"]
        session_id = feedback_data["session_id"]
        feedback_type = feedback_data["feedback_type"]
        timestamp = feedback_data["timestamp"]
        
        # Store the full feedback record
        await self.redis.set(f"feedback:{feedback_id}", json.dumps(feedback_data))
        
        # Index by session for easy retrieval
        await self.redis.lpush(f"feedback:session:{session_id}", feedback_id)
        
        # Index by type for analytics
        await self.redis.lpush(f"feedback:type:{feedback_type}", feedback_id)
        
        # Index by date for time-based analytics
        date_key = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
        await self.redis.lpush(f"feedback:date:{date_key}", feedback_id)
        
        # Index by intent if available
        if feedback_data.get("intent"):
            await self.redis.lpush(f"feedback:intent:{feedback_data['intent']}", feedback_id)
        
        # Set expiration (90 days)
        expiry = 90 * 24 * 60 * 60
        await self.redis.expire(f"feedback:{feedback_id}", expiry)
    
    async def _store_locally(self, feedback_data: Dict):
        """Fallback local storage when Redis is unavailable"""
        try:
            filename = f"feedback_{feedback_data['id']}.json"
            with open(filename, "w") as f:
                json.dump(feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to store feedback locally: {e}")
    
    async def get_session_feedback(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Retrieve all feedback for a session"""
        if not self.redis:
            return []
        
        try:
            feedback_ids = await self.redis.lrange(f"feedback:session:{session_id}", 0, limit-1)
            result = []
            
            for fid in feedback_ids:
                data = await self.redis.get(f"feedback:{fid}")
                if data:
                    result.append(json.loads(data))
            
            return sorted(result, key=lambda x: x["timestamp"], reverse=True)
        except Exception as e:
            logger.error(f"Error retrieving session feedback: {e}")
            return []
    
    async def get_feedback_analytics(self, 
                                   days: int = 7, 
                                   feedback_type: Optional[str] = None,
                                   intent: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive feedback analytics"""
        if not self.redis:
            return self._get_local_analytics()
        
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            analytics = {
                "summary": {
                    "total_feedback": 0,
                    "positive_feedback": 0,
                    "negative_feedback": 0,
                    "average_confidence": 0.0,
                    "average_response_time": 0.0
                },
                "by_type": defaultdict(int),
                "by_intent": defaultdict(int),
                "by_date": defaultdict(int),
                "trends": {
                    "satisfaction_score": 0.0,
                    "daily_feedback_count": [],
                    "top_issues": []
                }
            }
            
            # Collect feedback for the date range
            all_feedback = []
            current_date = start_date
            
            while current_date <= end_date:
                date_key = current_date.strftime("%Y-%m-%d")
                feedback_ids = await self.redis.lrange(f"feedback:date:{date_key}", 0, -1)
                
                for fid in feedback_ids:
                    data = await self.redis.get(f"feedback:{fid}")
                    if data:
                        feedback_item = json.loads(data)
                        
                        # Apply filters
                        if feedback_type and feedback_item["feedback_type"] != feedback_type:
                            continue
                        if intent and feedback_item.get("intent") != intent:
                            continue
                            
                        all_feedback.append(feedback_item)
                
                current_date += timedelta(days=1)
            
            # Calculate analytics
            if all_feedback:
                analytics["summary"]["total_feedback"] = len(all_feedback)
                
                total_weight = 0
                confidence_sum = 0
                response_time_sum = 0
                confidence_count = 0
                response_time_count = 0
                
                for item in all_feedback:
                    # Weight calculation for satisfaction
                    weight = item.get("weight", 0)
                    total_weight += weight
                    
                    if weight > 0:
                        analytics["summary"]["positive_feedback"] += 1
                    elif weight < 0:
                        analytics["summary"]["negative_feedback"] += 1
                    
                    # Count by type and intent
                    analytics["by_type"][item["feedback_type"]] += 1
                    if item.get("intent"):
                        analytics["by_intent"][item["intent"]] += 1
                    
                    # Date aggregation
                    date = datetime.fromisoformat(item["timestamp"]).strftime("%Y-%m-%d")
                    analytics["by_date"][date] += 1
                    
                    # Averages
                    if item.get("confidence") is not None:
                        confidence_sum += item["confidence"]
                        confidence_count += 1
                    
                    if item.get("response_time") is not None:
                        response_time_sum += item["response_time"]
                        response_time_count += 1
                
                # Calculate derived metrics
                if len(all_feedback) > 0:
                    analytics["trends"]["satisfaction_score"] = total_weight / len(all_feedback)
                
                if confidence_count > 0:
                    analytics["summary"]["average_confidence"] = confidence_sum / confidence_count
                
                if response_time_count > 0:
                    analytics["summary"]["average_response_time"] = response_time_sum / response_time_count
                
                # Convert defaultdict to regular dict for JSON serialization
                analytics["by_type"] = dict(analytics["by_type"])
                analytics["by_intent"] = dict(analytics["by_intent"])
                analytics["by_date"] = dict(analytics["by_date"])
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting feedback analytics: {e}")
            return self._get_local_analytics()
    
    def _get_local_analytics(self) -> Dict[str, Any]:
        """Get analytics from recent in-memory feedback when Redis unavailable"""
        analytics = {
            "summary": {
                "total_feedback": len(self.recent_feedback),
                "positive_feedback": 0,
                "negative_feedback": 0
            },
            "by_type": dict(self.session_counters),
            "note": "Limited analytics - Redis not available"
        }
        
        for item in self.recent_feedback:
            weight = item.get("weight", 0)
            if weight > 0:
                analytics["summary"]["positive_feedback"] += 1
            elif weight < 0:
                analytics["summary"]["negative_feedback"] += 1
        
        return analytics
    
    async def _handle_special_feedback(self, feedback_data: Dict):
        """Handle special feedback types that require immediate action"""
        feedback_type = feedback_data["feedback_type"]
        
        # Flag negative feedback for review
        if feedback_type in ["thumbs_down", "not_helpful", "problem_unsolved"]:
            await self._flag_for_review(feedback_data)
        
        # Track conversation abandonment
        elif feedback_type == "conversation_abandoned":
            await self._track_abandonment(feedback_data)
        
        # Positive feedback can trigger model confidence updates
        elif feedback_type in ["thumbs_up", "helpful", "problem_solved"]:
            await self._record_success(feedback_data)
    
    async def _flag_for_review(self, feedback_data: Dict):
        """Flag problematic interactions for human review"""
        review_id = str(uuid.uuid4())
        review_data = {
            "id": review_id,
            "feedback_id": feedback_data["id"],
            "session_id": feedback_data["session_id"],
            "status": "pending",
            "priority": "high" if feedback_data["feedback_type"] == "problem_unsolved" else "medium",
            "created_at": datetime.utcnow().isoformat(),
            "feedback_data": feedback_data
        }
        
        if self.redis:
            try:
                await self.redis.set(f"review:{review_id}", json.dumps(review_data))
                await self.redis.lpush("reviews:pending", review_id)
                await self.redis.expire(f"review:{review_id}", 30 * 24 * 60 * 60)  # 30 days
            except Exception as e:
                logger.error(f"Error flagging for review: {e}")
    
    async def _track_abandonment(self, feedback_data: Dict):
        """Track conversation abandonment patterns"""
        session_id = feedback_data["session_id"]
        
        if self.redis:
            try:
                await self.redis.incr(f"abandonment:session:{session_id}")
                
                # Track abandonment by intent
                if feedback_data.get("intent"):
                    await self.redis.incr(f"abandonment:intent:{feedback_data['intent']}")
            except Exception as e:
                logger.error(f"Error tracking abandonment: {e}")
    
    async def _record_success(self, feedback_data: Dict):
        """Record successful interactions for model improvement"""
        if self.redis:
            try:
                # Track success by intent
                if feedback_data.get("intent"):
                    await self.redis.incr(f"success:intent:{feedback_data['intent']}")
                
                # Track success by confidence level
                if feedback_data.get("confidence"):
                    confidence_bucket = int(feedback_data["confidence"] * 10) / 10  # Round to 0.1
                    await self.redis.incr(f"success:confidence:{confidence_bucket}")
            except Exception as e:
                logger.error(f"Error recording success: {e}")
    
    async def _analytics_worker(self):
        """Background worker to process feedback for advanced analytics"""
        while True:
            try:
                # Wait for feedback to process
                feedback = await self.analytics_queue.get()
                
                # Process the feedback for analytics
                await self._process_feedback_analytics(feedback)
                
                self.analytics_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Analytics worker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in analytics worker: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _process_feedback_analytics(self, feedback: Dict):
        """Process individual feedback for analytics"""
        try:
            feedback_type = feedback["feedback_type"]
            timestamp = datetime.fromisoformat(feedback["timestamp"])
            
            if not self.redis:
                return
            
            # Update time-based counters
            hour_key = timestamp.strftime("%Y-%m-%d-%H")
            await self.redis.hincrby("feedback:stats:hourly", f"{hour_key}:{feedback_type}", 1)
            
            day_key = timestamp.strftime("%Y-%m-%d")
            await self.redis.hincrby("feedback:stats:daily", f"{day_key}:{feedback_type}", 1)
            
            # Update global counters
            await self.redis.hincrby("feedback:stats:total", feedback_type, 1)
            
            # Track entity-specific feedback
            for entity in feedback.get("entities", []):
                await self.redis.hincrby(f"feedback:entity:{entity}", feedback_type, 1)
            
            # Track intent-specific feedback
            if feedback.get("intent"):
                await self.redis.hincrby(f"feedback:intent_stats:{feedback['intent']}", feedback_type, 1)
                
        except Exception as e:
            logger.error(f"Error processing feedback analytics: {e}")
    
    async def get_pending_reviews(self, limit: int = 50) -> List[Dict]:
        """Get pending reviews for human moderators"""
        if not self.redis:
            return []
        
        try:
            review_ids = await self.redis.lrange("reviews:pending", 0, limit-1)
            reviews = []
            
            for review_id in review_ids:
                data = await self.redis.get(f"review:{review_id}")
                if data:
                    reviews.append(json.loads(data))
            
            return reviews
        except Exception as e:
            logger.error(f"Error getting pending reviews: {e}")
            return []
    
    async def close(self):
        """Clean shutdown of the feedback system"""
        if self.analytics_task:
            self.analytics_task.cancel()
            try:
                await self.analytics_task
            except asyncio.CancelledError:
                pass
        
        if self.redis:
            await self.redis.close()


# Utility functions for integration
async def create_feedback_system(redis_url: str = "redis://localhost:6379") -> FeedbackSystem:
    """Factory function to create and initialize feedback system"""
    system = FeedbackSystem(redis_url)
    await system.initialize()
    return system
