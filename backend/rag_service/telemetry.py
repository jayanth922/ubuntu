"""
Telemetry Module for RAG Service
Provides distributed tracing and logging capabilities for monitoring and debugging
"""

import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class Telemetry:
    """
    Provides distributed tracing and logging capabilities for monitoring
    RAG service operations and performance
    """
    
    def __init__(self):
        """Initialize the telemetry system"""
        self.logs = []
        self.spans = {}
        self.current_span = None
        self.metrics = {}
        self.counters = {}
        
    def start_span(self, name: str, parent_id: Optional[str] = None) -> str:
        """
        Start a new tracing span
        
        Args:
            name (str): Name of the span
            parent_id (str, optional): ID of parent span for nested operations
            
        Returns:
            str: Unique span ID
        """
        span_id = str(uuid.uuid4())
        span = {
            "id": span_id,
            "name": name,
            "start_time": time.time(),
            "parent_id": parent_id,
            "tags": {},
            "logs": [],
            "status": "active"
        }
        
        self.spans[span_id] = span
        self.current_span = span_id
        
        # Log span creation
        self.log("DEBUG", f"Started span: {name}", span_id)
        
        return span_id
        
    def end_span(self, span_id: Optional[str] = None, status: str = "completed") -> None:
        """
        End a tracing span
        
        Args:
            span_id (str, optional): Span ID to end. Uses current span if None
            status (str): Final status of the span (completed, error, timeout)
        """
        if span_id is None:
            span_id = self.current_span
            
        if span_id and span_id in self.spans:
            span = self.spans[span_id]
            span["end_time"] = time.time()
            span["duration"] = span["end_time"] - span["start_time"]
            span["status"] = status
            
            # Log span completion
            self.log("DEBUG", f"Ended span: {span['name']} (duration: {span['duration']:.3f}s)", span_id)
            
            # Update current span to parent if this was the current span
            if self.current_span == span_id:
                self.current_span = span.get("parent_id")
                
    def add_tag(self, key: str, value: Any, span_id: Optional[str] = None) -> None:
        """
        Add a tag to a span for additional context
        
        Args:
            key (str): Tag key
            value (Any): Tag value
            span_id (str, optional): Span ID. Uses current span if None
        """
        if span_id is None:
            span_id = self.current_span
            
        if span_id and span_id in self.spans:
            self.spans[span_id]["tags"][key] = str(value)
            
    def log(self, level: str, message: str, span_id: Optional[str] = None, **kwargs) -> None:
        """
        Add a log entry
        
        Args:
            level (str): Log level (DEBUG, INFO, WARNING, ERROR)
            message (str): Log message
            span_id (str, optional): Associated span ID
            **kwargs: Additional log data
        """
        log_entry = {
            "timestamp": time.time(),
            "datetime": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "span_id": span_id or self.current_span,
            "extra": kwargs
        }
        
        self.logs.append(log_entry)
        
        # Add to span logs if span exists
        if span_id and span_id in self.spans:
            self.spans[span_id]["logs"].append(log_entry)
            
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric
        
        Args:
            name (str): Counter name
            value (int): Increment value
            tags (dict, optional): Additional tags for the metric
        """
        counter_key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        self.counters[counter_key] = self.counters.get(counter_key, 0) + value
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric value
        
        Args:
            name (str): Metric name
            value (float): Metric value
            tags (dict, optional): Additional tags for the metric
        """
        metric_key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        
        if metric_key not in self.metrics:
            self.metrics[metric_key] = {
                "values": [],
                "count": 0,
                "sum": 0,
                "min": value,
                "max": value
            }
        
        metric = self.metrics[metric_key]
        metric["values"].append(value)
        metric["count"] += 1
        metric["sum"] += value
        metric["min"] = min(metric["min"], value)
        metric["max"] = max(metric["max"], value)
        
        # Keep only recent values to prevent memory issues
        if len(metric["values"]) > 1000:
            metric["values"] = metric["values"][-500:]
            
    def get_span_summary(self, span_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information for a span
        
        Args:
            span_id (str): Span ID
            
        Returns:
            Dict[str, Any]: Span summary or None if not found
        """
        if span_id not in self.spans:
            return None
            
        span = self.spans[span_id]
        summary = {
            "id": span["id"],
            "name": span["name"],
            "status": span["status"],
            "duration": span.get("duration"),
            "tags": span["tags"],
            "log_count": len(span["logs"]),
            "error_count": len([log for log in span["logs"] if log["level"] == "ERROR"])
        }
        
        return summary
        
    def get_trace_summary(self, root_span_id: str) -> Dict[str, Any]:
        """
        Get summary for entire trace starting from root span
        
        Args:
            root_span_id (str): Root span ID
            
        Returns:
            Dict[str, Any]: Trace summary
        """
        def collect_children(span_id: str) -> List[str]:
            children = []
            for sid, span in self.spans.items():
                if span.get("parent_id") == span_id:
                    children.append(sid)
                    children.extend(collect_children(sid))
            return children
        
        if root_span_id not in self.spans:
            return {"error": "Root span not found"}
            
        all_span_ids = [root_span_id] + collect_children(root_span_id)
        all_spans = [self.spans[sid] for sid in all_span_ids if sid in self.spans]
        
        total_duration = max(
            (span.get("duration", 0) for span in all_spans if span.get("duration")),
            default=0
        )
        
        error_count = sum(
            len([log for log in span["logs"] if log["level"] == "ERROR"])
            for span in all_spans
        )
        
        return {
            "root_span_id": root_span_id,
            "total_spans": len(all_spans),
            "total_duration": total_duration,
            "error_count": error_count,
            "status": "error" if error_count > 0 else "success",
            "spans": [self.get_span_summary(sid) for sid in all_span_ids]
        }
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics
        
        Returns:
            Dict[str, Any]: Metrics summary
        """
        summary = {
            "counters": dict(self.counters),
            "metrics": {}
        }
        
        for metric_key, metric_data in self.metrics.items():
            avg = metric_data["sum"] / metric_data["count"] if metric_data["count"] > 0 else 0
            summary["metrics"][metric_key] = {
                "count": metric_data["count"],
                "sum": metric_data["sum"],
                "avg": avg,
                "min": metric_data["min"],
                "max": metric_data["max"]
            }
            
        return summary
        
    def get_recent_logs(self, level: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent log entries
        
        Args:
            level (str, optional): Filter by log level
            limit (int): Maximum number of logs to return
            
        Returns:
            List[Dict[str, Any]]: Recent log entries
        """
        logs = self.logs
        
        if level:
            logs = [log for log in logs if log["level"] == level]
            
        # Return most recent logs
        return logs[-limit:] if limit else logs
        
    def clear_old_data(self, max_age_hours: int = 24) -> None:
        """
        Clear old telemetry data to prevent memory issues
        
        Args:
            max_age_hours (int): Maximum age of data to keep in hours
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        # Clear old spans
        old_span_ids = [
            sid for sid, span in self.spans.items()
            if span["start_time"] < cutoff_time
        ]
        for sid in old_span_ids:
            del self.spans[sid]
            
        # Clear old logs
        self.logs = [
            log for log in self.logs
            if log["timestamp"] >= cutoff_time
        ]
        
        self.log("INFO", f"Cleared {len(old_span_ids)} old spans and old logs")
        
    def export_trace(self, root_span_id: str) -> Dict[str, Any]:
        """
        Export trace data in a format suitable for external tracing systems
        
        Args:
            root_span_id (str): Root span ID
            
        Returns:
            Dict[str, Any]: Exported trace data
        """
        trace_summary = self.get_trace_summary(root_span_id)
        
        if "error" in trace_summary:
            return trace_summary
            
        # Build trace in OpenTelemetry-like format
        trace_data = {
            "trace_id": root_span_id,
            "spans": [],
            "logs": []
        }
        
        for span_summary in trace_summary["spans"]:
            span_id = span_summary["id"]
            if span_id in self.spans:
                span = self.spans[span_id]
                trace_data["spans"].append({
                    "span_id": span["id"],
                    "name": span["name"],
                    "start_time": span["start_time"],
                    "end_time": span.get("end_time"),
                    "duration": span.get("duration"),
                    "parent_id": span.get("parent_id"),
                    "status": span["status"],
                    "tags": span["tags"]
                })
                
                # Add span logs
                trace_data["logs"].extend(span["logs"])
                
        return trace_data
