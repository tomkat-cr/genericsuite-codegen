"""
Enhanced Search Logging and Performance Monitoring

This module provides comprehensive logging, error handling, and performance
monitoring for the enhanced vector search system.
"""

import functools
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict, deque

from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    log_warning,
    log_error,
)

from .enhanced_search_types import (
    EnhancedSearchError,
    CodeGenerationContext,
    DualSearchResult
)


DEBUG = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for enhanced search operations."""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, success: bool = True,
                 error_message: Optional[str] = None):
        """Mark the operation as complete."""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


@dataclass
class SearchOperationLog:
    """Log entry for search operations."""
    timestamp: datetime
    # dual_search, context_determination, document_retrieval, etc.
    operation_type: str
    user_query: Optional[str] = None
    contextual_query: Optional[str] = None
    code_context: Optional[Dict[str, Any]] = None
    results_count: int = 0
    duration: Optional[float] = None
    success: bool = True
    error_details: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "operation_type": self.operation_type,
            "user_query": self.user_query,
            "contextual_query": self.contextual_query,
            "code_context": self.code_context,
            "results_count": self.results_count,
            "duration": self.duration,
            "success": self.success,
            "error_details": self.error_details,
            "performance_metrics": self.performance_metrics
        }


class PerformanceMonitor:
    """Performance monitoring and metrics collection for enhanced search."""

    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.

        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "total_duration": 0.0,
                "success_count": 0,
                "error_count": 0,
                "avg_duration": 0.0,
                "min_duration": float('inf'),
                "max_duration": 0.0,
                "last_execution": None
            }
        )

        # Performance thresholds (in seconds)
        self.performance_thresholds = {
            "dual_search": 10.0,
            "context_determination": 2.0,
            "document_retrieval": 5.0,
            "search_merge": 1.0,
            "template_load": 1.0
        }

    def start_operation(self, operation_name: str, **metadata
                        ) -> PerformanceMetrics:
        """Start tracking a new operation."""
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            metadata=metadata
        )

        _ = DEBUG and log_debug(f"Started operation: {operation_name}")
        return metrics

    def complete_operation(
        self,
        metrics: PerformanceMetrics,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Complete an operation and update statistics."""
        metrics.complete(success=success, error_message=error_message)

        # Add to history
        self.metrics_history.append(metrics)

        # Update operation statistics
        stats = self.operation_stats[metrics.operation_name]
        stats["count"] += 1
        stats["total_duration"] += metrics.duration
        stats["last_execution"] = metrics.end_time

        if success:
            stats["success_count"] += 1
        else:
            stats["error_count"] += 1

        # Update duration statistics
        if metrics.duration < stats["min_duration"]:
            stats["min_duration"] = metrics.duration
        if metrics.duration > stats["max_duration"]:
            stats["max_duration"] = metrics.duration

        stats["avg_duration"] = stats["total_duration"] / stats["count"]

        # Check performance thresholds
        threshold = self.performance_thresholds.get(metrics.operation_name)
        if threshold and metrics.duration > threshold:
            log_warning(
                "Performance threshold exceeded for "
                f"{metrics.operation_name}: "
                f"{metrics.duration:.2f}s > {threshold}s"
            )

        # Log completion
        msg = f"Completed operation: {metrics.operation_name} " \
            + f"(duration: {metrics.duration:.2f}s, "
        if success:
            msg += f"success: {success})"
            _ = DEBUG and log_debug(msg)
        else:
            msg += f"ERROR: {error_message})"
            log_error(msg)

    def get_operation_stats(self, operation_name: Optional[str] = None
                            ) -> Dict[str, Any]:
        """Get statistics for operations."""
        if operation_name:
            return dict(self.operation_stats.get(operation_name, {}))
        return {op: dict(stats) for op, stats in self.operation_stats.items()}

    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent performance metrics."""
        recent = list(self.metrics_history)[-limit:]
        return [metrics.to_dict() for metrics in recent]

    def reset_stats(self):
        """Reset all statistics."""
        self.metrics_history.clear()
        self.operation_stats.clear()
        _ = DEBUG and log_debug("Performance statistics reset")


class EnhancedSearchLogger:
    """Comprehensive logging system for enhanced search operations."""

    def __init__(
        self,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        """
        Initialize the enhanced search logger.

        Args:
            performance_monitor: Optional performance monitor instance
        """
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.operation_logs: deque = deque(maxlen=1000)

        # Configure structured logging
        # self.logger = get_logger("enhanced_search")

        # self.logger = logging.getLogger("enhanced_search")
        # # Ensure logger has appropriate handlers
        # if not self.logger.handlers:
        #     handler = logging.StreamHandler()
        #     formatter = logging.Formatter(
        #         '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        #     )
        #     handler.setFormatter(formatter)
        #     self.logger.addHandler(handler)
        #     self.logger.setLevel(logging.INFO)

    def log_dual_search_operation(
        self,
        user_query: str,
        contextual_query: str,
        code_context: CodeGenerationContext,
        result: DualSearchResult,
        duration: float,
        success: bool = True,
        error_details: Optional[Dict[str, Any]] = None
    ):
        """Log a dual search operation."""
        log_entry = SearchOperationLog(
            timestamp=datetime.now(),
            operation_type="dual_search",
            user_query=user_query,
            contextual_query=contextual_query,
            code_context={
                "code_type": code_context.code_type,
                "framework": code_context.framework,
                "confidence": code_context.confidence
            } if code_context else None,
            results_count=len(result.merged_results) if result else 0,
            duration=duration,
            success=success,
            error_details=error_details,
            performance_metrics={
                "user_results": len(result.user_results) if result else 0,
                "context_results": len(result.context_results)
                if result else 0,
                "merged_results": len(result.merged_results) if result else 0
            }
        )

        self.operation_logs.append(log_entry)

        # Log to standard logger
        if success:
            _ = DEBUG and log_debug(
                f"Dual search completed: query='{user_query[:50]}...', "
                f"context={code_context.code_type if code_context
                           else 'none'}, "
                f"results={len(result.merged_results) if result else 0}, "
                f"duration={duration:.2f}s"
            )
        else:
            log_error(
                f"Dual search failed: query='{user_query[:50]}...', "
                f"error={error_details.get('message', 'Unknown')
                         if error_details else 'Unknown'}, "
                f"duration={duration:.2f}s"
            )

    def log_context_determination(
        self,
        user_query: str,
        determined_context: Optional[CodeGenerationContext],
        duration: float,
        success: bool = True,
        error_details: Optional[Dict[str, Any]] = None
    ):
        """Log a context determination operation."""
        log_entry = SearchOperationLog(
            timestamp=datetime.now(),
            operation_type="context_determination",
            user_query=user_query,
            code_context={
                "code_type": determined_context.code_type,
                "framework": determined_context.framework,
                "confidence": determined_context.confidence,
                "detected_patterns": determined_context.detected_patterns
            } if determined_context else None,
            duration=duration,
            success=success,
            error_details=error_details
        )

        self.operation_logs.append(log_entry)

        # Log to standard logger
        if success and determined_context:
            _ = DEBUG and log_debug(
                f"Context determined: query='{user_query[:50]}...', "
                f"type={determined_context.code_type}, "
                f"confidence={determined_context.confidence:.2f}, "
                f"duration={duration:.2f}s"
            )
        else:
            log_error(
                f"Context determination failed: query='{user_query[:50]}...', "
                f"error={error_details.get('message', 'Unknown')
                         if error_details else 'Unknown'}, "
                f"duration={duration:.2f}s"
            )

    def log_document_retrieval(
        self,
        document_paths: List[str],
        successful_retrievals: int,
        failed_retrievals: int,
        duration: float,
        success: bool = True,
        error_details: Optional[Dict[str, Any]] = None
    ):
        """Log a document retrieval operation."""
        log_entry = SearchOperationLog(
            timestamp=datetime.now(),
            operation_type="document_retrieval",
            results_count=successful_retrievals,
            duration=duration,
            success=success,
            error_details=error_details,
            performance_metrics={
                "total_requested": len(document_paths),
                "successful_retrievals": successful_retrievals,
                "failed_retrievals": failed_retrievals,
                "success_rate": successful_retrievals / len(document_paths)
                if document_paths else 0
            }
        )

        self.operation_logs.append(log_entry)

        # Log to standard logger
        if success:
            _ = DEBUG and log_debug(
                f"Document retrieval completed: "
                f"requested={len(document_paths)}, "
                f"successful={successful_retrievals}, "
                f"failed={failed_retrievals}, "
                f"duration={duration:.2f}s"
            )
        else:
            log_error(
                f"Document retrieval failed: "
                f"requested={len(document_paths)}, "
                f"error={error_details.get('message', 'Unknown')
                         if error_details else 'Unknown'}, "
                f"duration={duration:.2f}s"
            )

    def log_error(
        self,
        error: EnhancedSearchError,
        operation_context: Optional[Dict[str, Any]] = None
    ):
        """Log an enhanced search error with full context."""
        error_dict = error.to_dict()

        # Add operation context if provided
        if operation_context:
            error_dict["operation_context"] = operation_context

        # Log structured error
        log_error(
            f"Enhanced search error: {error.__class__.__name__} - {error}",
            extra={"error_details": error_dict}
        )

        # Log to operation logs
        log_entry = SearchOperationLog(
            timestamp=datetime.now(),
            operation_type="error",
            success=False,
            error_details=error_dict
        )
        self.operation_logs.append(log_entry)

    def get_operation_logs(
        self,
        operation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent operation logs."""
        logs = list(self.operation_logs)

        if operation_type:
            logs = [log for log in logs
                    if log.operation_type == operation_type]

        # Get most recent logs
        recent_logs = logs[-limit:]
        return [log.to_dict() for log in recent_logs]

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        error_logs = [
            log for log in self.operation_logs
            if log.timestamp >= cutoff_time and not log.success
        ]

        # Group errors by type
        error_counts = defaultdict(int)
        error_details = defaultdict(list)

        for log in error_logs:
            error_type = "unknown"
            if log.error_details and "error_type" in log.error_details:
                error_type = log.error_details["error_type"]

            error_counts[error_type] += 1
            error_details[error_type].append({
                "timestamp": log.timestamp.isoformat(),
                "operation_type": log.operation_type,
                "message": log.error_details.get("message", "Unknown")
                if log.error_details else "Unknown"
            })

        return {
            "time_period_hours": hours,
            "total_errors": len(error_logs),
            "error_counts": dict(error_counts),
            "error_details": dict(error_details)
        }


# Global instances
_performance_monitor = PerformanceMonitor()
_enhanced_search_logger = EnhancedSearchLogger(_performance_monitor)


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


def get_enhanced_search_logger() -> EnhancedSearchLogger:
    """Get the global enhanced search logger instance."""
    return _enhanced_search_logger


@contextmanager
def performance_tracking(operation_name: str, **metadata):
    """Context manager for tracking operation performance."""
    monitor = get_performance_monitor()
    metrics = monitor.start_operation(operation_name, **metadata)

    try:
        yield metrics
        monitor.complete_operation(metrics, success=True)
    except Exception as e:
        monitor.complete_operation(
            metrics, success=False, error_message=str(e))
        raise


def log_performance(operation_name: str):
    """Decorator for automatic performance logging."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with performance_tracking(operation_name, function=func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def handle_enhanced_search_errors(
    fallback_enabled: bool = True,
    fallback_value: Any = None,
    log_errors: bool = True
):
    """Decorator for handling enhanced search errors with fallback."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except EnhancedSearchError as e:
                if log_errors:
                    # logger = get_enhanced_search_logger()
                    log_error(e, {"function": func.__name__})

                if fallback_enabled:
                    log_warning(
                        f"Enhanced search error in {func.__name__}, "
                        f"using fallback: {e}"
                    )
                    return fallback_value
                else:
                    raise
            except Exception as e:
                # Convert unexpected exceptions to EnhancedSearchError
                enhanced_error = EnhancedSearchError(
                    f"[1] Unexpected error in {func.__name__}: {e}",
                    error_code="UNEXPECTED_ERROR",
                    original_exception=e
                )

                if log_errors:
                    # logger = get_enhanced_search_logger()
                    log_error(enhanced_error, {
                        "function": func.__name__})

                if fallback_enabled:
                    log_warning(
                        f"[2] Unexpected error in {func.__name__}, "
                        f"using fallback: {e}"
                    )
                    return fallback_value
                else:
                    raise enhanced_error
        return wrapper
    return decorator
