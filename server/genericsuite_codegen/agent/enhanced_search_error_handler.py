"""
Enhanced Search Error Handler Integration

This module provides centralized error handling and recovery mechanisms
for the enhanced vector search system, ensuring graceful fallback to
original search behavior when enhanced features fail.
"""

import logging
from typing import Any, Optional, Dict, Callable
from functools import wraps

from genericsuite_codegen.database.setup import SearchResult

from .enhanced_search_types import (
    EnhancedSearchError,
    DualSearchError,
    ContextDeterminationError,
    DocumentRetrievalError,
    TemplateLoadError,
    SearchMergeError,
    ConfigurationError,
    PerformanceError,
    CodeGenerationContext,
    DualSearchResult
)
from .enhanced_search_logging import (
    get_enhanced_search_logger,
    get_performance_monitor
)

logger = logging.getLogger(__name__)


class EnhancedSearchErrorHandler:
    """Centralized error handling for enhanced search operations."""

    def __init__(self, fallback_enabled: bool = True):
        """
        Initialize the error handler.

        Args:
            fallback_enabled: Whether to enable fallback to original behavior
        """
        self.fallback_enabled = fallback_enabled
        self.search_logger = get_enhanced_search_logger()
        self.performance_monitor = get_performance_monitor()

        # Error recovery strategies
        self.recovery_strategies = {
            DualSearchError: self._handle_dual_search_error,
            ContextDeterminationError: (
                self._handle_context_determination_error
            ),
            DocumentRetrievalError: self._handle_document_retrieval_error,
            TemplateLoadError: self._handle_template_load_error,
            SearchMergeError: self._handle_search_merge_error,
            ConfigurationError: self._handle_configuration_error,
            PerformanceError: self._handle_performance_error
        }

    def handle_error(
        self,
        error: Exception,
        operation_context: Optional[Dict[str, Any]] = None,
        fallback_value: Any = None
    ) -> Any:
        """
        Handle an enhanced search error with appropriate recovery strategy.

        Args:
            error: The exception that occurred
            operation_context: Context information about the operation
            fallback_value: Value to return if fallback is enabled

        Returns:
            Fallback value if recovery is possible, otherwise re-raises
        """
        # Log the error
        if isinstance(error, EnhancedSearchError):
            self.search_logger.log_error(error, operation_context)
        else:
            # Convert to EnhancedSearchError for consistent handling
            enhanced_error = EnhancedSearchError(
                f"Unexpected error: {error}",
                error_code="UNEXPECTED_ERROR",
                original_exception=error
            )
            self.search_logger.log_error(enhanced_error, operation_context)
            error = enhanced_error

        # Try specific recovery strategy
        error_type = type(error)
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](
                    error, operation_context, fallback_value
                )
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")

        # Generic fallback if enabled
        if self.fallback_enabled and fallback_value is not None:
            logger.warning(
                f"Using generic fallback for {error_type.__name__}: {error}"
            )
            return fallback_value

        # Re-raise if no recovery possible
        raise error

    def _handle_dual_search_error(
        self,
        error: DualSearchError,
        context: Optional[Dict[str, Any]],
        fallback_value: Any
    ) -> Any:
        """Handle dual search errors with fallback to single search."""
        if not self.fallback_enabled:
            raise error

        logger.warning(
            f"Dual search failed, attempting single search fallback: {error}")

        # Try to extract user query from error details
        user_query = (error.details.get('user_query')
                      if error.details else None)

        if user_query and context and 'kb_tool' in context:
            try:
                # Attempt single search as fallback
                kb_tool = context['kb_tool']
                search_results = kb_tool.search(query=user_query, limit=10)

                # Convert to expected format
                results = []
                for result_model in search_results.results:
                    search_result = SearchResult(
                        content=result_model.content,
                        metadata=result_model.metadata,
                        similarity_score=result_model.similarity_score,
                        document_path=result_model.document_path
                    )
                    results.append(search_result)

                # Create minimal dual search result
                fallback_result = DualSearchResult(
                    user_results=results,
                    context_results=[],
                    merged_results=results,
                    context_used=CodeGenerationContext(
                        code_type="generic",
                        confidence=0.0
                    ),
                    user_query=user_query,
                    contextual_query="",
                    total_results=len(results),
                    sources=[]
                )

                logger.info(
                    f"Single search fallback successful: {len(results)} results")
                return fallback_result

            except Exception as fallback_error:
                logger.error(
                    f"Single search fallback failed: {fallback_error}")

        return fallback_value

    def _handle_context_determination_error(
        self,
        error: ContextDeterminationError,
        context: Optional[Dict[str, Any]],
        fallback_value: Any
    ) -> Any:
        """Handle context determination errors with generic context fallback."""
        if not self.fallback_enabled:
            raise error

        logger.warning(
            f"Context determination failed, using generic context: {error}")

        # Return generic context as fallback
        generic_context = CodeGenerationContext(
            code_type="generic",
            framework=None,
            confidence=0.0,
            detected_patterns=[]
        )

        return generic_context if fallback_value is None else fallback_value

    def _handle_document_retrieval_error(
        self,
        error: DocumentRetrievalError,
        context: Optional[Dict[str, Any]],
        fallback_value: Any
    ) -> Any:
        """Handle document retrieval errors with empty results fallback."""
        if not self.fallback_enabled:
            raise error

        logger.warning(
            f"Document retrieval failed, using empty results: {error}")

        # Return empty list as fallback for document retrieval
        return [] if fallback_value is None else fallback_value

    def _handle_template_load_error(
        self,
        error: TemplateLoadError,
        context: Optional[Dict[str, Any]],
        fallback_value: Any
    ) -> Any:
        """Handle template loading errors with default templates fallback."""
        if not self.fallback_enabled:
            raise error

        logger.warning(
            f"Template loading failed, using default templates: {error}")

        # Default template fallback
        default_template = "examples and rules for creating code in Genericsuite"
        return default_template if fallback_value is None else fallback_value

    def _handle_search_merge_error(
        self,
        error: SearchMergeError,
        context: Optional[Dict[str, Any]],
        fallback_value: Any
    ) -> Any:
        """Handle search merge errors with user results fallback."""
        if not self.fallback_enabled:
            raise error

        logger.warning(
            f"Search merge failed, using user results only: {error}")

        # Try to return user results if available in error details
        if error.details and 'user_results' in error.details:
            user_results = error.details['user_results']
            if isinstance(user_results, list):
                return user_results

        return [] if fallback_value is None else fallback_value

    def _handle_configuration_error(
        self,
        error: ConfigurationError,
        context: Optional[Dict[str, Any]],
        fallback_value: Any
    ) -> Any:
        """Handle configuration errors with default configuration fallback."""
        if not self.fallback_enabled:
            raise error

        logger.warning(f"Configuration error, using defaults: {error}")

        # Return default configuration or fallback value
        return fallback_value

    def _handle_performance_error(
        self,
        error: PerformanceError,
        context: Optional[Dict[str, Any]],
        fallback_value: Any
    ) -> Any:
        """Handle performance errors with timeout fallback."""
        logger.warning(f"Performance threshold exceeded: {error}")

        # Performance errors are warnings, not failures
        # Continue with the operation but log the performance issue
        return fallback_value

    def create_error_handler_decorator(
        self,
        fallback_value: Any = None,
        operation_name: Optional[str] = None
    ):
        """Create a decorator for automatic error handling."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    operation_context = {
                        "function": func.__name__,
                        "operation_name": operation_name or func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }

                    return self.handle_error(
                        error=e,
                        operation_context=operation_context,
                        fallback_value=fallback_value
                    )
            return wrapper
        return decorator


# Global error handler instance
_global_error_handler = EnhancedSearchErrorHandler()


def get_error_handler() -> EnhancedSearchErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler


def with_error_handling(
    fallback_value: Any = None,
    operation_name: Optional[str] = None,
    fallback_enabled: bool = True
):
    """Decorator for automatic error handling with fallback."""
    handler = EnhancedSearchErrorHandler(fallback_enabled=fallback_enabled)
    return handler.create_error_handler_decorator(
        fallback_value=fallback_value,
        operation_name=operation_name
    )


def handle_enhanced_search_error(
    error: Exception,
    operation_context: Optional[Dict[str, Any]] = None,
    fallback_value: Any = None,
    fallback_enabled: bool = True
) -> Any:
    """
    Handle an enhanced search error with appropriate recovery.

    Args:
        error: The exception that occurred
        operation_context: Context about the operation
        fallback_value: Value to return on fallback
        fallback_enabled: Whether fallback is enabled

    Returns:
        Fallback value if recovery possible, otherwise re-raises
    """
    handler = EnhancedSearchErrorHandler(fallback_enabled=fallback_enabled)
    return handler.handle_error(error, operation_context, fallback_value)


class ErrorRecoveryStats:
    """Statistics tracking for error recovery operations."""

    def __init__(self):
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self.recovery_by_type = {}

    def record_recovery_attempt(self, error_type: str, success: bool):
        """Record a recovery attempt."""
        self.recovery_attempts += 1

        if success:
            self.successful_recoveries += 1
        else:
            self.failed_recoveries += 1

        if error_type not in self.recovery_by_type:
            self.recovery_by_type[error_type] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0
            }

        self.recovery_by_type[error_type]["attempts"] += 1
        if success:
            self.recovery_by_type[error_type]["successes"] += 1
        else:
            self.recovery_by_type[error_type]["failures"] += 1

    def get_recovery_rate(self) -> float:
        """Get overall recovery success rate."""
        if self.recovery_attempts == 0:
            return 0.0
        return self.successful_recoveries / self.recovery_attempts

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        return {
            "total_attempts": self.recovery_attempts,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "recovery_rate": self.get_recovery_rate(),
            "recovery_by_type": self.recovery_by_type
        }


# Global recovery stats instance
_recovery_stats = ErrorRecoveryStats()


def get_recovery_stats() -> ErrorRecoveryStats:
    """Get the global recovery statistics instance."""
    return _recovery_stats
