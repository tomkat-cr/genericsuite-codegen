"""
Context Determination Service for Enhanced Vector Search

This module provides intelligent context determination for code generation
based on user queries, automatically selecting appropriate GenericSuite
rule searches and templates.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass

from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    log_error,
)

from .enhanced_search_types import (
    CodeGenerationContext,
    ContextDeterminationError,
    CONTEXT_PATTERNS,
    DEFAULT_SEARCH_TEMPLATES
)
from .search_templates import SearchTemplateManager
from .enhanced_search_logging import (
    get_enhanced_search_logger,
    performance_tracking,
    log_performance,
    handle_enhanced_search_errors
)


DEBUG = False


@dataclass
class ContextAnalysis:
    """Detailed analysis of context determination."""
    detected_patterns: List[str]
    pattern_scores: Dict[str, float]
    confidence_factors: Dict[str, float]
    ambiguity_score: float
    final_confidence: float


class ContextDeterminationService:
    """Service to determine code generation context from user queries."""

    def __init__(
        self,
        template_manager: Optional[SearchTemplateManager] = None
    ):
        """
        Initialize the ContextDeterminationService.

        Args:
            template_manager: Optional SearchTemplateManager instance.
                            If None, creates a default instance.
        """
        self.template_manager = (template_manager or
                                 SearchTemplateManager())

        # Confidence thresholds
        self.HIGH_CONFIDENCE_THRESHOLD = 0.8
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.5
        self.LOW_CONFIDENCE_THRESHOLD = 0.2

        # Pattern matching weights
        self.EXACT_MATCH_WEIGHT = 1.0
        self.PARTIAL_MATCH_WEIGHT = 0.7
        self.CONTEXT_MATCH_WEIGHT = 0.5
        self.FRAMEWORK_MATCH_WEIGHT = 0.8

        # Framework detection patterns
        self.FRAMEWORK_PATTERNS = {
            "react": ["react", "jsx", "tsx", "component", "hook", "state"],
            "fastapi": ["fastapi", "pydantic", "endpoint", "route", "api"],
            "flask": ["flask", "blueprint", "route", "app"],
            "chalice": ["chalice", "aws", "lambda", "serverless"],
            "langchain": ["langchain", "chain", "agent", "tool", "llm"],
            "fastmcp": ["fastmcp", "mcp", "model context protocol", "server"],
            "mongodb": ["mongodb", "mongo", "collection", "document"],
            "pydantic": ["pydantic", "model", "validation", "schema"]
        }

    @log_performance("context_determination")
    @handle_enhanced_search_errors(fallback_enabled=True, fallback_value=None)
    def determine_context(
        self,
        user_query: str,
        task_type: Optional[str] = None
    ) -> CodeGenerationContext:
        """
        Determine the appropriate context for code generation.

        Args:
            user_query: The user's query or request
            task_type: Optional explicit task type hint

        Returns:
            CodeGenerationContext with determined context information

        Raises:
            ContextDeterminationError: If context determination fails
        """
        search_logger = get_enhanced_search_logger()

        with performance_tracking(
            "context_determination_detailed",
            query_length=len(user_query),
            has_task_type=task_type is not None
        ) as metrics:
            try:
                if not user_query or not user_query.strip():
                    raise ContextDeterminationError(
                        "Empty or invalid user query provided",
                        query=user_query,
                        error_code="EMPTY_QUERY"
                    )

                # Normalize query for analysis
                normalized_query = self._normalize_query(user_query)

                # Perform context analysis
                analysis = self._analyze_query_context(
                    normalized_query, task_type)

                # Determine primary code type
                code_type = self._determine_primary_code_type(analysis)

                # Detect framework
                framework = self._detect_framework(normalized_query, code_type)

                # Calculate final confidence
                confidence = self._calculate_final_confidence(
                    analysis, framework)

                # Create context object
                context = CodeGenerationContext(
                    code_type=code_type,
                    framework=framework,
                    confidence=confidence,
                    detected_patterns=analysis.detected_patterns
                )

                # Log successful operation
                search_logger.log_context_determination(
                    user_query=user_query,
                    determined_context=context,
                    duration=metrics.duration or 0.0,
                    success=True
                )

                _ = DEBUG and log_debug(
                    f"Determined context: {code_type} "
                    f"(framework: {framework}, confidence: {confidence:.2f})"
                )

                return context

            except ContextDeterminationError as e:
                # Log failed operation
                search_logger.log_context_determination(
                    user_query=user_query,
                    determined_context=None,
                    duration=metrics.duration or 0.0,
                    success=False,
                    error_details=e.to_dict()
                )
                raise
            except Exception as e:
                # Convert to ContextDeterminationError and log
                context_error = ContextDeterminationError(
                    f"Failed to determine context: {e}",
                    query=user_query,
                    analysis_data={"task_type": task_type},
                    original_exception=e
                )

                search_logger.log_context_determination(
                    user_query=user_query,
                    determined_context=None,
                    duration=metrics.duration or 0.0,
                    success=False,
                    error_details=context_error.to_dict()
                )

                log_error(f"Context determination failed: {e}")
                raise context_error

    @log_performance("contextual_query_generation")
    @handle_enhanced_search_errors(
        fallback_enabled=True,
        fallback_value="examples and rules for creating code in Genericsuite"
    )
    def get_contextual_search_query(
        self,
        context: CodeGenerationContext
    ) -> str:
        """
        Get the appropriate contextual search query for the context.

        Args:
            context: CodeGenerationContext with determined context

        Returns:
            Contextual search query string

        Raises:
            ContextDeterminationError: If query generation fails
        """
        try:
            if not context:
                raise ContextDeterminationError(
                    "No context provided for query generation",
                    error_code="NO_CONTEXT"
                )

            # Get template from template manager
            template = self.template_manager.get_template(context.code_type)

            if not template:
                # Fallback to default templates
                template = DEFAULT_SEARCH_TEMPLATES.get(
                    context.code_type,
                    DEFAULT_SEARCH_TEMPLATES.get(
                        "generic",
                        "examples and rules for creating code in Genericsuite"
                    )
                )
                _ = DEBUG and log_debug(
                    f"Using default template for {context.code_type}")

            # Enhance template with framework-specific information if available
            if (context.framework and
                    context.confidence > self.MEDIUM_CONFIDENCE_THRESHOLD):
                enhanced_template = self._enhance_template_with_framework(
                    template, context.framework
                )
                _ = DEBUG and log_debug(
                    f"Enhanced template with framework: {context.framework}")
                return enhanced_template

            return template

        except Exception as e:
            log_error(f"Failed to get contextual search query: {e}")
            raise ContextDeterminationError(
                f"Failed to generate contextual search query: {e}",
                analysis_data={
                    "code_type": context.code_type if context else None,
                    "framework": context.framework if context else None,
                    "confidence": context.confidence if context else None
                },
                original_exception=e
            )

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent analysis."""
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', query.lower().strip())

        # Remove common stop words that don't affect context
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'how', 'what', 'when',
            'where', 'why', 'can', 'could', 'should', 'would', 'will'
        }

        words = normalized.split()
        filtered_words = [word for word in words if word not in stop_words]

        return ' '.join(filtered_words)

    def _analyze_query_context(
        self,
        normalized_query: str,
        task_type: Optional[str] = None
    ) -> ContextAnalysis:
        """Analyze query to determine context patterns and scores."""
        detected_patterns = []
        pattern_scores = {}
        confidence_factors = {}

        # Analyze each code type pattern
        for code_type, patterns in CONTEXT_PATTERNS.items():
            score = self._calculate_pattern_score(normalized_query, patterns)
            pattern_scores[code_type] = score

            if score > 0:
                detected_patterns.extend([p for p in patterns
                                          if p in normalized_query])

        # Factor in explicit task type if provided
        if task_type and task_type in CONTEXT_PATTERNS:
            pattern_scores[task_type] *= 1.5  # Boost explicit task type
            confidence_factors['explicit_task_type'] = 0.3

        # Calculate ambiguity score (how many types have significant scores)
        significant_scores = [score for score in pattern_scores.values()
                              if score > self.LOW_CONFIDENCE_THRESHOLD]
        ambiguity_score = len(significant_scores) / len(pattern_scores)

        # Calculate confidence factors
        max_score = max(pattern_scores.values()) if pattern_scores else 0
        confidence_factors['max_pattern_score'] = max_score * 0.4
        confidence_factors['pattern_clarity'] = (1 - ambiguity_score) * 0.3

        final_confidence = sum(confidence_factors.values())

        return ContextAnalysis(
            detected_patterns=list(set(detected_patterns)),
            pattern_scores=pattern_scores,
            confidence_factors=confidence_factors,
            ambiguity_score=ambiguity_score,
            final_confidence=min(final_confidence, 1.0)
        )

    def _calculate_pattern_score(
        self,
        query: str,
        patterns: List[str]
    ) -> float:
        """Calculate score for a set of patterns against the query."""
        total_score = 0.0
        query_words = set(query.split())

        for pattern in patterns:
            pattern_words = pattern.split()

            # Exact phrase match
            if pattern in query:
                total_score += self.EXACT_MATCH_WEIGHT
                continue

            # Partial word matches
            matching_words = sum(1 for word in pattern_words
                                 if word in query_words)
            if matching_words > 0:
                partial_score = (matching_words / len(pattern_words)
                                 ) * self.PARTIAL_MATCH_WEIGHT
                total_score += partial_score

        # Normalize by number of patterns
        return min(total_score / len(patterns), 1.0) if patterns else 0.0

    def _determine_primary_code_type(self, analysis: ContextAnalysis) -> str:
        """Determine the primary code type from analysis."""
        if not analysis.pattern_scores:
            return "generic"

        # Find the code type with highest score
        max_score = max(analysis.pattern_scores.values())

        if max_score < self.LOW_CONFIDENCE_THRESHOLD:
            return "generic"

        # Get all code types with the maximum score
        top_types = [
            code_type for code_type, score in analysis.pattern_scores.items()
            if score == max_score
        ]

        # If there's a tie, use priority order
        priority_order = [
            "json", "mcp", "langchain", "backend_ai", "frontend_ai",
            "backend", "frontend"
        ]

        for priority_type in priority_order:
            if priority_type in top_types:
                return priority_type

        # Return first if no priority match
        return top_types[0]

    def _detect_framework(
        self,
        normalized_query: str,
        code_type: str
    ) -> Optional[str]:
        """Detect specific framework from query and code type."""
        framework_scores = {}

        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            score = self._calculate_pattern_score(normalized_query, patterns)
            if score > 0:
                framework_scores[framework] = score

        if not framework_scores:
            return None

        # Get framework with highest score
        best_framework = max(framework_scores.items(), key=lambda x: x[1])

        # Only return if score is significant and relevant to code type
        if (best_framework[1] > self.LOW_CONFIDENCE_THRESHOLD and
                self._is_framework_relevant(best_framework[0], code_type)):
            return best_framework[0]

        return None

    def _is_framework_relevant(self, framework: str, code_type: str) -> bool:
        """Check if framework is relevant to the code type."""
        relevance_map = {
            "json": [],
            "langchain": ["langchain", "pydantic", "mongodb"],
            "mcp": ["fastmcp", "pydantic"],
            "frontend": ["react"],
            "frontend_ai": ["react"],
            "backend": [
                "fastapi", "flask", "chalice", "pydantic", "mongodb"
            ],
            "backend_ai": [
                "fastapi", "flask", "chalice", "pydantic", "mongodb",
                "langchain"
            ],
            "generic": list(self.FRAMEWORK_PATTERNS.keys())
        }

        relevant_frameworks = relevance_map.get(code_type, [])
        return framework in relevant_frameworks

    def _calculate_final_confidence(
        self,
        analysis: ContextAnalysis,
        framework: Optional[str]
    ) -> float:
        """Calculate final confidence score."""
        base_confidence = analysis.final_confidence

        # Boost confidence if framework detected
        if framework:
            base_confidence += 0.1

        # Reduce confidence for high ambiguity
        if analysis.ambiguity_score > 0.5:
            base_confidence *= (1 - analysis.ambiguity_score * 0.3)

        return min(base_confidence, 1.0)

    def _enhance_template_with_framework(
        self,
        template: str,
        framework: str
    ) -> str:
        """Enhance search template with framework-specific information."""
        framework_enhancements = {
            "react": "React components and hooks",
            "fastapi": "FastAPI endpoints and Pydantic models",
            "flask": "Flask applications and blueprints",
            "chalice": "AWS Chalice serverless applications",
            "langchain": "LangChain tools and agents",
            "fastmcp": "FastMCP server tools and resources",
            "mongodb": "MongoDB collections and queries",
            "pydantic": "Pydantic models and validation"
        }

        enhancement = framework_enhancements.get(framework)
        if enhancement:
            return f"{template} with {enhancement}"

        return template

    def get_confidence_level(self, confidence: float) -> str:
        """Get human-readable confidence level."""
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        elif confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return "medium"
        elif confidence >= self.LOW_CONFIDENCE_THRESHOLD:
            return "low"
        else:
            return "very_low"

    def analyze_query_details(
        self,
        user_query: str,
        task_type: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Provide detailed analysis of query for debugging and monitoring.

        Args:
            user_query: The user's query
            task_type: Optional task type hint

        Returns:
            Dictionary with detailed analysis information
        """
        try:
            normalized_query = self._normalize_query(user_query)
            analysis = self._analyze_query_context(normalized_query, task_type)
            context = self.determine_context(user_query, task_type)

            return {
                "original_query": user_query,
                "normalized_query": normalized_query,
                "detected_patterns": analysis.detected_patterns,
                "pattern_scores": analysis.pattern_scores,
                "confidence_factors": analysis.confidence_factors,
                "ambiguity_score": analysis.ambiguity_score,
                "determined_context": {
                    "code_type": context.code_type,
                    "framework": context.framework,
                    "confidence": context.confidence,
                    "confidence_level": self.get_confidence_level(
                        context.confidence
                    )
                },
                "contextual_search_query": self.get_contextual_search_query(
                    context
                )
            }

        except Exception as e:
            log_error(f"Failed to analyze query details: {e}")
            return {
                "error": str(e),
                "original_query": user_query
            }
