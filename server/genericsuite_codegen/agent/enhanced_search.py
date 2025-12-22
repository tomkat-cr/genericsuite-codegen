"""
Enhanced Vector Search Engine for GenericSuite CodeGen

This module provides enhanced vector search capabilities that combine user
queries with contextual GenericSuite rules and patterns to ensure generated
code follows established conventions.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import KnowledgeBaseTool
from dataclasses import asdict

from genericsuite_codegen.database.setup import (
    SearchResult, VectorSearchError
)

from .enhanced_search_types import (
    CodeGenerationContext,
    DualSearchResult,
    EnhancedSearchConfig,
    DualSearchError,
    SearchMergeError,
    MERGE_PRIORITY_WEIGHTS
)
from .context_determination import ContextDeterminationService
from .search_templates import SearchTemplateManager
from .enhanced_search_logging import (
    get_enhanced_search_logger,
    performance_tracking,
    log_performance,
    handle_enhanced_search_errors
)
# Avoid circular import - KnowledgeBaseTool will be passed as parameter

logger = logging.getLogger(__name__)


class EnhancedVectorSearch:
    """Enhanced vector search with dual query capability."""

    def __init__(
        self,
        kb_tool: "KnowledgeBaseTool",
        template_manager: Optional[SearchTemplateManager] = None,
        context_service: Optional[ContextDeterminationService] = None,
        config: Optional[EnhancedSearchConfig] = None
    ):
        """
        Initialize the EnhancedVectorSearch.

        Args:
            kb_tool: KnowledgeBaseTool instance for vector search operations
            template_manager: Optional SearchTemplateManager instance
            context_service: Optional ContextDeterminationService instance
            config: Optional EnhancedSearchConfig for customization
        """
        self.kb_tool = kb_tool
        self.template_manager = template_manager or SearchTemplateManager()
        self.context_service = (context_service or
                                ContextDeterminationService(
                                    self.template_manager))

        # Use provided config or create default
        self.config = config or EnhancedSearchConfig(
            templates=self.template_manager.get_all_templates(),
            local_repo_path="local_repo_files",
            max_context_length=10000,
            fallback_enabled=True,
            context_determination_enabled=True,
            document_retrieval_enabled=True,
            search_result_limit=10,
            similarity_threshold=0.7,
            merge_strategy="prioritize_context"
        )

        logger.info("Initialized EnhancedVectorSearch with dual search "
                    "capability")

    @log_performance("dual_search")
    @handle_enhanced_search_errors(fallback_enabled=True, fallback_value=None)
    async def dual_search(
        self,
        user_query: str,
        code_context: Optional[CodeGenerationContext] = None,
        max_context_length: int = None,
        file_type_filter: Optional[str] = None,
        limit: int = None
    ) -> DualSearchResult:
        """
        Perform dual search: user query + contextual rules.

        Args:
            user_query: The user's search query
            code_context: Optional pre-determined code generation context
            max_context_length: Optional max context length override
            file_type_filter: Optional file type filter
            limit: Optional result limit override

        Returns:
            DualSearchResult: Combined results from both searches

        Raises:
            DualSearchError: If dual search operation fails
        """
        try:
            result = self._sync_dual_search(
                user_query=user_query,
                code_context=code_context,
                max_context_length=max_context_length,
                file_type_filter=file_type_filter,
                limit=limit
            )

            return result

        except Exception as e:
            # Log the error with context
            error_details = {
                "user_query": user_query,
                "code_context": code_context.code_type if code_context
                else None,
                "file_type_filter": file_type_filter,
                "limit": limit
            }

            if isinstance(e, DualSearchError):
                raise
            else:
                raise DualSearchError(
                    f"Dual search operation failed: {e}",
                    user_query=user_query,
                    details=error_details,
                    original_exception=e
                )

    @log_performance("search_merge")
    @handle_enhanced_search_errors(fallback_enabled=True, fallback_value=[])
    def merge_search_results(
        self,
        user_results: List[SearchResult],
        context_results: List[SearchResult],
        merge_strategy: str = "prioritize_context"
    ) -> List[SearchResult]:
        """
        Merge and prioritize results from both searches.

        Args:
            user_results: Results from user query search
            context_results: Results from contextual rules search
            merge_strategy: Strategy for merging results

        Returns:
            List[SearchResult]: Merged and prioritized results

        Raises:
            SearchMergeError: If result merging fails
        """
        with performance_tracking(
            "search_merge_detailed",
            merge_strategy=merge_strategy,
            user_results_count=len(user_results),
            context_results_count=len(context_results)
        ):
            try:
                if not user_results and not context_results:
                    logger.info("No search results to merge")
                    return []

                # If only one type of results, return them
                if not context_results:
                    logger.info(
                        "No context results, returning user results only")
                    return user_results[:self.config.search_result_limit]

                if not user_results:
                    logger.info(
                        "No user results, returning context results only")
                    return context_results[:self.config.search_result_limit]

                # Validate merge strategy
                if merge_strategy not in MERGE_PRIORITY_WEIGHTS:
                    logger.warning(
                        f"Invalid merge strategy '{merge_strategy}'"
                        ", using default 'prioritize_context'")
                    merge_strategy = "prioritize_context"

                # Get merge weights based on strategy
                weights = MERGE_PRIORITY_WEIGHTS[merge_strategy]

                # Create combined results with adjusted scores
                combined_results = []
                seen_paths = set()

                # Add context results with priority weighting
                for result in context_results:
                    if result.document_path not in seen_paths:
                        # Boost context result scores
                        adjusted_result = SearchResult(
                            content=result.content,
                            metadata=result.metadata,
                            similarity_score=min(1.0, result.similarity_score *
                                                 (1 + weights["context"])),
                            document_path=result.document_path
                        )
                        combined_results.append(adjusted_result)
                        seen_paths.add(result.document_path)

                # Add user results with user weighting
                for result in user_results:
                    if result.document_path not in seen_paths:
                        # Apply user result weighting
                        adjusted_result = SearchResult(
                            content=result.content,
                            metadata=result.metadata,
                            similarity_score=min(1.0, result.similarity_score *
                                                 (1 + weights["user"])),
                            document_path=result.document_path
                        )
                        combined_results.append(adjusted_result)
                        seen_paths.add(result.document_path)
                    else:
                        # If document already exists from context results,
                        # potentially merge or keep the higher scored one
                        existing_idx = next(
                            (i for i, r in enumerate(combined_results)
                             if r.document_path == result.document_path),
                            None
                        )

                        if existing_idx is not None:
                            existing_result = combined_results[existing_idx]
                            user_adjusted_score = min(
                                1.0, result.similarity_score *
                                (1 + weights["user"]))

                            # Keep the result with higher adjusted score
                            if (user_adjusted_score >
                                    existing_result.similarity_score):
                                combined_results[existing_idx] = SearchResult(
                                    content=result.content,
                                    metadata=result.metadata,
                                    similarity_score=user_adjusted_score,
                                    document_path=result.document_path
                                )

                # Sort by adjusted similarity score (descending)
                combined_results.sort(key=lambda x: x.similarity_score,
                                      reverse=True)

                # Apply similarity threshold filter
                filtered_results = [
                    result for result in combined_results
                    if result.similarity_score >=
                    self.config.similarity_threshold
                ]

                # Limit results
                final_results = filtered_results[
                    : self.config.search_result_limit]

                logger.info("Merged results using "
                            f"'{merge_strategy}' strategy: "
                            f"{len(combined_results)} combined -> "
                            f"{len(filtered_results)} after threshold -> "
                            f"{len(final_results)} final")

                return final_results

            except Exception as e:
                logger.error(f"Failed to merge search results: {e}")
                raise SearchMergeError(
                    f"Result merging failed: {e}",
                    merge_strategy=merge_strategy,
                    user_results_count=len(user_results),
                    context_results_count=len(context_results),
                    original_exception=e
                )

    @log_performance("vector_search")
    @handle_enhanced_search_errors(fallback_enabled=True, fallback_value=[])
    def _perform_search(
        self,
        query: str,
        file_type_filter: Optional[str] = None,
        limit: int = None
    ) -> List[SearchResult]:
        """
        Perform a single vector search operation.

        Args:
            query: Search query
            file_type_filter: Optional file type filter
            limit: Optional result limit

        Returns:
            List[SearchResult]: Search results
        """
        with performance_tracking(
            "vector_search_detailed",
            query_length=len(query),
            file_type_filter=file_type_filter,
            limit=limit
        ):
            try:
                # Validate query
                if not query or not query.strip():
                    raise DualSearchError(
                        "Empty search query provided",
                        error_code="EMPTY_QUERY"
                    )

                search_limit = limit or self.config.search_result_limit

                # Check for reasonable limits
                if search_limit > 100:
                    logger.warning(
                        f"Large search limit requested: {search_limit}")
                    search_limit = 100

                # Use KnowledgeBaseTool to perform the search
                search_results = self.kb_tool.search(
                    query=query,
                    limit=search_limit,
                    file_type_filter=file_type_filter
                )

                # Convert KnowledgeBaseSearchResults to List[SearchResult]
                results = []
                for result_model in search_results.results:
                    search_result = SearchResult(
                        content=result_model.content,
                        metadata=result_model.metadata,
                        similarity_score=result_model.similarity_score,
                        document_path=result_model.document_path
                    )
                    results.append(search_result)

                logger.debug("Vector search completed: "
                             f"query='{query[:50]}...', "
                             f"results={len(results)}, "
                             f"filter={file_type_filter}")

                return results

            except VectorSearchError as e:
                logger.error(
                    f"Vector search failed for query '{query[:50]}...': {e}")
                if self.config.fallback_enabled:
                    logger.info("Using fallback: returning empty results")
                    return []  # Return empty results as fallback
                raise DualSearchError(
                    f"Vector search failed: {e}",
                    error_code="VECTOR_SEARCH_ERROR",
                    original_exception=e
                )
            except Exception as e:
                logger.error(
                    "Search operation failed for query "
                    f"'{query[:50]}...': {e}")
                if self.config.fallback_enabled:
                    logger.info("Using fallback: returning empty results")
                    return []  # Return empty results as fallback
                raise DualSearchError(
                    f"Search operation failed: {e}",
                    error_code="SEARCH_OPERATION_ERROR",
                    original_exception=e
                )

    def _sync_dual_search(
        self,
        user_query: str,
        code_context: Optional[CodeGenerationContext] = None,
        max_context_length: int = None,
        file_type_filter: Optional[str] = None,
        limit: int = None
    ) -> DualSearchResult:
        """
        Synchronous version of dual search for compatibility.
        """
        search_logger = get_enhanced_search_logger()

        with performance_tracking(
            "dual_search_complete",
            user_query_length=len(user_query),
            has_code_context=code_context is not None,
            file_type_filter=file_type_filter
        ) as metrics:
            try:
                # Validate input
                if not user_query or not user_query.strip():
                    raise DualSearchError(
                        "Empty or invalid user query provided",
                        user_query=user_query,
                        error_code="EMPTY_QUERY"
                    )

                # Use provided limits or defaults from config
                result_limit = limit or self.config.search_result_limit

                # Determine context if not provided
                if (code_context is None and
                        self.config.context_determination_enabled):
                    try:
                        code_context = self.context_service.determine_context(
                            user_query)
                        logger.info(f"Determined context: "
                                    f"{code_context.code_type} "
                                    "(confidence: "
                                    f"{code_context.confidence:.2f})")
                    except Exception as e:
                        logger.warning(f"Context determination failed: {e}")
                        # Create generic context as fallback
                        code_context = CodeGenerationContext(
                            code_type="generic",
                            confidence=0.0
                        )

                # Perform user query search
                logger.info(
                    f"Performing user query search: '{user_query[:50]}...'")
                user_results = self._perform_search(
                    query=user_query,
                    file_type_filter=file_type_filter,
                    limit=result_limit
                )

                # Perform contextual search if context is available
                context_results = []
                contextual_query = ""

                if (code_context and
                    code_context.code_type != "generic" and
                        code_context.confidence > 0.2):

                    try:
                        contextual_query = (
                            self.context_service.get_contextual_search_query(
                                code_context))

                        # Use file type filter from template if not provided
                        context_file_filter = (
                            file_type_filter or
                            self.template_manager.get_file_type_filter(
                                code_context.code_type))

                        logger.info(f"Performing contextual search: "
                                    f"'{contextual_query[:50]}...'")
                        context_results = self._perform_search(
                            query=contextual_query,
                            file_type_filter=context_file_filter,
                            limit=result_limit
                        )

                    except Exception as e:
                        logger.warning(f"Contextual search failed: {e}")
                        if not self.config.fallback_enabled:
                            raise DualSearchError(
                                f"Contextual search failed: {e}",
                                user_query=user_query,
                                contextual_query=contextual_query,
                                search_phase="contextual_search",
                                original_exception=e
                            )

                # Merge search results
                merged_results = self.merge_search_results(
                    user_results=user_results,
                    context_results=context_results,
                    merge_strategy=self.config.merge_strategy
                )

                # Create dual search result
                dual_result = DualSearchResult(
                    user_results=user_results,
                    context_results=context_results,
                    merged_results=merged_results,
                    context_used=code_context,
                    user_query=user_query,
                    contextual_query=contextual_query,
                    total_results=len(merged_results),
                    sources=[]  # Will be populated in __post_init__
                )

                # Log successful operation
                search_logger.log_dual_search_operation(
                    user_query=user_query,
                    contextual_query=contextual_query,
                    code_context=code_context,
                    result=dual_result,
                    duration=metrics.duration or 0.0,
                    success=True
                )

                logger.info(f"Dual search completed: {len(user_results)} user "
                            f"results, {len(context_results)} context results,"
                            f" {len(merged_results)} merged results")

                return dual_result

            except DualSearchError as e:
                # Log failed operation
                search_logger.log_dual_search_operation(
                    user_query=user_query,
                    contextual_query=contextual_query,
                    code_context=code_context,
                    result=None,
                    duration=metrics.duration or 0.0,
                    success=False,
                    error_details=e.to_dict()
                )
                raise
            except Exception as e:
                # Convert to DualSearchError and log
                dual_error = DualSearchError(
                    f"Dual search failed: {e}",
                    user_query=user_query,
                    contextual_query=contextual_query,
                    original_exception=e
                )

                search_logger.log_dual_search_operation(
                    user_query=user_query,
                    contextual_query=contextual_query,
                    code_context=code_context,
                    result=None,
                    duration=metrics.duration or 0.0,
                    success=False,
                    error_details=dual_error.to_dict()
                )

                logger.error(f"Dual search operation failed: {e}")
                raise dual_error

    def update_config(self, new_config: EnhancedSearchConfig) -> None:
        """
        Update the enhanced search configuration.

        Args:
            new_config: New configuration to apply
        """
        self.config = new_config
        logger.info("Updated enhanced search configuration")

    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about search operations.

        Returns:
            Dictionary with search statistics
        """
        return {
            "config": asdict(self.config),
            "supported_code_types": (
                self.template_manager.get_supported_code_types()),
            "available_templates": len(
                self.template_manager.get_all_templates()),
            "merge_strategies": list(MERGE_PRIORITY_WEIGHTS.keys())
        }

    def validate_search_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a search query.

        Args:
            query: Query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not isinstance(query, str):
            return False, "Query must be a non-empty string"

        query = query.strip()
        if len(query) < 3:
            return False, "Query must be at least 3 characters long"

        if len(query) > 1000:
            return False, "Query must be less than 1000 characters"

        return True, None
