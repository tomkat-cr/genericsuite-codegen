"""
Knowledge Base Tools
"""

from typing import List, Dict, Any, Optional, Tuple

from genericsuite_codegen.database.setup import (
    get_database_manager,
    SearchResult,
    VectorSearchError,
    DatabaseConnectionError
)
from genericsuite_codegen.document_processing.embeddings import \
    create_embedding_generator

from genericsuite_codegen.utilities import (
    local_path_to_url, get_file_extension)
from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    log_warning,
    log_error,
)

from genericsuite_codegen.agent.types import (
    KnowledgeBaseSearchResults,
    SearchResultModel,
    ContextRanking,
)
from genericsuite_codegen.agent.agent_super import \
    get_tool_context_default_max_length
from genericsuite_codegen.agent.enhanced_search_types import (
    CodeGenerationContext,
    DualSearchResult
)
from genericsuite_codegen.agent.enhanced_search import EnhancedVectorSearch
from genericsuite_codegen.agent.context_determination import (
    ContextDeterminationService)
from genericsuite_codegen.agent.search_templates import SearchTemplateManager


DEBUG = True

TOOL_CONTEXT_DEFAULT_MAX_LENGTH = get_tool_context_default_max_length()


class KnowledgeBaseTool:
    """
    Knowledge base search tool for vector similarity search and context
    retrieval.

    Provides methods for searching the knowledge base, ranking results,
    and generating context summaries with source attribution.
    Enhanced with dual search capability for contextual GenericSuite rules.
    """

    def __init__(self, enable_enhanced_search: bool = True):
        """Initialize the knowledge base tool."""
        self.db_manager = get_database_manager()
        # self.db_manager = initialize_database()
        self.embedding_provider = None
        self._initialize_embedding_provider()

        # Enhanced search components
        self.enable_enhanced_search = enable_enhanced_search
        self.enhanced_search = None
        self.template_manager = None
        self.context_service = None

        if self.enable_enhanced_search:
            self._initialize_enhanced_search()

    def _initialize_embedding_provider(self) -> None:
        """Initialize the embedding provider for query vectorization."""
        try:
            self.embedding_provider = create_embedding_generator()
            _ = DEBUG and log_debug(
                "Initialized embedding provider for knowledge base tool")
        except Exception as e:
            log_error(
                f"Failed to initialize embedding provider: {e}")
            raise RuntimeError(
                f"Embedding provider initialization failed: {e}")

    def _initialize_enhanced_search(self) -> None:
        """Initialize enhanced search components."""
        try:
            # Initialize template manager
            self.template_manager = SearchTemplateManager()

            # Initialize context determination service
            self.context_service = ContextDeterminationService(
                self.template_manager
            )

            # Initialize enhanced vector search
            self.enhanced_search = EnhancedVectorSearch(
                kb_tool=self,
                template_manager=self.template_manager,
                context_service=self.context_service
            )

            _ = DEBUG and log_debug("Initialized enhanced search components")

        except Exception as e:
            log_warning(f"Failed to initialize enhanced search: {e}")
            self.enable_enhanced_search = False
            _ = DEBUG and log_debug(
                "Falling back to standard search functionality")

    def search(
        self, query: str, limit: int = 5,
        file_type_filter: Optional[str] = None
    ) -> KnowledgeBaseSearchResults:
        """
        Search the knowledge base using vector similarity.

        Args:
            query: Search query text.
            limit: Maximum number of results to return.
            file_type_filter: Optional filter by file type.

        Returns:
            KnowledgeBaseSearchResults: Search results with context and
            attribution.

        Raises:
            VectorSearchError: If search operation fails.
            DatabaseConnectionError: If database connection fails.
        """
        try:
            _ = DEBUG and log_debug(
                f"Searching knowledge base for query: '{query}' "
                f"(limit: {limit})")

            # Generate query embedding
            query_embedding = self.embedding_provider \
                .generate_query_embedding(query)

            # Perform vector search
            search_results = self.db_manager.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                file_type_filter=file_type_filter
            )

            # Convert to response models
            result_models = []
            sources = set()

            for result in search_results:
                # Extract file type from metadata or path
                file_type = result.metadata.get(
                    'file_type',
                    result.document_path.split('.')[-1]
                    if '.' in result.document_path else 'unknown')

                result_model = SearchResultModel(
                    content=result.content,
                    document_path=result.document_path,
                    similarity_score=result.similarity_score,
                    file_type=file_type,
                    metadata=result.metadata
                )
                result_models.append(result_model)
                sources.add(result.document_path)

            # Generate context summary
            context_summary = self._generate_context_summary(
                search_results, query)

            # Create final results
            final_results = KnowledgeBaseSearchResults(
                results=result_models,
                total_results=len(result_models),
                query=query,
                context_summary=context_summary,
                sources=list(sources)
            )

            _ = DEBUG and log_debug(
                f"Found {len(result_models)} results from "
                f"{len(sources)} sources")
            return final_results

        except VectorSearchError as e:
            log_error(f"Vector search failed: {e}")
            raise
        except DatabaseConnectionError as e:
            log_error(f"Database connection failed: {e}")
            raise
        except Exception as e:
            log_error(f"Unexpected error during search: {e}")
            raise VectorSearchError(f"Search operation failed: {e}")

    def _generate_context_summary(self, results: List[SearchResult],
                                  query: str) -> str:
        """
        Generate a summary of the retrieved context.

        Args:
            results: List of search results.
            query: Original search query.

        Returns:
            str: Context summary describing the retrieved information.
        """
        if not results:
            return f"No relevant context found for query: '{query}'"

        # Analyze file types and sources
        file_types = {}
        sources = set()
        total_content_length = 0

        for result in results:
            file_type = result.metadata.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            sources.add(result.document_path)
            total_content_length += len(result.content)

        # Generate summary
        summary_parts = [
            f"Retrieved {len(results)} relevant document chunks for query"
            f" '{query}'."
        ]

        if len(sources) > 1:
            summary_parts.append(
                f"Information sourced from {len(sources)} different "
                "documents.")

        if file_types:
            file_type_desc = ", ".join(
                [f"{count} {ftype}" for ftype, count in file_types.items()])
            summary_parts.append(f"Content types: {file_type_desc}.")

        # Add relevance information
        if results:
            avg_score = sum(r.similarity_score for r in results) / len(results)
            summary_parts.append(f"Average relevance score: {avg_score:.3f}.")

        return " ".join(summary_parts)

    def rank_context_by_relevance(
        self,
        results: List[SearchResult],
        query: str
    ) -> List[ContextRanking]:
        """
        Rank and score context results by relevance to the query.

        Args:
            results: List of search results to rank.
            query: Original search query for relevance scoring.

        Returns:
            List[ContextRanking]: Ranked context with relevance scores.
        """
        ranked_context = []
        query_lower = query.lower()

        for result in results:
            # Calculate enhanced relevance score
            relevance_score = self._calculate_relevance_score(
                result, query_lower
            )

            _ = DEBUG and log_debug(
                f">>> rank_context_by_relevance | Result: {result}")

            context_ranking = ContextRanking(
                content=result.content,
                relevance_score=relevance_score,
                source_path=result.document_path,
                file_type=result.metadata.get('file_type', result.metadata.get(
                    'original_document_type', 'unknown')),
                metadata=result.metadata
            )
            ranked_context.append(context_ranking)

        # Sort by relevance score (descending)
        ranked_context.sort(key=lambda x: x.relevance_score, reverse=True)

        _ = DEBUG and log_debug(
            ">>> rank_context_by_relevance | "
            f"Ranked context: {ranked_context}")

        return ranked_context

    def _calculate_relevance_score(self, result: SearchResult,
                                   query_lower: str) -> float:
        """
        Calculate enhanced relevance score combining similarity and text
        matching.

        Args:
            result: Search result to score.
            query_lower: Lowercase query for text matching.

        Returns:
            float: Enhanced relevance score.
        """
        # Base similarity score (0.0 to 1.0)
        base_score = result.similarity_score

        # Text matching bonus
        content_lower = result.content.lower()
        query_words = query_lower.split()

        # Exact phrase match bonus
        phrase_bonus = 0.1 if query_lower in content_lower else 0.0

        # Word match bonus
        word_matches = sum(1 for word in query_words if word in content_lower)
        word_bonus = (word_matches / len(query_words)) * \
            0.05 if query_words else 0.0

        # File type relevance bonus
        file_type = result.metadata.get('file_type', '')
        file_type_bonus = 0.0

        # Prioritize documentation and code files
        if file_type in ['md', 'rst', 'txt']:
            file_type_bonus = 0.02  # Documentation bonus
        elif file_type in ['py', 'js', 'ts', 'jsx', 'tsx']:
            file_type_bonus = 0.01  # Code file bonus

        # Content length penalty for very short or very long chunks
        content_length = len(result.content)
        length_penalty = 0.0

        if content_length < 50:  # Very short content
            length_penalty = -0.02
        elif content_length > 2000:  # Very long content
            length_penalty = -0.01

        # Calculate final score
        final_score = base_score + phrase_bonus + \
            word_bonus + file_type_bonus + length_penalty

        # Ensure score stays within reasonable bounds
        return max(0.0, min(1.0, final_score))

    def get_context_for_generation(
        self,
        query: str,
        max_context_length: int = None,
        file_type_filter: Optional[str] = None,
        limit: int = 10,
        enable_dual_search: bool = True,
        code_context: Optional[CodeGenerationContext] = None,
        full_content: bool = False
    ) -> Tuple[str, List[str]]:
        """
        Get formatted context for code generation with length limits.
        Enhanced with dual search capability for contextual GenericSuite rules.

        Args:
            query: Search query for context retrieval.
            max_context_length: Maximum total context length in characters.
            file_type_filter: Optional filter by file type.
            limit: Maximum number of results to return. Default is 10.
            enable_dual_search: Enable enhanced dual search if available.
            code_context: Optional pre-determined code generation context.

        Returns:
            Tuple[str, List[str]]: Formatted context string, list of
            sources (only the document paths), and raw_results.
        """
        if not max_context_length:
            max_context_length = TOOL_CONTEXT_DEFAULT_MAX_LENGTH

        try:
            if (self.enable_enhanced_search and
                enable_dual_search and
                    self.enhanced_search is not None):

                # Use enhanced search if available and enabled
                final_context, sources, raw_results = \
                    self._get_enhanced_context_for_generation(
                        query=query,
                        max_context_length=max_context_length,
                        file_type_filter=file_type_filter,
                        limit=limit,
                        code_context=code_context,
                    )
            else:
                # Fallback to standard search
                final_context, sources, raw_results = \
                    self._get_standard_context_for_generation(
                        query=query,
                        max_context_length=max_context_length,
                        file_type_filter=file_type_filter,
                        limit=limit
                    )

        except Exception as e:
            log_error(f"Failed to get context for generation: {e}")
            raise Exception(f"Error retrieving context: {e}")

        try:
            if full_content:
                # We need to read the full content of the files, for example
                # for the code and json files generation in the agent.query()
                # method
                final_context = ""
                for result in raw_results:
                    if get_file_extension(result.document_path) in ['pdf']:
                        continue
                    with open(result.document_path, 'r') as f:
                        file_content = f.read()
                        if len(final_context)+len(file_content) >= \
                           max_context_length:
                            break
                        final_context += file_content
                return final_context, sources, raw_results

            return final_context, sources, raw_results

        except Exception as e:
            log_error(f"Failed to get FULL context for generation: {e}")
            raise Exception(f"Error retrieving FULL context: {e}")

    def _get_enhanced_context_for_generation(
        self,
        query: str,
        max_context_length: int,
        file_type_filter: Optional[str],
        limit: int,
        code_context: Optional[CodeGenerationContext]
    ) -> Tuple[str, List[str], List[SearchResult]]:
        """
        Get context using enhanced dual search capability.

        Args:
            query: Search query for context retrieval.
            max_context_length: Maximum total context length in characters.
            file_type_filter: Optional filter by file type.
            limit: Maximum number of results to return.
            code_context: Optional pre-determined code generation context.

        Returns:
            Tuple[str, List[str]]: Formatted context string, list of sources,
            raw_results.
        """
        try:
            _ = DEBUG and log_debug(
                f"Using enhanced dual search for query: '{query}'")

            # Perform dual search
            dual_result = self.enhanced_search._sync_dual_search(
                user_query=query,
                code_context=code_context,
                max_context_length=max_context_length,
                file_type_filter=file_type_filter,
                limit=limit
            )

            # Use merged results for context generation
            search_results = dual_result.merged_results

            if not search_results:
                _ = DEBUG and log_debug(
                    "No results from enhanced search, trying standard search")
                return self._get_standard_context_for_generation(
                    query=query,
                    max_context_length=max_context_length,
                    file_type_filter=file_type_filter,
                    limit=limit
                )

            # Add context information about dual search
            if dual_result.contextual_query:
                context_header = (
                    f"Enhanced context for: {query}\n"
                    f"Contextual search: {dual_result.contextual_query}\n"
                    f"Context type: {dual_result.context_used.code_type}"
                )
                if dual_result.context_used.framework:
                    context_header += (
                        f" ({dual_result.context_used.framework})")
                context_header += (
                    f"\nConfidence: "
                    f"{dual_result.context_used.confidence:.2f}\n")
                context_header += "=" * 60 + "\n\n"
            else:
                context_header = (
                    f"Relevant context for: {query}\n" +
                    "=" * 50 + "\n\n")

            return self.get_content_sources_raw_results(
                query=query,
                context_header=context_header,
                search_results=search_results,
                max_context_length=max_context_length,
                title="Enhanced search"
            )

        except Exception as e:
            log_error(f"Enhanced context generation failed: {e}"
                      + "Falling back to standard search")
            return self._get_standard_context_for_generation(
                query=query,
                max_context_length=max_context_length,
                file_type_filter=file_type_filter,
                limit=limit
            )

    def get_content_sources_raw_results(
        self,
        query: str,
        context_header: str,
        search_results: List[SearchResult],
        max_context_length: int,
        title: str
    ) -> Tuple[str, List[str]]:

        # Rank results by relevance
        ranked_context = self.rank_context_by_relevance(
            search_results, query)

        # Build context string within length limits
        available_length = max_context_length - len(context_header)
        context_parts = []
        current_length = 0
        sources = []

        for context in ranked_context:
            # Format context entry
            source_info = f"Source: {context.source_path}"
            content_with_source = f"{source_info}\n{context.content}\n"

            # Check if adding this context would exceed the limit
            if (current_length + len(content_with_source) >
                    available_length):
                # Try to fit a truncated version
                remaining_space = (available_length -
                                   current_length - len(source_info) - 20)
                if remaining_space > 100:
                    # Only add if we have reasonable space
                    truncated_content = (
                        context.content[:remaining_space] + "...")
                    context_parts.append(
                        f"{source_info}\n{truncated_content}\n")
                    sources.append(context.source_path)
                break

            context_parts.append(content_with_source)
            sources.append(context.source_path)
            current_length += len(content_with_source)

        # Join all context parts
        formatted_context = "\n---\n".join(context_parts)
        final_context = context_header + formatted_context

        _ = DEBUG and log_debug(
            f"{title} returned {len(search_results)} "
            f"results from {len(list(set(sources)))} sources")

        # Remove duplicate sources and include dual search results
        return final_context, list(set(sources)), search_results

    def _get_standard_context_for_generation(
        self,
        query: str,
        max_context_length: int,
        file_type_filter: Optional[str],
        limit: int
    ) -> Tuple[str, List[str]]:
        """
        Get context using standard single search (backward compatibility).

        Args:
            query: Search query for context retrieval.
            max_context_length: Maximum total context length in characters.
            file_type_filter: Optional filter by file type.
            limit: Maximum number of results to return.

        Returns:
            Tuple[str, List[str]]: Formatted context string, list of sources,
            raw_results.
        """
        # Search for relevant context
        search_results = self.search(
            query=query,
            limit=limit,  # Get more results for better selection
            file_type_filter=file_type_filter
        )

        if not search_results.results:
            return "No relevant context found.", [], []

        # Rank results by relevance
        raw_results = [
            SearchResult(
                content=r.content,
                metadata=r.metadata,
                similarity_score=r.similarity_score,
                document_path=r.document_path
            )
            for r in search_results.results
        ]

        context_header = f"Relevant context for: {query}\n" + "=" * 50 + "\n\n"

        return self.get_content_sources_raw_results(
            query=query,
            context_header=context_header,
            search_results=raw_results,
            max_context_length=max_context_length,
            title="Standard search"
        )

    def search_similar_documents(
        self,
        query: str,
        limit: int = 10,
        file_type_filter: Optional[str] = None,
        similarity_threshold: float = 0.7,
        translate_path: bool = False
    ) -> KnowledgeBaseSearchResults:
        """
        Search for similar documents in the knowledge base
        with a similarity threshold.
        """
        search_results = self.search(
            query=query,
            limit=limit,
            file_type_filter=file_type_filter,
        )
        search_results.results = [
            SearchResultModel(
                content=r.content,
                document_path=r.document_path if not translate_path
                else local_path_to_url(r.document_path, True),
                similarity_score=r.similarity_score,
                file_type=r.file_type,
                metadata=r.metadata,
            ) for r in search_results.results
            if r.similarity_score >= similarity_threshold
        ]
        return search_results

    def set_enhanced_search_enabled(self, enabled: bool) -> None:
        """
        Enable or disable enhanced search functionality.

        Args:
            enabled: Whether to enable enhanced search
        """
        if enabled and not self.enable_enhanced_search:
            # Try to initialize enhanced search if not already done
            self._initialize_enhanced_search()

        self.enable_enhanced_search = enabled
        _ = DEBUG and log_debug(
            f"Enhanced search {'enabled' if enabled else 'disabled'}")

    def is_enhanced_search_available(self) -> bool:
        """
        Check if enhanced search is available and properly initialized.

        Returns:
            True if enhanced search is available, False otherwise
        """
        return (self.enable_enhanced_search and
                self.enhanced_search is not None and
                self.template_manager is not None and
                self.context_service is not None)

    def get_enhanced_search_info(self) -> Dict[str, Any]:
        """
        Get information about enhanced search capabilities.

        Returns:
            Dictionary with enhanced search information
        """
        if not self.is_enhanced_search_available():
            return {
                "available": False,
                "reason": "Enhanced search not initialized or disabled"
            }

        return {
            "available": True,
            "supported_code_types": (
                self.template_manager.get_supported_code_types()),
            "available_templates": (
                len(self.template_manager.get_all_templates())),
            "statistics": self.enhanced_search.get_search_statistics()
        }

    def perform_dual_search(
        self,
        user_query: str,
        code_context: Optional[CodeGenerationContext] = None,
        max_context_length: Optional[int] = None,
        file_type_filter: Optional[str] = None,
        limit: int = 10
    ) -> Optional[DualSearchResult]:
        """
        Perform dual search directly and return detailed results.

        Args:
            user_query: The user's search query
            code_context: Optional pre-determined code generation context
            max_context_length: Maximum context length
            file_type_filter: Optional file type filter
            limit: Maximum number of results

        Returns:
            DualSearchResult if enhanced search is available, None otherwise
        """
        if not max_context_length:
            max_context_length = TOOL_CONTEXT_DEFAULT_MAX_LENGTH

        if not self.is_enhanced_search_available():
            log_warning("Enhanced search not available for dual search")
            return None

        try:
            return self.enhanced_search._sync_dual_search(
                user_query=user_query,
                code_context=code_context,
                max_context_length=max_context_length,
                file_type_filter=file_type_filter,
                limit=limit
            )
        except Exception as e:
            log_error(f"Dual search failed: {e}")
            return None

    def determine_code_context(
        self,
        user_query: str,
        task_type: Optional[str] = None
    ) -> Optional[CodeGenerationContext]:
        """
        Determine code generation context from user query.

        Args:
            user_query: The user's query
            task_type: Optional task type hint

        Returns:
            CodeGenerationContext if context service is available,
            None otherwise
        """
        if not self.is_enhanced_search_available():
            log_warning("Context determination not available")
            return None

        try:
            return self.context_service.determine_context(
                user_query=user_query,
                task_type=task_type
            )
        except Exception as e:
            log_error(f"Context determination failed: {e}")
            return None
