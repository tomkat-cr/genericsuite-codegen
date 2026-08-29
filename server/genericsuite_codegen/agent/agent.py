"""
Core Pydantic AI agent for GenericSuite CodeGen.

This module implements the main AI agent using Pydantic AI framework,
integrating knowledge base search, code generation capabilities, and
LLM provider configuration for GenericSuite development assistance.
"""

from typing import Dict, Any, Optional, List, Tuple

from pydantic_ai import Agent
# from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings

from genericsuite_codegen.agent.agent_super import (
    AgentSuper,
    create_agent_config_from_env,
)
from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    log_warning,
    log_error,
    # log_info,
)
from genericsuite_codegen.utilities.env_vars import get_envvar

from genericsuite_codegen.agent.types import (
    AgentConfig, QueryRequest, AgentContext, AgentResponse)
from genericsuite_codegen.agent.tools import (
    KnowledgeBaseTool,
    validate_search_query,
    format_sources_for_attribution,
)
from genericsuite_codegen.agent.enhanced_search_types \
    import EnhancedSearchConfig
from genericsuite_codegen.agent.search_templates import SearchTemplateManager
from genericsuite_codegen.agent.prompts import get_prompt_manager
from genericsuite_codegen.agent.tools import get_all_agent_tools

from genericsuite_codegen.agent.logfire import configure_logfire


DEBUG = True
DEBUG_DETAILED = False


class GenericSuiteAgent(AgentSuper):
    """
    Main AI agent for GenericSuite CodeGen using Pydantic AI.

    Provides intelligent assistance for GenericSuite development including
    code generation, documentation queries, and configuration creation.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        enhanced_search_config: Optional[EnhancedSearchConfig] = None
    ):
        """
        Initialize the GenericSuite AI agent.

        Args:
            config: Agent configuration. If None, uses environment defaults.
            enhanced_search_config: Enhanced search configuration.
                If None, uses defaults.
        """
        super().__init__(config, enhanced_search_config)

        self.prompt_manager = get_prompt_manager()

        # Initialize enhanced search configuration
        self.enhanced_search_config = (
            enhanced_search_config or
            self._create_default_enhanced_search_config()
        )

        # Create Pydantic AI agent and knowledge base tool when needed
        self.kb_tool = None
        self.agent = None

        enhanced_status = (
            'enabled' if self.enhanced_search_config.fallback_enabled
            else 'disabled'
        )

        _ = DEBUG and log_debug(
            "GenericSuite Agent initialized, with "
            f"Provider: '{self.config.model_provider}'"
            f", Model: '{self.config.model_name}'"
            ", Context window size: "
            f"{self.llm_data.context_window_size} tokens"
            ", Pricing: input USD: "
            f"{self.llm_data.input_tokens_price}"
            " | output USD: "
            f"{self.llm_data.output_tokens_price}"
            f", Enhanced search: {enhanced_status}"
        )

        if get_envvar("LOGFIRE_ENABLED", "false").lower() == "true":
            configure_logfire()

    def _create_agent(self) -> Agent:
        """
        Create the Pydantic AI agent with tools and configuration.

        Returns:
            Agent: Configured Pydantic AI agent.
        """
        # Get all agent tools (knowledge base + JSON generation)
        tools = get_all_agent_tools(self.kb_tool)

        # Create agent with system prompt
        system_prompt = self.prompt_manager.get_system_prompt("general")

        agent_args = {
            "model": self.model,
            "system_prompt": system_prompt,
            "tools": tools,
            "model_settings": ModelSettings(**self.inference_args),
        }
        _ = DEBUG_DETAILED and log_debug(f"Agent args: {agent_args}")
        agent = Agent(**agent_args)
        return agent

    async def query(
        self,
        request: QueryRequest,
        context: Optional[AgentContext] = None,
        run_context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a user query and generate a response.

        Args:
            request: Query request with user input and parameters.
            context: Optional agent context for personalization.
            run_context: Optional run context for personalization.

        Returns:
            AgentResponse: Generated response with sources and metadata.

        Raises:
            ValueError: If query is invalid.
            RuntimeError: If query processing fails.
        """
        try:
            # Validate query
            if not validate_search_query(request.query):
                raise ValueError("Invalid query: must be 3-1000 characters")

            _ = DEBUG and log_debug(
                f"Processing query: '{request.query}' "
                f"(type: {request.task_type})"
            )

            # Get relevant context from knowledge base
            kb_context, sources = await self._get_query_context(request)

            # Create task-specific prompt
            prompt = self._create_task_prompt(request, kb_context, sources)

            _ = DEBUG and log_debug(
                f">> Agent prompt: {prompt}")

            # Run agent with context
            run_context = self._create_run_context(context, request)

        except ValueError as e:
            log_error(f"Query preparation validation error: {e}")
            raise
        except Exception as e:
            log_error(
                "Query preparation failed for query:"
                f"\n'{request.query}'"
                f"\nError: {e}")
            raise RuntimeError(f"Failed to process query: {e}")

        try:
            # Execute query

            _ = DEBUG and log_debug(
                f">> Running agent with System prompt: {prompt}")
            _ = DEBUG and log_debug(
                f">> Running agent with User prompt: {request.query}")
            _ = DEBUG and log_debug(f">> Run context: {run_context}")

            self.set_kb_tool_and_agent()

            result = await self.agent.run(
                request.query,
                message_history=run_context.get("history", [])
            )

            _ = DEBUG and log_debug(
                f">> Agent result: {result}")

            # Format response
            response = self._format_response(
                result, request, sources, kb_context)

            _ = DEBUG and log_debug(
                f"Generated response ({len(response.content)} chars) with "
                f"{len(sources)} sources"
            )
            return response

        except ValueError as e:
            log_error(f"Query validation error: {e}")
            raise
        except Exception as e:
            log_error(
                f"Query processing failed for Query: {request.query}"
                + f"\n| Agent: {self.agent}"
                + f"\n| Configuration: {self.config}"
                + f"\n| Error: {e}")
            raise RuntimeError(f"Failed to process query: {e}")

    def set_kb_tool_and_agent(self) -> KnowledgeBaseTool:
        """Get the knowledge base tool."""
        if self.kb_tool is None:
            # Initialize knowledge base tool with enhanced search enabled
            enable_enhanced_search = (
                self.enhanced_search_config.fallback_enabled
            )
            self.kb_tool = KnowledgeBaseTool(
                enable_enhanced_search=enable_enhanced_search
            )

            # Configure enhanced search if available
            if (enable_enhanced_search and
                hasattr(self.kb_tool, 'enhanced_search') and
                    self.kb_tool.enhanced_search is not None):
                self.kb_tool.enhanced_search.update_config(
                    self.enhanced_search_config
                )
                _ = DEBUG and log_debug(
                    "Enhanced search configured with custom settings")

        if self.agent is None:
            self.agent = self._create_agent()

    async def _get_query_context(self, request: QueryRequest
                                 ) -> Tuple[str, List[str]]:
        """
        Retrieve relevant context from the knowledge base.

        Args:
            request: Query request.

        Returns:
            Tuple[str, List[str]]: context, sources, raw_results
                Formatted context string
                List of sources (only the document paths)
                Raw (KB search) results.
        """
        try:
            # Determine file type filter based on task type
            file_type_filter = self._get_file_type_filter(request.task_type)

        except Exception as e:
            log_error(f"Failed to get file type filter: {e}")
            raise

        try:
            self.set_kb_tool_and_agent()

        except Exception as e:
            log_error(f"Failed to set KB tool and agent: {e}")
            raise

        try:
            # Get context from knowledge base
            context, sources, raw_results = \
                self.kb_tool.get_context_for_generation(
                    query=request.query,
                    max_context_length=request.context_limit,
                    file_type_filter=file_type_filter,
                    full_content=True
                )

            return context, sources

        except Exception as e:
            log_warning(f"Failed to retrieve context: {e}")
            return "No context available due to retrieval error.", []

    def _get_file_type_filter(self, task_type: str) -> Optional[str]:
        """Get appropriate file type filter for task type."""
        filters = {
            "json": "json",
            "python": "py",
            "frontend": "jsx",  # Could also include tsx, js, ts
            "backend": "py",
        }
        return filters.get(task_type)

    def _create_task_prompt(
        self,
        request: QueryRequest,
        context: str,
        sources: List[str]
    ) -> str:
        """
        Create a task-specific prompt for the agent.

        Args:
            request: Query request.
            context: Retrieved context.
            sources: Source document paths.

        Returns:
            str: Formatted prompt for the agent.
        """

        _ = DEBUG and log_debug(
            f">> Creating task prompt for request: {request}")
        _ = DEBUG and log_debug(f">> Context: {context}")
        _ = DEBUG and log_debug(f">> Sources: {sources}")

        if request.task_type in ["json", "python", "frontend", "backend"]:
            # Add framework-specific guidance for backend tasks
            if request.task_type == "backend" and request.framework:
                framework_prompt = \
                    self.prompt_manager.get_framework_specific_prompt(
                        request.framework
                    )
                context += f"\n\n{framework_prompt}"

            return self.prompt_manager.create_generation_prompt(
                task_type=request.task_type,
                requirements=request.query,
                context=context,
                sources=sources,
            )
        else:
            return self.prompt_manager.create_query_prompt(
                query=request.query,
                context=context,
                sources=sources,
            )

    def _create_run_context(
        self,
        context: Optional[AgentContext],
        request: QueryRequest,
    ) -> Dict[str, Any]:
        """Create run context for the agent."""
        run_context = {}

        if context and context.conversation_history:
            # Convert conversation history to agent format
            history = []
            for msg in context.conversation_history[-5:]:  # Last 5 messages
                if msg.get("role") not in ["user", "assistant"]:
                    continue
                if msg.get("role") == "user":
                    history.append(ModelRequest(
                        parts=[
                            UserPromptPart(
                                content=msg.get("content"))]))
                if msg.get("role") == "assistant":
                    history.append(ModelResponse(
                        parts=[
                            TextPart(content=msg.get("content"))]))
            run_context["history"] = history

        _ = DEBUG and log_debug(
            f">> _create_run_context | Run context: {run_context}")

        return run_context

    def _format_response(
        self,
        agent_result: Any,
        request: QueryRequest,
        sources: List[str],
        context: str
    ) -> AgentResponse:
        """
        Format the agent result into a structured response.

        Args:
            agent_result: Agent execution result.
            request: Original query request.
            sources: Source document paths.
            context: Retrieved context.

        Returns:
            AgentResponse: Formatted response.
        """
        _ = DEBUG and log_debug(
            f">> _format_response | agent_result: {agent_result}")

        # Extract content from agent_result
        content = str(agent_result.output) if hasattr(
            agent_result, "output") else str(agent_result)

        # Add source attribution if requested
        if request.include_sources and sources:
            source_attribution = format_sources_for_attribution(sources)
            content += f"\n\n---\n{source_attribution}"

        model_used = self.config.model_name
        if DEBUG:
            # Add AI model and provider used
            model_used = \
                f"API: {self.config.model_api}" \
                + f" | Provider: {self.config.model_provider}" \
                + f" | Model: {self.config.model_name}"

        # Extract token usage if available
        token_usage = None
        if hasattr(agent_result, "usage") and agent_result.usage:
            _ = DEBUG and log_debug(
                ">> Agent | _format_response | Token usage:"
                f" {agent_result.usage}")
            token_usage = {
                "prompt_tokens": getattr(agent_result.usage,
                                         "prompt_tokens", 0),
                "completion_tokens": getattr(
                    agent_result.usage, "completion_tokens", 0),
                "total_tokens": getattr(agent_result.usage, "total_tokens", 0),
            }

        return AgentResponse(
            content=content,
            sources=sources,
            task_type=request.task_type,
            model_used=model_used,
            token_usage=token_usage,
        )

    async def generate_json_config(
        self,
        requirements: str,
        table_name: str,
        config_type: str = "table",
    ) -> AgentResponse:
        """
        Generate JSON configuration for GenericSuite.

        Args:
            requirements: Requirements for the configuration.
            table_name: Name of the table.
            config_type: Type of configuration (table, form, menu).

        Returns:
            AgentResponse: Generated JSON configuration.
        """
        user_requirements = \
            "1. Generate the table configuration for table(s)" \
            + f" {table_name}: {requirements}." \
            + "\n" \
            + "2. Generate the form configuration for the table(s)." \
            + ""
        # + "\n" \
        # + "3. Generate the menu configurations for the table(s)." \
        # + "\n" \
        # + "4. Generate the endpoints configurations for the table(s)." \
        # + ""

        request = QueryRequest(
            query=user_requirements,
            task_type="json",
            include_sources=True,
        )

        return await self.query(request)

    async def generate_python_code(
        self,
        requirements: str,
        tool_name: str,
        description: str,
        code_type: str = "tool",
    ) -> AgentResponse:
        """
        Generate Python code for GenericSuite.

        Args:
            requirements: Requirements for the code.
            code_type: Type of code (tool, langchain, mcp).

        Returns:
            AgentResponse: Generated Python code.
        """
        request = QueryRequest(
            query=f"Generate {code_type} Python code for a tool"
            f" named '{tool_name}' for '{description}' and completing the"
            f" following requirements: {requirements}",
            task_type="python",
            include_sources=True,
        )

        return await self.query(request)

    async def generate_frontend_code(self, requirements: str) -> AgentResponse:
        """
        Generate ReactJS frontend code.

        Args:
            requirements: Requirements for the frontend code.

        Returns:
            AgentResponse: Generated frontend code.
        """
        request = QueryRequest(
            query=f"Generate ReactJS frontend code: {requirements}",
            task_type="frontend",
            include_sources=True,
        )

        return await self.query(request)

    async def generate_backend_code(
        self, requirements: str, framework: str = "fastapi"
    ) -> AgentResponse:
        """
        Generate backend code for specified framework.

        Args:
            requirements: Requirements for the backend code.
            framework: Backend framework (fastapi, flask, chalice).

        Returns:
            AgentResponse: Generated backend code.
        """
        request = QueryRequest(
            query=f"Generate {framework} backend code: {requirements}",
            task_type="backend",
            framework=framework,
            include_sources=True,
        )

        return await self.query(request)

    def update_config(self, new_config: AgentConfig) -> None:
        """
        Update agent configuration.

        Args:
            new_config: New configuration to apply.
        """
        self.config = new_config
        self.model = self._initialize_model()
        self.agent = self._create_agent()
        _ = DEBUG and log_debug("Agent configuration updated")

    def update_enhanced_search_config(
        self, new_config: EnhancedSearchConfig
    ) -> None:
        """
        Update enhanced search configuration.

        Args:
            new_config: New enhanced search configuration to apply.
        """
        self.enhanced_search_config = new_config

        # Update existing knowledge base tool if it exists
        if (self.kb_tool is not None and
            hasattr(self.kb_tool, 'enhanced_search') and
                self.kb_tool.enhanced_search is not None):
            self.kb_tool.enhanced_search.update_config(new_config)
            _ = DEBUG and log_debug("Enhanced search configuration updated")
        else:
            # Reset kb_tool to force re-initialization with new config
            self.kb_tool = None
            _ = DEBUG and log_debug(
                "Enhanced search configuration updated - "
                "will apply on next use"
            )

    def get_enhanced_search_info(self) -> Dict[str, Any]:
        """
        Get information about enhanced search capabilities.

        Returns:
            Dict[str, Any]: Enhanced search information.
        """
        if self.kb_tool is None:
            return {
                "available": False,
                "reason": "Knowledge base tool not initialized"
            }

        if hasattr(self.kb_tool, 'get_enhanced_search_info'):
            return self.kb_tool.get_enhanced_search_info()
        else:
            return {
                "available": False,
                "reason": (
                    "Enhanced search not supported by knowledge base tool"
                )
            }

    def set_enhanced_search_enabled(self, enabled: bool) -> None:
        """
        Enable or disable enhanced search functionality.

        Args:
            enabled: Whether to enable enhanced search
        """
        # Update config
        self.enhanced_search_config.fallback_enabled = enabled

        # Update existing knowledge base tool if it exists
        if (self.kb_tool is not None and
                hasattr(self.kb_tool, 'set_enhanced_search_enabled')):
            self.kb_tool.set_enhanced_search_enabled(enabled)
            _ = DEBUG and log_debug(
                f"Enhanced search {'enabled' if enabled else 'disabled'}")
        else:
            # Reset kb_tool to force re-initialization with new setting
            self.kb_tool = None
            _ = DEBUG and log_debug(
                f"Enhanced search {'enabled' if enabled else 'disabled'} - "
                "will apply on next use"
            )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.

        Returns:
            Dict[str, Any]: Model information.
        """
        model_info = {
            "provider": self.config.model_provider,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }

        # Add enhanced search information
        enhanced_search_info = self.get_enhanced_search_info()
        model_info["enhanced_search"] = enhanced_search_info

        return model_info

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform agent health check.

        Returns:
            Dict[str, Any]: Health check results.
        """
        try:
            # Test basic query
            test_request = QueryRequest(
                query="In 10 words, what is GenericSuite?",
                task_type="general",
                context_limit=1000
            )

            response = await self.query(test_request)

            # Get enhanced search status
            enhanced_search_info = self.get_enhanced_search_info()

            return {
                "status": "healthy",
                "model": self.config.model_name,
                "provider": self.config.model_provider,
                "test_response_length": len(response.content),
                "sources_available": len(response.sources) > 0,
                "enhanced_search": enhanced_search_info,
            }

        except Exception as e:
            log_error(f"Agent health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.config.model_name,
                "provider": self.config.model_provider,
            }


# Global agent instance
_agent_instance: Optional[GenericSuiteAgent] = None


def get_agent(
    config: Optional[AgentConfig] = None,
    enhanced_search_config: Optional[EnhancedSearchConfig] = None
) -> GenericSuiteAgent:
    """
    Get or create the global agent instance.

    Args:
        config: Optional configuration for new agent.
        enhanced_search_config: Optional enhanced search configuration.

    Returns:
        GenericSuiteAgent: Global agent instance.
    """
    global _agent_instance

    if (_agent_instance is None):
        _agent_instance = GenericSuiteAgent(config, enhanced_search_config)

    return _agent_instance


def initialize_agent(
    config: Optional[AgentConfig] = None,
    enhanced_search_config: Optional[EnhancedSearchConfig] = None
) -> GenericSuiteAgent:
    """
    Initialize the GenericSuite AI agent.

    Args:
        config: Optional agent configuration.
        enhanced_search_config: Optional enhanced search configuration.

    Returns:
        GenericSuiteAgent: Initialized agent instance.
    """
    agent = get_agent(config, enhanced_search_config)
    _ = DEBUG and log_debug("GenericSuite AI agent initialized successfully")
    return agent


# Utility functions


def create_enhanced_search_config_from_env(
    config: AgentConfig = None,
) -> EnhancedSearchConfig:
    """
    Create enhanced search configuration from environment variables.

    Returns:
        EnhancedSearchConfig: Enhanced search configuration from environment.
    """
    try:
        # Initialize template manager to get templates
        template_manager = SearchTemplateManager()
        templates = template_manager.get_all_templates()

        if config is None:
            config = create_agent_config_from_env()

        return EnhancedSearchConfig(
            templates=templates,
            local_repo_path=get_envvar(
                "LOCAL_REPO_DIR", "./local_repo_files"
            ),
            max_context_length=int(
                get_envvar(
                    "ENHANCED_SEARCH_MAX_CONTEXT_LENGTH", str(
                        config.context_window_size)
                )
            ),
            fallback_enabled=get_envvar(
                "ENHANCED_SEARCH_FALLBACK_ENABLED", "true"
            ).lower() == "true",
            context_determination_enabled=get_envvar(
                "ENHANCED_SEARCH_CONTEXT_DETERMINATION_ENABLED",
                "true"
            ).lower() == "true",
            document_retrieval_enabled=get_envvar(
                "ENHANCED_SEARCH_ENABLE_DOC_RETRIEVAL", "true"
            ).lower() == "true",
            search_result_limit=int(
                get_envvar("ENHANCED_SEARCH_RESULT_LIMIT", "10")
            ),
            similarity_threshold=float(
                get_envvar(
                    "ENHANCED_SEARCH_SIMILARITY_THRESHOLD", "0.7"
                )
            ),
            merge_strategy=get_envvar(
                "ENHANCED_SEARCH_MERGE_STRATEGY",
                "prioritize_context"
            )
        )
    except Exception as e:
        log_warning(
            f"Failed to create enhanced search config from environment: {e}"
        )
        # Return minimal config with fallback enabled
        return EnhancedSearchConfig(
            templates={},
            local_repo_path="local_repo_files",
            max_context_length=10000,
            fallback_enabled=True
        )


def validate_agent_config(config: AgentConfig) -> bool:
    """
    Validate agent configuration.

    Args:
        config: Configuration to validate.

    Returns:
        bool: True if configuration is valid.
    """
    if not config.model_name:
        return False

    if config.temperature < 0.0 or config.temperature > 1.0:
        return False

    if config.max_tokens and config.max_tokens < 1:
        return False

    if config.timeout < 1:
        return False

    return True


def validate_enhanced_search_config(config: EnhancedSearchConfig) -> bool:
    """
    Validate enhanced search configuration.

    Args:
        config: Enhanced search configuration to validate.

    Returns:
        bool: True if configuration is valid.
    """
    if not config.local_repo_path:
        return False

    if config.max_context_length < 100:
        return False

    if config.search_result_limit < 1:
        return False

    valid_strategies = ["prioritize_context", "balanced", "prioritize_user"]
    if config.merge_strategy not in valid_strategies:
        return False

    if not (0.0 <= config.similarity_threshold <= 1.0):
        return False

    return True


if __name__ == "__main__":
    # Example usage and testing
    import asyncio

    async def test_agent():
        """Test the GenericSuite agent."""
        try:
            # Initialize agent with enhanced search
            config = create_agent_config_from_env()
            enhanced_config = create_enhanced_search_config_from_env()
            agent = initialize_agent(config, enhanced_config)

            # Test health check
            health = await agent.health_check()
            print(f"Health check: {health}")

            # Test enhanced search info
            enhanced_info = agent.get_enhanced_search_info()
            print(f"Enhanced search info: {enhanced_info}")

            # Test query
            request = QueryRequest(
                query="How do I create a GenericSuite table configuration?",
                task_type="general",
            )

            response = await agent.query(request)
            print(f"Response: {response.content[:200]}...")
            print(f"Sources: {response.sources}")

            # Test enhanced search toggle
            agent.set_enhanced_search_enabled(False)
            print("Enhanced search disabled")

            agent.set_enhanced_search_enabled(True)
            print("Enhanced search re-enabled")

        except Exception as e:
            print(f"Test error: {e}")

    asyncio.run(test_agent())
