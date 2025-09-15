"""
Core Pydantic AI agent for GenericSuite CodeGen.

This module implements the main AI agent using Pydantic AI framework,
integrating knowledge base search, code generation capabilities, and
LLM provider configuration for GenericSuite development assistance.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple

from pydantic_ai import Agent
# from pydantic_ai import RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from pydantic_ai.messages import (
    # ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from .types import AgentConfig, QueryRequest, AgentContext, AgentResponse

from .tools import (
    get_all_agent_tools,
    KnowledgeBaseTool,
    validate_search_query,
    format_sources_for_attribution,
)
from .prompts import get_prompt_manager


DEBUG = True
DEBUG_DETAILED = False

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if DEBUG else logging.WARNING)

try:
    import litellm
    if DEBUG:
        logging.info(f"LiteLLM loaded: {litellm}")
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logging.warning("LiteLLM not available, using OpenAI only")


class GenericSuiteAgent:
    """
    Main AI agent for GenericSuite CodeGen using Pydantic AI.

    Provides intelligent assistance for GenericSuite development including
    code generation, documentation queries, and configuration creation.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the GenericSuite AI agent.

        Args:
            config: Agent configuration. If None, uses environment defaults.
        """
        self.config = config or self._create_default_config()
        self.prompt_manager = get_prompt_manager()

        # Initialize LLM model
        self.inference_args = {}
        self.model = self._initialize_model()

        # Create Pydantic AI agent and knowledge base tool when needed
        self.kb_tool = None
        self.agent = None

        logger.info(
            "Initialized GenericSuite agent with "
            f"{self.config.model_provider} provider"
        )

    def _create_default_config(self) -> AgentConfig:
        """Create default configuration from environment variables."""
        config_args = {
            "model_provider": os.getenv("LLM_PROVIDER", "openai"),
            "model_name": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.5")),
            "max_tokens": (
                int(os.getenv("LLM_MAX_TOKENS", "4000"))
                if os.getenv("LLM_MAX_TOKENS")
                else None
            ),
            "timeout": int(os.getenv("LLM_TIMEOUT", "60")),
            "api_key": os.getenv("LLM_API_KEY"),
        }
        base_url = os.getenv("LLM_BASE_URL")
        if base_url is not None and base_url != '':
            config_args["base_url"] = base_url
        agent_config = AgentConfig(**config_args)
        logger.info(
            f"Agent config: {agent_config}"
        )
        return agent_config

    def _initialize_model(self) -> Model:
        """
        Initialize the LLM model based on configuration.

        Returns:
            Model: Configured Pydantic AI model.

        Raises:
            ValueError: If model configuration is invalid.
        """
        try:
            if self.config.model_provider == "openai":
                return self._create_openai_model()
            elif self.config.model_provider == "litellm" and LITELLM_AVAILABLE:
                return self._create_litellm_model()
            else:
                logger.warning(
                    "Unsupported provider "
                    f"{self.config.model_provider}, falling back to OpenAI"
                )
                return self._create_openai_model()

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise ValueError(f"Model initialization failed: {e}")

    def _create_openai_model(self) -> OpenAIChatModel:
        """Create OpenAI model configuration."""
        self.inference_args["temperature"] = self.config.temperature
        self.inference_args["timeout"] = self.config.timeout
        if self.config.max_tokens:
            self.inference_args["max_tokens"] = self.config.max_tokens

        model_kwargs = {}

        if self.config.api_key:
            model_kwargs["api_key"] = self.config.api_key

        if self.config.base_url is None or self.config.base_url == '':
            model_kwargs["base_url"] = 'https://api.openai.com/v1'
        else:
            model_kwargs["base_url"] = self.config.base_url

        logger.info(f"Model kwargs: {model_kwargs}")

        return OpenAIChatModel(
            self.config.model_name,
            provider=OpenAIProvider(**model_kwargs)
        )

    def _create_litellm_model(self) -> Model:
        """Create LiteLLM model configuration."""
        # TODO: Note: This would need to be implemented based on Pydantic AI's
        # LiteLLM support. For now, fall back to OpenAI
        logger.warning("LiteLLM integration not yet implemented, using OpenAI")
        return self._create_openai_model()

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
        _ = DEBUG_DETAILED and logger.info(f"Agent args: {agent_args}")
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

            logger.info(
                f"Processing query: '{request.query}' "
                f"(type: {request.task_type})"
            )

            # Get relevant context from knowledge base
            kb_context, sources = await self._get_query_context(request)

            # Create task-specific prompt
            prompt = self._create_task_prompt(request, kb_context, sources)

            # Run agent with context
            run_context = self._create_run_context(context, request)

        except ValueError as e:
            logger.error(f"Query preparation validation error: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Query preparation failed for query: {request.query}: {e}")
            raise RuntimeError(f"Failed to process query: {e}")

        try:
            # Execute query

            logger.info(f">> Running agent with System prompt: {prompt}")
            logger.info(f">> Running agent with User prompt: {request.query}")
            logger.info(f">> Run context: {run_context}")

            self.set_kb_tool_and_agent()

            result = await self.agent.run(
                request.query,
                message_history=run_context.get("history", [])
            )

            logger.info(f">> Agent result: {result}")

            # Format response
            response = self._format_response(
                result, request, sources, kb_context)

            logger.info(
                f"Generated response ({len(response.content)} chars) with"
                f" {len(sources)} sources"
            )
            return response

        except ValueError as e:
            logger.error(f"Query validation error: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Query processing failed for query: {request.query}: {e}")
            raise RuntimeError(f"Failed to process query: {e}")

    def set_kb_tool_and_agent(self) -> KnowledgeBaseTool:
        """Get the knowledge base tool."""
        if self.kb_tool is None:
            self.kb_tool = KnowledgeBaseTool()
        if self.agent is None:
            self.agent = self._create_agent()

    async def _get_query_context(self, request: QueryRequest
                                 ) -> Tuple[str, List[str]]:
        """
        Retrieve relevant context from the knowledge base.

        Args:
            request: Query request.

        Returns:
            Tuple[str, List[str]]: Context string and source paths.
        """
        try:
            # Determine file type filter based on task type
            file_type_filter = self._get_file_type_filter(request.task_type)

            self.set_kb_tool_and_agent()

            # Get context from knowledge base
            context, sources = self.kb_tool.get_context_for_generation(
                query=request.query,
                max_context_length=request.context_limit,
                file_type_filter=file_type_filter,
            )

            return context, sources

        except Exception as e:
            logger.warning(f"Failed to retrieve context: {e}")
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

        logger.info(f">> Creating task prompt for request: {request}")
        logger.info(f">> Context: {context}")
        logger.info(f">> Sources: {sources}")

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

        logger.info(f">> _create_run_context | Run context: {run_context}")

        return run_context

    def _format_response(
        self, result: Any,
        request: QueryRequest,
        sources: List[str],
        context: str
    ) -> AgentResponse:
        """
        Format the agent result into a structured response.

        Args:
            result: Agent execution result.
            request: Original query request.
            sources: Source document paths.
            context: Retrieved context.

        Returns:
            AgentResponse: Formatted response.
        """
        # Extract content from result
        logger.info(f">> _format_response | result: {result}")
        content = str(result.output) if hasattr(
            result, "output") else str(result)

        # Add source attribution if requested
        if request.include_sources and sources:
            source_attribution = format_sources_for_attribution(sources)
            content += f"\n\n---\n{source_attribution}"

        # Extract token usage if available
        token_usage = None
        if hasattr(result, "usage") and result.usage:
            token_usage = {
                "prompt_tokens": getattr(result.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(
                    result.usage, "completion_tokens", 0),
                "total_tokens": getattr(result.usage, "total_tokens", 0),
            }

        return AgentResponse(
            content=content,
            sources=sources,
            task_type=request.task_type,
            model_used=self.config.model_name,
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
        request = QueryRequest(
            query=f"Generate a {config_type} configuration for table"
            f" '{table_name}': {requirements}",
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
        logger.info("Agent configuration updated")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.

        Returns:
            Dict[str, Any]: Model information.
        """
        return {
            "provider": self.config.model_provider,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }

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

            return {
                "status": "healthy",
                "model": self.config.model_name,
                "provider": self.config.model_provider,
                "test_response_length": len(response.content),
                "sources_available": len(response.sources) > 0,
            }

        except Exception as e:
            logger.error(f"Agent health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.config.model_name,
                "provider": self.config.model_provider,
            }


# Global agent instance
_agent_instance: Optional[GenericSuiteAgent] = None


def get_agent(config: Optional[AgentConfig] = None) -> GenericSuiteAgent:
    """
    Get or create the global agent instance.

    Args:
        config: Optional configuration for new agent.

    Returns:
        GenericSuiteAgent: Global agent instance.
    """
    global _agent_instance

    if _agent_instance is None or config is not None:
        _agent_instance = GenericSuiteAgent(config)

    return _agent_instance


def initialize_agent(config: Optional[AgentConfig] = None
                     ) -> GenericSuiteAgent:
    """
    Initialize the GenericSuite AI agent.

    Args:
        config: Optional agent configuration.

    Returns:
        GenericSuiteAgent: Initialized agent instance.
    """
    agent = get_agent(config)
    logger.info("GenericSuite AI agent initialized successfully")
    return agent


# Utility functions


def create_agent_config_from_env() -> AgentConfig:
    """
    Create agent configuration from environment variables.

    Returns:
        AgentConfig: Configuration from environment.
    """
    return AgentConfig(
        model_provider=os.getenv("GENERICSUITE_LLM_PROVIDER", "openai"),
        model_name=os.getenv("GENERICSUITE_LLM_MODEL", "gpt-4"),
        temperature=float(os.getenv("GENERICSUITE_LLM_TEMPERATURE", "0.1")),
        max_tokens=(
            int(os.getenv("GENERICSUITE_LLM_MAX_TOKENS", "4000"))
            if os.getenv("GENERICSUITE_LLM_MAX_TOKENS")
            else None
        ),
        timeout=int(os.getenv("GENERICSUITE_LLM_TIMEOUT", "60")),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("GENERICSUITE_LLM_BASE_URL"),
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


if __name__ == "__main__":
    # Example usage and testing
    import asyncio

    async def test_agent():
        """Test the GenericSuite agent."""
        try:
            # Initialize agent
            config = create_agent_config_from_env()
            agent = initialize_agent(config)

            # Test health check
            health = await agent.health_check()
            print(f"Health check: {health}")

            # Test query
            request = QueryRequest(
                query="How do I create a GenericSuite table configuration?",
                task_type="general",
            )

            response = await agent.query(request)
            print(f"Response: {response.content[:200]}...")
            print(f"Sources: {response.sources}")

        except Exception as e:
            print(f"Test error: {e}")

    asyncio.run(test_agent())
