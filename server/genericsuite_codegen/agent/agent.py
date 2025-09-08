"""
Core Pydantic AI agent for GenericSuite CodeGen.

This module implements the main AI agent using Pydantic AI framework,
integrating knowledge base search, code generation capabilities, and
LLM provider configuration for GenericSuite development assistance.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from pydantic import BaseModel, Field

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logging.warning("LiteLLM not available, using OpenAI only")

from .tools import (
    get_all_agent_tools,
    KnowledgeBaseTool,
    validate_search_query,
    format_sources_for_attribution,
)
from .prompts import get_prompt_manager, PromptManager

# Configure logging
logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """Configuration for the GenericSuite AI agent."""

    model_provider: str = Field(
        default="openai", description="LLM provider (openai, litellm)"
    )
    model_name: str = Field(default="gpt-4", description="Model name to use")
    temperature: float = Field(
        default=0.1, description="Model temperature (0.0 to 1.0)"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens for responses"
    )
    timeout: int = Field(default=60, description="Request timeout in seconds")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Custom base URL for API")


class QueryRequest(BaseModel):
    """Request model for agent queries."""

    query: str = Field(description="User query or request")
    task_type: str = Field(
        default="general",
        description="Type of task (general, json, python, frontend, backend)",
    )
    framework: Optional[str] = Field(
        default=None, description="Backend framework for code generation"
    )
    context_limit: int = Field(default=4000, description="Maximum context length")
    include_sources: bool = Field(
        default=True, description="Include source attribution in response"
    )


class AgentResponse(BaseModel):
    """Response model from the agent."""

    content: str = Field(description="Generated response content")
    sources: List[str] = Field(
        default_factory=list, description="Source documents used"
    )
    task_type: str = Field(description="Type of task performed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    model_used: str = Field(description="Model used for generation")
    token_usage: Optional[Dict[str, int]] = Field(
        default=None, description="Token usage statistics"
    )


@dataclass
class AgentContext:
    """Context information for agent operations."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = None
    preferences: Dict[str, Any] = None


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
        self.kb_tool = KnowledgeBaseTool()

        # Initialize LLM model
        self.inference_args = {}
        self.model = self._initialize_model()

        # Create Pydantic AI agent
        self.agent = self._create_agent()

        logger.info(
            f"Initialized GenericSuite agent with {self.config.model_provider} provider"
        )

    def _create_default_config(self) -> AgentConfig:
        """Create default configuration from environment variables."""
        return AgentConfig(
            model_provider=os.getenv("LLM_PROVIDER", "openai"),
            model_name=os.getenv("LLM_MODEL", "gpt-4"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=(
                int(os.getenv("LLM_MAX_TOKENS", "4000"))
                if os.getenv("LLM_MAX_TOKENS")
                else None
            ),
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )

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
                    f"Unsupported provider {self.config.model_provider}, falling back to OpenAI"
                )
                return self._create_openai_model()

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise ValueError(f"Model initialization failed: {e}")

    def _create_openai_model(self) -> OpenAIModel:
        """Create OpenAI model configuration."""
        self.inference_args["temperature"] = self.config.temperature
        self.inference_args["timeout"] = self.config.timeout
        if self.config.max_tokens:
            self.inference_args["max_tokens"] = self.config.max_tokens

        model_kwargs = {}

        if self.config.api_key:
            model_kwargs["api_key"] = self.config.api_key

        if self.config.base_url:
            model_kwargs["base_url"] = self.config.base_url

        return OpenAIModel(
            self.config.model_name, provider=OpenAIProvider(**model_kwargs)
        )

    def _create_litellm_model(self) -> Model:
        """Create LiteLLM model configuration."""
        # Note: This would need to be implemented based on Pydantic AI's LiteLLM support
        # For now, fall back to OpenAI
        logger.warning("LiteLLM integration not yet implemented, using OpenAI")
        return self._create_openai_model()

    def _create_agent(self) -> Agent:
        """
        Create the Pydantic AI agent with tools and configuration.

        Returns:
            Agent: Configured Pydantic AI agent.
        """
        # Get all agent tools (knowledge base + JSON generation)
        tools = get_all_agent_tools()

        # Create agent with system prompt
        system_prompt = self.prompt_manager.get_system_prompt("general")

        agent = Agent(
            model=self.model,
            system_prompt=system_prompt,
            tools=tools,
            model_settings=ModelSettings(**self.inference_args),
        )

        return agent

    async def query(
        self, request: QueryRequest, context: Optional[AgentContext] = None
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
                f"Processing query: '{request.query}' (type: {request.task_type})"
            )

            # Get relevant context from knowledge base
            kb_context, sources = await self._get_query_context(request)

            # Create task-specific prompt
            prompt = self._create_task_prompt(request, kb_context, sources)

            # Run agent with context
            run_context = self._create_run_context(context, request)

            # Execute query
            result = await self.agent.run(
                prompt, message_history=run_context.get("history", [])
            )

            # Format response
            response = self._format_response(result, request, sources, kb_context)

            logger.info(
                f"Generated response ({len(response.content)} chars) with {len(sources)} sources"
            )
            return response

        except ValueError as e:
            logger.error(f"Query validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise RuntimeError(f"Failed to process query: {e}")

    async def _get_query_context(self, request: QueryRequest) -> Tuple[str, List[str]]:
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
        self, request: QueryRequest, context: str, sources: List[str]
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
        if request.task_type in ["json", "python", "frontend", "backend"]:
            # Add framework-specific guidance for backend tasks
            if request.task_type == "backend" and request.framework:
                framework_prompt = self.prompt_manager.get_framework_specific_prompt(
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
                query=request.query, context=context, sources=sources
            )

    def _create_run_context(
        self, context: Optional[AgentContext], request: QueryRequest
    ) -> Dict[str, Any]:
        """Create run context for the agent."""
        run_context = {}

        if context and context.conversation_history:
            # Convert conversation history to agent format
            history = []
            for msg in context.conversation_history[-5:]:  # Last 5 messages
                if msg.get("role") in ["user", "assistant"]:
                    history.append({"role": msg["role"], "content": msg["content"]})
            run_context["history"] = history

        return run_context

    def _format_response(
        self, result: Any, request: QueryRequest, sources: List[str], context: str
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
        content = str(result.data) if hasattr(result, "data") else str(result)

        # Add source attribution if requested
        if request.include_sources and sources:
            source_attribution = format_sources_for_attribution(sources)
            content += f"\n\n---\n{source_attribution}"

        # Extract token usage if available
        token_usage = None
        if hasattr(result, "usage") and result.usage:
            token_usage = {
                "prompt_tokens": getattr(result.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(result.usage, "completion_tokens", 0),
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
        self, requirements: str, config_type: str = "table"
    ) -> AgentResponse:
        """
        Generate JSON configuration for GenericSuite.

        Args:
            requirements: Requirements for the configuration.
            config_type: Type of configuration (table, form, menu).

        Returns:
            AgentResponse: Generated JSON configuration.
        """
        request = QueryRequest(
            query=f"Generate a {config_type} configuration: {requirements}",
            task_type="json",
            include_sources=True,
        )

        return await self.query(request)

    async def generate_python_code(
        self, requirements: str, code_type: str = "tool"
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
            query=f"Generate {code_type} Python code: {requirements}",
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
                query="What is GenericSuite?", task_type="general", context_limit=1000
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


def initialize_agent(config: Optional[AgentConfig] = None) -> GenericSuiteAgent:
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
