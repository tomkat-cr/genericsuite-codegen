"""
Unit tests for AI agent core functionality.
"""

import pytest
import os
from unittest.mock import Mock, patch, AsyncMock

from genericsuite_codegen.agent.agent import GenericSuiteAgent
from genericsuite_codegen.agent.types import (
    AgentConfig,
    QueryRequest,
    AgentContext,
    AgentResponse
)
from genericsuite_codegen.agent.enhanced_search_types \
    import EnhancedSearchConfig


class TestGenericSuiteAgent:
    """Test GenericSuiteAgent class."""

    def test_agent_initialization_default(self):
        """Test agent initialization with default configuration."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()

            assert agent.config is not None
            assert agent.enhanced_search_config is not None
            assert agent.prompt_manager is not None

    def test_agent_initialization_custom_config(self):
        """Test agent initialization with custom configuration."""
        config = AgentConfig(
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=2000,
            api_key="custom-key"
        )

        enhanced_config = EnhancedSearchConfig(
            fallback_enabled=True,
            dual_search_enabled=True,
            context_merge_strategy="priority"
        )

        agent = GenericSuiteAgent(
            config=config, enhanced_search_config=enhanced_config)

        assert agent.config is config
        assert agent.enhanced_search_config is enhanced_config

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_create_default_config(self):
        """Test creating default configuration."""
        agent = GenericSuiteAgent()
        config = agent._create_default_config()

        assert isinstance(config, AgentConfig)
        assert config.model_provider in ["openai", "litellm"]
        assert config.api_key == "test-key"

    def test_create_default_enhanced_search_config(self):
        """Test creating default enhanced search configuration."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()
            config = agent._create_default_enhanced_search_config()

            assert isinstance(config, EnhancedSearchConfig)
            assert config.fallback_enabled is True
            assert config.dual_search_enabled is True

    @patch('genericsuite_codegen.agent.agent.OpenAIChatModel')
    def test_initialize_model_openai(self, mock_openai_model):
        """Test model initialization with OpenAI."""
        config = AgentConfig(
            model_provider="openai",
            model_name="gpt-4",
            api_key="test-key"
        )

        mock_model = Mock()
        mock_openai_model.return_value = mock_model

        agent = GenericSuiteAgent(config=config)
        model = agent._initialize_model()

        assert model is mock_model
        mock_openai_model.assert_called_once()

    @patch('genericsuite_codegen.agent.agent.LITELLM_AVAILABLE', True)
    @patch('genericsuite_codegen.agent.agent.litellm')
    def test_initialize_model_litellm(self, mock_litellm):
        """Test model initialization with LiteLLM."""
        config = AgentConfig(
            model_provider="litellm",
            model_name="claude-3-sonnet",
            api_key="test-key"
        )

        agent = GenericSuiteAgent(config=config)
        model = agent._initialize_model()

        # Should create a LiteLLM model wrapper
        assert model is not None

    def test_initialize_model_invalid_provider(self):
        """Test model initialization with invalid provider."""
        config = AgentConfig(
            model_provider="invalid",
            model_name="test-model",
            api_key="test-key"
        )

        with pytest.raises(ValueError):
            GenericSuiteAgent(config=config)

    @patch('genericsuite_codegen.agent.agent.KnowledgeBaseTool')
    def test_get_knowledge_base_tool(self, mock_kb_tool_class):
        """Test getting knowledge base tool."""
        mock_kb_tool = Mock()
        mock_kb_tool_class.return_value = mock_kb_tool

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()
            kb_tool = agent._get_knowledge_base_tool()

            assert kb_tool is mock_kb_tool
            assert agent.kb_tool is mock_kb_tool

    @patch('genericsuite_codegen.agent.agent.Agent')
    @patch('genericsuite_codegen.agent.agent.get_all_agent_tools')
    def test_get_pydantic_agent(self, mock_get_tools, mock_agent_class):
        """Test getting Pydantic AI agent."""
        mock_tools = [Mock(), Mock()]
        mock_get_tools.return_value = mock_tools

        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()
            pydantic_agent = agent._get_pydantic_agent()

            assert pydantic_agent is mock_agent
            assert agent.agent is mock_agent
            mock_agent_class.assert_called_once()

    @patch('genericsuite_codegen.agent.agent.Agent')
    @patch('genericsuite_codegen.agent.agent.get_all_agent_tools')
    async def test_query_async_success(self, mock_get_tools, mock_agent_class):
        """Test successful async query."""
        # Mock tools and agent
        mock_tools = [Mock()]
        mock_get_tools.return_value = mock_tools

        mock_agent = Mock()
        mock_run_result = Mock()
        mock_run_result.data = "Test response"
        mock_run_result.all_messages.return_value = []
        mock_agent.run.return_value = mock_run_result
        mock_agent_class.return_value = mock_agent

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()

            request = QueryRequest(
                query="Test query",
                context=AgentContext(user_id="test_user")
            )

            response = await agent.query_async(request)

            assert isinstance(response, AgentResponse)
            assert response.response == "Test response"
            assert response.success is True

    @patch('genericsuite_codegen.agent.agent.Agent')
    @patch('genericsuite_codegen.agent.agent.get_all_agent_tools')
    async def test_query_async_error(self, mock_get_tools, mock_agent_class):
        """Test async query with error."""
        # Mock tools and agent
        mock_tools = [Mock()]
        mock_get_tools.return_value = mock_tools

        mock_agent = Mock()
        mock_agent.run.side_effect = Exception("Query failed")
        mock_agent_class.return_value = mock_agent

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()

            request = QueryRequest(
                query="Test query",
                context=AgentContext(user_id="test_user")
            )

            response = await agent.query_async(request)

            assert isinstance(response, AgentResponse)
            assert response.success is False
            assert "Query failed" in response.error

    @patch('genericsuite_codegen.agent.agent.asyncio.run')
    def test_query_sync(self, mock_asyncio_run):
        """Test synchronous query wrapper."""
        mock_response = AgentResponse(
            response="Test response",
            success=True,
            sources=[],
            metadata={}
        )
        mock_asyncio_run.return_value = mock_response

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()

            request = QueryRequest(
                query="Test query",
                context=AgentContext(user_id="test_user")
            )

            response = agent.query(request)

            assert response is mock_response
            mock_asyncio_run.assert_called_once()

    def test_validate_query_request_valid(self):
        """Test query request validation with valid request."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()

            request = QueryRequest(
                query="Valid query",
                context=AgentContext(user_id="test_user")
            )

            # Should not raise exception
            agent._validate_query_request(request)

    def test_validate_query_request_empty_query(self):
        """Test query request validation with empty query."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()

            request = QueryRequest(
                query="",
                context=AgentContext(user_id="test_user")
            )

            with pytest.raises(ValueError):
                agent._validate_query_request(request)

    def test_validate_query_request_too_long(self):
        """Test query request validation with too long query."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()

            request = QueryRequest(
                query="x" * 10000,  # Very long query
                context=AgentContext(user_id="test_user")
            )

            with pytest.raises(ValueError):
                agent._validate_query_request(request)

    def test_validate_query_request_no_context(self):
        """Test query request validation without context."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()

            request = QueryRequest(query="Valid query")

            with pytest.raises(ValueError):
                agent._validate_query_request(request)

    def test_format_response_success(self):
        """Test response formatting for successful query."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()

            mock_run_result = Mock()
            mock_run_result.data = "Test response"
            mock_run_result.all_messages.return_value = []

            request = QueryRequest(
                query="Test query",
                context=AgentContext(user_id="test_user")
            )

            response = agent._format_response(mock_run_result, request)

            assert isinstance(response, AgentResponse)
            assert response.response == "Test response"
            assert response.success is True
            assert response.query == "Test query"

    def test_format_response_with_sources(self):
        """Test response formatting with sources."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()

            mock_run_result = Mock()
            mock_run_result.data = "Test response"

            # Mock messages with tool calls
            mock_message = Mock()
            mock_message.parts = [Mock(tool_name="search_knowledge_base")]
            mock_run_result.all_messages.return_value = [mock_message]

            request = QueryRequest(
                query="Test query",
                context=AgentContext(user_id="test_user")
            )

            with patch('genericsuite_codegen.agent.'
                       'agent.format_sources_for_attribution') \
                    as mock_format:
                mock_format.return_value = ["source1.py", "source2.py"]

                response = agent._format_response(mock_run_result, request)

                assert response.sources == ["source1.py", "source2.py"]

    def test_format_error_response(self):
        """Test error response formatting."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()

            request = QueryRequest(
                query="Test query",
                context=AgentContext(user_id="test_user")
            )

            error = Exception("Test error")

            response = agent._format_error_response(error, request)

            assert isinstance(response, AgentResponse)
            assert response.success is False
            assert "Test error" in response.error
            assert response.query == "Test query"

    def test_get_model_info(self):
        """Test getting model information."""
        config = AgentConfig(
            model_provider="openai",
            model_name="gpt-4",
            api_key="test-key"
        )

        agent = GenericSuiteAgent(config=config)
        model_info = agent.get_model_info()

        assert model_info["provider"] == "openai"
        assert model_info["model"] == "gpt-4"
        assert "temperature" in model_info
        assert "max_tokens" in model_info

    def test_get_agent_status(self):
        """Test getting agent status."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()
            status = agent.get_agent_status()

            assert "model_info" in status
            assert "enhanced_search" in status
            assert "tools_available" in status
            assert "initialized" in status

    @patch('genericsuite_codegen.agent.agent.KnowledgeBaseTool')
    def test_health_check_success(self, mock_kb_tool_class):
        """Test successful health check."""
        mock_kb_tool = Mock()
        mock_kb_tool.health_check.return_value = True
        mock_kb_tool_class.return_value = mock_kb_tool

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()
            is_healthy = agent.health_check()

            assert is_healthy is True

    @patch('genericsuite_codegen.agent.agent.KnowledgeBaseTool')
    def test_health_check_failure(self, mock_kb_tool_class):
        """Test health check failure."""
        mock_kb_tool = Mock()
        mock_kb_tool.health_check.side_effect = Exception(
            "Health check failed")
        mock_kb_tool_class.return_value = mock_kb_tool

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()
            is_healthy = agent.health_check()

            assert is_healthy is False

    def test_update_config(self):
        """Test updating agent configuration."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()

            new_config = AgentConfig(
                model_provider="openai",
                model_name="gpt-3.5-turbo",
                temperature=0.8,
                api_key="new-key"
            )

            agent.update_config(new_config)

            assert agent.config is new_config
            # Should reinitialize model and agent
            assert agent.model is not None
            assert agent.agent is None  # Should be reset


class TestAgentIntegration:
    """Test agent integration with other components."""

    @patch('genericsuite_codegen.agent.agent.KnowledgeBaseTool')
    @patch('genericsuite_codegen.agent.agent.Agent')
    @patch('genericsuite_codegen.agent.agent.get_all_agent_tools')
    async def test_full_query_workflow(self, mock_get_tools, mock_agent_class,
                                       mock_kb_tool_class):
        """Test complete query workflow."""
        # Mock knowledge base tool
        mock_kb_tool = Mock()
        mock_kb_tool.search_knowledge_base = AsyncMock(
            return_value="Knowledge base results")
        mock_kb_tool_class.return_value = mock_kb_tool

        # Mock tools
        mock_tools = [mock_kb_tool]
        mock_get_tools.return_value = mock_tools

        # Mock Pydantic agent
        mock_agent = Mock()
        mock_run_result = Mock()
        mock_run_result.data = "Generated response based on knowledge base"
        mock_run_result.all_messages.return_value = []
        mock_agent.run = AsyncMock(return_value=mock_run_result)
        mock_agent_class.return_value = mock_agent

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent()

            request = QueryRequest(
                query="How do I create a GenericSuite table configuration?",
                context=AgentContext(user_id="test_user")
            )

            response = await agent.query_async(request)

            assert response.success is True
            assert "Generated response" in response.response
            mock_agent.run.assert_called_once()

    @patch('genericsuite_codegen.agent.agent.SearchTemplateManager')
    def test_enhanced_search_integration(self, mock_template_manager):
        """Test integration with enhanced search."""
        mock_manager = Mock()
        mock_template_manager.return_value = mock_manager

        enhanced_config = EnhancedSearchConfig(
            fallback_enabled=True,
            dual_search_enabled=True,
            context_merge_strategy="priority"
        )

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = GenericSuiteAgent(enhanced_search_config=enhanced_config)

            assert agent.enhanced_search_config.dual_search_enabled is True
            assert agent.enhanced_search_config.fallback_enabled is True

    def test_environment_configuration(self):
        """Test agent configuration from environment variables."""
        env_vars = {
            'OPENAI_API_KEY': 'env-key',
            'MODEL_PROVIDER': 'openai',
            'MODEL_NAME': 'gpt-4',
            'MODEL_TEMPERATURE': '0.7',
            'MODEL_MAX_TOKENS': '3000'
        }

        with patch.dict(os.environ, env_vars):
            agent = GenericSuiteAgent()

            assert agent.config.api_key == 'env-key'
            assert agent.config.model_provider == 'openai'
            assert agent.config.model_name == 'gpt-4'
            assert agent.config.temperature == 0.7
            assert agent.config.max_tokens == 3000
