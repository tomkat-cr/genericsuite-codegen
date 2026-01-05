import json
from typing import Optional
from pathlib import Path

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from genericsuite_codegen.utilities.utilities import (
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL_NAME,
    DEFAULT_LLM_TEMPERATURE,
    CONTEXT_DEFAULT_MAX_LENGTH,
)
from genericsuite_codegen.utilities.env_vars import get_envvar
from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    log_warning,
    log_error,
)

from genericsuite_codegen.api.types import LlmData
from genericsuite_codegen.agent.types import AgentConfig
from genericsuite_codegen.agent.enhanced_search_types \
    import EnhancedSearchConfig
from genericsuite_codegen.agent.search_templates import SearchTemplateManager

from genericsuite_codegen.agent.patch_openai_service_tier \
    import patch_openai_service_tier
from genericsuite_codegen.agent.patch_tokenizers import patch_tokenizers


DEBUG = False
DEBUG_DETAILED = False


class AgentSuper:

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
        self.config = config or self._create_default_config()

        # Initialize LLM model
        self.inference_args = {}
        self.model = self._initialize_model()

        # Get LLM data (input_tokens_price, output_tokens_price,
        # context_window_size)
        self.llm_data = get_llm_data(
            self.config.model_name,
            self.config.model_provider
        )

    def _create_default_config(self) -> AgentConfig:
        """Create default configuration from environment variables."""
        config_args = {
            "model_api": get_envvar("LLM_API", "openai"),
            "model_provider": get_envvar("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
            "model_name": get_envvar("LLM_MODEL_NAME", DEFAULT_LLM_MODEL_NAME),
            "temperature": float(get_envvar("LLM_TEMPERATURE",
                                            DEFAULT_LLM_TEMPERATURE)),
            "max_tokens": (
                int(get_envvar("LLM_MAX_TOKENS", "4000"))
                if get_envvar("LLM_MAX_TOKENS")
                else None
            ),
            "timeout": int(get_envvar("LLM_TIMEOUT", "60")),
            "api_key": None,
        }

        base_url = get_envvar("LLM_BASE_URL")
        if base_url is not None and base_url != '':
            config_args["base_url"] = base_url

        agent_config = AgentConfig(**config_args)
        _ = DEBUG and log_debug(
            f"Agent - Default config: {agent_config}"
        )
        return agent_config

    def _create_default_enhanced_search_config(self) -> EnhancedSearchConfig:
        """Create default enhanced search configuration from environment."""
        try:
            # Initialize template manager to get templates
            template_manager = SearchTemplateManager()
            templates = template_manager.get_all_templates()

            # Create enhanced search config
            enhanced_config = EnhancedSearchConfig(
                templates=templates,
                local_repo_path=get_envvar(
                    "ENHANCED_SEARCH_LOCAL_REPO_PATH", "local_repo_files"
                ),
                max_context_length=int(
                    get_envvar("ENHANCED_SEARCH_MAX_CONTEXT_LENGTH", "10000")),
                fallback_enabled=get_envvar(
                    "ENHANCED_SEARCH_FALLBACK_ENABLED", "true"
                ).lower() == "true",
                search_result_limit=int(
                    get_envvar("ENHANCED_SEARCH_RESULT_LIMIT", "10")
                ),
                context_determination_enabled=get_envvar(
                    "ENHANCED_SEARCH_CONTEXT_DETERMINATION_ENABLED", "true"
                ).lower() == "true",
                document_retrieval_enabled=get_envvar(
                    "ENHANCED_SEARCH_ENABLE_DOC_RETRIEVAL", "true"
                ).lower() == "true",
                similarity_threshold=float(
                    get_envvar("ENHANCED_SEARCH_SIMILARITY_THRESHOLD", "0.7")
                ),
                merge_strategy=get_envvar(
                    "ENHANCED_SEARCH_MERGE_STRATEGY", "prioritize_context"
                )
            )

            _ = DEBUG and log_debug(
                f"Agent - Enhanced search config: "
                f"local_repo_path={enhanced_config.local_repo_path}, "
                f"fallback_enabled={enhanced_config.fallback_enabled}, "
                f"document_retrieval="
                f"{enhanced_config.document_retrieval_enabled}"
            )

            return enhanced_config

        except Exception as e:
            log_warning(f"Failed to create enhanced search config: {e}")
            # Return minimal config with fallback enabled
            return EnhancedSearchConfig(
                templates={},
                local_repo_path="local_repo_files",
                max_context_length=10000,
                fallback_enabled=True
            )

    def _initialize_model(self) -> Model:
        """
        Initialize the LLM model based on configuration.

        Returns:
            Model: Configured Pydantic AI model.

        Raises:
            ValueError: If model configuration is invalid.
        """
        model_params = {
            "api_key": get_envvar("OPENAI_API_KEY"),
            "base_url": get_envvar("OPENAI_BASE_URL")
        }
        inference_params = {
            "model_name": get_envvar("OPENAI_MODEL_NAME",
                                     self.config.model_name),

        }
        if self.config.model_provider == "openai":
            pass
        elif self.config.model_provider == "huggingface":
            model_params = {
                "api_key": get_envvar("HF_TOKEN"),
                "base_url": get_envvar("HF_BASE_URL",
                                       "https://router.huggingface.co/v1")
            }
            inference_params = {
                "model_name": get_envvar("HF_MODEL_NAME",
                                         self.config.model_name),
            }
        elif self.config.model_provider == "groq":
            model_params = {
                "api_key": get_envvar("GROQ_API_KEY"),
                "base_url": get_envvar("GROQ_BASE_URL",
                                       "https://api.groq.com/openai/v1")
            }
            inference_params = {
                "model_name": get_envvar("GROQ_MODEL_NAME",
                                         self.config.model_name),
            }
        elif self.config.model_provider == "aimlapi":
            model_params = {
                "api_key": get_envvar("AIMLAPI_API_KEY"),
                "base_url": get_envvar("AIMLAPI_BASE_URL",
                                       "https://api.aimlapi.com/v1")
            }
            inference_params = {
                "model_name": get_envvar("AIMLAPI_MODEL_NAME",
                                         self.config.model_name),
            }
        elif self.config.model_provider == "openrouter":
            model_params = {
                "api_key": get_envvar("OPENROUTER_API_KEY"),
                "base_url": get_envvar("OPENROUTER_BASE_URL",
                                       "https://openrouter.ai/api/v1")
            }
            inference_params = {
                "model_name": get_envvar("OPENROUTER_MODEL_NAME",
                                         self.config.model_name),
            }
        elif self.config.model_provider == "together":
            model_params = {
                "api_key": get_envvar("TOGETHER_API_KEY"),
                "base_url": get_envvar("TOGETHER_BASE_URL",
                                       "https://api.together.xyz/v1"),
            }
            inference_params = {
                "model_name": get_envvar("TOGETHER_MODEL_NAME",
                                         self.config.model_name),
                "stop": json.loads(
                    get_envvar("TOGETHER_STOP",
                               '["<|eot_id|>", "<|eom_id|>"]'))
            }
        elif self.config.model_provider == "nvidia":
            model_params = {
                "api_key": get_envvar("NVIDIA_API_KEY"),
                "base_url": get_envvar(
                    "NVIDIA_BASE_URL",
                    "https://integrate.api.nvidia.com/v1")
            }
            inference_params = {
                "model_name": get_envvar("NVIDIA_MODEL_NAME",
                                         self.config.model_name),
            }
        elif self.config.model_provider == "xai":
            model_params = {
                "api_key": get_envvar("XAI_API_KEY"),
                "base_url": get_envvar("XAI_BASE_URL",
                                       "https://api.x.ai/v1")
            }
            inference_params = {
                "model_name": get_envvar("XAI_MODEL_NAME",
                                         self.config.model_name),
            }
        elif self.config.model_provider == "rhymes":
            model_params = {
                "api_key": get_envvar("RHYMES_API_KEY"),
                "base_url": get_envvar("RHYMES_BASE_URL",
                                       "https://api.rhymes.ai/v1")
            }
            inference_params = {
                "model_name": get_envvar("RHYMES_MODEL_NAME",
                                         self.config.model_name),
            }
        elif self.config.model_provider == "ollama":
            model_params = {
                "base_url": get_envvar("OLLAMA_BASE_URL",
                                       "http://localhost:11434/v1")
            }
            inference_params = {
                "model_name": get_envvar("OLLAMA_MODEL_NAME",
                                         self.config.model_name),
            }
        else:
            log_warning(
                "Unsupported provider "
                f"{self.config.model_provider}, falling back to OpenAI"
            )

        patch_tokenizers()
        patch_openai_service_tier()

        try:
            if self.config.model_api == "openai":
                return self._create_openai_model(
                    model_params, inference_params)
            elif self.config.model_api == "litellm":
                return self._create_litellm_model(
                    model_params, inference_params)
            else:
                log_warning(
                    "Unsupported API "
                    f"{self.config.model_api}, falling back to OpenAI"
                )
                return self._create_openai_model(model_params)

        except Exception as e:
            log_error(f"Failed to initialize model: {e}")
            raise ValueError(f"Model initialization failed: {e}")

    def _create_openai_model(
        self,
        model_params: dict,
        inference_params: dict
    ) -> OpenAIChatModel:
        """Create OpenAI model configuration."""
        if model_params is None:
            model_params = {}
        if inference_params is None:
            inference_params = {}

        self.inference_args["temperature"] = self.config.temperature
        self.inference_args["timeout"] = self.config.timeout
        if self.config.max_tokens:
            self.inference_args["max_tokens"] = self.config.max_tokens
        if self.config.stop:
            self.inference_args["stop"] = self.config.stop

        model_kwargs = {}

        if self.config.api_key:
            model_kwargs["api_key"] = self.config.api_key

        if self.config.base_url is None or self.config.base_url == '':
            model_kwargs["base_url"] = 'https://api.openai.com/v1'
        else:
            model_kwargs["base_url"] = self.config.base_url

        model_kwargs.update(model_params)
        self.inference_args.update(inference_params)

        self.config.model_name = self.inference_args["model_name"]
        self.config.base_url = model_kwargs["base_url"]

        _ = DEBUG and log_debug(f"Agent - Model kwargs: {model_kwargs}"
                                + f"\nInference Args: {self.inference_args}")

        return OpenAIChatModel(
            self.config.model_name,
            provider=OpenAIProvider(**model_kwargs)
        )

    def _create_litellm_model(
        self,
        model_params: dict = None,
        inference_params: dict = None
    ) -> Model:
        """Create LiteLLM model configuration."""
        # TODO: Note: This would need to be implemented based on Pydantic AI's
        # LiteLLM support. For now, fall back to OpenAI
        try:
            # type: ignore[import]
            import litellm
            _ = DEBUG and log_debug(f"LiteLLM loaded: {litellm}")
            # LITELLM_AVAILABLE = True
        except ImportError:
            # LITELLM_AVAILABLE = False
            log_warning("LiteLLM not available, using OpenAI only")

        log_warning("LiteLLM integration not yet implemented, using OpenAI")
        return self._create_openai_model(model_params, inference_params)


def get_llm_data(model_name: str, llm_provider: str) -> dict[str, int]:
    """
    Get LLM data from JSON file.

    Args:
        model_name: LLM model name.
        llm_provider: LLM provider.

    Returns:
        dict[str, int]: LLM data.
    """
    llm_data = get_all_llm_data()
    default_llm_data = {
        "input_tokens_price": None,
        "output_tokens_price": None,
        "context_window_size": CONTEXT_DEFAULT_MAX_LENGTH,
    }
    return LlmData(**llm_data.get(
        llm_provider, {}).get(
            "models", {}).get(
                model_name, default_llm_data))


def get_all_llm_data() -> dict[str, int]:
    """
    Get LLM data from JSON file.

    Returns:
        dict[str, int]: LLM data. For example:
        ```json
        {
            "openai": {
                "price_per_n_tokens": 1000000,
                "models": {
                    "gpt-5-mini": {
                        "input_tokens_price": 0.25,
                        "output_tokens_price": 2.00,
                        "context_window_size": 400000
                    },
                    "gpt-4.1-nano": {
                        "input_tokens_price": 0.10,
                        "output_tokens_price": 0.40,
                        "context_window_size": 1047576
                    },
                    "gpt-4o-mini": {
                        "input_tokens_price": 0.15,
                        "output_tokens_price": 0.60,
                        "context_window_size": 128000
                    },
                },
            },
        }
        ```

    """
    lib_root_dir = Path(Path(__file__).parent).parent
    with open(f"{lib_root_dir}/assets/llm_models_data.json", "r") as f:
        return json.load(f)


def create_agent_config_from_env() -> AgentConfig:
    """
    Create agent configuration from environment variables.

    Returns:
        AgentConfig: Configuration from environment.
    """
    agent = AgentSuper()
    return agent.config


def get_context_default_max_length() -> int:
    """
    Get context default max length from environment variables.

    Returns:
        int: Context default max length from environment.
    """
    agent = AgentSuper()
    return agent.llm_data.context_window_size


def get_tool_context_default_max_length() -> int:
    """
    Get tool context default max length

    Returns:
        int: Tool context default max length minus 
                a buffer of 500 tokens.
    """
    return get_context_default_max_length() - 500
