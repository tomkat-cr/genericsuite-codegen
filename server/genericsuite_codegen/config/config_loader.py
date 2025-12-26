"""Configuration loader for enhanced search functionality."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    log_warning,
    log_error,
)

DEBUG = False


@dataclass
class EnhancedSearchConfig:
    """Enhanced search configuration data class."""

    # Core settings
    enabled: bool = True
    max_context_length: int = 10000
    fallback_enabled: bool = True
    dual_search_enabled: bool = True
    context_determination_enabled: bool = True
    document_retrieval_enabled: bool = True

    # Local storage settings
    local_repo_path: str = "local_repo_files"
    max_file_size_mb: int = 10
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        ".md", ".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".yaml", ".yml",
        ".txt", ".rst", ".html", ".css", ".sh", ".toml", ".cfg", ".ini"
    ])
    excluded_directories: List[str] = field(default_factory=lambda: [
        ".git", "__pycache__", "node_modules", ".pytest_cache",
        "dist", "build", ".vscode", ".idea"
    ])

    # Performance settings
    max_concurrent_searches: int = 5
    search_timeout_seconds: int = 30
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600

    # Context determination settings
    confidence_threshold: float = 0.6
    default_context: str = "generic"
    context_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "json": ["json", "configuration", "config", "table", "schema"],
        "langchain": ["langchain", "tool", "chain", "agent"],
        "mcp": ["mcp", "server", "protocol", "model context"],
        "frontend": ["react", "component", "ui", "frontend", "client"],
        "backend": ["fastapi", "api", "server", "backend", "endpoint"],
        "frontend_ai": ["react", "ai", "chat", "llm", "frontend"],
        "backend_ai": ["fastapi", "ai", "agent", "llm", "backend"]
    })

    # Logging settings
    log_level: str = "INFO"
    log_search_queries: bool = True
    log_document_retrieval: bool = True
    log_context_determination: bool = True
    log_performance_metrics: bool = True


class ConfigLoader:
    """Configuration loader for enhanced search functionality."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration loader.

        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)
        self._config_cache: Dict[str, Any] = {}

    def load_enhanced_search_config(
        self,
        config_file: Optional[str] = None,
        environment: Optional[str] = None
    ) -> EnhancedSearchConfig:
        """Load enhanced search configuration.

        Args:
            config_file: Specific config file to load
            environment: Environment-specific config (development,
                production, docker)

        Returns:
            EnhancedSearchConfig instance
        """
        try:
            # Determine config file to load
            if config_file is None:
                config_file = self._get_config_filename(environment)

            # Load from file
            config_data = self._load_config_file(config_file)

            # Load from environment variables (overrides file config)
            env_config = self._load_from_environment()

            # Merge configurations (env overrides file)
            merged_config = self._merge_configs(config_data, env_config)

            # Create config object
            return self._create_config_object(merged_config)

        except Exception as e:
            log_warning(f"Failed to load enhanced search config: {e}"
                        + "\nUsing default enhanced search configuration")
            return EnhancedSearchConfig()

    def load_search_templates(
        self,
        templates_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load search templates configuration.

        Args:
            templates_file: Specific templates file to load

        Returns:
            Dictionary containing search templates
        """
        try:
            if templates_file is None:
                templates_file = "search_templates.json"

            return self._load_config_file(templates_file)

        except Exception as e:
            log_warning(f"Failed to load search templates: {e}"
                        + "\nUsing default search templates")
            return self._get_default_templates()

    def _get_config_filename(self, environment: Optional[str]) -> str:
        """Get configuration filename based on environment."""
        if environment:
            return f"enhanced_search_config.{environment}.json"

        # Check environment variable
        env_name = os.getenv("ENVIRONMENT", "").lower()
        if env_name in ["development", "dev"]:
            return "enhanced_search_config.development.json"
        elif env_name in ["production", "prod"]:
            return "enhanced_search_config.production.json"
        elif env_name == "docker":
            return "enhanced_search_config.docker.json"

        return "enhanced_search_config.json"

    def _load_config_file(self, filename: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        cache_key = f"file:{filename}"
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        config_path = self.config_dir / filename

        if not config_path.exists():
            log_warning(f"Config file not found: {config_path}")
            return {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            self._config_cache[cache_key] = config_data
            _ = DEBUG and log_debug(
                f"Loaded configuration from: {config_path}")
            return config_data

        except json.JSONDecodeError as e:
            log_error(f"Invalid JSON in config file {config_path}: {e}")
            return {}
        except Exception as e:
            log_error(f"Error loading config file {config_path}: {e}")
            return {}

    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}

        # Enhanced search settings
        if os.getenv("ENHANCED_SEARCH_ENABLED"):
            env_config.setdefault(
                "enhanced_search", {})["enabled"] = (
                os.getenv("ENHANCED_SEARCH_ENABLED", "true").lower() == "true"
            )

        if os.getenv("ENHANCED_SEARCH_MAX_CONTEXT_LENGTH"):
            env_config.setdefault(
                "enhanced_search", {})["max_context_length"] = int(
                os.getenv("ENHANCED_SEARCH_MAX_CONTEXT_LENGTH", "10000")
            )

        if os.getenv("ENHANCED_SEARCH_FALLBACK_ENABLED"):
            env_config.setdefault(
                "enhanced_search", {})["fallback_enabled"] = (
                os.getenv("ENHANCED_SEARCH_FALLBACK_ENABLED",
                          "true").lower() == "true"
            )

        # Local storage settings
        if os.getenv("LOCAL_REPO_DIR"):
            env_config.setdefault("local_storage", {})["local_repo_path"] = (
                os.getenv("LOCAL_REPO_DIR", "local_repo_files")
            )

        if os.getenv("DOCUMENT_RETRIEVAL_MAX_FILE_SIZE_MB"):
            env_config.setdefault(
                "local_storage", {})["max_file_size_mb"] = int(
                os.getenv("DOCUMENT_RETRIEVAL_MAX_FILE_SIZE_MB", "10")
            )

        # Performance settings
        if os.getenv("SEARCH_MAX_CONCURRENT_SEARCHES"):
            env_config.setdefault(
                "search_performance", {})["max_concurrent_searches"] = int(
                os.getenv("SEARCH_MAX_CONCURRENT_SEARCHES", "5")
            )

        if os.getenv("SEARCH_TIMEOUT_SECONDS"):
            env_config.setdefault(
                "search_performance", {})["search_timeout_seconds"] = int(
                os.getenv("SEARCH_TIMEOUT_SECONDS", "30")
            )

        # Context determination settings
        if os.getenv("CONTEXT_DETERMINATION_CONFIDENCE_THRESHOLD"):
            env_config.setdefault(
                "context_determination", {})["confidence_threshold"] = float(
                os.getenv("CONTEXT_DETERMINATION_CONFIDENCE_THRESHOLD", "0.6")
            )

        # Logging settings
        if os.getenv("ENHANCED_SEARCH_LOG_LEVEL"):
            env_config.setdefault("logging", {})["level"] = (
                os.getenv("ENHANCED_SEARCH_LOG_LEVEL", "INFO")
            )

        return env_config

    def _merge_configs(self, file_config: Dict[str, Any],
                       env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge file and environment configurations."""
        merged = file_config.copy()

        for section, values in env_config.items():
            if section in merged and isinstance(merged[section], dict):
                merged[section].update(values)
            else:
                merged[section] = values

        return merged

    def _create_config_object(self, config_data: Dict[str, Any]
                              ) -> EnhancedSearchConfig:
        """Create EnhancedSearchConfig object from configuration data."""
        config = EnhancedSearchConfig()

        # Enhanced search settings
        if "enhanced_search" in config_data:
            es_config = config_data["enhanced_search"]
            config.enabled = es_config.get("enabled", config.enabled)
            config.max_context_length = es_config.get(
                "max_context_length", config.max_context_length)
            config.fallback_enabled = es_config.get(
                "fallback_enabled", config.fallback_enabled)
            config.dual_search_enabled = es_config.get(
                "dual_search_enabled", config.dual_search_enabled)
            config.context_determination_enabled = es_config.get(
                "context_determination_enabled",
                config.context_determination_enabled)
            config.document_retrieval_enabled = es_config.get(
                "document_retrieval_enabled",
                config.document_retrieval_enabled)

        # Local storage settings
        if "local_storage" in config_data:
            ls_config = config_data["local_storage"]
            config.local_repo_path = ls_config.get(
                "local_repo_path", config.local_repo_path)
            config.max_file_size_mb = ls_config.get(
                "max_file_size_mb", config.max_file_size_mb)
            config.allowed_file_extensions = ls_config.get(
                "allowed_file_extensions", config.allowed_file_extensions)
            config.excluded_directories = ls_config.get(
                "excluded_directories", config.excluded_directories)

        # Performance settings
        if "search_performance" in config_data:
            sp_config = config_data["search_performance"]
            config.max_concurrent_searches = sp_config.get(
                "max_concurrent_searches", config.max_concurrent_searches)
            config.search_timeout_seconds = sp_config.get(
                "search_timeout_seconds", config.search_timeout_seconds)
            config.cache_enabled = sp_config.get(
                "cache_enabled", config.cache_enabled)
            config.cache_ttl_seconds = sp_config.get(
                "cache_ttl_seconds", config.cache_ttl_seconds)

        # Context determination settings
        if "context_determination" in config_data:
            cd_config = config_data["context_determination"]
            config.confidence_threshold = cd_config.get(
                "confidence_threshold", config.confidence_threshold)
            config.default_context = cd_config.get(
                "default_context", config.default_context)
            config.context_keywords = cd_config.get(
                "context_keywords", config.context_keywords)

        # Logging settings
        if "logging" in config_data:
            log_config = config_data["logging"]
            config.log_level = log_config.get("level", config.log_level)
            config.log_search_queries = log_config.get(
                "log_search_queries", config.log_search_queries)
            config.log_document_retrieval = log_config.get(
                "log_document_retrieval", config.log_document_retrieval)
            config.log_context_determination = log_config.get(
                "log_context_determination", config.log_context_determination)
            config.log_performance_metrics = log_config.get(
                "log_performance_metrics", config.log_performance_metrics)

        return config

    def _get_default_templates(self) -> Dict[str, Any]:
        """Get default search templates."""
        return {
            "templates": {
                "json": {
                    "template": "examples of how to create a JSON table"
                    + " configuration files in Genericsuite",
                    "file_type_filter": "json",
                    "priority": 1
                },
                "langchain": {
                    "template": "examples of how to create a Python Langchain"
                    + " Tool in Genericsuite",
                    "file_type_filter": "py",
                    "priority": 1
                },
                "mcp": {
                    "template": "examples of how to create a MCP server"
                    + "tool in Genericsuite",
                    "file_type_filter": "py",
                    "priority": 1
                },
                "frontend": {
                    "template": "examples of how to create frontend code"
                    + " in Genericsuite",
                    "file_type_filter": "jsx",
                    "priority": 1
                },
                "backend": {
                    "template": "examples of how to create backend code"
                    + " in Genericsuite",
                    "file_type_filter": "py",
                    "priority": 1
                },
                "generic": {
                    "template": "examples and rules for creating code"
                    + " in Genericsuite",
                    "file_type_filter": None,
                    "priority": 0
                }
            }
        }

    def reload_config(self) -> None:
        """Clear configuration cache to force reload."""
        self._config_cache.clear()
        _ = DEBUG and log_debug("Configuration cache cleared")
