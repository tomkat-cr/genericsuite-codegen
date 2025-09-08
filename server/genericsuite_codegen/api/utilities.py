"""
Shared utility functions for the FastAPI application.

This module provides common utilities for logging, configuration,
request handling, and other shared functionality across the API.
"""

import os
import logging
import uuid
import time
from typing import Dict, Any, Optional
from datetime import datetime
import json

from .types import AppInfo


def setup_logging() -> None:
    """
    Setup application logging configuration.
    """
    # Get log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    debug_mode = os.getenv("SERVER_DEBUG", "0") == "1"
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if debug_mode:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    
    # Setup basic logging
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    if not debug_mode:
        # Reduce noise from external libraries in production
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("pymongo").setLevel(logging.WARNING)


def get_app_info() -> AppInfo:
    """
    Get application information.
    
    Returns:
        AppInfo: Application information object.
    """
    return AppInfo(
        name=os.getenv("APP_NAME", "GenericSuite CodeGen"),
        version="1.0.0",  # Should match pyproject.toml
        description="RAG AI system for GenericSuite documentation and code generation",
        docs_url="/docs",
        health_url="/health"
    )


def create_correlation_id() -> str:
    """
    Create a unique correlation ID for request tracking.
    
    Returns:
        str: Unique correlation ID.
    """
    return str(uuid.uuid4())


def log_request_response(
    method: str,
    url: str,
    correlation_id: str,
    event_type: str,
    status_code: Optional[int] = None,
    duration: Optional[float] = None,
    user_id: Optional[str] = None
) -> None:
    """
    Log request and response information.
    
    Args:
        method: HTTP method.
        url: Request URL.
        correlation_id: Request correlation ID.
        event_type: Type of event (request/response).
        status_code: HTTP status code (for responses).
        duration: Request duration in seconds (for responses).
        user_id: User ID if available.
    """
    logger = logging.getLogger("api.requests")
    
    log_data = {
        "correlation_id": correlation_id,
        "method": method,
        "url": url,
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if status_code is not None:
        log_data["status_code"] = status_code
    
    if duration is not None:
        log_data["duration"] = duration
    
    if user_id:
        log_data["user_id"] = user_id
    
    logger.info(json.dumps(log_data))


def validate_environment() -> Dict[str, Any]:
    """
    Validate required environment variables.
    
    Returns:
        Dict[str, Any]: Validation results with missing variables and warnings.
    """
    required_vars = [
        "MONGODB_URI",
        "OPENAI_API_KEY"
    ]
    
    optional_vars = [
        "HF_TOKEN",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "EMBEDDINGS_PROVIDER",
        "EMBEDDINGS_MODEL"
    ]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    return {
        "valid": len(missing_required) == 0,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "warnings": []
    }


def get_database_config() -> Dict[str, Any]:
    """
    Get database configuration from environment.
    
    Returns:
        Dict[str, Any]: Database configuration.
    """
    return {
        "uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
        "database_name": os.getenv("DATABASE_NAME", "genericsuite_codegen"),
        "connection_timeout": int(os.getenv("DB_CONNECTION_TIMEOUT", "10")),
        "server_selection_timeout": int(os.getenv("DB_SERVER_SELECTION_TIMEOUT", "5"))
    }


def get_agent_config() -> Dict[str, Any]:
    """
    Get agent configuration from environment.
    
    Returns:
        Dict[str, Any]: Agent configuration.
    """
    return {
        "provider": os.getenv("LLM_PROVIDER", "openai"),
        "model": os.getenv("LLM_MODEL", "gpt-4"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4000")) if os.getenv("LLM_MAX_TOKENS") else None,
        "timeout": int(os.getenv("LLM_TIMEOUT", "60")),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL")
    }


def get_embedding_config() -> Dict[str, Any]:
    """
    Get embedding configuration from environment.
    
    Returns:
        Dict[str, Any]: Embedding configuration.
    """
    return {
        "provider": os.getenv("EMBEDDINGS_PROVIDER", "openai"),
        "model": os.getenv("EMBEDDINGS_MODEL", "text-embedding-ada-002"),
        "api_key": os.getenv("OPENAI_API_KEY") if os.getenv("EMBEDDINGS_PROVIDER", "openai") == "openai" else os.getenv("HF_TOKEN"),
        "dimension": int(os.getenv("EMBEDDINGS_DIMENSION", "1536"))
    }


def get_server_config() -> Dict[str, Any]:
    """
    Get server configuration from environment.
    
    Returns:
        Dict[str, Any]: Server configuration.
    """
    return {
        "host": os.getenv("SERVER_HOST", "0.0.0.0"),
        "port": int(os.getenv("SERVER_PORT", "8000")),
        "debug": os.getenv("SERVER_DEBUG", "0") == "1",
        "cors_origins": os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
        "allowed_hosts": os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(","),
        "max_request_size": int(os.getenv("MAX_REQUEST_SIZE", "10485760"))  # 10MB default
    }


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes.
        
    Returns:
        str: Formatted size string.
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def sanitize_path(path: str) -> str:
    """
    Sanitize file path to prevent directory traversal.
    
    Args:
        path: Original file path.
        
    Returns:
        str: Sanitized path.
    """
    import os.path
    
    # Remove any path traversal attempts
    path = path.replace("../", "").replace("..\\", "")
    
    # Normalize path separators
    path = os.path.normpath(path)
    
    # Remove leading slashes
    path = path.lstrip("/\\")
    
    return path


def validate_json_content(content: str) -> tuple[bool, Optional[str]]:
    """
    Validate JSON content.
    
    Args:
        content: JSON content to validate.
        
    Returns:
        tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        json.loads(content)
        return True, None
    except json.JSONDecodeError as e:
        return False, str(e)


def extract_code_blocks(content: str, language: Optional[str] = None) -> list[Dict[str, str]]:
    """
    Extract code blocks from markdown content.
    
    Args:
        content: Markdown content.
        language: Optional language filter.
        
    Returns:
        list[Dict[str, str]]: List of code blocks with metadata.
    """
    import re
    
    # Pattern to match code blocks
    pattern = r'```(\w+)?\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    code_blocks = []
    for lang, code in matches:
        if language is None or lang == language:
            code_blocks.append({
                "language": lang or "text",
                "code": code.strip()
            })
    
    return code_blocks


def create_file_hash(content: str) -> str:
    """
    Create a hash for file content.
    
    Args:
        content: File content.
        
    Returns:
        str: SHA-256 hash of the content.
    """
    import hashlib
    
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def measure_execution_time(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure.
        
    Returns:
        Decorated function that logs execution time.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger("performance")
        logger.info(f"{func.__name__} executed in {end_time - start_time:.3f} seconds")
        
        return result
    
    return wrapper


async def async_measure_execution_time(func):
    """
    Decorator to measure async function execution time.
    
    Args:
        func: Async function to measure.
        
    Returns:
        Decorated async function that logs execution time.
    """
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger("performance")
        logger.info(f"{func.__name__} executed in {end_time - start_time:.3f} seconds")
        
        return result
    
    return wrapper


def get_client_ip(request) -> str:
    """
    Get client IP address from request.
    
    Args:
        request: FastAPI request object.
        
    Returns:
        str: Client IP address.
    """
    # Check for forwarded headers first (for reverse proxy setups)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fall back to direct client IP
    return request.client.host if request.client else "unknown"


def create_pagination_info(
    total: int,
    page: int,
    page_size: int
) -> Dict[str, Any]:
    """
    Create pagination information.
    
    Args:
        total: Total number of items.
        page: Current page number (1-based).
        page_size: Number of items per page.
        
    Returns:
        Dict[str, Any]: Pagination information.
    """
    total_pages = (total + page_size - 1) // page_size
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
        "next_page": page + 1 if page < total_pages else None,
        "prev_page": page - 1 if page > 1 else None
    }


def validate_pagination_params(
    page: int,
    page_size: int,
    max_page_size: int = 100
) -> tuple[int, int]:
    """
    Validate and normalize pagination parameters.
    
    Args:
        page: Page number.
        page_size: Items per page.
        max_page_size: Maximum allowed page size.
        
    Returns:
        tuple[int, int]: Validated (page, page_size).
    """
    # Ensure page is at least 1
    page = max(1, page)
    
    # Ensure page_size is within bounds
    page_size = max(1, min(page_size, max_page_size))
    
    return page, page_size


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window.
            window_seconds: Time window in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.
        
        Args:
            client_id: Client identifier.
            
        Returns:
            bool: True if request is allowed.
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old entries
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > window_start
            ]
        else:
            self.requests[client_id] = []
        
        # Check if under limit
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True
        
        return False
    
    def get_reset_time(self, client_id: str) -> Optional[float]:
        """
        Get time when rate limit resets for client.
        
        Args:
            client_id: Client identifier.
            
        Returns:
            Optional[float]: Reset time as timestamp, None if no limit.
        """
        if client_id not in self.requests or not self.requests[client_id]:
            return None
        
        oldest_request = min(self.requests[client_id])
        return oldest_request + self.window_seconds


# Global rate limiter instance
rate_limiter = RateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
    window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60"))
)