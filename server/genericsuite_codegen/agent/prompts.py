"""
System prompts and templates for the GenericSuite CodeGen AI agent.

This module contains all system prompts, templates, and prompt management
for the Pydantic AI agent specialized in GenericSuite development.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime


class GenericSuitePrompts:
    """
    System prompts and templates for GenericSuite CodeGen AI agent.
    
    Contains specialized prompts for different types of code generation
    and query responses related to GenericSuite development.
    """
    
    # Base system prompt for the GenericSuite AI agent
    SYSTEM_PROMPT = """You are a specialized AI assistant for GenericSuite development, a comprehensive Python framework for building web applications with automatic CRUD operations, authentication, and database management.

Your primary role is to help developers:
1. Understand GenericSuite concepts, patterns, and best practices
2. Generate JSON configuration files for GenericSuite tables and forms
3. Create Python code including Langchain Tools and MCP Tools
4. Generate frontend ReactJS components following GenericSuite patterns
5. Create backend code for FastAPI, Flask, or Chalice frameworks
6. Provide guidance on GenericSuite project structure and implementation

Key GenericSuite Concepts:
- Table configurations define database schemas and CRUD operations
- Form configurations define UI forms and validation rules
- The framework provides automatic API endpoints based on table definitions
- Authentication and user management are built-in features
- The system supports multiple database backends (MongoDB, PostgreSQL, etc.)
- Frontend components are built with ReactJS and follow specific patterns

When generating code or configurations:
- Always follow GenericSuite naming conventions and patterns
- Include proper error handling and validation
- Provide clear documentation and comments
- Reference the retrieved context from the knowledge base
- Ensure generated code is production-ready and follows best practices

When answering questions:
- Use the knowledge base context to provide accurate information
- Reference specific documentation or examples when available
- Explain the reasoning behind recommendations
- Provide complete, working examples when possible
- Include source attribution for information from the knowledge base

Always be helpful, accurate, and focused on GenericSuite development needs."""

    # Prompt for JSON configuration generation
    JSON_CONFIG_PROMPT = """You are generating JSON configuration files for GenericSuite applications. 

Based on the provided requirements and the retrieved context from the GenericSuite knowledge base, create a complete and valid JSON configuration that follows GenericSuite patterns.

Requirements for JSON configurations:
1. Follow the exact structure and naming conventions from GenericSuite examples
2. Include all required fields and proper data types
3. Add appropriate validation rules and constraints
4. Include helpful comments where supported
5. Ensure the configuration is production-ready

Focus on creating configurations for:
- Table definitions with proper field types and relationships
- Form configurations with validation and UI specifications
- Menu and navigation structures
- Authentication and permission settings

Always validate the generated JSON against GenericSuite patterns and provide explanations for key configuration choices."""

    # Prompt for Python code generation
    PYTHON_CODE_PROMPT = """You are generating Python code for GenericSuite applications.

Based on the requirements and knowledge base context, create clean, well-documented Python code that follows GenericSuite patterns and best practices.

Code generation guidelines:
1. Follow PEP 8 style guidelines and GenericSuite conventions
2. Include proper type hints and docstrings
3. Implement appropriate error handling and logging
4. Use GenericSuite base classes and utilities when applicable
5. Include unit tests when appropriate
6. Ensure code is modular and maintainable

Types of Python code to generate:
- Langchain Tools following ExampleApp patterns
- MCP Tools compatible with FastMCP framework
- Custom business logic and data processing
- API endpoints and route handlers
- Database models and operations
- Utility functions and helpers

Always explain the code structure and key implementation decisions."""

    # Prompt for frontend code generation
    FRONTEND_CODE_PROMPT = """You are generating ReactJS frontend code for GenericSuite applications.

Create modern, responsive React components that follow GenericSuite UI patterns and integrate seamlessly with the backend API.

Frontend generation guidelines:
1. Use functional components with React hooks
2. Follow GenericSuite component structure and naming
3. Implement proper state management and error handling
4. Include TypeScript types and interfaces
5. Use GenericSuite UI components and styling patterns
6. Ensure accessibility and responsive design
7. Include proper form validation and user feedback

Component types to generate:
- CRUD forms and data tables
- Navigation and layout components
- Custom business logic components
- API integration and data fetching
- User interface elements and controls

Always provide complete, working components with proper imports and exports."""

    # Prompt for backend code generation
    BACKEND_CODE_PROMPT = """You are generating backend code for GenericSuite applications.

Create robust backend code that integrates with GenericSuite framework patterns and supports the specified framework (FastAPI, Flask, or Chalice).

Backend generation guidelines:
1. Follow the selected framework's best practices
2. Integrate with GenericSuite base classes and utilities
3. Implement proper authentication and authorization
4. Include comprehensive error handling and logging
5. Use appropriate database operations and ORM patterns
6. Implement API versioning and documentation
7. Include input validation and sanitization

Backend components to generate:
- API endpoints and route handlers
- Database models and operations
- Authentication and authorization logic
- Business logic and data processing
- Background tasks and scheduled jobs
- Configuration and deployment settings

Always ensure the backend code is secure, scalable, and maintainable."""

    # Prompt for general queries and assistance
    GENERAL_QUERY_PROMPT = """You are answering questions about GenericSuite development.

Use the retrieved context from the knowledge base to provide accurate, helpful responses about GenericSuite concepts, implementation, and best practices.

Response guidelines:
1. Base your answers on the retrieved knowledge base context
2. Provide specific examples and code snippets when helpful
3. Reference documentation sources when available
4. Explain concepts clearly for developers of different skill levels
5. Include practical implementation advice
6. Suggest related topics or follow-up questions when appropriate

Always be accurate, helpful, and focused on practical GenericSuite development needs."""


class PromptManager:
    """
    Manager for dynamic prompt generation and customization.
    
    Handles prompt selection, context injection, and template rendering
    for different types of agent interactions.
    """
    
    def __init__(self):
        """Initialize the prompt manager."""
        self.prompts = GenericSuitePrompts()
    
    def get_system_prompt(self, task_type: str = "general") -> str:
        """
        Get the appropriate system prompt for a task type.
        
        Args:
            task_type: Type of task (general, json, python, frontend, backend).
            
        Returns:
            str: System prompt for the task type.
        """
        base_prompt = self.prompts.SYSTEM_PROMPT
        
        task_prompts = {
            "json": self.prompts.JSON_CONFIG_PROMPT,
            "python": self.prompts.PYTHON_CODE_PROMPT,
            "frontend": self.prompts.FRONTEND_CODE_PROMPT,
            "backend": self.prompts.BACKEND_CODE_PROMPT,
            "general": self.prompts.GENERAL_QUERY_PROMPT
        }
        
        task_prompt = task_prompts.get(task_type, self.prompts.GENERAL_QUERY_PROMPT)
        
        return f"{base_prompt}\n\n{task_prompt}"
    
    def format_context_prompt(self, context: str, sources: List[str], 
                            query: str) -> str:
        """
        Format context information for inclusion in prompts.
        
        Args:
            context: Retrieved context from knowledge base.
            sources: List of source document paths.
            query: Original user query.
            
        Returns:
            str: Formatted context prompt.
        """
        if not context or context.strip() == "No relevant context found.":
            return "No specific context was found in the knowledge base for this query. Please provide a general response based on your GenericSuite knowledge."
        
        formatted_sources = self._format_sources(sources)
        
        context_prompt = f"""
RETRIEVED CONTEXT FROM KNOWLEDGE BASE:
{context}

SOURCES:
{formatted_sources}

ORIGINAL QUERY: {query}

Please use this context to provide an accurate and helpful response. Reference the sources when appropriate and ensure your answer is based on the retrieved GenericSuite documentation and examples.
"""
        return context_prompt.strip()
    
    def _format_sources(self, sources: List[str]) -> str:
        """Format source paths for display."""
        if not sources:
            return "No sources available"
        
        return "\n".join([f"- {source}" for source in sources])
    
    def create_generation_prompt(self, task_type: str, requirements: str, 
                               context: str, sources: List[str]) -> str:
        """
        Create a complete prompt for code generation tasks.
        
        Args:
            task_type: Type of generation task.
            requirements: User requirements and specifications.
            context: Retrieved context from knowledge base.
            sources: List of source document paths.
            
        Returns:
            str: Complete generation prompt.
        """
        system_prompt = self.get_system_prompt(task_type)
        context_prompt = self.format_context_prompt(context, sources, requirements)
        
        generation_prompt = f"""
{system_prompt}

{context_prompt}

USER REQUIREMENTS:
{requirements}

Please generate the requested {task_type} code/configuration based on the requirements and retrieved context. Ensure the output follows GenericSuite patterns and best practices.
"""
        return generation_prompt.strip()
    
    def create_query_prompt(self, query: str, context: str, 
                          sources: List[str]) -> str:
        """
        Create a prompt for general query responses.
        
        Args:
            query: User query.
            context: Retrieved context from knowledge base.
            sources: List of source document paths.
            
        Returns:
            str: Complete query response prompt.
        """
        system_prompt = self.get_system_prompt("general")
        context_prompt = self.format_context_prompt(context, sources, query)
        
        query_prompt = f"""
{system_prompt}

{context_prompt}

Please provide a comprehensive answer to the user's query based on the retrieved context and your GenericSuite knowledge.
"""
        return query_prompt.strip()
    
    def get_framework_specific_prompt(self, framework: str) -> str:
        """
        Get framework-specific guidance for backend generation.
        
        Args:
            framework: Backend framework (fastapi, flask, chalice).
            
        Returns:
            str: Framework-specific prompt addition.
        """
        framework_prompts = {
            "fastapi": """
FASTAPI SPECIFIC GUIDELINES:
- Use FastAPI decorators and dependency injection
- Implement Pydantic models for request/response validation
- Use async/await for database operations when possible
- Include OpenAPI documentation with proper tags and descriptions
- Implement proper exception handling with HTTPException
- Use FastAPI's built-in security features for authentication
""",
            "flask": """
FLASK SPECIFIC GUIDELINES:
- Use Flask blueprints for modular organization
- Implement proper request validation with Flask-WTF or marshmallow
- Use Flask-SQLAlchemy for database operations
- Include proper error handling with Flask error handlers
- Implement authentication with Flask-Login or Flask-JWT-Extended
- Use Flask's application factory pattern
""",
            "chalice": """
CHALICE SPECIFIC GUIDELINES:
- Use Chalice decorators for AWS Lambda integration
- Implement proper CORS configuration for web APIs
- Use Chalice's built-in authentication and authorization
- Include proper error handling with ChaliceViewError
- Optimize for serverless deployment and cold starts
- Use Chalice's local development features
"""
        }
        
        return framework_prompts.get(framework.lower(), "")
    
    def add_timestamp_context(self) -> str:
        """
        Add current timestamp context for generation.
        
        Returns:
            str: Timestamp context string.
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Generated on: {current_time}"


# Global prompt manager instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """
    Get or create the global prompt manager instance.
    
    Returns:
        PromptManager: Global prompt manager instance.
    """
    global _prompt_manager
    
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    
    return _prompt_manager


# Utility functions for prompt management

def format_code_block(code: str, language: str = "python") -> str:
    """
    Format code in markdown code blocks.
    
    Args:
        code: Code content to format.
        language: Programming language for syntax highlighting.
        
    Returns:
        str: Formatted code block.
    """
    return f"```{language}\n{code}\n```"


def create_example_prompt(example_type: str, example_content: str) -> str:
    """
    Create an example prompt for demonstration.
    
    Args:
        example_type: Type of example (table, form, component, etc.).
        example_content: Content of the example.
        
    Returns:
        str: Formatted example prompt.
    """
    return f"""
EXAMPLE {example_type.upper()}:
{example_content}

Use this example as a reference for structure and patterns, but adapt it to the specific requirements.
"""


def validate_prompt_length(prompt: str, max_length: int = 8000) -> bool:
    """
    Validate that a prompt doesn't exceed maximum length.
    
    Args:
        prompt: Prompt text to validate.
        max_length: Maximum allowed length.
        
    Returns:
        bool: True if prompt is within limits, False otherwise.
    """
    return len(prompt) <= max_length


if __name__ == "__main__":
    # Example usage and testing
    prompt_manager = get_prompt_manager()
    
    # Test system prompt generation
    system_prompt = prompt_manager.get_system_prompt("json")
    print(f"System prompt length: {len(system_prompt)}")
    
    # Test context formatting
    context_prompt = prompt_manager.format_context_prompt(
        "Example context content",
        ["example/file.py", "docs/readme.md"],
        "How to create a table?"
    )
    print(f"Context prompt: {context_prompt}")
    
    # Test generation prompt
    gen_prompt = prompt_manager.create_generation_prompt(
        "python",
        "Create a user management tool",
        "Example context",
        ["source.py"]
    )
    print(f"Generation prompt length: {len(gen_prompt)}")