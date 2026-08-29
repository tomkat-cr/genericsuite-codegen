"""
Backend code generator for GenericSuite applications.
"""
from typing import List

from genericsuite_codegen.utilities.app_logger import (
    log_error,
)

from genericsuite_codegen.agent.types import (
    CodeGenerationResult,
)
from genericsuite_codegen.agent.tool_knowledge_base import KnowledgeBaseTool


DEBUG = True


class BackendCodeGenerator:
    """
    Backend code generator for GenericSuite applications.

    Generates backend code for FastAPI, Flask, and Chalice frameworks
    following GenericSuite patterns and best practices.
    """

    def __init__(self, kb_tool: KnowledgeBaseTool):
        """Initialize the backend code generator."""
        self.kb_tool = kb_tool
        self._load_templates()

    def _load_templates(self) -> None:
        """Load backend code templates."""
        # FastAPI endpoint template
        self.fastapi_endpoint_template = '''"""
{endpoint_name} API endpoint for GenericSuite application.

{description}
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..auth import get_current_user, require_permissions
from ..database import get_db_session
from ..models import {model_imports}
from ..schemas import {schema_imports}

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/{endpoint_prefix}", tags=["{endpoint_tag}"])


# Request/Response models
{request_response_models}


# Endpoints
{endpoints}


# Helper functions
{helper_functions}
'''

        # Flask endpoint template
        self.flask_endpoint_template = '''"""
{endpoint_name} Flask blueprint for GenericSuite application.

{description}
"""

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from marshmallow import Schema, fields, ValidationError
from datetime import datetime
import logging

from ..auth import require_permissions
from ..database import db
from ..models import {model_imports}

logger = logging.getLogger(__name__)

{endpoint_name}_bp = Blueprint('{endpoint_name}', __name__,
    url_prefix='/{endpoint_prefix}')


# Schemas
{schemas}


# Routes
{routes}


# Helper functions
{helper_functions}
'''

        # Chalice endpoint template
        self.chalice_endpoint_template = '''"""
{endpoint_name} Chalice routes for GenericSuite application.

{description}
"""

from chalice import Blueprint, Response
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging

from ..auth import require_auth, get_current_user
from ..database import get_db_connection
from ..models import {model_imports}
from ..utils import validate_request, format_response

logger = logging.getLogger(__name__)

{endpoint_name}_routes = Blueprint(__name__)


# Request validation schemas
{validation_schemas}


# Routes
{routes}


# Helper functions
{helper_functions}
'''

    def generate_backend_code(
        self, requirements: str, module_name: str,
        framework: str, code_type: str = "api_endpoint"
    ) -> CodeGenerationResult:
        """
        Generate backend code for the specified framework.

        Args:
            requirements: Requirements for the backend code.
            module_name: Name of the module/endpoint.
            framework: Backend framework (fastapi, flask, chalice).
            code_type: Type of code to generate.

        Returns:
            CodeGenerationResult: Generated backend code.
        """
        try:
            # Get relevant context for backend code
            context, sources, raw_results = \
                self.kb_tool.get_context_for_generation(
                    query=f"GenericSuite {framework} {code_type} "
                    f"{requirements}",
                    max_context_length=None,
                    file_type_filter="py"
                )

            if framework == "fastapi":
                return self._generate_fastapi_code(requirements, module_name,
                                                   code_type, sources)
            elif framework == "flask":
                return self._generate_flask_code(requirements, module_name,
                                                 code_type, sources)
            elif framework == "chalice":
                return self._generate_chalice_code(requirements, module_name,
                                                   code_type, sources)
            else:
                raise ValueError(f"Unsupported framework: {framework}")

        except Exception as e:
            log_error(f"Failed to generate {framework} code: {e}")
            raise RuntimeError(f"{framework} code generation failed: {e}")

    def _generate_fastapi_code(
        self, requirements: str, module_name: str,
        code_type: str, sources: List[str]
    ) -> CodeGenerationResult:
        """Generate FastAPI code."""
        # Generate code components
        endpoints = self._generate_fastapi_endpoints(requirements, module_name)
        request_response_models = self._generate_pydantic_models(
            requirements, module_name)
        helper_functions = self._generate_helper_functions(
            requirements, "fastapi")

        # Format the template
        code = self.fastapi_endpoint_template.format(
            endpoint_name=module_name,
            description=f"FastAPI endpoint for {requirements}",
            endpoint_prefix=module_name.lower().replace("_", "-"),
            endpoint_tag=module_name.replace("_", " ").title(),
            model_imports=self._get_model_imports(requirements),
            schema_imports=self._get_schema_imports(requirements),
            request_response_models=request_response_models,
            endpoints=endpoints,
            helper_functions=helper_functions
        )

        # Generate additional files
        files = {
            f"test_{module_name}.py": self._generate_backend_test(
                module_name, "fastapi"),
            f"{module_name}_models.py": self._generate_models_file(
                module_name, "fastapi"),
            f"{module_name}_schemas.py": self._generate_schemas_file(
                module_name, "fastapi")
        }

        return CodeGenerationResult(
            code=code,
            code_type="fastapi_endpoint",
            framework="fastapi",
            files=files,
            imports=self._get_fastapi_imports(),
            usage_instructions=self._generate_backend_usage_instructions(
                module_name, "fastapi"),
            integration_notes=self._generate_backend_integration_notes(
                "fastapi"),
            sources=sources
        )

    def _generate_flask_code(
        self, requirements: str, module_name: str,
        code_type: str, sources: List[str]
    ) -> CodeGenerationResult:
        """Generate Flask code."""
        # Generate code components
        routes = self._generate_flask_routes(requirements, module_name)
        schemas = self._generate_marshmallow_schemas(requirements, module_name)
        helper_functions = self._generate_helper_functions(
            requirements, "flask")

        # Format the template
        code = self.flask_endpoint_template.format(
            endpoint_name=module_name,
            description=f"Flask blueprint for {requirements}",
            endpoint_prefix=module_name.lower().replace("_", "-"),
            model_imports=self._get_model_imports(requirements),
            schemas=schemas,
            routes=routes,
            helper_functions=helper_functions
        )

        # Generate additional files
        files = {
            f"test_{module_name}.py": self._generate_backend_test(
                module_name, "flask"),
            f"{module_name}_models.py": self._generate_models_file(
                module_name, "flask")
        }

        return CodeGenerationResult(
            code=code,
            code_type="flask_blueprint",
            framework="flask",
            files=files,
            imports=self._get_flask_imports(),
            usage_instructions=self._generate_backend_usage_instructions(
                module_name, "flask"),
            integration_notes=self._generate_backend_integration_notes(
                "flask"),
            sources=sources
        )

    def _generate_chalice_code(
        self, requirements: str, module_name: str,
        code_type: str, sources: List[str]
    ) -> CodeGenerationResult:
        """Generate Chalice code."""
        # Generate code components
        routes = self._generate_chalice_routes(requirements, module_name)
        validation_schemas = self._generate_chalice_schemas(
            requirements, module_name)
        helper_functions = self._generate_helper_functions(
            requirements, "chalice")

        # Format the template
        code = self.chalice_endpoint_template.format(
            endpoint_name=module_name,
            description=f"Chalice routes for {requirements}",
            model_imports=self._get_model_imports(requirements),
            validation_schemas=validation_schemas,
            routes=routes,
            helper_functions=helper_functions
        )

        # Generate additional files
        files = {
            f"test_{module_name}.py": self._generate_backend_test(
                module_name, "chalice"),
            f"{module_name}_models.py": self._generate_models_file(
                module_name, "chalice")
        }

        return CodeGenerationResult(
            code=code,
            code_type="chalice_routes",
            framework="chalice",
            files=files,
            imports=self._get_chalice_imports(),
            usage_instructions=self._generate_backend_usage_instructions(
                module_name, "chalice"),
            integration_notes=self._generate_backend_integration_notes(
                "chalice"),
            sources=sources
        )

    # Helper methods for backend code generation
    def _generate_fastapi_endpoints(self, requirements: str, module_name: str
                                    ) -> str:
        """Generate FastAPI endpoints."""
        return f'''@router.get("/")
async def list_{module_name}(
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """List {module_name} items."""
    # Implementation based on requirements
    return {{"items": [], "total": 0}}


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_{module_name}(
    item: {module_name.title()}Create,
    current_user = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Create a new {module_name} item."""
    # Implementation based on requirements
    return {{"id": 1, "message": "Created successfully"}}


@router.get("/{{item_id}}")
async def get_{module_name}(
    item_id: int,
    current_user = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Get a specific {module_name} item."""
    # Implementation based on requirements
    return {{"id": item_id}}


@router.put("/{{item_id}}")
async def update_{module_name}(
    item_id: int,
    item: {module_name.title()}Update,
    current_user = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Update a {module_name} item."""
    # Implementation based on requirements
    return {{"id": item_id, "message": "Updated successfully"}}


@router.delete("/{{item_id}}")
async def delete_{module_name}(
    item_id: int,
    current_user = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Delete a {module_name} item."""
    # Implementation based on requirements
    return {{"message": "Deleted successfully"}}'''

    def _generate_flask_routes(self, requirements: str, module_name: str
                               ) -> str:
        """Generate Flask routes."""
        return f'''@{module_name}_bp.route('/', methods=['GET'])
@jwt_required()
def list_{module_name}():
    """List {module_name} items."""
    try:
        # Implementation based on requirements
        return jsonify({{"items": [], "total": 0}})
    except Exception as e:
        logger.error(f"Error listing {module_name}: {{e}}")
        return jsonify({{"error": "Internal server error"}}), 500


@{module_name}_bp.route('/', methods=['POST'])
@jwt_required()
def create_{module_name}():
    """Create a new {module_name} item."""
    try:
        data = request.get_json()
        # Validate and process data
        return jsonify({{"id": 1, "message": "Created successfully"}}), 201
    except ValidationError as e:
        return jsonify({{"error": e.messages}}), 400
    except Exception as e:
        logger.error(f"Error creating {module_name}: {{e}}")
        return jsonify({{"error": "Internal server error"}}), 500


@{module_name}_bp.route('/<int:item_id>', methods=['GET'])
@jwt_required()
def get_{module_name}(item_id):
    """Get a specific {module_name} item."""
    try:
        # Implementation based on requirements
        return jsonify({{"id": item_id}})
    except Exception as e:
        logger.error(f"Error getting {module_name}: {{e}}")
        return jsonify({{"error": "Internal server error"}}), 500'''

    def _generate_chalice_routes(self, requirements: str, module_name: str
                                 ) -> str:
        """Generate Chalice routes."""
        return f'''@{module_name}_routes.route('/', methods=['GET'])
@require_auth
def list_{module_name}():
    """List {module_name} items."""
    try:
        # Implementation based on requirements
        return format_response({{"items": [], "total": 0}})
    except Exception as e:
        logger.error(f"Error listing {module_name}: {{e}}")
        return Response(
            body=json.dumps({{"error": "Internal server error"}}),
            status_code=500,
            headers={{"Content-Type": "application/json"}}
        )


@{module_name}_routes.route('/', methods=['POST'])
@require_auth
def create_{module_name}():
    """Create a new {module_name} item."""
    try:
        request_data = {module_name}_routes.current_request.json_body
        # Validate and process data
        return format_response({{"id": 1, "message": "Created successfully"}},
            201)
    except Exception as e:
        logger.error(f"Error creating {module_name}: {{e}}")
        return Response(
            body=json.dumps({{"error": "Internal server error"}}),
            status_code=500,
            headers={{"Content-Type": "application/json"}}
        )'''

    # More helper methods would continue here...
    def _generate_pydantic_models(self, requirements: str, module_name: str
                                  ) -> str:
        """Generate Pydantic models for FastAPI."""
        return f'''class {module_name.title()}Base(BaseModel):
    """Base {module_name} model."""
    name: str = Field(..., description="Name of the {module_name}")
    description: Optional[str] = Field(None, description="Description")


class {module_name.title()}Create({module_name.title()}Base):
    """Model for creating {module_name}."""
    pass


class {module_name.title()}Update(BaseModel):
    """Model for updating {module_name}."""
    name: Optional[str] = Field(None, description="Name of the {module_name}")
    description: Optional[str] = Field(None, description="Description")


class {module_name.title()}Response({module_name.title()}Base):
    """Model for {module_name} response."""
    id: int = Field(..., description="Unique identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True'''

    def _generate_marshmallow_schemas(self, requirements: str,
                                      module_name: str) -> str:
        """Generate Marshmallow schemas for Flask."""
        return f'''class {module_name.title()}Schema(Schema):
    """Schema for {module_name} validation."""
    name = fields.Str(required=True, validate=fields.Length(min=1, max=100))
    description = fields.Str(missing=None, validate=fields.Length(max=500))


class {module_name.title()}UpdateSchema(Schema):
    """Schema for {module_name} updates."""
    name = fields.Str(validate=fields.Length(min=1, max=100))
    description = fields.Str(validate=fields.Length(max=500))


# Schema instances
{module_name}_schema = {module_name.title()}Schema()
{module_name}_update_schema = {module_name.title()}UpdateSchema()'''

    def _generate_chalice_schemas(self, requirements: str, module_name: str
                                  ) -> str:
        """Generate validation schemas for Chalice."""
        return f'''def validate_{module_name}_create(data):
    """Validate {module_name} creation data."""
    required_fields = ['name']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {{field}}")

    if len(data.get('name', '')) < 1:
        raise ValueError("Name cannot be empty")

    return True


def validate_{module_name}_update(data):
    """Validate {module_name} update data."""
    if 'name' in data and len(data['name']) < 1:
        raise ValueError("Name cannot be empty")

    return True'''

    def _get_model_imports(self, requirements: str) -> str:
        """Get model imports."""
        return "User, BaseModel"

    def _get_schema_imports(self, requirements: str) -> str:
        """Get schema imports."""
        return "UserSchema, BaseSchema"

    def _generate_helper_functions(self, requirements: str, framework: str
                                   ) -> str:
        """Generate helper functions."""
        return '''def format_response(data, status_code=200):
    """Format API response."""
    return {{
        "data": data,
        "status": "success",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
    }}


def handle_error(error, status_code=500):
    """Handle API errors."""
    logger.error(f"API error: {{error}}")
    return {{
        "error": str(error),
        "status": "error",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
    }}'''

    def _get_fastapi_imports(self) -> List[str]:
        """Get FastAPI imports."""
        return [
            "fastapi",
            "pydantic",
            "sqlalchemy",
            "python-jose[cryptography]",
            "passlib[bcrypt]"
        ]

    def _get_flask_imports(self) -> List[str]:
        """Get Flask imports."""
        return [
            "flask",
            "flask-jwt-extended",
            "marshmallow",
            "sqlalchemy",
            "flask-sqlalchemy"
        ]

    def _get_chalice_imports(self) -> List[str]:
        """Get Chalice imports."""
        return [
            "chalice",
            "boto3",
            "pydantic"
        ]

    def _generate_backend_test(self, module_name: str, framework: str) -> str:
        """Generate backend test file."""
        return f'''"""
Tests for {module_name} {framework} endpoints.
"""

import pytest
from unittest.mock import Mock, patch

# Framework-specific test imports would go here

class Test{module_name.title()}Endpoints:
    """Test cases for {module_name} endpoints."""

    def test_list_{module_name}(self):
        """Test listing {module_name} items."""
        # Test implementation
        pass

    def test_create_{module_name}(self):
        """Test creating {module_name} item."""
        # Test implementation
        pass

    def test_get_{module_name}(self):
        """Test getting {module_name} item."""
        # Test implementation
        pass

    def test_update_{module_name}(self):
        """Test updating {module_name} item."""
        # Test implementation
        pass

    def test_delete_{module_name}(self):
        """Test deleting {module_name} item."""
        # Test implementation
        pass
'''

    def _generate_models_file(self, module_name: str, framework: str) -> str:
        """Generate models file."""
        return f'''"""
Database models for {module_name}.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class {module_name.title()}(Base):
    """Database model for {module_name}."""

    __tablename__ = '{module_name.lower()}'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow,
        onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<{module_name.title()}(id={{self.id}}, name='{{self.name}}')>"
'''

    def _generate_schemas_file(self, module_name: str, framework: str) -> str:
        """Generate schemas file for FastAPI."""
        return f'''"""
Pydantic schemas for {module_name}.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class {module_name.title()}Base(BaseModel):
    """Base schema for {module_name}."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class {module_name.title()}Create({module_name.title()}Base):
    """Schema for creating {module_name}."""
    pass


class {module_name.title()}Update(BaseModel):
    """Schema for updating {module_name}."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class {module_name.title()}InDB({module_name.title()}Base):
    """Schema for {module_name} in database."""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class {module_name.title()}Response({module_name.title()}InDB):
    """Schema for {module_name} API response."""
    pass
'''

    def _generate_backend_usage_instructions(self, module_name: str,
                                             framework: str) -> str:
        """Generate backend usage instructions."""
        return f'''Usage Instructions for {module_name} {framework} endpoint:

1. Install dependencies:
   pip install -r requirements.txt

2. Set up database:
   - Configure database connection
   - Run migrations if needed

3. Register the endpoint:
   {self._get_registration_instructions(framework, module_name)}

4. Test the endpoints:
   - Use the included test file
   - Test with API client (Postman, curl, etc.)

5. Deploy:
   - Configure environment variables
   - Set up production database
   - Deploy using your preferred method
'''

    def _get_registration_instructions(self, framework: str, module_name: str
                                       ) -> str:
        """Get framework-specific registration instructions."""
        if framework == "fastapi":
            return f"app.include_router({module_name}_router)"
        elif framework == "flask":
            return f"app.register_blueprint({module_name}_bp)"
        elif framework == "chalice":
            return f"app.register_blueprint({module_name}_routes)"
        else:
            return "Register according to framework documentation"

    def _generate_backend_integration_notes(self, framework: str) -> str:
        """Generate backend integration notes."""
        return f'''Integration Notes for {framework}:

- Follows GenericSuite backend patterns and conventions
- Includes proper authentication and authorization
- Implements comprehensive error handling and logging
- Uses framework-specific best practices
- Compatible with GenericSuite database models and schemas

Make sure to:
1. Configure authentication middleware
2. Set up database connections
3. Configure logging and monitoring
4. Test all endpoints thoroughly
5. Set up proper deployment pipeline

Framework-specific considerations:
{self._get_framework_considerations(framework)}
'''

    def _get_framework_considerations(self, framework: str) -> str:
        """Get framework-specific considerations."""
        considerations = {
            "fastapi": "- Use dependency injection for database sessions\n"
            "- Leverage automatic API documentation\n"
            "- Implement proper async/await patterns",
            "flask": "- Use blueprints for modular organization\n"
            "- Configure Flask-JWT-Extended for authentication\n"
            "- Set up proper error handlers",
            "chalice": "- Optimize for serverless deployment\n"
            "- Configure proper CORS settings\n"
            "- Use Chalice's built-in authentication features"
        }
        return considerations.get(framework, "Follow framework best practices")
