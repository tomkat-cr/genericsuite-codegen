# deploy/docker_images/Dockerfile

FROM python:3.12-slim

# Set working directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    pipx \
    && rm -rf /var/lib/apt/lists/*

#  Finish pipx installation
RUN pipx ensurepath

# Install Poetry
RUN pipx install poetry

# Create directories
RUN mkdir -p /code/local_repo_files

# Copy Poetry configuration files
# COPY ../../server/pyproject.toml /code/pyproject.toml
# COPY ../../server/poetry.lock /code/poetry.lock* 

# Configure Poetry
# RUN poetry config virtualenvs.create false

# Install Python dependencies
# RUN poetry install --no-dev

# Copy application code
# COPY ../../server/genericsuite_codegen /code/genericsuite_codegen
# COPY ../../mcp-server /code/mcp-server

# Make scripts executable
# RUN chmod +x /code/server-entrypoint.sh

RUN pip install --upgrade pip && pip install --no-cache-dir \
    "aiofiles>=23.2.0,<24.0.0" \
    "fastapi>=0.116.1,<0.117.0" \
    "fastmcp>=2.12.2,<3.0.0" \
    "gitpython>=3.1.40,<4.0.0" \
    "httpx>=0.28.1,<0.29.0" \
    "litellm>=1.17.0,<2.0.0" \
    "openai>=1.3.0,<2.0.0" \
    "pydantic>=2.5.0,<3.0.0" \
    "pydantic-ai>=1.0.1,<2.0.0" \
    "pymongo>=4.6.0,<5.0.0" \
    "pypdf2>=3.0.1,<4.0.0" \
    "python-dotenv>=1.0.0,<2.0.0" \
    "python-multipart>=0.0.20,<0.0.21" \
    "sentence-transformers>=2.2.2,<3.0.0" \
    "torch>=2.1.0,<3.0.0" \
    "transformers>=4.36.0,<5.0.0" \
    "uvicorn[standard]>=0.35.0,<0.36.0"





EXPOSE 8000 8070

# Default command
# CMD ["bash", "/code/server-entrypoint.sh"]
