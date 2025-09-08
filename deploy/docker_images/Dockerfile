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
    "fastapi>=0.104.1" \
    "fastmcp>=0.104.1" \
    "uvicorn[standard]>=0.24.0" \
    "pydantic-ai>=0.0.1"3 \
    "pymongo>=4.6.0" \
    "openai>=1.3.0" \
    "sentence-transformers>=2.2.2" \
    "transformers>=4.36.0" \
    "torch>=2.1.0" \
    "pypdf2>=3.0.1" \
    "python-multipart>=0.0.6" \
    "python-dotenv>=1.0.0" \
    "litellm>=1.17.0" \
    "httpx>=0.27.2" \
    "aiofiles>=23.2.0" \
    "gitpython>=3.1.40"

# Expose ports
EXPOSE 8000 8070

# Default command
# CMD ["bash", "/code/server-entrypoint.sh"]
