# Design Document

## Overview

The Dependency Sync system is a Python-based tool that automatically synchronizes Poetry dependencies from multiple pyproject.toml files to Docker pip installation commands. The system consists of a core Python module with TOML parsing, version constraint translation, and Dockerfile manipulation capabilities, wrapped by a convenient bash script for easy integration with deployment workflows.

## Architecture

The system follows a modular architecture with clear separation of concerns:

```
deploy/dependency-sync/
├── sync_dependencies.py    # Main Python implementation
├── sync_dependencies.sh    # Bash wrapper script
└── README.md              # Usage documentation
```

### Core Components

1. **TOML Parser Module**: Handles reading and parsing pyproject.toml files
2. **Version Translator Module**: Converts Poetry version constraints to pip format
3. **Dependency Merger Module**: Combines and deduplicates dependencies from multiple sources
4. **Dockerfile Updater Module**: Locates and replaces pip install commands in Dockerfiles
5. **CLI Interface Module**: Provides command-line argument parsing and user interaction

## Components and Interfaces

### 1. TOML Parser Module

```python
class TOMLParser:
    def parse_pyproject_file(self, file_path: str) -> Dict[str, str]:
        """Parse a pyproject.toml file and extract Poetry dependencies."""
        
    def extract_dependencies(self, toml_data: Dict) -> Dict[str, str]:
        """Extract dependencies from parsed TOML data."""
```

**Responsibilities:**
- Read and parse pyproject.toml files using the `tomllib` standard library
- Extract `[tool.poetry.dependencies]` sections
- Handle file I/O errors and malformed TOML gracefully
- Return dependency name-to-constraint mappings

### 2. Version Translator Module

```python
class VersionTranslator:
    def translate_poetry_to_pip(self, constraint: str) -> str:
        """Convert Poetry version constraint to pip-compatible format."""
        
    def handle_caret_constraint(self, version: str) -> str:
        """Convert ^1.2.3 to >=1.2.3,<2.0.0"""
        
    def handle_tilde_constraint(self, version: str) -> str:
        """Convert ~1.2.3 to >=1.2.3,<1.3.0"""
        
    def handle_extras(self, dependency: str, constraint: str) -> str:
        """Handle dependencies with extras like uvicorn[standard]"""
```

**Responsibilities:**
- Convert Poetry caret constraints (`^1.2.3`) to pip range constraints
- Convert Poetry tilde constraints (`~1.2.3`) to pip range constraints
- Preserve exact version constraints and minimum version constraints
- Handle complex version ranges and extras syntax
- Validate version constraint formats

### 3. Dependency Merger Module

```python
class DependencyMerger:
    def merge_dependencies(self, *dependency_dicts: Dict[str, str]) -> Dict[str, str]:
        """Merge multiple dependency dictionaries, resolving conflicts."""
        
    def resolve_version_conflict(self, dep_name: str, constraints: List[str]) -> str:
        """Resolve conflicting version constraints for a dependency."""
        
    def compare_constraints(self, constraint1: str, constraint2: str) -> str:
        """Compare two constraints and return the more restrictive one."""
```

**Responsibilities:**
- Merge dependency dictionaries from multiple pyproject.toml files
- Detect and resolve version constraint conflicts
- Apply conflict resolution strategy (most restrictive constraint wins)
- Provide warnings for conflicting constraints

### 4. Dockerfile Updater Module

```python
class DockerfileUpdater:
    def update_pip_install(self, dockerfile_path: str, dependencies: Dict[str, str]) -> None:
        """Update pip install command in Dockerfile with new dependencies."""
        
    def find_pip_install_block(self, content: str) -> Tuple[int, int]:
        """Locate the pip install command block in Dockerfile content."""
        
    def format_pip_dependencies(self, dependencies: Dict[str, str]) -> str:
        """Format dependencies for pip install command with proper line breaks."""
        
    def backup_dockerfile(self, dockerfile_path: str) -> str:
        """Create a backup of the original Dockerfile."""
```

**Responsibilities:**
- Locate existing `RUN pip install` command blocks in Dockerfiles
- Replace dependency lists while preserving command structure
- Format dependencies with proper line continuation and readability
- Create backups before making changes
- Handle file I/O errors and permission issues

### 5. CLI Interface Module

```python
class CLIInterface:
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        
    def validate_file_paths(self, paths: List[str]) -> None:
        """Validate that all specified files exist."""
        
    def print_summary(self, changes: Dict[str, Any]) -> None:
        """Print a summary of changes made."""
        
    def handle_errors(self, error: Exception) -> int:
        """Handle and report errors, return appropriate exit code."""
```

**Responsibilities:**
- Parse command-line arguments for source and target files
- Validate file existence and permissions
- Provide verbose and quiet output modes
- Handle errors gracefully with meaningful messages
- Return appropriate exit codes for CI/CD integration

## Data Models

### Dependency Model

```python
@dataclass
class Dependency:
    name: str
    constraint: str
    extras: Optional[List[str]] = None
    source_file: str = ""
    
    def to_pip_format(self) -> str:
        """Convert to pip-compatible dependency string."""
```

### Configuration Model

```python
@dataclass
class SyncConfig:
    source_files: List[str]
    target_dockerfile: str
    verbose: bool = False
    quiet: bool = False
    backup: bool = True
    dry_run: bool = False
```

### Result Model

```python
@dataclass
class SyncResult:
    success: bool
    dependencies_processed: int
    conflicts_resolved: int
    changes_made: List[str]
    errors: List[str]
    backup_file: Optional[str] = None
```

## Error Handling

### Error Categories

1. **File System Errors**
   - Missing pyproject.toml files
   - Missing or unwritable Dockerfile
   - Permission denied errors
   - Disk space issues

2. **Parsing Errors**
   - Malformed TOML files
   - Invalid version constraints
   - Missing dependency sections

3. **Processing Errors**
   - Unresolvable version conflicts
   - Invalid Dockerfile format
   - Regex matching failures

### Error Handling Strategy

```python
class DependencySyncError(Exception):
    """Base exception for dependency sync operations."""
    pass

class TOMLParsingError(DependencySyncError):
    """Raised when TOML parsing fails."""
    pass

class VersionConstraintError(DependencySyncError):
    """Raised when version constraint translation fails."""
    pass

class DockerfileUpdateError(DependencySyncError):
    """Raised when Dockerfile update fails."""
    pass
```

**Error Recovery:**
- Create backups before making changes
- Provide detailed error messages with file paths and line numbers
- Continue processing other files when one fails (where possible)
- Rollback changes on critical failures

## Testing Strategy

### Unit Tests

1. **TOML Parser Tests**
   - Valid pyproject.toml parsing
   - Malformed TOML handling
   - Missing dependency sections
   - File not found scenarios

2. **Version Translator Tests**
   - Caret constraint conversion (`^1.2.3` → `>=1.2.3,<2.0.0`)
   - Tilde constraint conversion (`~1.2.3` → `>=1.2.3,<1.3.0`)
   - Exact version preservation (`==1.2.3` → `==1.2.3`)
   - Extras handling (`uvicorn[standard]^0.24.0`)
   - Complex version ranges

3. **Dependency Merger Tests**
   - Simple dependency merging
   - Version conflict resolution
   - Multiple file processing
   - Constraint comparison logic

4. **Dockerfile Updater Tests**
   - Pip install block detection
   - Dependency replacement
   - Formatting preservation
   - Backup creation
   - Error handling

### Integration Tests

1. **End-to-End Workflow Tests**
   - Complete sync process with real files
   - Multiple pyproject.toml files
   - Various Dockerfile formats
   - Error scenarios and recovery

2. **CLI Interface Tests**
   - Command-line argument parsing
   - File validation
   - Output formatting
   - Exit code verification

### Test Data

```
tests/
├── fixtures/
│   ├── pyproject_server.toml
│   ├── pyproject_mcp.toml
│   ├── Dockerfile_valid
│   ├── Dockerfile_invalid
│   └── Dockerfile_no_pip
├── test_toml_parser.py
├── test_version_translator.py
├── test_dependency_merger.py
├── test_dockerfile_updater.py
├── test_cli_interface.py
└── test_integration.py
```

## Implementation Details

### Version Constraint Translation Logic

```python
def translate_poetry_to_pip(self, constraint: str) -> str:
    """
    Poetry → Pip Translation Rules:
    ^1.2.3 → >=1.2.3,<2.0.0 (compatible release)
    ~1.2.3 → >=1.2.3,<1.3.0 (patch level changes)
    >=1.2.3 → >=1.2.3 (minimum version)
    ==1.2.3 → ==1.2.3 (exact version)
    1.2.3 → ==1.2.3 (implicit exact)
    """
```

### Dockerfile Pattern Matching

The system will use regex patterns to locate and replace pip install commands:

```python
PIP_INSTALL_PATTERN = re.compile(
    r'(RUN pip install --upgrade pip && pip install --no-cache-dir\s+\\?\s*\n)'
    r'((?:\s*"[^"]+"\s*\\?\s*\n?)*)',
    re.MULTILINE | re.DOTALL
)
```

### Dependency Formatting

Dependencies will be formatted for readability:

```python
def format_pip_dependencies(self, dependencies: Dict[str, str]) -> str:
    formatted_deps = []
    for name, constraint in sorted(dependencies.items()):
        if constraint and constraint != "*":
            formatted_deps.append(f'    "{name}{constraint}"')
        else:
            formatted_deps.append(f'    "{name}"')
    
    return " \\\n".join(formatted_deps)
```

## Security Considerations

1. **File System Security**
   - Validate file paths to prevent directory traversal
   - Check file permissions before reading/writing
   - Use temporary files for atomic updates

2. **Input Validation**
   - Sanitize version constraints to prevent injection
   - Validate TOML content before processing
   - Limit file sizes to prevent resource exhaustion

3. **Backup and Recovery**
   - Always create backups before modifying files
   - Provide rollback functionality
   - Log all changes for audit trails