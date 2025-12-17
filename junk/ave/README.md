# Schema Agent with Clean Architecture

A refactored database schema exploration agent using clean architecture principles with OpenRouter.

## Architecture

The project follows clean architecture with three layers:

- **Domain Layer** (`src/domain/`): Core entities and repository interfaces
  - `entities.py`: Table and Column domain models
  - `repositories.py`: Abstract repository interface

- **Application Layer** (`src/application/`): Business logic and use cases
  - `schema_service.py`: Service for schema operations

- **Infrastructure Layer** (`src/infrastructure/`): External concerns and implementations
  - `json_schema_repository.py`: JSON-based repository implementation
  - `tools.py`: Custom tool definitions with OpenAI function schemas
  - `agent.py`: Agent implementation using OpenRouter directly

## Changes from Previous Version

1. **Clean Architecture**: Separated concerns into domain, application, and infrastructure layers
2. **Repository Pattern**: Data access abstracted through repository interface
3. **Updated API**: `list_tables()` now returns tables with columns as name-description pairs
4. **Removed Function**: `list_columns()` removed (info included in `list_tables()`)
5. **Custom Implementation**: Direct OpenRouter integration with custom tool execution
6. **Tools**: `list_tables` for overview, `describe_column` for detailed column metadata

## Usage

```python
from src import JsonSchemaRepository, SchemaService, SchemaAgent

# Initialize layers
repository = JsonSchemaRepository("combined.json")
service = SchemaService(repository)
agent = SchemaAgent(service)

# Use the agent
response = agent.run("Show me all available tables")
```

## Running the Application

```bash
# Install dependencies
uv sync

# Set environment variable
export OPENROUTER_API_KEY=your_key_here

# Run the agent
python main_new.py
```

## Environment Variables

- `OPENROUTER_API_KEY`: OpenRouter API key for model access

## Testing

```bash
pytest tests/
```

## API Reference

### SchemaService

- `list_tables()`: Returns list of tables, each with:
  - `name`: Table name
  - `description`: Table description
  - `columns`: List of column objects with `name` and `description`

- `describe_column(table, column)`: Returns detailed column information with:
  - `name`: Column name
  - `description`: Column description
  - `type`: Data type
  - `units`: Units of measurement (if applicable)
  - `is_pk`: Primary key flag
  - `is_fk`: Foreign key flag
  - `references`: Referenced table/column
  - `passage`: Detailed passage/explanation
  - `hint`: Usage hints for queries

### Tools (for smolagents)

- `list_tables`: Lists all available tables with their columns (name-description pairs)
- `describe_column`: Gets detailed metadata for a specific column including type, keys, references, and hints
