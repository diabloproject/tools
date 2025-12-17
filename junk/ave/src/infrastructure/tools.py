"""Tools for agent."""
import logging
from typing import Any, Dict
from ..application.schema_service import SchemaService

logger = logging.getLogger(__name__)


class Tool:
    """Base class for tools."""

    name: str
    description: str

    def __init__(self, schema_service: SchemaService):
        logger.debug(f"Initializing tool: {self.__class__.__name__}")
        self.schema_service = schema_service

    def execute(self, **kwargs) -> str:
        """Execute the tool."""
        raise NotImplementedError


class ListTablesTool(Tool):
    """Tool to list all available database tables with columns."""

    name = "list_tables"
    description = "Get a list of all available database tables with their descriptions and columns (name-description pairs for each column)"

    def get_schema(self) -> Dict[str, Any]:
        """Get OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

    def execute(self, **kwargs) -> str:
        """Execute the tool."""
        logger.info("Executing ListTablesTool")
        tables = self.schema_service.list_tables()
        logger.debug(f"Retrieved {len(tables)} tables from service")
        result = "Available tables:\n\n"
        for table in tables:
            result += f"Table: {table['name']}\n"
            result += f"Description: {table['description']}\n"
            result += "Columns:\n"
            for col in table.get('columns', []):
                result += f"  - {col['name']}: {col['description']}\n"
            result += "\n"
        logger.info(f"Successfully formatted {len(tables)} tables for output")
        return result


class DescribeColumnTool(Tool):
    """Tool to get detailed information about a specific column."""

    name = "describe_column"
    description = "Get detailed information about a specific column in a table, including type, units, primary/foreign key status, references, passage, and hints"

    def get_schema(self) -> Dict[str, Any]:
        """Get OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "The name of the table"
                        },
                        "column": {
                            "type": "string",
                            "description": "The name of the column"
                        }
                    },
                    "required": ["table", "column"]
                }
            }
        }

    def execute(self, table: str, column: str, **kwargs) -> str:
        """Execute the tool."""
        logger.info(f"Executing DescribeColumnTool for {table}.{column}")
        col_info = self.schema_service.describe_column(table, column)
        if col_info is None:
            logger.warning(f"Column '{column}' not found in table '{table}'")
            return f"Column '{column}' not found in table '{table}'"

        logger.debug(f"Successfully retrieved column info for {table}.{column}")
        result = f"Column: {col_info['name']}\n"
        result += f"Description: {col_info['description']}\n"
        if col_info.get('type'):
            result += f"Type: {col_info['type']}\n"
        if col_info.get('units'):
            result += f"Units: {col_info['units']}\n"
        if col_info.get('is_pk'):
            result += "Primary Key: Yes\n"
        if col_info.get('is_fk'):
            result += f"Foreign Key: Yes (references {col_info.get('references')})\n"
        if col_info.get('passage'):
            result += f"Details: {col_info['passage']}\n"
        if col_info.get('hint'):
            result += f"Hint: {col_info['hint']}\n"

        logger.info(f"Successfully formatted column description for {table}.{column}")
        return result
