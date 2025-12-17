"""Application service for schema operations."""
import logging
from typing import List, Dict, Optional

from ..domain.repositories import SchemaRepository

logger = logging.getLogger(__name__)


class SchemaService:
    """Service for managing schema queries."""

    def __init__(self, repository: SchemaRepository):
        """Initialize with a schema repository."""
        logger.info("Initializing SchemaService")
        self.repository = repository
        logger.debug("SchemaService initialized with repository")

    def list_tables(self) -> List[Dict]:
        """Get list of tables with name-description pairs including columns."""
        logger.info("Listing all tables")
        tables = self.repository.get_all_tables()
        logger.debug(f"Retrieved {len(tables)} tables from repository")
        result = []

        for table in tables:
            logger.debug(f"Processing table: {table.name}")
            # Get full table details to access columns
            table_details = self.repository.get_table_details(table.name)

            table_dict = {
                "name": table.name,
                "description": table.description,
                "columns": []
            }

            if table_details and "columns" in table_details:
                column_count = len(table_details["columns"])
                logger.debug(f"Table {table.name} has {column_count} columns")
                table_dict["columns"] = [
                    {
                        "name": col["name"],
                        "description": col["description"]
                    }
                    for col in table_details["columns"]
                ]

            result.append(table_dict)

        logger.info(f"Successfully listed {len(result)} tables")
        return result

    def describe_column(self, table: str, column: str) -> Optional[Dict]:
        """Get detailed information about a column with all metadata."""
        logger.info(f"Describing column: {table}.{column}")
        col = self.repository.get_column(table, column)
        if col is None:
            logger.warning(f"Column {table}.{column} not found")
            return None

        logger.debug(f"Successfully retrieved column details for {table}.{column}")
        return {
            "name": col.name,
            "description": col.description,
            "type": col.type,
            "units": col.units,
            "is_pk": col.is_pk,
            "is_fk": col.is_fk,
            "references": col.references,
            "passage": col.passage,
            "hint": col.hint
        }
