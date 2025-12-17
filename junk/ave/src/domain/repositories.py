"""Repository interfaces for data access."""
from abc import ABC, abstractmethod
from typing import List, Optional

from .entities import Table, Column


class SchemaRepository(ABC):
    """Abstract repository for schema data access."""

    @abstractmethod
    def get_all_tables(self) -> List[Table]:
        """Get all tables with name and description."""
        pass

    @abstractmethod
    def get_column(self, table: str, column: str) -> Optional[Column]:
        """Get detailed information about a specific column."""
        pass

    @abstractmethod
    def get_table_details(self, table: str) -> Optional[dict]:
        """Get full details for a table including all metadata."""
        pass
