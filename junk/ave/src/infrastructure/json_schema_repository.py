"""JSON-based implementation of schema repository."""
import json
from pathlib import Path
from typing import List, Optional

from ..domain.entities import Table, Column
from ..domain.repositories import SchemaRepository


class JsonSchemaRepository(SchemaRepository):
    """Repository that loads schema from a JSON file."""

    def __init__(self, json_path: str = "combined.json"):
        """Initialize with path to JSON file."""
        self.json_path = Path(json_path)
        self._data = None

    def _load_data(self) -> List[dict]:
        """Load data from JSON file."""
        if self._data is None:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
        return self._data

    def get_all_tables(self) -> List[Table]:
        """Get all tables with name and description."""
        data = self._load_data()
        return [
            Table(
                name=table["table"],
                description=table.get("purpose", ""),
                granularity=table.get("granularity"),
                purpose=table.get("purpose"),
                passage=table.get("passage")
            )
            for table in data
        ]

    def get_column(self, table: str, column: str) -> Optional[Column]:
        """Get detailed information about a specific column."""
        data = self._load_data()
        for t in data:
            if t["table"] == table:
                for col in t["columns"]:
                    if col["name"] == column:
                        return Column(
                            name=col["name"],
                            description=col["description"],
                            type=col.get("type"),
                            units=col.get("units"),
                            is_pk=col.get("is_pk", False),
                            is_fk=col.get("is_fk", False),
                            references=col.get("references"),
                            passage=col.get("passage"),
                            hint=col.get("hint")
                        )
        return None

    def get_table_details(self, table: str) -> Optional[dict]:
        """Get full details for a table including all metadata."""
        data = self._load_data()
        for t in data:
            if t["table"] == table:
                return t
        return None
