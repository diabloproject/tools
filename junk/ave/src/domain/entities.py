"""Domain entities representing core business objects."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Table:
    """Represents a database table with metadata."""
    name: str
    description: str
    granularity: Optional[str] = None
    purpose: Optional[str] = None
    passage: Optional[str] = None


@dataclass
class Column:
    """Represents a table column with detailed metadata."""
    name: str
    description: str
    type: Optional[str] = None
    units: Optional[str] = None
    is_pk: bool = False
    is_fk: bool = False
    references: Optional[str] = None
    passage: Optional[str] = None
    hint: Optional[str] = None
