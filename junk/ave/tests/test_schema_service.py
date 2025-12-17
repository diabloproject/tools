"""Tests for schema service."""
import pytest
from src.infrastructure.json_schema_repository import JsonSchemaRepository
from src.application.schema_service import SchemaService


@pytest.fixture
def schema_service():
    """Create a schema service for testing."""
    repository = JsonSchemaRepository("combined.json")
    return SchemaService(repository)


def test_list_tables(schema_service):
    """Test that list_tables returns tables with columns as name-description pairs."""
    tables = schema_service.list_tables()

    assert isinstance(tables, list)
    assert len(tables) > 0

    # Check structure
    for table in tables:
        assert "name" in table
        assert "description" in table
        assert "columns" in table
        assert isinstance(table["name"], str)
        assert isinstance(table["description"], str)
        assert isinstance(table["columns"], list)

        # Check columns structure
        for col in table["columns"]:
            assert "name" in col
            assert "description" in col
            assert isinstance(col["name"], str)
            assert isinstance(col["description"], str)


def test_describe_column(schema_service):
    """Test describe_column returns full column details."""
    # Test with a known column
    col_info = schema_service.describe_column(
        "public.video_file_duplicates",
        "file_id"
    )

    assert col_info is not None
    assert col_info["name"] == "file_id"
    assert "description" in col_info
    assert "type" in col_info
    assert "is_pk" in col_info
    assert "is_fk" in col_info
    assert "references" in col_info
    assert "passage" in col_info
    assert "hint" in col_info


def test_describe_column_not_found(schema_service):
    """Test describe_column with non-existent column."""
    col_info = schema_service.describe_column(
        "public.video_file_duplicates",
        "nonexistent_column"
    )

    assert col_info is None
