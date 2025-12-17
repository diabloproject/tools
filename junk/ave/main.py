import json

def list_tables():
    data = json.load(open('combined.json'))
    return [{
        "name": table["table"],
        "granularity": table["granularity"],
        "purpose": table["purpose"],
        "passage": table["passage"],
        "relationships": table["relationships"]
    } for table in data]


def list_columns(table: str):
    data = json.load(open('combined.json'))
    for t in data:
        if t["table"] == table:
            return [{
                "name": col["name"],
                "description": col["description"],
            } for col in t["columns"]]
    return []


def describe_column(table: str, column: str):
    data = json.load(open('combined.json'))
    for t in data:
        if t["table"] == table:
            for col in t["columns"]:
                if col["name"] == column:
                    return col
    return None