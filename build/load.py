import json
from typing import Literal
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class UVExtra:
    step: Literal["uv"]
    pyproject: str


@dataclass
class SystemExtra:
    step: Literal["system"]
    command: str


Extra = UVExtra | SystemExtra


@dataclass
class DependsFile:
    language: Literal["python", "rust"]
    type: Literal["executable", "binary"]
    project: Path
    includes: list[str] = field(default_factory=list)
    extras: list[Extra] = field(default_factory=list)


def load_depends(location: Path):
    depends_file = location / "depends.json"
    if not depends_file.exists():
        raise FileNotFoundError(f"Depends file not found at {depends_file}")
    with open(depends_file, 'r') as file:
        data = json.load(file)
    language = data.get("language", None)
    if language is None:
        raise ValueError("Language not specified")
    if language not in ["python", "rust"]:
        raise ValueError(f"Unknown language: {language}")
    type = data.get("type", None)
    if type is None:
        raise ValueError("Type not specified")
    if language not in ["executable", "library"]:
        raise ValueError(f"Unknown type: {type}")
    extras: list[Extra] = []
    for e in data.get("extras", []):
        if e["step"] == "uv":
            extras.append(UVExtra(**e))
        elif e["step"] == "system":
            extras.append(SystemExtra(**e))
        else:
            raise ValueError(f"Unknown step type: {e['step']}")
    return DependsFile(
        language=language,
        type=type,
        project=location,
        includes=data.get("includes", []),
        extras=extras
    )
