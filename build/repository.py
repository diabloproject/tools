from dataclasses import dataclass
from pathlib import Path


@dataclass
class Repository:
    root: Path
