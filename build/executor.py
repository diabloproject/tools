import os
import subprocess
from typing import TYPE_CHECKING
from pathlib import Path
from venv import EnvBuilder


from build.uv import Uv


if TYPE_CHECKING:
    from build.load import DependsFile
    from build.repository import Repository


def generate_venv_name(repo: Repository, depends: "DependsFile"):
    rel = depends.project.relative_to(repo.root)
    return str(rel).replace("\\", "_")


def python_execute(repo: "Repository", depends: "DependsFile"):
    builder = EnvBuilder(
        system_site_packages=False,
        clear=False,
        symlinks=True,
        upgrade=False,
        with_pip=False,
        prompt=depends.project.name,
        upgrade_deps=False,
    )
    name = generate_venv_name(repo, depends)
    path = Path(f"~/.tools/venvs/{name}").expanduser().resolve()
    builder.create(path)
    bin_dir = path / ("Scripts" if os.name == "nt" else "bin")
    python_bin = bin_dir / ("python.exe" if os.name == "nt" else "python")
    base_env = os.environ.copy()
    base_env["VIRTUAL_ENV"] = str(path)
    base_env["PATH"] = f"{bin_dir}{os.pathsep}{base_env['PATH']}"
    base_env["PYTHONPATH"] = f"{repo.root.absolute()}{os.pathsep}{base_env.get('PYTHONPATH', '')}"

    def run_in_venv(command: str, **kwargs):
        subprocess.run(command, shell=True, check=True, env=base_env, **kwargs)

    for extra in depends.extras:
        match extra.step:
            case "uv":
                uv = Uv()
                uv




def execute(repo: "Repository", depends: "DependsFile"):
    match depends.language:
        case "python":
            python_execute(repo, depends)
        case _:
            raise ValueError(f"Unsupported language: {depends.language}")
