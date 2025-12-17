import os
import zipfile
import tarfile
from pathlib import Path
from http.client import HTTPSConnection
import platform

from libraries.python.http.client import HttpClient


_RELEASE = '0.9.7'

_PLATFORM_TRIPLET = None
if not hasattr(os, 'uname'):
    _PLATFORM_TRIPLET = 'x86_64-pc-windows-msvc'
elif os.uname().sysname == 'Linux':  # type: ignore
    _PLATFORM_TRIPLET = f"{platform.machine()}-unknown-linux-musl"
elif os.uname().sysname == 'Darwin':  # type: ignore
    if platform.machine() == 'arm64':
        _PLATFORM_TRIPLET = 'aarch64-apple-darwin'
    elif platform.machine() == 'x86_64':
        _PLATFORM_TRIPLET = 'x86_64-apple-darwin'
    else:
        raise Exception(f'Unsupported platform: {os.name}')
else:
    raise Exception(f'Unsupported platform: {os.name}')

_IS_ZIP = False
if not hasattr(os, 'uname'):
    _IS_ZIP = True


_DOWNLOAD_URL = ""
if _IS_ZIP:
    _DOWNLOAD_URL = f"https://github.com/astral-sh/uv/releases/download/{_RELEASE}/uv-{_PLATFORM_TRIPLET}.zip"
else:
    _DOWNLOAD_URL = f"https://github.com/astral-sh/uv/releases/download/{_RELEASE}/uv-{_PLATFORM_TRIPLET}.tar.gz"



print(_PLATFORM_TRIPLET)

class Uv:
    def __init__(self):
        Path("~/.tools/cache").resolve().mkdir(parents=True, exist_ok=True)
        bin_path = Path(f"~/.tools/cache/{'uv.zip' if _IS_ZIP else 'uv.tar.gz'}").resolve()
        self._download_binary(bin_path)
        self._unpack_binary(bin_path, )

    def _download_binary(self, target_path: Path):
        client = HttpClient()
        response = client.get(_DOWNLOAD_URL)
        with open(target_path, 'wb') as f:
            f.write(response.content)

    def _unpack_binary(self, target_path: Path):
        if _IS_ZIP:
            with zipfile.ZipFile(target_path) as zip_file:
                zip_file.extractall(Path("~/.tools/uv").resolve())
        else:
            with tarfile.open(target_path) as tar_file:
                tar_file.extractall(Path("~/.tools/uv").resolve())

    def sync(self, target: Path):
        pass


Uv()
