#!/usr/bin/env python3
"""
File Storage Server - Handles actual file operations
Communicates with FUSE client via stdio using JSON protocol
"""

import os
import sys
import json
import stat
import base64
import shutil
from pathlib import Path


class FileStorageServer:
    """
    Server that handles file operations on the local filesystem
    """

    def __init__(self, storage_root):
        self.storage_root = Path(storage_root).resolve()

        # Ensure storage directory exists
        self.storage_root.mkdir(parents=True, exist_ok=True)

        # Ensure we stay within storage root for security
        self.storage_root_str = str(self.storage_root)

    def _get_real_path(self, virtual_path):
        """Convert virtual path to real filesystem path, ensuring security"""
        # Remove leading slash and resolve relative to storage root
        clean_path = virtual_path.lstrip('/')
        real_path = (self.storage_root / clean_path).resolve()

        # Security check: ensure path is within storage root
        if not str(real_path).startswith(self.storage_root_str):
            raise ValueError(f"Path {virtual_path} is outside storage root")

        return real_path

    def handle_getattr(self, path):
        """Get file/directory attributes"""
        try:
            real_path = self._get_real_path(path)

            if not real_path.exists():
                return {'error': 'ENOENT'}

            st = real_path.stat()

            attrs = {
                'mode': st.st_mode,
                'ino': st.st_ino,
                'dev': st.st_dev,
                'nlink': st.st_nlink,
                'uid': st.st_uid,
                'gid': st.st_gid,
                'size': st.st_size,
                'atime': int(st.st_atime),
                'mtime': int(st.st_mtime),
                'ctime': int(st.st_ctime)
            }

            return {'attrs': attrs}

        except Exception as e:
            return {'error': f'IO error: {e}'}

    def handle_readdir(self, path):
        """List directory contents"""
        try:
            real_path = self._get_real_path(path)

            if not real_path.exists():
                return {'error': 'ENOENT'}

            if not real_path.is_dir():
                return {'error': 'ENOTDIR'}

            entries = ['.', '..']

            try:
                for item in real_path.iterdir():
                    entries.append(item.name)
            except PermissionError:
                return {'error': 'EACCES'}

            return {'entries': entries}

        except Exception as e:
            return {'error': f'IO error: {e}'}

    def handle_open(self, path, flags):
        """Open file (just check if it exists and is accessible)"""
        try:
            real_path = self._get_real_path(path)

            if not real_path.exists():
                return {'error': 'ENOENT'}

            if real_path.is_dir():
                return {'error': 'EISDIR'}

            return {'success': True}

        except Exception as e:
            return {'error': f'IO error: {e}'}

    def handle_read(self, path, length, offset):
        """Read file content"""
        try:
            real_path = self._get_real_path(path)

            if not real_path.exists():
                return {'error': 'ENOENT'}

            if real_path.is_dir():
                return {'error': 'EISDIR'}

            with open(real_path, 'rb') as f:
                f.seek(offset)
                content = f.read(length)

            # Encode content as base64 to handle binary data in JSON
            content_b64 = base64.b64encode(content).decode('utf-8')

            return {'content': content_b64}

        except Exception as e:
            return {'error': f'IO error: {e}'}

    def handle_write(self, path, content_b64, offset):
        """Write file content"""
        try:
            real_path = self._get_real_path(path)

            if real_path.is_dir():
                return {'error': 'EISDIR'}

            # Decode base64 content
            content = base64.b64decode(content_b64)

            # Ensure parent directory exists
            real_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            with open(real_path, 'r+b' if real_path.exists() else 'wb') as f:
                f.seek(offset)
                f.write(content)

            return {'bytes_written': len(content)}

        except Exception as e:
            return {'error': f'IO error: {e}'}

    def handle_create(self, path, mode):
        """Create a new file"""
        try:
            real_path = self._get_real_path(path)

            if real_path.exists():
                return {'error': 'EEXIST'}

            # Ensure parent directory exists
            real_path.parent.mkdir(parents=True, exist_ok=True)

            # Create empty file
            real_path.touch()

            # Set permissions
            real_path.chmod(mode & 0o777)

            return {'success': True}

        except Exception as e:
            return {'error': f'IO error: {e}'}

    def handle_mkdir(self, path, mode):
        """Create a directory"""
        try:
            real_path = self._get_real_path(path)

            if real_path.exists():
                return {'error': 'EEXIST'}

            # Create directory
            real_path.mkdir(parents=True, exist_ok=False)

            # Set permissions
            real_path.chmod(mode & 0o777)

            return {'success': True}

        except FileExistsError:
            return {'error': 'EEXIST'}
        except Exception as e:
            return {'error': f'IO error: {e}'}

    def handle_unlink(self, path):
        """Delete a file"""
        try:
            real_path = self._get_real_path(path)

            if not real_path.exists():
                return {'error': 'ENOENT'}

            if real_path.is_dir():
                return {'error': 'EISDIR'}

            real_path.unlink()

            return {'success': True}

        except Exception as e:
            return {'error': f'IO error: {e}'}

    def handle_rmdir(self, path):
        """Remove a directory"""
        try:
            real_path = self._get_real_path(path)

            if not real_path.exists():
                return {'error': 'ENOENT'}

            if not real_path.is_dir():
                return {'error': 'ENOTDIR'}

            # Check if directory is empty
            try:
                real_path.rmdir()  # Only removes empty directories
            except OSError:
                return {'error': 'ENOTEMPTY'}

            return {'success': True}

        except Exception as e:
            return {'error': f'IO error: {e}'}

    def handle_request(self, request):
        """Handle a single request from client"""
        operation = request.get('operation')

        if operation == 'getattr':
            return self.handle_getattr(request['path'])
        elif operation == 'readdir':
            return self.handle_readdir(request['path'])
        elif operation == 'open':
            return self.handle_open(request['path'], request['flags'])
        elif operation == 'read':
            return self.handle_read(request['path'], request['length'], request['offset'])
        elif operation == 'write':
            return self.handle_write(request['path'], request['content'], request['offset'])
        elif operation == 'create':
            return self.handle_create(request['path'], request['mode'])
        elif operation == 'mkdir':
            return self.handle_mkdir(request['path'], request['mode'])
        elif operation == 'unlink':
            return self.handle_unlink(request['path'])
        elif operation == 'rmdir':
            return self.handle_rmdir(request['path'])
        else:
            return {'error': f'Unknown operation: {operation}'}

    def run(self):
        """Main server loop - read requests from stdin, send responses to stdout"""
        try:
            while True:
                # Read request from stdin
                line = sys.stdin.readline()
                if not line:
                    break

                try:
                    request = json.loads(line.strip())
                    response = self.handle_request(request)
                    response_line = json.dumps(response) + '\n'
                    sys.stdout.write(response_line)
                    sys.stdout.flush()

                except json.JSONDecodeError as e:
                    error_response = {'error': f'Invalid JSON: {e}'}
                    response_line = json.dumps(error_response) + '\n'
                    sys.stdout.write(response_line)
                    sys.stdout.flush()

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Server error: {e}", file=sys.stderr)


def main():
    if len(sys.argv) != 2:
        print("Usage: python fuse_server.py <storage_directory>", file=sys.stderr)
        print("Example: python fuse_server.py /path/to/storage", file=sys.stderr)
        sys.exit(1)

    storage_root = sys.argv[1]

    print(f"Starting file storage server with root: {storage_root}", file=sys.stderr)
    print("Server ready, waiting for requests...", file=sys.stderr)

    server = FileStorageServer(storage_root)
    server.run()


if __name__ == '__main__':
    main()