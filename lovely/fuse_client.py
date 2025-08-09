#!/usr/bin/env python3
"""
FUSE Client - Portable component that runs on remote server
Communicates with storage server via stdio using JSON protocol
"""

import os
import sys
import json
import errno
import stat
from time import time

try:
    import fuse
    from fuse import Fuse
except ImportError:
    print("Error: python-fuse not installed. Install with: pip install fusepy", file=sys.stderr)
    sys.exit(1)

fuse.fuse_python_api = (0, 2)


class RemoteFUSEClient(Fuse):
    """
    FUSE client that forwards all operations to a remote server via stdio
    """

    def __init__(self, server_command, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_command = server_command
        self.server_process = None
        self._start_server()

    def _start_server(self):
        """Start the server process"""
        import subprocess
        try:
            self.server_process = subprocess.Popen(
                self.server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True,
                bufsize=0
            )
        except Exception as e:
            print(f"Failed to start server: {e}", file=sys.stderr)
            sys.exit(1)

    def _send_request(self, operation, **kwargs):
        """Send request to server and get response"""
        if not self.server_process or self.server_process.poll() is not None:
            return {'error': 'Server not running'}

        request = {
            'operation': operation,
            **kwargs
        }

        try:
            # Send request
            request_line = json.dumps(request) + '\n'
            self.server_process.stdin.write(request_line)
            self.server_process.stdin.flush()

            # Read response
            response_line = self.server_process.stdout.readline()
            if not response_line:
                return {'error': 'No response from server'}

            response = json.loads(response_line.strip())
            return response

        except Exception as e:
            return {'error': f'Communication error: {e}'}

    def getattr(self, path):
        """Get file attributes"""
        response = self._send_request('getattr', path=path)

        if 'error' in response:
            if response['error'] == 'ENOENT':
                return -errno.ENOENT
            return -errno.EIO

        # Convert response to fuse.Stat
        st = fuse.Stat()
        attrs = response['attrs']
        st.st_mode = attrs['mode']
        st.st_ino = attrs.get('ino', 0)
        st.st_dev = attrs.get('dev', 0)
        st.st_nlink = attrs.get('nlink', 1)
        st.st_uid = attrs.get('uid', os.getuid())
        st.st_gid = attrs.get('gid', os.getgid())
        st.st_size = attrs.get('size', 0)
        st.st_atime = attrs.get('atime', int(time()))
        st.st_mtime = attrs.get('mtime', int(time()))
        st.st_ctime = attrs.get('ctime', int(time()))

        return st

    def readdir(self, path, offset):
        """List directory contents"""
        response = self._send_request('readdir', path=path)

        if 'error' in response:
            if response['error'] == 'ENOENT':
                return -errno.ENOENT
            elif response['error'] == 'ENOTDIR':
                return -errno.ENOTDIR
            return -errno.EIO

        entries = response['entries']
        for entry in entries:
            yield fuse.Direntry(entry)

    def open(self, path, flags):
        """Open a file"""
        response = self._send_request('open', path=path, flags=flags)

        if 'error' in response:
            if response['error'] == 'ENOENT':
                return -errno.ENOENT
            elif response['error'] == 'EISDIR':
                return -errno.EISDIR
            return -errno.EIO

        return 0

    def read(self, path, length, offset):
        """Read file content"""
        response = self._send_request('read', path=path, length=length, offset=offset)

        if 'error' in response:
            if response['error'] == 'ENOENT':
                return -errno.ENOENT
            elif response['error'] == 'EISDIR':
                return -errno.EISDIR
            return -errno.EIO

        # Content is base64 encoded to handle binary data
        import base64
        content = base64.b64decode(response['content'])
        return content

    def write(self, path, buf, offset):
        """Write file content"""
        import base64
        content_b64 = base64.b64encode(buf).decode('utf-8')

        response = self._send_request('write', path=path, content=content_b64, offset=offset)

        if 'error' in response:
            if response['error'] == 'ENOENT':
                return -errno.ENOENT
            elif response['error'] == 'EISDIR':
                return -errno.EISDIR
            return -errno.EIO

        return response['bytes_written']

    def create(self, path, mode):
        """Create a new file"""
        response = self._send_request('create', path=path, mode=mode)

        if 'error' in response:
            if response['error'] == 'EEXIST':
                return -errno.EEXIST
            return -errno.EIO

        return 0

    def mkdir(self, path, mode):
        """Create a directory"""
        response = self._send_request('mkdir', path=path, mode=mode)

        if 'error' in response:
            if response['error'] == 'EEXIST':
                return -errno.EEXIST
            return -errno.EIO

        return 0

    def unlink(self, path):
        """Delete a file"""
        response = self._send_request('unlink', path=path)

        if 'error' in response:
            if response['error'] == 'ENOENT':
                return -errno.ENOENT
            elif response['error'] == 'EISDIR':
                return -errno.EISDIR
            return -errno.EIO

        return 0

    def rmdir(self, path):
        """Remove a directory"""
        response = self._send_request('rmdir', path=path)

        if 'error' in response:
            if response['error'] == 'ENOENT':
                return -errno.ENOENT
            elif response['error'] == 'ENOTDIR':
                return -errno.ENOTDIR
            elif response['error'] == 'ENOTEMPTY':
                return -errno.ENOTEMPTY
            return -errno.EIO

        return 0

    def destroy(self, path):
        """Cleanup when filesystem is unmounted"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except:
                self.server_process.kill()


def main():
    if len(sys.argv) < 3:
        print("Usage: python fuse_client.py <server_command> <mountpoint> [fuse_options]")
        print("Example: python fuse_client.py 'python fuse_server.py /path/to/storage' /tmp/myfuse -f")
        sys.exit(1)

    server_command = sys.argv[1]
    mountpoint = sys.argv[2]

    # Create client with server command
    fs = RemoteFUSEClient(server_command)

    # Modify sys.argv for FUSE argument parsing
    sys.argv = [sys.argv[0], mountpoint] + sys.argv[3:]

    # Parse FUSE arguments
    fs.parse(values=fs, errex=1)

    print(f"Starting FUSE client, mounting at: {mountpoint}")
    print(f"Server command: {server_command}")
    print("To unmount: fusermount -u <mountpoint>")

    try:
        fs.main()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if fs.server_process:
            fs.server_process.terminate()


if __name__ == '__main__':
    main()