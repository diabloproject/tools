#ifdef _WIN32

#include <acho/ipc/ipc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define ACHO_PIPE_NAME "\\\\.\\pipe\\acho"
#define ACHO_MAX_CLIENTS 16
#define ACHO_PIPE_BUF    4096

struct acho_ipc_server {
	HANDLE listen_pipe;
	HANDLE clients[ACHO_MAX_CLIENTS];
	int    nclients;
};

struct acho_ipc_client {
	HANDLE pipe;
};

static HANDLE create_pipe_instance(void)
{
	return CreateNamedPipeA(
		ACHO_PIPE_NAME,
		PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
		PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
		ACHO_MAX_CLIENTS,
		ACHO_PIPE_BUF,
		ACHO_PIPE_BUF,
		0,
		NULL
	);
}

acho_ipc_server *acho_ipc_server_create(void)
{
	acho_ipc_server *srv = calloc(1, sizeof(*srv));
	if (!srv) return NULL;

	srv->listen_pipe = create_pipe_instance();
	if (srv->listen_pipe == INVALID_HANDLE_VALUE) {
		free(srv);
		return NULL;
	}

	return srv;
}

void acho_ipc_server_destroy(acho_ipc_server *srv)
{
	if (!srv) return;
	for (int i = 0; i < srv->nclients; i++) {
		DisconnectNamedPipe(srv->clients[i]);
		CloseHandle(srv->clients[i]);
	}
	CloseHandle(srv->listen_pipe);
	free(srv);
}

int acho_ipc_server_poll(acho_ipc_server *srv, int timeout_ms)
{
	OVERLAPPED ov = {0};
	ov.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
	if (!ov.hEvent) return 0;

	int added = 0;
	BOOL connected = ConnectNamedPipe(srv->listen_pipe, &ov);

	if (!connected) {
		DWORD err = GetLastError();
		if (err == ERROR_IO_PENDING) {
			DWORD wait = WaitForSingleObject(ov.hEvent, (DWORD)timeout_ms);
			if (wait == WAIT_OBJECT_0)
				connected = TRUE;
		} else if (err == ERROR_PIPE_CONNECTED) {
			connected = TRUE;
		}
	}

	if (connected && srv->nclients < ACHO_MAX_CLIENTS) {
		srv->clients[srv->nclients++] = srv->listen_pipe;
		srv->listen_pipe = create_pipe_instance();
		added = 1;
	}

	CloseHandle(ov.hEvent);
	return added;
}

int acho_ipc_server_broadcast(acho_ipc_server *srv, const char *data, size_t len)
{
	int i = 0;
	while (i < srv->nclients) {
		DWORD written;
		BOOL ok = WriteFile(srv->clients[i], data, (DWORD)len, &written, NULL);
		if (!ok) {
			DisconnectNamedPipe(srv->clients[i]);
			CloseHandle(srv->clients[i]);
			srv->clients[i] = srv->clients[--srv->nclients];
		} else {
			i++;
		}
	}
	return srv->nclients;
}

int acho_ipc_server_read_cmd(acho_ipc_server *srv, char *buf, size_t len, int *client_id)
{
	for (int i = 0; i < srv->nclients; i++) {
		DWORD avail = 0;
		if (!PeekNamedPipe(srv->clients[i], NULL, 0, NULL, &avail, NULL)) {
			/* broken pipe */
			DisconnectNamedPipe(srv->clients[i]);
			CloseHandle(srv->clients[i]);
			srv->clients[i] = srv->clients[--srv->nclients];
			i--;
			continue;
		}
		if (avail == 0) continue;

		DWORD nread;
		BOOL ok = ReadFile(srv->clients[i], buf, (DWORD)(len - 1), &nread, NULL);
		if (ok && nread > 0) {
			buf[nread] = '\0';
			if (client_id) *client_id = i;
			return (int)nread;
		}
	}
	return 0;
}

acho_ipc_client *acho_ipc_client_connect(void)
{
	HANDLE pipe = CreateFileA(
		ACHO_PIPE_NAME,
		GENERIC_READ | GENERIC_WRITE,
		0, NULL, OPEN_EXISTING, 0, NULL
	);
	if (pipe == INVALID_HANDLE_VALUE)
		return NULL;

	acho_ipc_client *cl = calloc(1, sizeof(*cl));
	if (!cl) { CloseHandle(pipe); return NULL; }
	cl->pipe = pipe;
	return cl;
}

void acho_ipc_client_destroy(acho_ipc_client *cl)
{
	if (!cl) return;
	CloseHandle(cl->pipe);
	free(cl);
}

int acho_ipc_client_send(acho_ipc_client *cl, const char *data, size_t len)
{
	DWORD written;
	BOOL ok = WriteFile(cl->pipe, data, (DWORD)len, &written, NULL);
	return ok ? (int)written : -1;
}

int acho_ipc_client_recv(acho_ipc_client *cl, char *buf, size_t len, int timeout_ms)
{
	/* Set read timeout via pipe mode isn't straightforward;
	 * use a simple peek + sleep loop */
	int elapsed = 0;
	while (elapsed < timeout_ms) {
		DWORD avail = 0;
		if (!PeekNamedPipe(cl->pipe, NULL, 0, NULL, &avail, NULL))
			return -1;
		if (avail > 0) {
			DWORD nread;
			BOOL ok = ReadFile(cl->pipe, buf, (DWORD)(len - 1), &nread, NULL);
			if (!ok) return -1;
			buf[nread] = '\0';
			return (int)nread;
		}
		Sleep(10);
		elapsed += 10;
	}
	return 0;
}

const char *acho_ipc_endpoint_path(void)
{
	return ACHO_PIPE_NAME;
}

#endif /* _WIN32 */
