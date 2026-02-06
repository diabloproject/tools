#ifdef __linux__

#include <acho/ipc/ipc.h>

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

#define ACHO_MAX_CLIENTS 16

static const char *sock_path(void)
{
	static char path[512];
	if (!path[0]) {
		const char *runtime = getenv("XDG_RUNTIME_DIR");
		if (!runtime) runtime = "/tmp";
		snprintf(path, sizeof(path), "%s/acho.sock", runtime);
	}
	return path;
}

struct acho_ipc_server {
	int fd;
	int clients[ACHO_MAX_CLIENTS];
	int nclients;
};

struct acho_ipc_client {
	int fd;
};

static int set_nonblock(int fd)
{
	int flags = fcntl(fd, F_GETFL, 0);
	if (flags < 0) return -1;
	return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

acho_ipc_server *acho_ipc_server_create(void)
{
	acho_ipc_server *srv = calloc(1, sizeof(*srv));
	if (!srv) return NULL;

	srv->fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (srv->fd < 0) { free(srv); return NULL; }

	unlink(sock_path());

	struct sockaddr_un addr = {0};
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, sock_path(), sizeof(addr.sun_path) - 1);

	if (bind(srv->fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		close(srv->fd);
		free(srv);
		return NULL;
	}

	if (listen(srv->fd, 4) < 0) {
		close(srv->fd);
		unlink(sock_path());
		free(srv);
		return NULL;
	}

	set_nonblock(srv->fd);
	return srv;
}

void acho_ipc_server_destroy(acho_ipc_server *srv)
{
	if (!srv) return;
	for (int i = 0; i < srv->nclients; i++)
		close(srv->clients[i]);
	close(srv->fd);
	unlink(sock_path());
	free(srv);
}

int acho_ipc_server_poll(acho_ipc_server *srv, int timeout_ms)
{
	struct pollfd pfd = { .fd = srv->fd, .events = POLLIN };
	int ret = poll(&pfd, 1, timeout_ms);
	if (ret <= 0) return 0;

	int added = 0;
	while (srv->nclients < ACHO_MAX_CLIENTS) {
		int cfd = accept(srv->fd, NULL, NULL);
		if (cfd < 0) break;
		set_nonblock(cfd);
		srv->clients[srv->nclients++] = cfd;
		added++;
	}
	return added;
}

int acho_ipc_server_broadcast(acho_ipc_server *srv, const char *data, size_t len)
{
	int i = 0;
	while (i < srv->nclients) {
		ssize_t w = write(srv->clients[i], data, len);
		if (w < 0 && errno != EAGAIN) {
			close(srv->clients[i]);
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
		ssize_t r = read(srv->clients[i], buf, len - 1);
		if (r > 0) {
			buf[r] = '\0';
			if (client_id) *client_id = i;
			return (int)r;
		}
		if (r == 0) {
			close(srv->clients[i]);
			srv->clients[i] = srv->clients[--srv->nclients];
			i--;
		}
	}
	return 0;
}

acho_ipc_client *acho_ipc_client_connect(void)
{
	acho_ipc_client *cl = calloc(1, sizeof(*cl));
	if (!cl) return NULL;

	cl->fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (cl->fd < 0) { free(cl); return NULL; }

	struct sockaddr_un addr = {0};
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, sock_path(), sizeof(addr.sun_path) - 1);

	if (connect(cl->fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		close(cl->fd);
		free(cl);
		return NULL;
	}

	return cl;
}

void acho_ipc_client_destroy(acho_ipc_client *cl)
{
	if (!cl) return;
	close(cl->fd);
	free(cl);
}

int acho_ipc_client_send(acho_ipc_client *cl, const char *data, size_t len)
{
	ssize_t w = write(cl->fd, data, len);
	return w < 0 ? -1 : (int)w;
}

int acho_ipc_client_recv(acho_ipc_client *cl, char *buf, size_t len, int timeout_ms)
{
	struct pollfd pfd = { .fd = cl->fd, .events = POLLIN };
	int ret = poll(&pfd, 1, timeout_ms);
	if (ret <= 0) return 0;

	ssize_t r = read(cl->fd, buf, len - 1);
	if (r <= 0) return -1;
	buf[r] = '\0';
	return (int)r;
}

const char *acho_ipc_endpoint_path(void)
{
	return sock_path();
}

#endif /* __linux__ */
