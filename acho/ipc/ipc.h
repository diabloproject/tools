#ifndef ACHO_IPC_H
#define ACHO_IPC_H

#include <stddef.h>

/*
 * Opaque IPC types. Implemented per-platform.
 */
typedef struct acho_ipc_server acho_ipc_server;
typedef struct acho_ipc_client acho_ipc_client;

/*
 * Server (used by daemon)
 */
acho_ipc_server *acho_ipc_server_create(void);
void             acho_ipc_server_destroy(acho_ipc_server *srv);

/* Accept new clients, non-blocking. Returns number of new clients. */
int  acho_ipc_server_poll(acho_ipc_server *srv, int timeout_ms);

/* Broadcast data to all connected clients. Drops dead ones. */
int  acho_ipc_server_broadcast(acho_ipc_server *srv, const char *data, size_t len);

/* Read a command from any client. Returns bytes read into buf, 0 if none, -1 error.
 * client_id is set to identify which client sent the command. */
int  acho_ipc_server_read_cmd(acho_ipc_server *srv, char *buf, size_t len, int *client_id);

/*
 * Client (used by --attach and --stop)
 */
acho_ipc_client *acho_ipc_client_connect(void);
void             acho_ipc_client_destroy(acho_ipc_client *cl);

int  acho_ipc_client_send(acho_ipc_client *cl, const char *data, size_t len);
int  acho_ipc_client_recv(acho_ipc_client *cl, char *buf, size_t len, int timeout_ms);

/*
 * Helpers
 */

/* Get the path/name of the IPC endpoint (for PID file co-location etc.) */
const char *acho_ipc_endpoint_path(void);

#endif
