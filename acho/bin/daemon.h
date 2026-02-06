#ifndef ACHO_DAEMON_H
#define ACHO_DAEMON_H

#include <acho/config.h>

/* Start the daemon in background (fork/CreateProcess). Returns 0 parent, -1 error. */
int daemon_start(const acho_config *cfg);

/* Run the streaming loop in the foreground. Returns exit code. */
int daemon_foreground(const acho_config *cfg);

/* Send STOP to a running daemon. */
int daemon_stop(void);

/* Attach to a running daemon and stream stats to stdout. */
int daemon_attach(void);

#endif
