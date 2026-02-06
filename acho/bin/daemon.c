#include <acho/bin/daemon.h>
#include <acho/acho.h>
#include <acho/ipc/ipc.h>
#include <acho/log/log.h>

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#include <pwd.h>
#endif

static volatile int g_running = 1;

static void signal_handler(int sig)
{
	(void)sig;
	g_running = 0;
}

#ifndef _WIN32
static void setup_signal_handlers(void)
{
	struct sigaction sa;
	sa.sa_handler = signal_handler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = 0;
	sigaction(SIGINT, &sa, NULL);
	sigaction(SIGTERM, &sa, NULL);
}
#endif

/* ── PID file ────────────────────────────────────────────────────── */

static int pid_path(char *buf, size_t len)
{
#ifdef _WIN32
	char tmp[MAX_PATH];
	GetTempPathA(MAX_PATH, tmp);
	snprintf(buf, len, "%sacho.pid", tmp);
#else
	const char *runtime = getenv("XDG_RUNTIME_DIR");
	if (!runtime) {
		const char *tmpdir = getenv("TMPDIR");
		if (!tmpdir) tmpdir = "/tmp";
		runtime = tmpdir;
	}
	snprintf(buf, len, "%s/acho.pid", runtime);
#endif
	return 0;
}

static int write_pid(void)
{
	char path[512];
	pid_path(path, sizeof(path));

	FILE *f = fopen(path, "w");
	if (!f) return -1;

#ifdef _WIN32
	fprintf(f, "%lu", (unsigned long)GetCurrentProcessId());
#else
	fprintf(f, "%d", getpid());
#endif

	fclose(f);
	return 0;
}

static void remove_pid(void)
{
	char path[512];
	pid_path(path, sizeof(path));
#ifdef _WIN32
	DeleteFileA(path);
#else
	unlink(path);
#endif
}

static int is_running(void)
{
	char path[512];
	pid_path(path, sizeof(path));
	FILE *f = fopen(path, "r");
	if (!f) return 0;

	long pid = 0;
	if (fscanf(f, "%ld", &pid) != 1) { fclose(f); return 0; }
	fclose(f);

#ifdef _WIN32
	HANDLE h = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, (DWORD)pid);
	if (!h) return 0;
	CloseHandle(h);
	return 1;
#else
	return (kill((pid_t)pid, 0) == 0);
#endif
}

/* ── streaming loop ──────────────────────────────────────────────── */

static int run_stream(const acho_config *cfg, acho_ipc_server *srv)
{
	acho_info("initializing stream context");
	acho_debug("video: %dx%d @ %d fps, bitrate %d kbps",
	           cfg->width, cfg->height, cfg->fps, cfg->video_bitrate);
	acho_debug("audio: %d Hz, bitrate %d kbps", cfg->sample_rate, cfg->audio_bitrate);
	acho_debug("rtmp_url: %s", cfg->rtmp_url);

	if (cfg->video_device[0])
		acho_debug("video_device: %s", cfg->video_device);
	if (cfg->mic_device[0])
		acho_debug("mic_device: %s", cfg->mic_device);
	if (cfg->system_audio_device[0])
		acho_debug("system_audio_device: %s", cfg->system_audio_device);
	if (cfg->encoder[0])
		acho_debug("encoder: %s", cfg->encoder);

	int err;
	acho_ctx *ctx = acho_init(cfg, &err);
	if (!ctx) {
		acho_error("init failed: %s", acho_err_msg(err));
		return err;
	}

	acho_info("stream context initialized successfully");
	acho_info("starting capture loop");

	acho_stats stats;
	char stat_line[256];
	int64_t last_stats_log = 0;

	while (g_running) {
		int ret = acho_tick(ctx, &stats);
		if (ret < 0) {
			acho_error("tick failed: %s", acho_err_msg(ret));
			break;
		}
		if (ret == ACHO_EOF) {
			acho_info("stream ended (EOF)");
			break;
		}

		/* log stats every 5 seconds */
		if (stats.uptime_sec >= last_stats_log + 5) {
			last_stats_log = stats.uptime_sec;
			acho_debug("stats: fps=%.1f frames=%lld dropped=%lld bytes=%lld uptime=%llds",
			           stats.fps,
			           (long long)stats.frames_encoded,
			           (long long)stats.frames_dropped,
			           (long long)stats.bytes_sent,
			           (long long)stats.uptime_sec);
		}

		/* push stats to connected clients */
		if (srv) {
			acho_ipc_server_poll(srv, 0);

			snprintf(stat_line, sizeof(stat_line),
			         "STAT fps=%.1f bitrate=%d dropped=%lld uptime=%lld bytes=%lld\n",
			         stats.fps, stats.video_bitrate,
			         (long long)stats.frames_dropped,
			         (long long)stats.uptime_sec,
			         (long long)stats.bytes_sent);

			acho_ipc_server_broadcast(srv, stat_line, strlen(stat_line));

			/* check for commands */
			char cmd[64];
			int client_id;
			int r = acho_ipc_server_read_cmd(srv, cmd, sizeof(cmd), &client_id);
			if (r > 0) {
				acho_debug("received IPC command: %s (client %d)", cmd, client_id);
				if (strncmp(cmd, "STOP", 4) == 0) {
					acho_info("stop command received via IPC");
					g_running = 0;
				} else if (strncmp(cmd, "STATUS", 6) == 0) {
					acho_ipc_server_broadcast(srv, stat_line, strlen(stat_line));
				}
			}
		}
	}

	acho_info("stopping stream (frames_encoded=%lld, bytes_sent=%lld)",
	          (long long)stats.frames_encoded, (long long)stats.bytes_sent);

	acho_stop(ctx);
	acho_free(ctx);

	acho_info("stream stopped and resources freed");

	return ACHO_OK;
}

/* ── public API ──────────────────────────────────────────────────── */

int daemon_foreground(const acho_config *cfg)
{
	acho_info("starting acho in foreground mode");

#ifdef _WIN32
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);
#else
	setup_signal_handlers();
#endif
	acho_debug("signal handlers registered (SIGINT, SIGTERM)");

	acho_ipc_server *srv = acho_ipc_server_create();
	if (srv) {
		acho_debug("IPC server created");
	} else {
		acho_warn("IPC server creation failed, continuing without IPC");
	}

	write_pid();
	acho_debug("PID file written");

	acho_info("streaming in foreground (Ctrl+C to stop)");
	int ret = run_stream(cfg, srv);

	acho_debug("cleaning up IPC server");
	acho_ipc_server_destroy(srv);

	acho_debug("removing PID file");
	remove_pid();

	acho_info("foreground mode exiting with code %d", ret);
	return ret;
}

int daemon_start(const acho_config *cfg)
{
	if (is_running()) {
		fprintf(stderr, "acho: already running\n");
		return -1;
	}

#ifdef _WIN32
	/* re-launch self with --daemon-internal */
	char exe[MAX_PATH];
	GetModuleFileNameA(NULL, exe, MAX_PATH);

	/* serialize config path via env or just re-read default */
	STARTUPINFOA si = { .cb = sizeof(si) };
	PROCESS_INFORMATION pi;

	char cmdline[MAX_PATH + 32];
	snprintf(cmdline, sizeof(cmdline), "\"%s\" --daemon-internal", exe);

	if (!CreateProcessA(NULL, cmdline, NULL, NULL, FALSE,
	                    DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
	                    NULL, NULL, &si, &pi)) {
		fprintf(stderr, "acho: failed to start daemon\n");
		return -1;
	}

	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);
	fprintf(stderr, "acho: started (pid %lu)\n", pi.dwProcessId);
	return 0;

#else
	pid_t pid = fork();
	if (pid < 0) {
		perror("acho: fork");
		return -1;
	}
	if (pid > 0) {
		/* parent */
		fprintf(stderr, "acho: started (pid %d)\n", pid);
		return 0;
	}

	/* child — become daemon */
	setsid();
	setup_signal_handlers();

	/* redirect stdio */
	freopen("/dev/null", "r", stdin);
	freopen("/dev/null", "w", stdout);
	/* keep stderr for logging, or redirect too:
	 * freopen("/dev/null", "w", stderr); */

	acho_ipc_server *srv = acho_ipc_server_create();
	write_pid();

	run_stream(cfg, srv);

	acho_ipc_server_destroy(srv);
	remove_pid();
	exit(0);
#endif
}

int daemon_stop(void)
{
	if (!is_running()) {
		fprintf(stderr, "acho: not running\n");
		return -1;
	}

	acho_ipc_client *cl = acho_ipc_client_connect();
	if (!cl) {
		fprintf(stderr, "acho: failed to connect to daemon\n");
		return -1;
	}

	acho_ipc_client_send(cl, "STOP\n", 5);

	/* wait for ack */
	char buf[64];
	int r = acho_ipc_client_recv(cl, buf, sizeof(buf), 3000);
	if (r > 0)
		fprintf(stderr, "acho: %s", buf);
	else
		fprintf(stderr, "acho: stop signal sent\n");

	acho_ipc_client_destroy(cl);
	return 0;
}

int daemon_attach(void)
{
	if (!is_running()) {
		fprintf(stderr, "acho: not running\n");
		return -1;
	}

	acho_ipc_client *cl = acho_ipc_client_connect();
	if (!cl) {
		fprintf(stderr, "acho: failed to connect to daemon\n");
		return -1;
	}

#ifdef _WIN32
	signal(SIGINT, signal_handler);
#else
	setup_signal_handlers();
#endif
	fprintf(stderr, "acho: attached (Ctrl+C to detach)\n");

	char buf[512];
	while (g_running) {
		int r = acho_ipc_client_recv(cl, buf, sizeof(buf), 1000);
		if (r > 0) {
			/* overwrite current line for live stats */
			fprintf(stdout, "\r%s", buf);
			fflush(stdout);
		} else if (r < 0) {
			fprintf(stderr, "\nacho: daemon disconnected\n");
			break;
		}
	}

	fprintf(stdout, "\n");
	acho_ipc_client_destroy(cl);
	return 0;
}
