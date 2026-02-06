#include <acho/log/log.h>
#include <acho/bin/daemon.h>
#include <acho/acho.h>
#include <acho/config.h>
#include <acho/platform/devices.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

static void usage(const char *argv0)
{
	fprintf(stderr,
		"usage: %s <command>\n"
		"\n"
		"commands:\n"
		"  --start      start streaming in background\n"
		"  --stop       stop the running stream\n"
		"  --attach     attach to running stream (show stats)\n"
		"  --fg         start streaming in foreground\n"
		"  --devices    list available capture devices\n"
		"  --edit       open config file in $EDITOR\n"
		"  --help       show this help\n",
		argv0
	);
}

static int cmd_devices(void)
{
    char* log_level_str = getenv("ACHO_LOG");
    acho_log_set_level(acho_log_level_from_str(log_level_str));
    
	acho_device_info *devs = NULL;
	int n = acho_devices_list(&devs);

	if (n < 0) {
		fprintf(stderr, "acho: %s\n", acho_err_msg(n));
		return 1;
	}

	if (n == 0) {
		fprintf(stderr, "acho: no capture devices found\n");
		return 0;
	}

	fprintf(stdout, "%-8s  %-30s  %s\n", "TYPE", "NAME", "DESCRIPTION");
	fprintf(stdout, "%-8s  %-30s  %s\n", "----", "----", "-----------");

	for (int i = 0; i < n; i++) {
		fprintf(stdout, "%-8s  %-30s  %s\n",
		        devs[i].is_audio ? "audio" : "video",
		        devs[i].name,
		        devs[i].description);
	}

	acho_devices_free(devs);
	return 0;
}

static int mkdirp(const char *path)
{
	char tmp[ACHO_MAX_PATH];
	char *p;
	size_t len;

	snprintf(tmp, sizeof(tmp), "%s", path);
	len = strlen(tmp);

	/* remove trailing slash */
	if (len > 0 && tmp[len - 1] == '/')
		tmp[len - 1] = '\0';

	for (p = tmp + 1; *p; p++) {
		if (*p == '/') {
			*p = '\0';
			if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
				return -1;
			*p = '/';
		}
	}
	if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
		return -1;

	return 0;
}

static int cmd_edit(void)
{
	char path[ACHO_MAX_PATH];
	if (acho_config_path(path, sizeof(path)) != ACHO_OK) {
		fprintf(stderr, "acho: could not determine config path\n");
		return 1;
	}

	/* extract directory from path */
	char dir[ACHO_MAX_PATH];
	snprintf(dir, sizeof(dir), "%s", path);
	char *last_slash = strrchr(dir, '/');
	if (last_slash) {
		*last_slash = '\0';
		if (mkdirp(dir) != 0) {
			fprintf(stderr, "acho: could not create config directory: %s\n", dir);
			return 1;
		}
	}

	/* create config file with defaults if it doesn't exist */
	FILE *f = fopen(path, "r");
	if (!f) {
		f = fopen(path, "w");
		if (!f) {
			fprintf(stderr, "acho: could not create config file: %s\n", path);
			return 1;
		}
		fprintf(f,
			"# acho configuration\n"
			"\n"
			"# RTMP endpoint\n"
			"rtmp_url = rtmp://live.twitch.tv/app\n"
			"stream_key = YOUR_STREAM_KEY_HERE\n"
			"\n"
			"# Video\n"
			"width = 1920\n"
			"height = 1080\n"
			"fps = 30\n"
			"video_bitrate = 4500\n"
			"# encoder = h264_videotoolbox\n"
			"\n"
			"# Audio\n"
			"audio_bitrate = 160\n"
			"sample_rate = 44100\n"
			"\n"
			"# Devices (run `acho --devices` to list available)\n"
			"# mic_device = \n"
			"# system_audio_device = \n"
		);
		fclose(f);
		fprintf(stderr, "acho: created %s\n", path);
	} else {
		fclose(f);
	}

	const char *editor = getenv("EDITOR");
	if (!editor || !*editor)
		editor = "vi";

	char cmd[ACHO_MAX_PATH + 256];
	snprintf(cmd, sizeof(cmd), "%s \"%s\"", editor, path);
	return system(cmd);
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	const char *cmd = argv[1];

	/* internal: daemon re-exec on Windows */
	if (!strcmp(cmd, "--daemon-internal")) {
		acho_config cfg;
		int ret = acho_config_load(&cfg, NULL);
		if (ret != ACHO_OK) {
			fprintf(stderr, "acho: %s\n", acho_err_msg(ret));
			return 1;
		}
		return daemon_foreground(&cfg);
	}

	if (!strcmp(cmd, "--help") || !strcmp(cmd, "-h")) {
		usage(argv[0]);
		return 0;
	}

	if (!strcmp(cmd, "--devices")) {
		return cmd_devices();
	}

	if (!strcmp(cmd, "--edit")) {
		return cmd_edit();
	}

	if (!strcmp(cmd, "--stop")) {
		return daemon_stop() == 0 ? 0 : 1;
	}

	if (!strcmp(cmd, "--attach")) {
		return daemon_attach() == 0 ? 0 : 1;
	}

	/* commands that need config */
	acho_config cfg;
	const char *config_path = NULL;

	/* optional: --config <path> before the command */
	if (argc >= 4 && !strcmp(argv[1], "--config")) {
		config_path = argv[2];
		cmd = argv[3];
	}

	int ret = acho_config_load(&cfg, config_path);
	if (ret != ACHO_OK) {
		fprintf(stderr, "acho: %s\n", acho_err_msg(ret));
		return 1;
	}

	if (!strcmp(cmd, "--start")) {
		return daemon_start(&cfg) == 0 ? 0 : 1;
	}

	if (!strcmp(cmd, "--fg")) {
		return daemon_foreground(&cfg) == 0 ? 0 : 1;
	}

	fprintf(stderr, "acho: unknown command '%s'\n", cmd);
	usage(argv[0]);
	return 1;
}
