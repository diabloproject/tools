#include <acho/config.h>
#include <acho/acho_err.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifdef _WIN32
#include <shlobj.h>
#else
#include <pwd.h>
#include <unistd.h>
#include <sys/types.h>
#endif

void acho_config_defaults(acho_config *cfg)
{
	memset(cfg, 0, sizeof(*cfg));

	strncpy(cfg->rtmp_url, "rtmp://live.twitch.tv/app", ACHO_MAX_URL - 1);

	cfg->width         = 1920;
	cfg->height        = 1080;
	cfg->fps           = 30;
	cfg->video_bitrate = 4500;
	cfg->audio_bitrate = 160;
	cfg->sample_rate   = 44100;
	cfg->capture_x     = 0;
	cfg->capture_y     = 0;
}

static char *strip(char *s)
{
	while (isspace((unsigned char)*s)) s++;
	char *end = s + strlen(s) - 1;
	while (end > s && isspace((unsigned char)*end)) *end-- = '\0';
	return s;
}

static void apply(acho_config *cfg, const char *key, const char *val)
{
	if (!strcmp(key, "rtmp_url"))
		strncpy(cfg->rtmp_url, val, ACHO_MAX_URL - 1);
	else if (!strcmp(key, "stream_key"))
		strncpy(cfg->stream_key, val, ACHO_MAX_KEY - 1);
	else if (!strcmp(key, "width"))
		cfg->width = atoi(val);
	else if (!strcmp(key, "height"))
		cfg->height = atoi(val);
	else if (!strcmp(key, "fps"))
		cfg->fps = atoi(val);
	else if (!strcmp(key, "video_bitrate"))
		cfg->video_bitrate = atoi(val);
	else if (!strcmp(key, "audio_bitrate"))
		cfg->audio_bitrate = atoi(val);
	else if (!strcmp(key, "sample_rate"))
		cfg->sample_rate = atoi(val);
	else if (!strcmp(key, "encoder"))
		strncpy(cfg->encoder, val, ACHO_MAX_NAME - 1);
	else if (!strcmp(key, "mic_device"))
		strncpy(cfg->mic_device, val, ACHO_MAX_NAME - 1);
	else if (!strcmp(key, "system_audio_device"))
		strncpy(cfg->system_audio_device, val, ACHO_MAX_NAME - 1);
	else if (!strcmp(key, "video_device"))
		strncpy(cfg->video_device, val, ACHO_MAX_NAME - 1);
	else if (!strcmp(key, "capture_x"))
		cfg->capture_x = atoi(val);
	else if (!strcmp(key, "capture_y"))
		cfg->capture_y = atoi(val);
}

int acho_config_path(char *buf, int len)
{
#ifdef _WIN32
	char appdata[MAX_PATH];
	if (SHGetFolderPathA(NULL, CSIDL_APPDATA, NULL, 0, appdata) != S_OK)
		return ACHO_ERR_CONFIG;
	snprintf(buf, len, "%s\\acho\\config", appdata);
#elif defined(__APPLE__)
	const char *home = getenv("HOME");
	if (!home) {
		struct passwd *pw = getpwuid(getuid());
		if (!pw) return ACHO_ERR_CONFIG;
		home = pw->pw_dir;
	}
	snprintf(buf, len, "%s/Library/Application Support/acho/config", home);
#else
	const char *xdg = getenv("XDG_CONFIG_HOME");
	if (xdg) {
		snprintf(buf, len, "%s/acho/config", xdg);
	} else {
		const char *home = getenv("HOME");
		if (!home) {
			struct passwd *pw = getpwuid(getuid());
			if (!pw) return ACHO_ERR_CONFIG;
			home = pw->pw_dir;
		}
		snprintf(buf, len, "%s/.config/acho/config", home);
	}
#endif
	return ACHO_OK;
}

int acho_config_load(acho_config *cfg, const char *path)
{
	char pathbuf[ACHO_MAX_PATH];

	acho_config_defaults(cfg);

	if (!path) {
		int ret = acho_config_path(pathbuf, sizeof(pathbuf));
		if (ret != ACHO_OK) return ret;
		path = pathbuf;
	}

	FILE *f = fopen(path, "r");
	if (!f) return ACHO_ERR_CONFIG;

	char line[1024];
	while (fgets(line, sizeof(line), f)) {
		char *s = strip(line);

		/* skip empty lines and comments */
		if (*s == '\0' || *s == '#')
			continue;

		char *eq = strchr(s, '=');
		if (!eq) continue;

		*eq = '\0';
		char *key = strip(s);
		char *val = strip(eq + 1);

		apply(cfg, key, val);
	}

	fclose(f);

	/* stream_key is mandatory */
	if (cfg->stream_key[0] == '\0')
		return ACHO_ERR_CONFIG;

	return ACHO_OK;
}
