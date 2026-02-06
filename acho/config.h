#ifndef ACHO_CONFIG_H
#define ACHO_CONFIG_H

#define ACHO_MAX_PATH   512
#define ACHO_MAX_URL    1024
#define ACHO_MAX_KEY    256
#define ACHO_MAX_NAME   256

typedef struct acho_config {
	char rtmp_url[ACHO_MAX_URL];
	char stream_key[ACHO_MAX_KEY];

	/* video */
	int  width;
	int  height;
	int  fps;
	int  video_bitrate;       /* kbps */
	char encoder[ACHO_MAX_NAME];  /* override, empty = auto probe */

	/* audio */
	int  audio_bitrate;       /* kbps */
	int  sample_rate;
	char mic_device[ACHO_MAX_NAME];
	char system_audio_device[ACHO_MAX_NAME];

	/* capture */
	char video_device[ACHO_MAX_NAME];
	int  capture_x;
	int  capture_y;
} acho_config;

/*
 * Load config from file. Returns ACHO_OK or ACHO_ERR_CONFIG.
 * If path is NULL, uses platform default config location.
 */
int  acho_config_load(acho_config *cfg, const char *path);

/*
 * Fill cfg with sensible defaults.
 */
void acho_config_defaults(acho_config *cfg);

/*
 * Get the default config file path for the current platform.
 * Returns ACHO_OK on success, ACHO_ERR_CONFIG on failure.
 */
int acho_config_path(char *buf, int len);

#endif
