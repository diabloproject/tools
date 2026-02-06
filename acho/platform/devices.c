#include <acho/platform/devices.h>
#include <acho/platform/platform.h>
#include <acho/acho_err.h>

#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/log.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_LOG_SIZE (16 * 1024)

static char  *g_log_buf  = NULL;
static size_t g_log_len  = 0;

static void log_callback(void *ptr, int level, const char *fmt, va_list vl)
{
	(void)ptr;
	if (level > AV_LOG_INFO || !g_log_buf)
		return;

	char line[1024];
	vsnprintf(line, sizeof(line), fmt, vl);

	size_t n = strlen(line);
	if (g_log_len + n + 1 < MAX_LOG_SIZE) {
		memcpy(g_log_buf + g_log_len, line, n);
		g_log_len += n;
		g_log_buf[g_log_len] = '\0';
	}
}

static int parse_avfoundation_log(const char *log,
                                  acho_device_info **out, int *count, int *cap)
{
	int is_audio = 0;
	const char *p = log;

	while (*p) {
		const char *eol = strchr(p, '\n');
		if (!eol) eol = p + strlen(p);

		size_t linelen = (size_t)(eol - p);
		char line[1024];
		if (linelen >= sizeof(line)) linelen = sizeof(line) - 1;
		memcpy(line, p, linelen);
		line[linelen] = '\0';

		if (strstr(line, "video devices"))
			is_audio = 0;
		else if (strstr(line, "audio devices"))
			is_audio = 1;
		else {
			char *bracket = strchr(line, '[');
			if (bracket) {
				char *close = strchr(bracket, ']');
				if (close && close[1] == ' ') {
					int idx;
					if (sscanf(bracket, "[%d]", &idx) == 1) {
						char *name = close + 2;
						char *end = name + strlen(name) - 1;
						while (end > name && (*end == ' ' || *end == '\r'))
							*end-- = '\0';

						if (*count >= *cap) {
							*cap *= 2;
							acho_device_info *tmp = realloc(*out, *cap * sizeof(**out));
							if (!tmp) return ACHO_ERR_OOM;
							*out = tmp;
						}

						acho_device_info *d = &(*out)[*count];
						memset(d, 0, sizeof(*d));
						strncpy(d->name, name, sizeof(d->name) - 1);
						snprintf(d->description, sizeof(d->description),
						         "[%d] %s", idx, name);
						d->is_audio = is_audio;
						(*count)++;
					}
				}
			}
		}

		p = *eol ? eol + 1 : eol;
	}

	return 0;
}

static int enumerate_via_log(acho_device_info **out, int *count, int *cap)
{
	const AVInputFormat *ifmt = av_find_input_format(acho_platform_video_format());
	if (!ifmt) return 0;

	g_log_buf = calloc(1, MAX_LOG_SIZE);
	if (!g_log_buf) return ACHO_ERR_OOM;
	g_log_len = 0;

	av_log_set_callback(log_callback);

	AVFormatContext *fmt_ctx = NULL;
	AVDictionary *opts = NULL;
	av_dict_set(&opts, "list_devices", "true", 0);

	avformat_open_input(&fmt_ctx, "", ifmt, &opts);

	av_dict_free(&opts);
	if (fmt_ctx)
		avformat_close_input(&fmt_ctx);

	av_log_set_callback(av_log_default_callback);

	int ret = parse_avfoundation_log(g_log_buf, out, count, cap);

	free(g_log_buf);
	g_log_buf = NULL;
	g_log_len = 0;

	return ret;
}

static int enumerate_format(const char *fmt_name, int is_audio,
                            acho_device_info **out, int *count, int *cap)
{
	const AVInputFormat *fmt = av_find_input_format(fmt_name);
	if (!fmt) return 0;

	AVDeviceInfoList *list = NULL;
	int ret = avdevice_list_input_sources(fmt, NULL, NULL, &list);
	if (ret < 0 || !list)
		return 0;

	for (int i = 0; i < list->nb_devices; i++) {
		if (*count >= *cap) {
			*cap *= 2;
			acho_device_info *tmp = realloc(*out, *cap * sizeof(**out));
			if (!tmp) {
				avdevice_free_list_devices(&list);
				return ACHO_ERR_OOM;
			}
			*out = tmp;
		}

		acho_device_info *d = &(*out)[*count];
		memset(d, 0, sizeof(*d));
		strncpy(d->name, list->devices[i]->device_name, sizeof(d->name) - 1);
		strncpy(d->description, list->devices[i]->device_description,
		        sizeof(d->description) - 1);
		d->is_audio = is_audio;
		(*count)++;
	}

	avdevice_free_list_devices(&list);
	return 0;
}

int acho_devices_list(acho_device_info **out)
{
	avdevice_register_all();

	int count = 0;
	int cap   = 16;
	*out = malloc(cap * sizeof(**out));
	if (!*out) return ACHO_ERR_OOM;

	int ret;

#if defined(__APPLE__)
	ret = enumerate_via_log(out, &count, &cap);
	if (ret < 0) { free(*out); *out = NULL; return ret; }
#else
	ret = enumerate_format(acho_platform_video_format(), 0, out, &count, &cap);
	if (ret < 0) { free(*out); *out = NULL; return ret; }

	const char *afmt = acho_platform_audio_format();
	if (strcmp(afmt, acho_platform_video_format()) != 0) {
		ret = enumerate_format(afmt, 1, out, &count, &cap);
		if (ret < 0) { free(*out); *out = NULL; return ret; }
	}
#endif

	if (count == 0) {
		free(*out);
		*out = NULL;
	}

	return count;
}

void acho_devices_free(acho_device_info *devices)
{
	free(devices);
}
