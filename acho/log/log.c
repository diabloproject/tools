#include <acho/log/log.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

static acho_log_level g_level = ACHO_LOG_INFO;
static int g_color = -1;  /* -1 = auto-detect */
static int g_timestamp = 1;

/* ANSI color codes */
static const char *level_colors[] = {
	"\x1b[90m",    /* TRACE: gray */
	"\x1b[36m",    /* DEBUG: cyan */
	"\x1b[32m",    /* INFO:  green */
	"\x1b[33m",    /* WARN:  yellow */
	"\x1b[31m",    /* ERROR: red */
	"\x1b[35m",    /* FATAL: magenta */
};

static const char *level_names[] = {
	"TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL", "OFF"
};

void acho_log_set_level(acho_log_level level)
{
	g_level = level;
}

acho_log_level acho_log_get_level(void)
{
	return g_level;
}

void acho_log_set_color(int enabled)
{
	g_color = enabled;
}

void acho_log_set_timestamp(int enabled)
{
	g_timestamp = enabled;
}

const char *acho_log_level_name(acho_log_level level)
{
	if (level < 0 || level > ACHO_LOG_OFF)
		return "UNKNOWN";
	return level_names[level];
}

acho_log_level acho_log_level_from_str(const char *str)
{
	if (!str) return ACHO_LOG_INFO;

	if (!strcasecmp(str, "trace")) return ACHO_LOG_TRACE;
	if (!strcasecmp(str, "debug")) return ACHO_LOG_DEBUG;
	if (!strcasecmp(str, "info"))  return ACHO_LOG_INFO;
	if (!strcasecmp(str, "warn"))  return ACHO_LOG_WARN;
	if (!strcasecmp(str, "warning")) return ACHO_LOG_WARN;
	if (!strcasecmp(str, "error")) return ACHO_LOG_ERROR;
	if (!strcasecmp(str, "fatal")) return ACHO_LOG_FATAL;
	if (!strcasecmp(str, "off"))   return ACHO_LOG_OFF;

	return ACHO_LOG_INFO;
}

static int use_color(void)
{
	if (g_color >= 0)
		return g_color;
	/* auto-detect */
	return isatty(fileno(stderr));
}

void acho_log(acho_log_level level, const char *file, int line,
              const char *fmt, ...)
{
	if (level < g_level)
		return;

	int color = use_color();
	const char *reset = "\x1b[0m";

	/* timestamp */
	if (g_timestamp) {
		time_t now = time(NULL);
		struct tm *tm = localtime(&now);
		char timebuf[32];
		strftime(timebuf, sizeof(timebuf), "%H:%M:%S", tm);

		if (color)
			fprintf(stderr, "\x1b[90m%s\x1b[0m ", timebuf);
		else
			fprintf(stderr, "%s ", timebuf);
	}

	/* level */
	if (color)
		fprintf(stderr, "%s%-5s%s ", level_colors[level], level_names[level], reset);
	else
		fprintf(stderr, "%-5s ", level_names[level]);

	/* source location for debug/trace */
	if (level <= ACHO_LOG_DEBUG) {
		const char *basename = strrchr(file, '/');
		if (!basename)
			basename = strrchr(file, '\\');
		basename = basename ? basename + 1 : file;

		if (color)
			fprintf(stderr, "\x1b[90m%s:%d:\x1b[0m ", basename, line);
		else
			fprintf(stderr, "%s:%d: ", basename, line);
	}

	/* message */
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	fprintf(stderr, "\n");
	fflush(stderr);

	/* fatal = abort */
	if (level == ACHO_LOG_FATAL)
		abort();
}
