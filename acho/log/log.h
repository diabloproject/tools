#ifndef ACHO_LOG_H
#define ACHO_LOG_H

#include <stdarg.h>

typedef enum {
	ACHO_LOG_TRACE,
	ACHO_LOG_DEBUG,
	ACHO_LOG_INFO,
	ACHO_LOG_WARN,
	ACHO_LOG_ERROR,
	ACHO_LOG_FATAL,
	ACHO_LOG_OFF
} acho_log_level;

/*
 * Set the minimum log level. Messages below this level are discarded.
 * Default is ACHO_LOG_INFO.
 */
void acho_log_set_level(acho_log_level level);

/*
 * Get current log level.
 */
acho_log_level acho_log_get_level(void);

/*
 * Enable/disable colored output. Enabled by default if stderr is a tty.
 */
void acho_log_set_color(int enabled);

/*
 * Enable/disable timestamps. Enabled by default.
 */
void acho_log_set_timestamp(int enabled);

/*
 * Core logging function. Use the macros below instead.
 */
void acho_log(acho_log_level level, const char *file, int line,
              const char *fmt, ...);

/*
 * Logging macros - include file and line info.
 */
#define acho_trace(...) acho_log(ACHO_LOG_TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define acho_debug(...) acho_log(ACHO_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define acho_info(...)  acho_log(ACHO_LOG_INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define acho_warn(...)  acho_log(ACHO_LOG_WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define acho_error(...) acho_log(ACHO_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define acho_fatal(...) acho_log(ACHO_LOG_FATAL, __FILE__, __LINE__, __VA_ARGS__)

/*
 * Log level name string.
 */
const char *acho_log_level_name(acho_log_level level);

/*
 * Parse log level from string (case insensitive).
 * Returns ACHO_LOG_INFO if unrecognized.
 */
acho_log_level acho_log_level_from_str(const char *str);

#endif
