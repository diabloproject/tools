#ifndef ACHO_H
#define ACHO_H

#include <acho/acho_err.h>
#include <acho/config.h>
#include <acho/platform/devices.h>

#include <stdint.h>

typedef struct acho_ctx acho_ctx;

typedef struct acho_stats {
	double   fps;
	int      video_bitrate;   /* kbps */
	int      audio_bitrate;   /* kbps */
	int64_t  frames_encoded;
	int64_t  frames_dropped;
	int64_t  uptime_sec;
	int64_t  bytes_sent;
} acho_stats;

/*
 * Initialize the streaming context.
 * Sets up capture devices, encoders, muxer, and RTMP connection.
 * Returns NULL on failure; call acho_err_msg() for details.
 * On failure, *err is set to the error code.
 */
acho_ctx *acho_init(const acho_config *cfg, int *err);

/*
 * Process one tick: capture frame(s), encode, mux, send.
 * Populates stats if non-NULL.
 * Returns ACHO_OK on success, ACHO_EOF when done, or negative error.
 */
int acho_tick(acho_ctx *ctx, acho_stats *stats);

/*
 * Gracefully stop: flush encoders, close RTMP, release resources.
 * Returns ACHO_OK or negative error.
 */
int acho_stop(acho_ctx *ctx);

/*
 * Free all resources. Call after acho_stop().
 */
void acho_free(acho_ctx *ctx);

#endif
