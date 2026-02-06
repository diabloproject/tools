#ifndef ACHO_PROBE_H
#define ACHO_PROBE_H

#include <libavcodec/avcodec.h>

/*
 * Find the best available H.264 encoder for this platform.
 * If override is non-empty, try that first.
 * Returns the codec, or NULL if nothing works.
 *
 * Probe order:
 *   macOS:   h264_videotoolbox → libx264
 *   Linux:   h264_nvenc → h264_vaapi → libx264
 *   Windows: h264_nvenc → h264_amf → libx264
 */
const AVCodec *acho_probe_video_encoder(const char *override);

/*
 * Find the best available AAC encoder.
 *   macOS:   aac_at → aac
 *   Others:  aac
 */
const AVCodec *acho_probe_audio_encoder(void);

#endif
