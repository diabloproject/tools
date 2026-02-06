#ifndef ACHO_PLATFORM_H
#define ACHO_PLATFORM_H

#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>

/*
 * Returns the platform-specific input format name for screen capture.
 *   macOS:   "avfoundation"
 *   Linux:   "x11grab"
 *   Windows: "gdigrab"
 */
const char *acho_platform_video_format(void);

/*
 * Returns the platform-specific input format name for audio capture.
 *   macOS:   "avfoundation"
 *   Linux:   "pulse"
 *   Windows: "dshow"
 */
const char *acho_platform_audio_format(void);

/*
 * Build the platform-specific device string for video capture.
 * e.g. "1:" on macOS, ":0.0+0,0" on Linux, "desktop" on Windows.
 * Writes into buf, returns buf.
 */
char *acho_platform_video_device_str(char *buf, size_t len,
                                     const char *device,
                                     int x, int y);

/*
 * Build the platform-specific device string for audio capture.
 * Writes into buf, returns buf.
 */
char *acho_platform_audio_device_str(char *buf, size_t len,
                                     const char *device);

#endif
