#include <acho/platform/platform.h>

#include <stdio.h>
#include <string.h>

#if defined(__APPLE__)

const char *acho_platform_video_format(void) { return "avfoundation"; }
const char *acho_platform_audio_format(void) { return "avfoundation"; }

char *acho_platform_video_device_str(char *buf, size_t len,
                                     const char *device,
                                     int x, int y)
{
	(void)x; (void)y;
	if (device && device[0])
		snprintf(buf, len, "%s:", device);
	else
		snprintf(buf, len, "1:");  /* default screen */
	return buf;
}

char *acho_platform_audio_device_str(char *buf, size_t len,
                                     const char *device)
{
	if (device && device[0])
		snprintf(buf, len, "none:%s", device);
	else
		snprintf(buf, len, "none:0");
	return buf;
}

#elif defined(_WIN32)

const char *acho_platform_video_format(void) { return "gdigrab"; }
const char *acho_platform_audio_format(void) { return "dshow"; }

char *acho_platform_video_device_str(char *buf, size_t len,
                                     const char *device,
                                     int x, int y)
{
	(void)device; (void)x; (void)y;
	snprintf(buf, len, "desktop");
	return buf;
}

char *acho_platform_audio_device_str(char *buf, size_t len,
                                     const char *device)
{
	if (device && device[0])
		snprintf(buf, len, "audio=%s", device);
	else
		snprintf(buf, len, "audio=Microphone");
	return buf;
}

#else /* Linux */

const char *acho_platform_video_format(void) { return "x11grab"; }
const char *acho_platform_audio_format(void) { return "pulse"; }

char *acho_platform_video_device_str(char *buf, size_t len,
                                     const char *device,
                                     int x, int y)
{
	const char *display = device && device[0] ? device : ":0.0";
	snprintf(buf, len, "%s+%d,%d", display, x, y);
	return buf;
}

char *acho_platform_audio_device_str(char *buf, size_t len,
                                     const char *device)
{
	if (device && device[0])
		snprintf(buf, len, "%s", device);
	else
		snprintf(buf, len, "default");
	return buf;
}

#endif
