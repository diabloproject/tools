#include <acho/platform/probe.h>

#include <string.h>

static const AVCodec *try_encoder(const char *name)
{
	if (!name || !name[0])
		return NULL;
	return avcodec_find_encoder_by_name(name);
}

const AVCodec *acho_probe_video_encoder(const char *override)
{
	const AVCodec *c;

	/* user override first */
	if (override && override[0]) {
		c = try_encoder(override);
		if (c) return c;
	}

#if defined(__APPLE__)
	static const char *probes[] = {
		"h264_videotoolbox",
		"libx264",
		NULL
	};
#elif defined(_WIN32)
	static const char *probes[] = {
		"h264_nvenc",
		"h264_amf",
		"libx264",
		NULL
	};
#else
	static const char *probes[] = {
		"h264_nvenc",
		"h264_vaapi",
		"libx264",
		NULL
	};
#endif

	for (int i = 0; probes[i]; i++) {
		c = try_encoder(probes[i]);
		if (c) return c;
	}

	/* absolute fallback */
	return avcodec_find_encoder(AV_CODEC_ID_H264);
}

const AVCodec *acho_probe_audio_encoder(void)
{
#if defined(__APPLE__)
	const AVCodec *c = try_encoder("aac_at");
	if (c) return c;
#endif
	const AVCodec *c2 = try_encoder("aac");
	if (c2) return c2;

	return avcodec_find_encoder(AV_CODEC_ID_AAC);
}
