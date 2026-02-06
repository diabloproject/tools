#include <acho/acho.h>
#include <acho/log/log.h>
#include <acho/platform/platform.h>
#include <acho/platform/probe.h>

#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>

struct acho_ctx {
	/* config */
	acho_config cfg;

	/* video capture */
	AVFormatContext *vid_fmt_ctx;
	int              vid_stream_idx;
	AVCodecContext  *vid_dec_ctx;

	/* audio capture: mic */
	AVFormatContext *mic_fmt_ctx;
	int              mic_stream_idx;
	AVCodecContext  *mic_dec_ctx;

	/* audio capture: system */
	AVFormatContext *sys_fmt_ctx;
	int              sys_stream_idx;
	AVCodecContext  *sys_dec_ctx;

	/* video encoder */
	AVCodecContext  *vid_enc_ctx;

	/* audio encoder */
	AVCodecContext  *aud_enc_ctx;

	/* output */
	AVFormatContext *out_fmt_ctx;
	AVStream        *out_vid_stream;
	AVStream        *out_aud_stream;

	/* scaling */
	struct SwsContext *sws_ctx;

	/* audio resampling + mixing */
	SwrContext      *swr_mic;
	SwrContext      *swr_sys;

	/* working frames/packets */
	AVFrame  *vid_frame;
	AVFrame  *enc_frame;
	AVFrame  *aud_frame;
	AVPacket *pkt;

	/* timing */
	int64_t  start_time;
	int64_t  vid_pts;
	int64_t  aud_pts;

	/* stats */
	acho_stats stats;
};

/* ── helpers ─────────────────────────────────────────────────────── */

static int open_input_device(AVFormatContext **fmt_ctx, const char *format_name,
                             const char *device, AVDictionary **opts,
                             int *stream_idx, AVCodecContext **dec_ctx,
                             enum AVMediaType type)
{
	const char *type_str = (type == AVMEDIA_TYPE_VIDEO) ? "video" : "audio";
	acho_debug("opening %s input: format=%s device=%s", type_str, format_name, device);

	const AVInputFormat *ifmt = av_find_input_format(format_name);
	if (!ifmt) {
		acho_error("input format '%s' not found", format_name);
		return ACHO_ERR_DEVICE;
	}

	*fmt_ctx = NULL;
	int ret = avformat_open_input(fmt_ctx, device, ifmt, opts);
	if (ret < 0) {
		char errbuf[128];
		av_strerror(ret, errbuf, sizeof(errbuf));
		acho_error("failed to open %s device '%s': %s", type_str, device, errbuf);
		return ACHO_ERR_DEVICE;
	}
	acho_trace("avformat_open_input succeeded for %s", device);

	ret = avformat_find_stream_info(*fmt_ctx, NULL);
	if (ret < 0) {
		acho_error("failed to find stream info for %s device", type_str);
		return ACHO_ERR_DEVICE;
	}
	acho_trace("found %u streams in %s device", (*fmt_ctx)->nb_streams, type_str);

	*stream_idx = -1;
	for (unsigned i = 0; i < (*fmt_ctx)->nb_streams; i++) {
		if ((*fmt_ctx)->streams[i]->codecpar->codec_type == type) {
			*stream_idx = (int)i;
			break;
		}
	}
	if (*stream_idx < 0) {
		acho_error("no %s stream found in device", type_str);
		return ACHO_ERR_DEVICE;
	}
	acho_trace("%s stream found at index %d", type_str, *stream_idx);

	AVCodecParameters *par = (*fmt_ctx)->streams[*stream_idx]->codecpar;
	const AVCodec *dec = avcodec_find_decoder(par->codec_id);
	if (!dec) {
		acho_error("decoder not found for codec_id %d", par->codec_id);
		return ACHO_ERR_DEVICE;
	}
	acho_debug("%s decoder: %s", type_str, dec->long_name ? dec->long_name : dec->name);

	*dec_ctx = avcodec_alloc_context3(dec);
	if (!*dec_ctx) {
		acho_error("failed to allocate decoder context");
		return ACHO_ERR_OOM;
	}

	avcodec_parameters_to_context(*dec_ctx, par);
	ret = avcodec_open2(*dec_ctx, dec, NULL);
	if (ret < 0) {
		char errbuf[128];
		av_strerror(ret, errbuf, sizeof(errbuf));
		acho_error("failed to open %s decoder: %s", type_str, errbuf);
		return ACHO_ERR_DEVICE;
	}

	if (type == AVMEDIA_TYPE_VIDEO) {
		acho_debug("%s capture ready: %dx%d pix_fmt=%d",
		           type_str, (*dec_ctx)->width, (*dec_ctx)->height, (*dec_ctx)->pix_fmt);
	} else {
		acho_debug("%s capture ready: sample_rate=%d channels=%d fmt=%d",
		           type_str, (*dec_ctx)->sample_rate,
		           (*dec_ctx)->ch_layout.nb_channels, (*dec_ctx)->sample_fmt);
	}

	return ACHO_OK;
}

/* ── init ────────────────────────────────────────────────────────── */

acho_ctx *acho_init(const acho_config *cfg, int *err)
{
	int ret;

	acho_info("initializing acho context");
	acho_trace("registering all avdevices");
	avdevice_register_all();

	acho_ctx *ctx = calloc(1, sizeof(*ctx));
	if (!ctx) {
		acho_error("failed to allocate context (out of memory)");
		*err = ACHO_ERR_OOM;
		return NULL;
	}

	ctx->cfg = *cfg;

	/* ── video capture ──────────────────────────────────────────── */
	acho_info("setting up video capture");
	char devstr[512];
	acho_platform_video_device_str(devstr, sizeof(devstr),
	                               cfg->video_device, cfg->capture_x, cfg->capture_y);
	acho_debug("video device string: %s", devstr);

	AVDictionary *vid_opts = NULL;
	char sizebuf[32];
	snprintf(sizebuf, sizeof(sizebuf), "%dx%d", cfg->width, cfg->height);
	av_dict_set(&vid_opts, "video_size", sizebuf, 0);
	char fpsbuf[16];
	snprintf(fpsbuf, sizeof(fpsbuf), "%d", cfg->fps);
	av_dict_set(&vid_opts, "framerate", fpsbuf, 0);
	acho_trace("video options: video_size=%s framerate=%s", sizebuf, fpsbuf);

	ret = open_input_device(&ctx->vid_fmt_ctx, acho_platform_video_format(),
	                        devstr, &vid_opts, &ctx->vid_stream_idx,
	                        &ctx->vid_dec_ctx, AVMEDIA_TYPE_VIDEO);
	av_dict_free(&vid_opts);
	if (ret != ACHO_OK) {
		acho_error("video capture setup failed");
		*err = ret;
		goto fail;
	}
	acho_info("video capture initialized successfully");

	/* ── mic capture ────────────────────────────────────────────── */
	if (cfg->mic_device[0]) {
		acho_info("setting up microphone capture");
		char micstr[512];
		acho_platform_audio_device_str(micstr, sizeof(micstr), cfg->mic_device);
		acho_debug("mic device string: %s", micstr);

		ret = open_input_device(&ctx->mic_fmt_ctx, acho_platform_audio_format(),
		                        micstr, NULL, &ctx->mic_stream_idx,
		                        &ctx->mic_dec_ctx, AVMEDIA_TYPE_AUDIO);
		if (ret != ACHO_OK) {
			acho_error("mic capture setup failed");
			*err = ret;
			goto fail;
		}
		acho_info("microphone capture initialized");
	} else {
		acho_debug("no mic device configured, skipping mic capture");
	}

	/* ── system audio capture ───────────────────────────────────── */
	if (cfg->system_audio_device[0]) {
		acho_info("setting up system audio capture");
		char sysstr[512];
		acho_platform_audio_device_str(sysstr, sizeof(sysstr), cfg->system_audio_device);
		acho_debug("system audio device string: %s", sysstr);

		ret = open_input_device(&ctx->sys_fmt_ctx, acho_platform_audio_format(),
		                        sysstr, NULL, &ctx->sys_stream_idx,
		                        &ctx->sys_dec_ctx, AVMEDIA_TYPE_AUDIO);
		if (ret != ACHO_OK) {
			acho_error("system audio capture setup failed");
			*err = ret;
			goto fail;
		}
		acho_info("system audio capture initialized");
	} else {
		acho_debug("no system audio device configured, skipping");
	}

	/* ── video encoder ──────────────────────────────────────────── */
	acho_info("setting up video encoder");
	const AVCodec *venc = acho_probe_video_encoder(cfg->encoder);
	if (!venc) {
		acho_error("no suitable video encoder found (requested: %s)",
		           cfg->encoder[0] ? cfg->encoder : "auto");
		*err = ACHO_ERR_ENCODER;
		goto fail;
	}
	acho_debug("video encoder selected: %s", venc->long_name ? venc->long_name : venc->name);

	ctx->vid_enc_ctx = avcodec_alloc_context3(venc);
	if (!ctx->vid_enc_ctx) {
		acho_error("failed to allocate video encoder context");
		*err = ACHO_ERR_OOM;
		goto fail;
	}

	ctx->vid_enc_ctx->width       = cfg->width;
	ctx->vid_enc_ctx->height      = cfg->height;
	ctx->vid_enc_ctx->time_base   = (AVRational){1, cfg->fps};
	ctx->vid_enc_ctx->framerate   = (AVRational){cfg->fps, 1};
	ctx->vid_enc_ctx->pix_fmt     = AV_PIX_FMT_YUV420P;
	ctx->vid_enc_ctx->bit_rate    = (int64_t)cfg->video_bitrate * 1000;
	ctx->vid_enc_ctx->gop_size    = cfg->fps * 2;
	ctx->vid_enc_ctx->max_b_frames = 0; /* low latency for streaming */

	acho_trace("video encoder params: %dx%d, %d fps, %d kbps, gop=%d",
	           cfg->width, cfg->height, cfg->fps, cfg->video_bitrate, ctx->vid_enc_ctx->gop_size);

	if (ctx->vid_enc_ctx->codec->id == AV_CODEC_ID_H264) {
		av_opt_set(ctx->vid_enc_ctx->priv_data, "preset", "fast", 0);
		acho_trace("h264 preset set to 'fast'");
	}

	ret = avcodec_open2(ctx->vid_enc_ctx, venc, NULL);
	if (ret < 0) {
		char errbuf[128];
		av_strerror(ret, errbuf, sizeof(errbuf));
		acho_error("failed to open video encoder: %s", errbuf);
		*err = ACHO_ERR_ENCODER;
		goto fail;
	}
	acho_info("video encoder initialized");

	/* ── audio encoder ──────────────────────────────────────────── */
	acho_info("setting up audio encoder");
	const AVCodec *aenc = acho_probe_audio_encoder();
	if (!aenc) {
		acho_error("no suitable audio encoder found");
		*err = ACHO_ERR_ENCODER;
		goto fail;
	}
	acho_debug("audio encoder selected: %s", aenc->long_name ? aenc->long_name : aenc->name);

	ctx->aud_enc_ctx = avcodec_alloc_context3(aenc);
	if (!ctx->aud_enc_ctx) {
		acho_error("failed to allocate audio encoder context");
		*err = ACHO_ERR_OOM;
		goto fail;
	}

	ctx->aud_enc_ctx->sample_rate    = cfg->sample_rate;
	ctx->aud_enc_ctx->bit_rate       = (int64_t)cfg->audio_bitrate * 1000;
	ctx->aud_enc_ctx->time_base      = (AVRational){1, cfg->sample_rate};
	av_channel_layout_default(&ctx->aud_enc_ctx->ch_layout, 2);

	if (aenc->sample_fmts)
		ctx->aud_enc_ctx->sample_fmt = aenc->sample_fmts[0];
	else
		ctx->aud_enc_ctx->sample_fmt = AV_SAMPLE_FMT_FLTP;

	acho_trace("audio encoder params: %d Hz, %d kbps, stereo, fmt=%d",
	           cfg->sample_rate, cfg->audio_bitrate, ctx->aud_enc_ctx->sample_fmt);

	ret = avcodec_open2(ctx->aud_enc_ctx, aenc, NULL);
	if (ret < 0) {
		char errbuf[128];
		av_strerror(ret, errbuf, sizeof(errbuf));
		acho_error("failed to open audio encoder: %s", errbuf);
		*err = ACHO_ERR_ENCODER;
		goto fail;
	}
	acho_info("audio encoder initialized");

	/* ── RTMP output ────────────────────────────────────────────── */
	acho_info("setting up RTMP output");
	char url[ACHO_MAX_URL + ACHO_MAX_KEY + 2];
	snprintf(url, sizeof(url), "%s/%s", cfg->rtmp_url, cfg->stream_key);
	acho_debug("RTMP URL: %s/***", cfg->rtmp_url);  /* don't log stream key */

	ret = avformat_alloc_output_context2(&ctx->out_fmt_ctx, NULL, "flv", url);
	if (ret < 0 || !ctx->out_fmt_ctx) {
		acho_error("failed to allocate output context");
		*err = ACHO_ERR_MUXER;
		goto fail;
	}
	acho_trace("FLV muxer context allocated");

	/* video output stream */
	ctx->out_vid_stream = avformat_new_stream(ctx->out_fmt_ctx, NULL);
	if (!ctx->out_vid_stream) {
		acho_error("failed to create video output stream");
		*err = ACHO_ERR_MUXER;
		goto fail;
	}
	avcodec_parameters_from_context(ctx->out_vid_stream->codecpar, ctx->vid_enc_ctx);
	ctx->out_vid_stream->time_base = ctx->vid_enc_ctx->time_base;
	acho_trace("video output stream created (index=%d)", ctx->out_vid_stream->index);

	/* audio output stream */
	ctx->out_aud_stream = avformat_new_stream(ctx->out_fmt_ctx, NULL);
	if (!ctx->out_aud_stream) {
		acho_error("failed to create audio output stream");
		*err = ACHO_ERR_MUXER;
		goto fail;
	}
	avcodec_parameters_from_context(ctx->out_aud_stream->codecpar, ctx->aud_enc_ctx);
	ctx->out_aud_stream->time_base = ctx->aud_enc_ctx->time_base;
	acho_trace("audio output stream created (index=%d)", ctx->out_aud_stream->index);

	/* open RTMP connection */
	acho_info("connecting to RTMP server...");
	ret = avio_open(&ctx->out_fmt_ctx->pb, url, AVIO_FLAG_WRITE);
	if (ret < 0) {
		char errbuf[128];
		av_strerror(ret, errbuf, sizeof(errbuf));
		acho_error("RTMP connection failed: %s", errbuf);
		*err = ACHO_ERR_RTMP;
		goto fail;
	}
	acho_info("RTMP connection established");

	acho_trace("writing FLV header");
	ret = avformat_write_header(ctx->out_fmt_ctx, NULL);
	if (ret < 0) {
		char errbuf[128];
		av_strerror(ret, errbuf, sizeof(errbuf));
		acho_error("failed to write stream header: %s", errbuf);
		*err = ACHO_ERR_RTMP;
		goto fail;
	}
	acho_debug("stream header written successfully");

	/* ── scaler ─────────────────────────────────────────────────── */
	acho_debug("setting up video scaler: %dx%d (fmt=%d) -> %dx%d (YUV420P)",
	           ctx->vid_dec_ctx->width, ctx->vid_dec_ctx->height, ctx->vid_dec_ctx->pix_fmt,
	           cfg->width, cfg->height);
	ctx->sws_ctx = sws_getContext(
		ctx->vid_dec_ctx->width, ctx->vid_dec_ctx->height, ctx->vid_dec_ctx->pix_fmt,
		cfg->width, cfg->height, AV_PIX_FMT_YUV420P,
		SWS_BILINEAR, NULL, NULL, NULL
	);
	if (!ctx->sws_ctx) {
		acho_error("failed to create video scaler");
		*err = ACHO_ERR_FILTER;
		goto fail;
	}
	acho_trace("video scaler initialized (bilinear)");

	/* ── audio resamplers ───────────────────────────────────────── */
	if (ctx->mic_dec_ctx) {
		acho_debug("setting up mic audio resampler: %d Hz -> %d Hz",
		           ctx->mic_dec_ctx->sample_rate, cfg->sample_rate);
		ret = swr_alloc_set_opts2(&ctx->swr_mic,
			&ctx->aud_enc_ctx->ch_layout, ctx->aud_enc_ctx->sample_fmt, cfg->sample_rate,
			&ctx->mic_dec_ctx->ch_layout, ctx->mic_dec_ctx->sample_fmt, ctx->mic_dec_ctx->sample_rate,
			0, NULL);
		if (ret < 0 || swr_init(ctx->swr_mic) < 0) {
			acho_error("failed to initialize mic resampler");
			*err = ACHO_ERR_RESAMPLE;
			goto fail;
		}
		acho_trace("mic resampler initialized");
	}

	if (ctx->sys_dec_ctx) {
		acho_debug("setting up system audio resampler: %d Hz -> %d Hz",
		           ctx->sys_dec_ctx->sample_rate, cfg->sample_rate);
		ret = swr_alloc_set_opts2(&ctx->swr_sys,
			&ctx->aud_enc_ctx->ch_layout, ctx->aud_enc_ctx->sample_fmt, cfg->sample_rate,
			&ctx->sys_dec_ctx->ch_layout, ctx->sys_dec_ctx->sample_fmt, ctx->sys_dec_ctx->sample_rate,
			0, NULL);
		if (ret < 0 || swr_init(ctx->swr_sys) < 0) {
			acho_error("failed to initialize system audio resampler");
			*err = ACHO_ERR_RESAMPLE;
			goto fail;
		}
		acho_trace("system audio resampler initialized");
	}

	/* ── working buffers ────────────────────────────────────────── */
	acho_trace("allocating working frames and packets");
	ctx->vid_frame = av_frame_alloc();
	ctx->enc_frame = av_frame_alloc();
	ctx->aud_frame = av_frame_alloc();
	ctx->pkt       = av_packet_alloc();
	if (!ctx->vid_frame || !ctx->enc_frame || !ctx->aud_frame || !ctx->pkt) {
		acho_error("failed to allocate working buffers");
		*err = ACHO_ERR_OOM;
		goto fail;
	}

	ctx->enc_frame->format = AV_PIX_FMT_YUV420P;
	ctx->enc_frame->width  = cfg->width;
	ctx->enc_frame->height = cfg->height;
	av_frame_get_buffer(ctx->enc_frame, 0);
	acho_trace("working buffers allocated");

	/* ── timing ─────────────────────────────────────────────────── */
	ctx->start_time = av_gettime_relative();
	ctx->vid_pts    = 0;
	ctx->aud_pts    = 0;
	acho_trace("timing initialized, start_time=%lld", (long long)ctx->start_time);

	acho_info("acho context initialization complete");

	*err = ACHO_OK;
	return ctx;

fail:
	acho_error("initialization failed, cleaning up");
	acho_free(ctx);
	return NULL;
}

/* ── tick ─────────────────────────────────────────────────────────── */

static int encode_and_mux(acho_ctx *ctx, AVCodecContext *enc,
                          AVFrame *frame, AVStream *stream)
{
	const char *type = (enc == ctx->vid_enc_ctx) ? "video" : "audio";

	int ret = avcodec_send_frame(enc, frame);
	if (ret < 0) {
		char errbuf[128];
		av_strerror(ret, errbuf, sizeof(errbuf));
		acho_error("%s encode send_frame failed: %s", type, errbuf);
		return ACHO_ERR_ENCODE;
	}

	while (1) {
		ret = avcodec_receive_packet(enc, ctx->pkt);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
			break;
		if (ret < 0) {
			char errbuf[128];
			av_strerror(ret, errbuf, sizeof(errbuf));
			acho_error("%s encode receive_packet failed: %s", type, errbuf);
			return ACHO_ERR_ENCODE;
		}

		av_packet_rescale_ts(ctx->pkt, enc->time_base, stream->time_base);
		ctx->pkt->stream_index = stream->index;

		ret = av_interleaved_write_frame(ctx->out_fmt_ctx, ctx->pkt);
		if (ret < 0) {
			char errbuf[128];
			av_strerror(ret, errbuf, sizeof(errbuf));
			acho_error("RTMP write failed: %s", errbuf);
			return ACHO_ERR_RTMP;
		}

		acho_trace("%s packet written: size=%d pts=%lld",
		           type, ctx->pkt->size, (long long)ctx->pkt->pts);

		ctx->stats.bytes_sent += ctx->pkt->size;
		av_packet_unref(ctx->pkt);
	}

	return ACHO_OK;
}

int acho_tick(acho_ctx *ctx, acho_stats *stats)
{
	int ret;

	/* ── capture + encode video ─────────────────────────────────── */
	ret = av_read_frame(ctx->vid_fmt_ctx, ctx->pkt);
	if (ret < 0) {
		if (ret == AVERROR_EOF) {
			acho_debug("video capture EOF");
			return ACHO_EOF;
		}
		if (ret == AVERROR(EAGAIN)) {
			/* no frame available yet, not an error */
			goto update_stats;
		}
		char errbuf[128];
		av_strerror(ret, errbuf, sizeof(errbuf));
		acho_error("video read_frame failed: %s", errbuf);
		return ACHO_ERR_CAPTURE;
	}

	if (ctx->pkt->stream_index == ctx->vid_stream_idx) {
		ret = avcodec_send_packet(ctx->vid_dec_ctx, ctx->pkt);
		av_packet_unref(ctx->pkt);
		if (ret < 0) {
			char errbuf[128];
			av_strerror(ret, errbuf, sizeof(errbuf));
			acho_error("video decode send_packet failed: %s", errbuf);
			return ACHO_ERR_CAPTURE;
		}

		ret = avcodec_receive_frame(ctx->vid_dec_ctx, ctx->vid_frame);
		if (ret == 0) {
			av_frame_make_writable(ctx->enc_frame);

			sws_scale(ctx->sws_ctx,
			          (const uint8_t *const *)ctx->vid_frame->data, ctx->vid_frame->linesize,
			          0, ctx->vid_dec_ctx->height,
			          ctx->enc_frame->data, ctx->enc_frame->linesize);

			ctx->enc_frame->pts = ctx->vid_pts++;

			ret = encode_and_mux(ctx, ctx->vid_enc_ctx, ctx->enc_frame, ctx->out_vid_stream);
			if (ret != ACHO_OK) return ret;

			ctx->stats.frames_encoded++;
			acho_trace("video frame encoded: pts=%lld total=%lld",
			           (long long)ctx->enc_frame->pts, (long long)ctx->stats.frames_encoded);
		}
	} else {
		av_packet_unref(ctx->pkt);
	}

	/* ── capture + encode audio (mic) ───────────────────────────── */
	if (ctx->mic_fmt_ctx) {
		ret = av_read_frame(ctx->mic_fmt_ctx, ctx->pkt);
		if (ret == 0 && ctx->pkt->stream_index == ctx->mic_stream_idx) {
			ret = avcodec_send_packet(ctx->mic_dec_ctx, ctx->pkt);
			av_packet_unref(ctx->pkt);

			if (ret == 0) {
				AVFrame *raw = av_frame_alloc();
				if (!raw) return ACHO_ERR_OOM;

				while (avcodec_receive_frame(ctx->mic_dec_ctx, raw) == 0) {
					ctx->aud_frame->format      = ctx->aud_enc_ctx->sample_fmt;
					ctx->aud_frame->nb_samples   = ctx->aud_enc_ctx->frame_size;
					av_channel_layout_copy(&ctx->aud_frame->ch_layout, &ctx->aud_enc_ctx->ch_layout);
					av_frame_get_buffer(ctx->aud_frame, 0);
					av_frame_make_writable(ctx->aud_frame);

					swr_convert(ctx->swr_mic,
					            ctx->aud_frame->data, ctx->aud_frame->nb_samples,
					            (const uint8_t **)raw->data, raw->nb_samples);

					/* TODO: mix system audio into aud_frame here */

					ctx->aud_frame->pts = ctx->aud_pts;
					ctx->aud_pts += ctx->aud_frame->nb_samples;

					ret = encode_and_mux(ctx, ctx->aud_enc_ctx, ctx->aud_frame, ctx->out_aud_stream);
					av_frame_unref(ctx->aud_frame);
					if (ret != ACHO_OK) { av_frame_free(&raw); return ret; }
				}

				av_frame_free(&raw);
			}
		} else {
			av_packet_unref(ctx->pkt);
		}
	}

	/* ── update stats ───────────────────────────────────────────── */
update_stats:;
	int64_t elapsed = av_gettime_relative() - ctx->start_time;
	ctx->stats.uptime_sec    = elapsed / 1000000;
	ctx->stats.fps           = ctx->stats.uptime_sec > 0
	                         ? (double)ctx->stats.frames_encoded / ctx->stats.uptime_sec
	                         : 0.0;
	ctx->stats.video_bitrate = ctx->cfg.video_bitrate;
	ctx->stats.audio_bitrate = ctx->cfg.audio_bitrate;

	if (stats)
		*stats = ctx->stats;

	return ACHO_OK;
}

/* ── stop ─────────────────────────────────────────────────────────── */

int acho_stop(acho_ctx *ctx)
{
	if (!ctx) return ACHO_OK;

	acho_info("stopping stream");

	/* flush video encoder */
	acho_debug("flushing video encoder");
	avcodec_send_frame(ctx->vid_enc_ctx, NULL);
	int flushed_video = 0;
	while (1) {
		int ret = avcodec_receive_packet(ctx->vid_enc_ctx, ctx->pkt);
		if (ret) break;
		av_packet_rescale_ts(ctx->pkt, ctx->vid_enc_ctx->time_base,
		                     ctx->out_vid_stream->time_base);
		ctx->pkt->stream_index = ctx->out_vid_stream->index;
		av_interleaved_write_frame(ctx->out_fmt_ctx, ctx->pkt);
		av_packet_unref(ctx->pkt);
		flushed_video++;
	}
	acho_trace("flushed %d video packets", flushed_video);

	/* flush audio encoder */
	acho_debug("flushing audio encoder");
	avcodec_send_frame(ctx->aud_enc_ctx, NULL);
	int flushed_audio = 0;
	while (1) {
		int ret = avcodec_receive_packet(ctx->aud_enc_ctx, ctx->pkt);
		if (ret) break;
		av_packet_rescale_ts(ctx->pkt, ctx->aud_enc_ctx->time_base,
		                     ctx->out_aud_stream->time_base);
		ctx->pkt->stream_index = ctx->out_aud_stream->index;
		av_interleaved_write_frame(ctx->out_fmt_ctx, ctx->pkt);
		av_packet_unref(ctx->pkt);
		flushed_audio++;
	}
	acho_trace("flushed %d audio packets", flushed_audio);

	acho_debug("writing trailer");
	av_write_trailer(ctx->out_fmt_ctx);
	acho_info("stream stopped successfully");

	return ACHO_OK;
}

/* ── free ─────────────────────────────────────────────────────────── */

void acho_free(acho_ctx *ctx)
{
	if (!ctx) return;

	acho_debug("freeing acho context");

	acho_trace("freeing frames and packets");
	av_frame_free(&ctx->vid_frame);
	av_frame_free(&ctx->enc_frame);
	av_frame_free(&ctx->aud_frame);
	av_packet_free(&ctx->pkt);

	acho_trace("freeing scalers and resamplers");
	sws_freeContext(ctx->sws_ctx);
	swr_free(&ctx->swr_mic);
	swr_free(&ctx->swr_sys);

	acho_trace("freeing codec contexts");
	avcodec_free_context(&ctx->vid_dec_ctx);
	avcodec_free_context(&ctx->mic_dec_ctx);
	avcodec_free_context(&ctx->sys_dec_ctx);
	avcodec_free_context(&ctx->vid_enc_ctx);
	avcodec_free_context(&ctx->aud_enc_ctx);

	if (ctx->out_fmt_ctx) {
		acho_trace("closing RTMP connection");
		if (ctx->out_fmt_ctx->pb)
			avio_closep(&ctx->out_fmt_ctx->pb);
		avformat_free_context(ctx->out_fmt_ctx);
	}

	acho_trace("closing input devices");
	if (ctx->vid_fmt_ctx) avformat_close_input(&ctx->vid_fmt_ctx);
	if (ctx->mic_fmt_ctx) avformat_close_input(&ctx->mic_fmt_ctx);
	if (ctx->sys_fmt_ctx) avformat_close_input(&ctx->sys_fmt_ctx);

	free(ctx);
	acho_debug("acho context freed");
}
