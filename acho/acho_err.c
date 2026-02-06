#include <acho/acho_err.h>

const char *acho_err_msg(int code)
{
	switch (code) {
	case ACHO_OK:            return "success";
	case ACHO_ERR_CONFIG:    return "invalid configuration";
	case ACHO_ERR_DEVICE:    return "capture device not found or unavailable";
	case ACHO_ERR_ENCODER:   return "no suitable encoder found";
	case ACHO_ERR_MUXER:     return "failed to initialize muxer";
	case ACHO_ERR_RTMP:      return "RTMP connection failed";
	case ACHO_ERR_CAPTURE:   return "frame capture failed";
	case ACHO_ERR_ENCODE:    return "encoding failed";
	case ACHO_ERR_AUDIO_MIX: return "audio mixing failed";
	case ACHO_ERR_OOM:       return "out of memory";
	case ACHO_ERR_RESAMPLE:  return "audio resampling failed";
	case ACHO_ERR_FILTER:    return "filter initialization failed";
	case ACHO_EOF:           return "stream ended";
	default:                 return "unknown error";
	}
}
