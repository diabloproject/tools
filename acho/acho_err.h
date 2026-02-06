#ifndef ACHO_ERR_H
#define ACHO_ERR_H

#define ACHO_OK              0
#define ACHO_ERR_CONFIG     -1
#define ACHO_ERR_DEVICE     -2
#define ACHO_ERR_ENCODER    -3
#define ACHO_ERR_MUXER      -4
#define ACHO_ERR_RTMP       -5
#define ACHO_ERR_CAPTURE    -6
#define ACHO_ERR_ENCODE     -7
#define ACHO_ERR_AUDIO_MIX  -8
#define ACHO_ERR_OOM        -9
#define ACHO_ERR_RESAMPLE   -10
#define ACHO_ERR_FILTER     -11

#define ACHO_EOF              1

const char *acho_err_msg(int code);

#endif
