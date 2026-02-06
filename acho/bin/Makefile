TOOLS_ROOT := ../..
include $(TOOLS_ROOT)/build/Makefile.base

CFLAGS += $(shell pkg-config --cflags libavdevice libavformat libavcodec \
                                       libavutil libswresample libswscale)

LDFLAGS += $(shell pkg-config --libs libavdevice libavformat libavcodec \
                                      libavutil libswresample libswscale)

ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -framework CoreFoundation -framework CoreMedia \
               -framework CoreVideo -framework AVFoundation \
               -framework VideoToolbox -framework AudioToolbox \
               -framework Security
else ifeq ($(UNAME_S),Linux)
    LDFLAGS += -lm -lpthread -lX11 -lXext -lpulse
else
    LDFLAGS += -lws2_32 -lole32 -lstrmiids
endif

DEPS    = acho acho/ipc acho/log
SRCS    = main.c daemon.c
OBJS    = $(SRCS:.c=.o)
TARGET  = acho

LIBACHO     = ../libacho.a
LIBACHO_IPC = ../ipc/libacho_ipc.a
LIBACHO_LOG = ../log/libacho_log.a

.DEFAULT_GOAL := all
all: deps $(TARGET)

$(TARGET): $(OBJS) $(LIBACHO) $(LIBACHO_IPC) $(LIBACHO_LOG)
	$(CC) $(OBJS) $(LIBACHO) $(LIBACHO_IPC) $(LIBACHO_LOG) $(LDFLAGS) -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
