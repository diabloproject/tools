TOOLS_ROOT := ..
include $(TOOLS_ROOT)/build/Makefile.base

CFLAGS += $(shell pkg-config --cflags libavdevice libavformat libavcodec \
                                       libavutil libswresample libswscale)

DEPS    = acho/platform acho/log
SRCS    = acho.c acho_err.c config.c
OBJS    = $(SRCS:.c=.o)
TARGET  = libacho.a

.DEFAULT_GOAL := all
all: deps $(TARGET)

$(TARGET): $(OBJS) platform/libacho_platform.a log/libacho_log.a
	cp platform/libacho_platform.a $@
	$(AR) rs $@ $(OBJS)
	$(AR) rs $@ log/*.o

clean:
	rm -f $(OBJS) $(TARGET)
	$(MAKE) -C platform clean
	$(MAKE) -C log clean

.PHONY: all clean
