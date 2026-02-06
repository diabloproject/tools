TOOLS_ROOT := ../..
include $(TOOLS_ROOT)/build/Makefile.base

CFLAGS += $(shell pkg-config --cflags libavdevice libavformat libavcodec libavutil)

SRCS    = platform.c probe.c devices.c
OBJS    = $(SRCS:.c=.o)
TARGET  = libacho_platform.a

.DEFAULT_GOAL := all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(AR) rcs $@ $^

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
